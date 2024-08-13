# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from collections import defaultdict
from hashlib import sha256
import math
import os
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.hub as hub
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision.transforms.v2 as transforms

from datasets import load_dataset_builder, load_dataset
from datasets.iterable_dataset import DistributedConfig
from datasets.distributed import split_dataset_by_node

from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

from common import collate, round_up, get_standard_transform, get_rank, get_world_size, rank_print, load_model


def main(rank: int = 0, world_size: int = 1):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    device = torch.device('cuda', local_rank)
    parser = argparse.ArgumentParser(description='ZeroShot Classification Demo')

    parser.add_argument('-v', '--model-version', default='radio_v2',
                        help='Which radio model to load.'
    )
    parser.add_argument('-a', '--adaptor-name', default='', help='Which head to use')
    parser.add_argument('-r', '--resolution', nargs='+', type=int, default=None,
                        help='The input image resolution.'
                             ' If one value is specified, the shortest dimension is resized to this.'
                             ' If two, the image is center cropped.'
                             ' If not specified, center cropped 378px is used.'
                             ' Default: The RADIO model\'s preferred resolution.'
    )
    parser.add_argument('-d', '--dataset', default='imagenet-1k',
                        help='The name of the dataset to classify'
    )
    parser.add_argument('--split', default='validation',
                        help='The dataset split to use.'
    )
    parser.add_argument('--resize-multiple', type=int, default=16,
                        help='Resize images with dimensions a multiple of this value.'
                             ' This should be equal to the patch size of a ViT (e.g. RADIOv1)'
    )
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The batch size. If the input is variable sized, then this argument becomes a maximum.'
    )
    parser.add_argument('-w', '--workers', default=8, type=int,
                        help='The number of data loader workers to use per GPU'
    )
    parser.add_argument('--vitdet-window-size', default=None, type=int,
                        help='The ViTDet window size to use, if desired. Default: Global attention'
    )
    parser.add_argument('--use-local-lib', default=False, action='store_true',
                        help='Use the library locally, instead of through TorchHub'
    )
    parser.add_argument('--force-reload', default=False, action='store_true',
                        help='Force reload RADIO library'
    )
    parser.add_argument('--amp', default=False, action='store_true', help='Run in amp')
    parser.add_argument('--torchhub-repo',
                        help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )
    parser.add_argument('-o', '--output', type=str, default='',
                        help='Output the histogram to the specified csv file')

    args, _ = parser.parse_known_args()

    rank_print('Loading model...')
    model, preprocessor, info = load_model(args.model_version, adaptor_names=args.adaptor_name, return_spatial_features=False,
                                           vitdet_window_size=args.vitdet_window_size, force_reload=args.force_reload,
                                           torchhub_repo=args.torchhub_repo)
    model.to(device=device).eval()
    rank_print('Done')

    rank_print('Loading dataset...')
    ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    ds_builder.download_and_prepare()
    num_examples = min(ds_builder.info.splits[args.split].num_examples, 1000)

    if args.resolution is None:
        args.resolution = (model.preferred_resolution.height, model.preferred_resolution.width)

    if args.resize_multiple is None:
        args.resize_multiple = getattr(model, 'min_resolution_step', model.patch_size)

    transform = get_standard_transform(args.resolution, args.resize_multiple, preprocessor=preprocessor)
    dataset = ds_builder.as_dataset(split=args.split)
    dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, args.workers))
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.shuffle(42)
    dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=args.workers > 0,
                        drop_last=False,
    )
    num_steps = round_up(num_examples, args.batch_size * world_size)
    rank_print('Done')
    rank_print(f'Description: {ds_builder.info.description}')

    all_feats = []

    mean_feats = None
    cov_feats = None
    cov_samples = torch.tensor(0, dtype=torch.float64, device='cuda')

    num_processed = 0
    with torch.no_grad(), tqdm(total=num_examples, disable=rank > 0) as t:
        for batches in loader:
            for images, targets in batches:
                images = images.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)

                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                    output = model(images)
                    if args.adaptor_name:
                        output = output[args.adaptor_name]
                    features = output[1].flatten(0, 1)
                    all_feats.append(features)

                    cov_samples += features.shape[0]

                    sample_mean_feats = features.sum(dim=0, dtype=torch.float64)
                    if mean_feats is None:
                        mean_feats = sample_mean_feats
                    else:
                        mean_feats += sample_mean_feats

                    curr_mean = mean_feats / cov_samples
                    zc_features = (features - curr_mean).double()

                    outer = zc_features.T @ zc_features
                    if cov_feats is None:
                        cov_feats = outer.double()
                    else:
                        cov_feats += outer.double()

            num_processed += world_size * args.batch_size

            t.update(world_size * args.batch_size)
            if num_processed >= num_examples:
                break

    del model
    del preprocessor
    all_feats = torch.cat(all_feats)

    flat_all_feats = all_feats.flatten()
    rank_global_mean = torch.mean(flat_all_feats)
    global_mean = rank_global_mean.clone()
    if world_size > 1:
        dist.all_reduce(global_mean, dist.ReduceOp.AVG)

        dist.reduce(mean_feats, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(cov_feats, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(cov_samples, dst=0, op=dist.ReduceOp.SUM)

    mean_feats /= cov_samples
    cov_feats /= cov_samples - 1

    var_feats = cov_feats.diag()
    idx_min_mean = torch.argmin(mean_feats)
    idx_max_mean = torch.argmax(mean_feats)
    idx_min_var = torch.argmin(var_feats)
    idx_max_var = torch.argmax(var_feats)

    hist_idxs = [None, idx_min_mean, idx_max_mean, idx_min_var, idx_max_var]

    op_data = []

    for feat_idx in hist_idxs:
        if feat_idx is None:
            curr_feats = flat_all_feats
            curr_mean = global_mean
        else:
            curr_feats = all_feats[:, feat_idx]
            curr_mean = mean_feats[feat_idx]

        zm_curr_feats = curr_feats - curr_mean
        var_curr_feats = torch.sum(zm_curr_feats ** 2, dtype=torch.float64)
        num_curr_feats = torch.tensor(zm_curr_feats.shape[0], dtype=torch.float64, device=zm_curr_feats.device)
        if world_size > 1:
            dist.all_reduce(var_curr_feats, dist.ReduceOp.SUM)
            dist.all_reduce(num_curr_feats, dist.ReduceOp.SUM)

        var_curr_feats = var_curr_feats / (num_curr_feats - 1)
        std_curr_feats = torch.sqrt(var_curr_feats)

        std_max = 5
        valid_mask = (curr_feats > (curr_mean - std_max * std_curr_feats)) & (curr_feats < (curr_mean + std_max * std_curr_feats))

        valid_curr_feats = curr_feats[valid_mask]

        min_val = torch.amin(valid_curr_feats)
        max_val = torch.amax(valid_curr_feats)

        if world_size > 1:
            dist.all_reduce(min_val, dist.ReduceOp.MIN)
            dist.all_reduce(max_val, dist.ReduceOp.MAX)

        num_buckets = 100
        bucket_size = (max_val - min_val) / num_buckets

        histogram = torch.zeros(num_buckets, dtype=torch.int64, device=min_val.device)

        bucket_idxs = torch.floor_divide(valid_curr_feats - min_val, bucket_size).long().clamp_(0, num_buckets - 1)

        histogram.scatter_add_(0, bucket_idxs, torch.ones_like(bucket_idxs, dtype=histogram.dtype))

        if world_size > 1:
            dist.reduce(histogram, 0, op=dist.ReduceOp.SUM)

        if rank != 0:
            continue

        total_weight = histogram.sum() * bucket_size
        histogram = histogram.double() / total_weight

        histogram = histogram.cpu()

        hist_bin_edges = torch.linspace(min_val.item(), max_val.item(), num_buckets + 1)
        hist_vals = torch.cat([
            histogram,
            torch.zeros_like(histogram[:1]),
        ]).cpu()

        op_data.append(torch.stack([hist_bin_edges, hist_vals], dim=1))

    if rank == 0:
        header = 'Global Boundary,Global Weight,MinMean Boundary,MinMean Weight,MaxMean Boundary,MaxMean Weight,MinVar Boundary,MinVar Weight,MaxVar Boundary,MaxVar Weight'

        op_data = torch.cat(op_data, dim=1)

        with open(args.output, 'w') as fd:
            fd.write(f'{header}\n')
            for row in op_data:
                s_row = ','.join(f'{c.item():.04f}' for c in row)
                fd.write(f'{s_row}\n')

        output_fname, output_fext = os.path.splitext(args.output)
        cov_path = f'{output_fname}_cov.pth.tar'
        torch.save(dict(cov=cov_feats, mean=mean_feats), cov_path)


if __name__ == '__main__':
    rank = 0
    world_size = 1

    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    main(rank, world_size)
