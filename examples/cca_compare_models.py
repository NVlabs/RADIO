# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from collections import defaultdict
from functools import partial
import gc
import math
import os
import hashlib
from PIL import Image
import random
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from sklearn.cross_decomposition import CCA
import torch
from timm.layers import to_2tuple
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms

from einops import rearrange

from datasets import load_dataset_builder, load_dataset
from datasets.distributed import split_dataset_by_node

from common import rank_print, load_model, get_standard_transform, collate, ResizeTransform, PadToSize
from common.phi_s import get_phi_s_matrix

try:
    import wandb
except ImportError:
    wandb = None


LAYER_STATS = dict()


def get_cache_filename(cache_dir: str, model_name: str, which: str):
    hashname = hashlib.md5(model_name.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f'{which}_{hashname}.pth')
    return cache_file


@torch.inference_mode()
def compute_feature_matrix(
    model: nn.Module,
    preproc: nn.Module,
    num_samples: int,
    resolution: int,
    which: str = 'features',
    dataset: str = 'imagenet-1k',
    split: str = 'validation',
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 32,
) -> torch.Tensor:
    transform = transforms.Compose([
        ResizeTransform(to_2tuple(resolution)),
        transforms.CenterCrop(to_2tuple(resolution)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    ds_builder = load_dataset_builder(dataset, trust_remote_code=True)
    ds_builder.download_and_prepare()
    dataset = ds_builder.as_dataset(split=split)
    dataset = dataset.to_iterable_dataset(num_shards=world_size)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))
    rank_print(f'Description: {ds_builder.info.description}')

    model.cuda().eval()
    preproc.cuda()

    num_batches = int(math.ceil(num_samples / batch_size / world_size))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate,
                        pin_memory=True,
                        drop_last=False,
    )

    all_outputs = []

    for i, batch in tqdm(enumerate(loader), total=num_batches, disable=rank > 0):
        if i == num_batches:
            break

        images: torch.Tensor = batch[0][0].to(device='cuda', non_blocking=True)
        images = preproc(images)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images)
            my_output: torch.Tensor = getattr(outputs, which)

            if my_output.ndim == 3:
                d = int(round(math.sqrt(my_output.shape[1])))
                my_output = rearrange(my_output, 'b (h w) c -> b c h w', h=d, w=d)

            all_outputs.append(my_output)

    all_outputs = torch.cat(all_outputs, dim=0)

    if world_size > 1:
        w_outputs = torch.empty(all_outputs.shape[0] * world_size, *all_outputs.shape[1:], dtype=all_outputs.dtype, device=all_outputs.device)
        dist.all_gather_into_tensor(w_outputs, all_outputs)
        all_outputs = w_outputs

    return all_outputs


def get_feature_matrix(
    cache_dir: str, model_name: str, which: str,
    *args, **kwargs
):
    cache_filename = get_cache_filename(cache_dir, model_name, which)

    if os.path.exists(cache_filename):
        features = torch.load(cache_filename, map_location='cpu', weights_only=True)
    else:
        model, preproc, _ = load_model(model_name)
        features = compute_feature_matrix(model, preproc, *args, **kwargs)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(features, cache_filename)

    return features


@torch.inference_mode()
def main(rank: int = 0, world_size: int = 1):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description='Compute SSL embedding rank estimates')
    parser.add_argument('-a', '--model-a', default='radio_v2.5-l')
    parser.add_argument('-b', '--model-b', default='radio_v2.5-b')
    parser.add_argument('--a-res', type=int, default=432)
    parser.add_argument('--b-res', type=int, default=432)
    parser.add_argument('-d', '--dataset', default='imagenet-1k', help='The name of the dataset to classify')
    parser.add_argument('--split', default='validation', help='The dataset split to use.')
    parser.add_argument('-n', default=1000, type=int, help='The number of samples to load')
    parser.add_argument('--which', type=str, choices=['summary', 'features'], default='features')
    parser.add_argument('--cache-dir', default='cca_cache', type=str)
    parser.add_argument('--batch_size', type=int, default=32)

    args, _ = parser.parse_known_args()

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    common_kwargs = dict(
        num_samples=args.n, which=args.which, cache_dir=args.cache_dir, dataset=args.dataset,
        split=args.split, rank=rank, world_size=world_size, batch_size=args.batch_size,
    )
    a_features = get_feature_matrix(model_name=args.model_a, resolution=args.a_res, **common_kwargs)
    b_features = get_feature_matrix(model_name=args.model_b, resolution=args.b_res, **common_kwargs)

    if a_features.ndim == 4:
        min_shape = tuple(min(d1, d2) for d1, d2 in zip(a_features.shape[-2:], b_features.shape[-2:]))
        a_features = F.interpolate(a_features, size=min_shape, mode='bilinear', align_corners=False)
        b_features = F.interpolate(b_features, size=min_shape, mode='bilinear', align_corners=False)
        a_features = rearrange(a_features, 'b c h w -> (b h w) c')
        b_features = rearrange(b_features, 'b c h w -> (b h w) c')

    assert a_features.ndim == 2

    if rank > 0:
        return

    n_samples = 10000
    if a_features.shape[0] > n_samples:
        weights = torch.full((a_features.shape[0],), 1.0, device=a_features.device)
        sel_idxs = torch.multinomial(weights, num_samples=n_samples, replacement=False)
        a_features = a_features[sel_idxs]
        b_features = b_features[sel_idxs]

    a_mean, a_phi_s = get_phi_s_matrix(a_features)
    b_mean, b_phi_s = get_phi_s_matrix(b_features)

    a_features = (a_features - a_mean) @ a_phi_s.T
    b_features = (b_features - b_mean) @ b_phi_s.T

    a_features = a_features.cpu().numpy()
    b_features = b_features.cpu().numpy()

    n_components = min(a_features.shape[1], b_features.shape[1])
    cca = CCA(scale=False, n_components=n_components)
    cca.fit(a_features, b_features)

    a_cca, b_cca = cca.transform(a_features, b_features)

    comp_corr = np.corrcoef(a_cca.T, b_cca.T)
    comp_corr = np.diag(comp_corr)

    print('Correlates:', comp_corr)

    score = cca.score(a_features, b_features)
    print('Score:', score)

    b_pred = cca.predict(a_features)
    pred_sq_err = ((b_pred - b_features) ** 2).mean()

    print('Pred MSE:', pred_sq_err)

    pass


if __name__ == '__main__':
    rank = 0
    world_size = 1

    # if 'WORLD_SIZE' in os.environ:
    #     dist.init_process_group(backend='nccl')
    #     rank = dist.get_rank()
    #     world_size = dist.get_world_size()

    main(rank, world_size)