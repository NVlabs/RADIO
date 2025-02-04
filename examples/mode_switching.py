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
from PIL import Image
import random
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
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

try:
    import wandb
except ImportError:
    wandb = None


LAYER_STATS = dict()


@torch.inference_mode()
def main(rank: int = 0, world_size: int = 1):
    '''
    Computes the RankMe (http://arxiv.org/abs/2210.02885) and LiDAR (http://arxiv.org/abs/2312.04000)
    estimates of the rank of the produced embeddings. While RADIO doesn't train in a multi-view setting
    which is an assumption of LiDAR, the metric does integrate an important concept of the invariance of the
    summary features to different view/augmentations of the same image.
    '''

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    cv2.setNumThreads(1)

    device = torch.device('cuda', local_rank)
    parser = argparse.ArgumentParser(description='Compute SSL embedding rank estimates')
    parser.add_argument('-v', '--version', default='radio_v2.1')
    parser.add_argument('-d', '--dataset', default='imagenet-1k',
                        help='The name of the dataset to classify'
    )
    parser.add_argument('--split', default='validation',
                        help='The dataset split to use.'
    )
    parser.add_argument('--teacher', default='dino_v2', type=str, help='Which teacher to test mode switching against')
    parser.add_argument('-n', default=100, type=int, help='The number of samples to load')
    parser.add_argument('--name', default='', type=str)

    args, _ = parser.parse_known_args()

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    rank_print('Loading RADIO...')
    radio, radio_preprocessor, radio_info = load_model(args.version, adaptor_names=args.teacher.split(',')[0])
    rank_print('Done')

    rank_print(f'Loading {args.teacher}...')

    teacher_args = next(t for t in radio_info.checkpoint['args'].teachers if args.teacher.startswith(t['name']))
    teacher_name = teacher_args['name']

    multi_res_teacher = False
    embed_teacher = False
    teacher_input_size = teacher_args['input_size']
    if teacher_args['type'] == 'open_clip':
        teacher, teacher_preprocessor, _ = load_model(f"open_clip,{teacher_args['model']},{teacher_args['pretrained']}")
    elif teacher_args['type'] == 'openai_clip':
        teacher, teacher_preprocessor, _ = load_model(f"openai_clip,{teacher_args['model']}")
    elif teacher_args['type'] == 'dino_v2':
        multi_res_teacher = True
        teacher, teacher_preprocessor, _ = load_model(teacher_args['model'])
    elif teacher_args['type'] == 'sam':
        teacher, teacher_preprocessor, _ = load_model(args.teacher)
        embed_teacher = True
        multi_res_teacher = True
    teacher_patch_size = teacher.patch_size
    rank_print('Done')

    radio.cuda().eval()
    radio_preprocessor.cuda().eval()
    teacher.cuda().eval()
    teacher_preprocessor.cuda().eval()

    first_res = int(math.floor(196 / radio.patch_size) * radio.patch_size)
    final_res = int(math.ceil(1536 / radio.patch_size) * radio.patch_size)
    resolutions = list(range(first_res, final_res + radio.patch_size, radio.patch_size))

    # Get the subset of resolutions for this rank
    resolutions = resolutions[rank::world_size]

    transform_teacher = transforms.Compose([
        ResizeTransform(to_2tuple(teacher_input_size), resize_multiple=teacher_patch_size),
        transforms.CenterCrop(to_2tuple(teacher_input_size)),
        PadToSize(to_2tuple(teacher_input_size)) if embed_teacher else nn.Identity(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    transform = transforms.Compose([
        ResizeTransform([512, 512], resize_multiple=radio.patch_size),
        transforms.CenterCrop([512, 512]),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    def update_resolution(tx, res: int):
        tx.transforms[0].size = [res, res]
        tx.transforms[1] = transforms.CenterCrop([res, res])

    ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    ds_builder.download_and_prepare()
    dataset = ds_builder.as_dataset(split=args.split)
    dataset = dataset.to_iterable_dataset(num_shards=world_size)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.map(lambda ex: dict(image=ex['image'], label=torch.as_tensor(ex['label'], dtype=torch.int64)))
    rank_print(f'Description: {ds_builder.info.description}')

    bins = dict()

    for res in tqdm(resolutions, desc="Resolutions", disable=rank > 0, position=0, leave=True):
        if multi_res_teacher:
            curr_teacher_res = res * teacher_patch_size // radio.patch_size
            update_resolution(transform_teacher, curr_teacher_res)
        else:
            curr_teacher_res = teacher_input_size

        update_resolution(transform, res)

        teacher_features = []
        student_features = []

        cov_bb: torch.Tensor = None
        cov_bb_ct = 0

        for i, sample in tqdm(enumerate(dataset), total=args.n, disable=rank > 0, desc=f'{res}', position=1, leave=False):
            if i == args.n:
                break

            image = transform(sample['image'])
            image_dv2 = transform_teacher(sample['image'])
            image_dv2 = image_dv2.unsqueeze(0).cuda()
            input_dv2 = teacher_preprocessor(image_dv2)
            _, curr_teacher_features = teacher(input_dv2)

            ncol = (curr_teacher_res if not embed_teacher else teacher_input_size) // teacher_patch_size
            curr_teacher_features = rearrange(curr_teacher_features, 'b (h w) d -> b d h w', h=ncol, w=ncol)

            if embed_teacher:
                ncol = curr_teacher_res // teacher_patch_size
                curr_teacher_features = curr_teacher_features[..., :ncol, :ncol]

            image = image.unsqueeze(0).cuda()
            input_radio = radio_preprocessor(image)
            r_out = radio(input_radio)
            _, bb_feat = r_out['backbone']
            _, curr_student_features = r_out[teacher_name]

            cov_curr = bb_feat.flatten(0, 1)
            cov_bb_ct += cov_curr.shape[0]
            if cov_bb is None:
                cov_bb = cov_curr.T @ cov_curr
            else:
                cov_bb.addmm_(cov_curr.T, cov_curr)

            ncol = int(round(math.sqrt(curr_student_features.shape[1])))
            curr_student_features = rearrange(curr_student_features, 'b (h w) d -> b d h w', h=ncol, w=ncol)

            teacher_features.append(curr_teacher_features)
            student_features.append(curr_student_features)

            del r_out
            del bb_feat
            del curr_student_features

        teacher_features = torch.cat(teacher_features)
        student_features = torch.cat(student_features)

        cov_bb /= cov_bb_ct - 1

        # if rank == 0:
        #     print(f'Backbone Feature Variance:\n{cov_bb.diag()}')

        res_matched_teacher = teacher_features
        if student_features.shape != teacher_features.shape:
            res_matched_teacher = F.interpolate(teacher_features, size=student_features.shape[-2:], mode='bilinear', align_corners=True)

        stud_variance = student_features.var()
        teacher_variance = teacher_features.var()

        cos_error = 1 - F.cosine_similarity(student_features, res_matched_teacher, dim=1).mean()
        fidelity = teacher_variance / F.mse_loss(student_features, res_matched_teacher, reduction='mean')

        bins[res] = (cos_error.item(), fidelity.item(), stud_variance.item(), teacher_variance.item())

    if dist.is_initialized():
        all_bins = [None for _ in range(world_size)]
        dist.all_gather_object(all_bins, bins)
        new_bins = []
        for rank_bins in all_bins:
            for res, stats in rank_bins.items():
                new_bins.append((res, stats))
        new_bins.sort(key=lambda t: t[0])
        bins = {k: v for k, v in new_bins}

    if rank > 0:
        return

    if not args.name and not os.path.isfile(args.version):
        args.name = args.version
    suffix = f'_{args.name.replace("_", "-")}' if args.name else ''
    f_teacher_name = teacher_name.replace('_', '-')

    with open(f'mode-switching_{f_teacher_name}{suffix}.csv', 'w') as fd:
        fd.write('Resolution,Cos Fidelity,Fidelity,Pred Variance,Teacher Variance\n')
        for res, t in bins.items():
            parts = ','.join(f'{v:.4f}' for v in t)
            fd.write(f'{res},{parts}\n')


if __name__ == '__main__':
    rank = 0
    world_size = 1

    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    main(rank, world_size)
