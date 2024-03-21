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

from common import rank_print, load_model, get_standard_transform, collate, ResizeTransform

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
    parser.add_argument('-d', '--dataset', default='imagenet-1k',
                        help='The name of the dataset to classify'
    )
    parser.add_argument('--split', default='validation',
                        help='The dataset split to use.'
    )
    parser.add_argument('-n', default=100, type=int, help='The number of samples to load')

    args, _ = parser.parse_known_args()

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    rank_print('Loading RADIO...')
    radio, radio_preprocessor, _ = load_model('radio_v2.1', adaptor_names='dino_v2')
    radio_vitdet, _, _ = load_model('radio_v2.1', vitdet_window_size=16, adaptor_names='dino_v2')
    rank_print('Done')

    rank_print('Loading DINOv2...')
    dinov2, dinov2_preprocessor, _ = load_model('dinov2_vitg14_reg')
    rank_print('Done')

    radio.cuda().eval()
    radio_preprocessor.cuda().eval()
    radio_vitdet.cuda().eval()
    dinov2.cuda().eval()
    dinov2_preprocessor.cuda().eval()

    resolutions = list(range(224, 1024 + 16, 16))

    transform = transforms.Compose([
        ResizeTransform([518, 518], resize_multiple=14),
        transforms.CenterCrop([518, 518]),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    dataset = ds_builder.as_dataset(split=args.split)
    dataset = dataset.to_iterable_dataset(num_shards=world_size)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))
    rank_print(f'Description: {ds_builder.info.description}')

    bins = dict()

    dino_features = []

    for i, sample in tqdm(enumerate(dataset), desc='DINOv2 Features', total=args.n):
        if i == args.n:
            break
        image = sample['image'].unsqueeze(0).cuda()
        input_dino = dinov2_preprocessor(image)
        _, features = dinov2(input_dino)

        ncol = int(round(math.sqrt(features.shape[1])))
        features = rearrange(features, 'b (h w) d -> b d h w', h=ncol, w=ncol)

        dino_features.append(features)

    dino_features = torch.cat(dino_features)

    for res in tqdm(resolutions, desc="Resolutions"):
        transform.transforms[0].size = [res, res]
        transform.transforms[1] = transforms.CenterCrop([res, res])

        res_features = []

        for i, sample in tqdm(enumerate(dataset), total=args.n, leave=None, desc=f'{res}'):
            if i == args.n:
                break

            image = sample['image'].unsqueeze(0).cuda()
            input_radio = radio_preprocessor(image)
            _, features = radio(input_radio)['dino_v2']

            ncol = int(round(math.sqrt(features.shape[1])))
            features = rearrange(features, 'b (h w) d -> b d h w', h=ncol, w=ncol)

            res_features.append(features)

        res_features = torch.cat(res_features)

        res_matched = F.interpolate(res_features, size=dino_features.shape[-2:], mode='bilinear', align_corners=True)

        cos_error = 1 - F.cosine_similarity(res_matched, dino_features, dim=1).mean()
        mse_error = F.mse_loss(res_matched, dino_features, reduction='mean')

        bins[res] = (cos_error.item(), mse_error.item())

    with open('mode_switching_results.csv', 'w') as fd:
        fd.write('Resolution,Cos Error, MSE Error\n')
        for res, (cos_error, mse_error) in bins.items():
            fd.write(f'{res},{cos_error:.4f},{mse_error:.4f}\n')


if __name__ == '__main__':
    rank = 0
    world_size = 1

    # if 'WORLD_SIZE' in os.environ:
    #     dist.init_process_group(backend='nccl')
    #     rank = dist.get_rank()
    #     world_size = dist.get_world_size()

    main(rank, world_size)
