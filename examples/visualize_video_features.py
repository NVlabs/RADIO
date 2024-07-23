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
from torchvision.io import read_video, write_video

from einops import rearrange

from datasets import load_dataset_builder, load_dataset
from datasets.distributed import split_dataset_by_node

from common import rank_print, load_model, get_standard_transform, collate
from radio.input_conditioner import InputConditioner
from visualize_features import get_robust_pca, get_pca_map


@torch.inference_mode()
def main(rank: int = 0, world_size: int = 1):
    '''
    Computes the PCA features for every frame in a supplied video and renders them into a new video.
    '''

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    cv2.setNumThreads(1)

    device = torch.device('cuda', local_rank)
    parser = argparse.ArgumentParser(description='Visual Model Features in Video')
    parser.add_argument('-v', '--model-version', default='radio_v2',
                        help='Which radio model to load.'
    )
    parser.add_argument('--video', type=str, required=True,
                        help='Path to the video')
    parser.add_argument('--output', type=str, required=True,
                        help='Where to store the output video')
    parser.add_argument('-r', '--resolution', nargs='+', type=int, default=None,
                        help='The input image resolution.'
                             ' If one value is specified, the shortest dimension is resized to this.'
                             ' If two, the image is center cropped.'
                             ' If not specified, center cropped 378px is used.'
                             ' Default: The RADIO model\'s preferred resolution.'
    )
    parser.add_argument('--max-dim', default=False, action='store_true', help='Resize the max dimension to the specified resolution')
    parser.add_argument('--resize-multiple', type=int, default=16,
                        help='Resize images with dimensions a multiple of this value.'
                             ' This should be equal to the patch size of a ViT (e.g. RADIOv1)'
    )
    parser.add_argument('--vitdet-window-size', default=None, type=int, help='Enable ViTDet at the specific window size')
    parser.add_argument('--adaptor-name', default=None, type=str, help='Generate features from a teacher adaptor')
    parser.add_argument('--patch-size', default=16, type=int, help='The model patch size')
    parser.add_argument('--torchhub-repo',
                        help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )
    parser.add_argument('--side-by-side', default=False, action='store_true',
                        help='Render the original frame and the PCA frame side-by-side')
    parser.add_argument('--audio', default=False, action='store_true',
                        help='Encode the audio in the output video')
    parser.add_argument('--video-codec', default='libx264', type=str, help='The video codec to use')
    parser.add_argument('--batch-size', type=int, default=16, help='The processing batch size')
    parser.add_argument('--force-reload', default=False, action='store_true', help='Reload the torch.hub codebase')

    args, _ = parser.parse_known_args()

    rank_print(f'Loading model: "{args.model_version}", ViTDet: {args.vitdet_window_size}, Adaptor: "{args.adaptor_name}", Resolution: {args.resolution}, Max: {args.max_dim}...')
    model, preprocessor, info = load_model(args.model_version, vitdet_window_size=args.vitdet_window_size, adaptor_names=args.adaptor_name,
                                           torchhub_repo=args.torchhub_repo, force_reload=args.force_reload)
    model.to(device=device).eval()
    if isinstance(preprocessor, nn.Module):
        preprocessor.to(device).eval()
    rank_print('Done')

    if args.resolution is None:
        args.resolution = (model.preferred_resolution.height, model.preferred_resolution.width)

    patch_size = getattr(model, 'patch_size', None) or args.patch_size

    if args.resize_multiple is None:
        args.resize_multiple = getattr(model, 'min_resolution_step', patch_size)

    transform = get_standard_transform(args.resolution, args.resize_multiple, max_dim=args.max_dim,
                                       pad_mean=preprocessor.norm_mean if isinstance(preprocessor, InputConditioner) else None,)

    input_video = read_video(args.video, output_format='TCHW')
    input_frames = input_video[0]

    all_features = []
    tx_frames = []

    batch_size = args.batch_size
    for b in tqdm(range(0, len(input_frames), batch_size)):
        curr_frames = input_frames[b:b+batch_size]
        curr_frames = transform(curr_frames)

        tx_frames.append(curr_frames)

        curr_frames = curr_frames.cuda()

        with torch.autocast(device.type, dtype=torch.bfloat16):
            p_frames = preprocessor(curr_frames)

            output = model(p_frames)
            if args.adaptor_name:
                output = output[args.adaptor_name].features
            else:
                output = output[1]

        num_rows = curr_frames.shape[-2] // patch_size
        num_cols = curr_frames.shape[-1] // patch_size

        output = rearrange(output, 'b (h w) c -> b h w c', h=num_rows, w=num_cols).float()

        all_features.append(output.cpu())

    all_features = torch.cat(all_features)
    tx_frames = torch.cat(tx_frames)

    num_keyframes = 30
    kf_stride = max(all_features.shape[0] // num_keyframes, 1)

    # We'll use this to compute the PCA
    sub_features = all_features[::kf_stride]
    pca_stats = get_robust_pca(sub_features.flatten(0, 2))

    output_frames = []
    for raw_frame, features in zip(tx_frames, all_features):
        pca_features = torch.from_numpy(get_pca_map(features, raw_frame.shape[-2:], pca_stats=pca_stats, interpolation='bilinear'))

        if args.side_by_side:
            raw_frame = raw_frame.permute(1, 2, 0).cpu()
            pca_features = torch.cat((raw_frame, pca_features), dim=1)

        pca_features = pca_features.mul_(255).byte()
        output_frames.append(pca_features)

    output_frames = torch.stack(output_frames)
    extra_args = dict()
    if args.audio:
        extra_args.update(dict(
            audio_array=input_video[1],
            audio_fps=input_video[2]['audio_fps'],
        ))

    dirname = os.path.dirname(args.output)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    options = {
        'crf': '18',  # Lower CRF for better quality
        'preset': 'slow',  # Use a slower preset for better compression efficiency
        'profile': 'high',  # Use high profile for advanced features
    }
    write_video(args.output, output_frames, input_video[2]['video_fps'], video_codec=args.video_codec, options=options, **extra_args)


if __name__ == '__main__':
    main()
