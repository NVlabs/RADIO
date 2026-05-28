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

from einops import rearrange

from datasets import load_dataset_builder, load_dataset
from datasets.distributed import split_dataset_by_node

from common import rank_print, load_model, get_standard_transform, collate
from radio.input_conditioner import InputConditioner
from radio.radio_model import RadioOutput

try:
    import wandb
except ImportError:
    wandb = None


LAYER_STATS = dict()


def parse_int_list(string):
    """Parse a comma-separated list of integers and sort them."""
    try:
        return sorted([int(i) for i in string.split(',')])
    except ValueError:
        raise argparse.ArgumentTypeError(f"{string} is not a valid list of integers")


@torch.no_grad()
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
    parser.add_argument('-v', '--model-version', default='radio_v2',
                        help='Which radio model to load.'
    )
    parser.add_argument('-d', '--dataset', default='detection-datasets/coco',
                        help='The name of the dataset to classify'
    )
    parser.add_argument('--split', default='train',
                        help='The dataset split to use.'
    )
    parser.add_argument('-n', default=128, type=int, help='The number of samples to load')
    parser.add_argument('-r', '--resolution', nargs='+', type=int, default=None,
                        help='The input image resolution.'
                             ' If one value is specified, the shortest dimension is resized to this.'
                             ' If two, the image is center cropped.'
                             ' If not specified, center cropped 378px is used.'
                             ' Default: The RADIO model\'s preferred resolution.'
    )
    parser.add_argument('--max-dim', default=False, action='store_true', help='Resize the max dimension to the specified resolution')
    parser.add_argument('--resize-multiple', type=int, default=None,
                        help='Resize images with dimensions a multiple of this value.'
                             ' This should be equal to the patch size of a ViT (e.g. RADIOv1)'
    )
    parser.add_argument('--batch-size', type=int, default=16,
                        help='The batch size. If the input is variable sized, then this argument becomes a maximum.'
    )
    parser.add_argument('--workers', default=8, type=int, help='Number of loader workers to use')
    parser.add_argument('--vitdet-window-size', default=None, type=int, help='Enable ViTDet at the specific window size')
    parser.add_argument('--output-dir', default='vis_denoise', type=str)
    parser.add_argument('--adaptor-name', default=None, type=str, help='Generate features from a teacher adaptor')
    parser.add_argument('--patch-size', default=None, type=int, help='The model patch size')
    parser.add_argument('--torchhub-repo',
                        help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )
    parser.add_argument('--intermediates',
                        type=parse_int_list,
                        help='Visualize intermediate layers, specified as a comma-separated list of integers (e.g., 1,2,3,4,5)'
    )
    parser.add_argument('--intermediate-aggregation',
                        default='sparse',
                        type=str,
                        help='How to aggregate intermediate layers',
                        choices=['sparse', 'dense'])
    parser.add_argument('--interpolation', default='bilinear', type=str, help='Interpolation mode')
    parser.add_argument('--skip', default=0, type=int, help='Skip the first N components')
    parser.add_argument('--neck', default=None, type=str, help='Generate features from specified neck')
    parser.add_argument('--animate-radio1d', action='store_true', help='Create animated PNGs sweeping radio1d_size')
    parser.add_argument('--radio1d-start', default=1, type=int, help='Starting radio1d_size for animation')
    parser.add_argument('--radio1d-end', default=512, type=int, help='Ending radio1d_size for animation')
    parser.add_argument('--radio1d-step', default=32, type=int, help='Step size for radio1d_size sweep')
    parser.add_argument('--animation-duration', default=100, type=int, help='Duration of each frame in ms')

    args, _ = parser.parse_known_args()

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    rank_print(f'Loading model: "{args.model_version}", ViTDet: {args.vitdet_window_size}, Adaptor: "{args.adaptor_name}", Resolution: {args.resolution}, Max: {args.max_dim}...')
    if os.path.isdir(args.model_version):
        for chk_name in ['last_release_half.pth.tar', 'last_release.pth.tar', 'last.pth.tar']:
            chk_path = os.path.join(args.model_version, chk_name)
            if os.path.exists(chk_path):
                args.model_version = chk_path
                print(f'Using "{chk_path}" as model version.')
                break

    model, preprocessor, info = load_model(args.model_version, vitdet_window_size=args.vitdet_window_size, adaptor_names=args.adaptor_name,
                                           torchhub_repo=args.torchhub_repo, neck_name=args.neck)
    model.to(device=device).eval()
    if isinstance(preprocessor, nn.Module):
        preprocessor.to(device).eval()
    rank_print('Done')

    rank_print('Loading dataset...')

    if args.resolution is None:
        args.resolution = (model.preferred_resolution.height, model.preferred_resolution.width)

    # if len(args.resolution) == 1:
    #     args.batch_size = 1
    args.batch_size = 1

    patch_size = getattr(model, 'patch_size', None) or args.patch_size

    if args.resize_multiple is None:
        args.resize_multiple = getattr(model, 'min_resolution_step', patch_size)

    transform = get_standard_transform(args.resolution, args.resize_multiple, max_dim=args.max_dim,
                                       pad_mean=preprocessor.norm_mean if isinstance(preprocessor, InputConditioner) else None,)

    if not os.path.isdir(args.dataset):
        try:
            ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
        except:
            ds_builder = load_dataset_builder("imagefolder", data_dir=args.dataset)
        ds_builder.download_and_prepare()
        ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
        dataset = ds_builder.as_dataset(split=args.split)
        dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, args.workers))
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.zeros(1, dtype=torch.int64)))
        rank_print(f'Description: {ds_builder.info.description}')
    else:
        dataset = ImageFolder(args.dataset, transform=transform)
        dataset.samples.sort(key=lambda s: s[0])

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=partial(collate, group=False),
                        pin_memory=args.workers > 0,
                        drop_last=False,
    )
    rank_print('Done')

    dirs = dict(
        orig=os.path.join(args.output_dir, 'orig'),
        viz=os.path.join(args.output_dir, 'viz'),
        sbs=os.path.join(args.output_dir, 'sbs'),
        grid=os.path.join(args.output_dir, 'grid'),
    )

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    if args.animate_radio1d:
        # Create animation directories
        anim_dirs = dict(
            viz=os.path.join(args.output_dir, 'anim_viz'),
            sbs=os.path.join(args.output_dir, 'anim_sbs'),
            grid=os.path.join(args.output_dir, 'anim_grid'),
        )
        for d in anim_dirs.values():
            os.makedirs(d, exist_ok=True)

        process_with_animation(
            loader, model, preprocessor, device, args, dirs, anim_dirs
        )
    else:
        # Original behavior
        ctr = 0

        for batches in loader:
            if ctr >= args.n:
                break

            for images, _ in batches:
                images = images.to(device=device, non_blocking=True)

                all_feat = []
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    p_images = preprocessor(images)

                    if args.intermediates:
                        outputs = model.forward_intermediates(
                            p_images,
                            indices=args.intermediates,
                            return_prefix_tokens=False,
                            norm=False,
                            stop_early=True,
                            output_fmt='NCHW',
                            intermediates_only=True,
                            aggregation=args.intermediate_aggregation,
                            norm_alpha_scheme="none",
                        )
                        assert args.adaptor_name is None
                        all_feat = outputs
                    else:
                        kwargs = {}
                        if args.neck == "encoder":
                            all_tokens = math.prod(p_images.shape[-2:]) // (4 * model.patch_size ** 2)
                            # Assuming 1D encoder with 2x2 downsampling.
                            kwargs['num_tokens'] = all_tokens
                            output = model(p_images, feature_fmt='NLC', **kwargs)
                            features = output.features.reshape(
                                output.features.shape[0],
                                p_images.shape[-2] // model.patch_size // 2,
                                p_images.shape[-1] // model.patch_size // 2,
                                output.features.shape[2]).permute(0, 3, 1, 2)
                            output = RadioOutput(output.summary, features)
                        else:
                            output = model(p_images, feature_fmt='NCHW', **kwargs)
                        if args.adaptor_name:
                            all_feat = [
                                output['backbone'].features,
                                output[args.adaptor_name].features,
                            ]
                        else:
                            all_feat = [output[1]]

                all_feat = [rearrange(f, 'b c h w -> b h w c').float() for f in all_feat]

                # b m h w c
                all_feat = list(zip(*all_feat))

                for i, feats in enumerate(all_feat):
                    colored = []
                    for features in feats:
                        color = get_pca_map(features, images.shape[-2:], interpolation=args.interpolation, skip_components=args.skip)
                        colored.append(color)

                    orig = cv2.cvtColor(images[i].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

                    cv2.imwrite(f'{dirs["orig"]}/vis_{ctr}.jpg', orig * 255)

                    if args.intermediates:
                        annotations = [f'layer_{i}_{args.intermediate_aggregation}' for i in args.intermediates]
                    else:
                        annotations = ["backbone"]
                        if args.adaptor_name is not None:
                            annotations.append(args.adaptor_name)
                    for annotation, img in zip(annotations, colored):
                        rdir = f'{dirs["viz"]}/{annotation}'
                        os.makedirs(rdir, exist_ok=True)
                        cv2.imwrite(f'{rdir}/vis_{ctr}.jpg', img * 255)

                    grid_image = create_image_grid_with_annotations(
                        images=[[orig] + colored],
                        annotations=["Original"] + annotations)
                    if args.intermediates:
                        grid_filename = f'{dirs["grid"]}/vis_{ctr}_{args.intermediate_aggregation}.jpg'
                    else:
                        grid_filename = f'{dirs["grid"]}/vis_{ctr}.jpg'
                    cv2.imwrite(grid_filename, grid_image * 255)

                    sbs_images = [orig] + list(colored)
                    op = np.concatenate(sbs_images, axis=1) * 255

                    cv2.imwrite(f'{dirs["sbs"]}/vis_{ctr}.jpg', op)

                    ctr += 1


def process_with_animation(loader, model, preprocessor, device, args, dirs, anim_dirs):
    """
    Process images with radio1d_size animation.
    First pass: Extract PCA stats with max radio1d_size.
    Second pass: Generate frames for all radio1d_size values using those stats.
    """
    rank_print(f'Processing with radio1d animation: {args.radio1d_start} to {args.radio1d_end} step {args.radio1d_step}')

    # Collect all images first
    all_images = []
    ctr = 0
    for batches in loader:
        if ctr >= args.n:
            break
        for images, _ in batches:
            images = images.to(device=device, non_blocking=True)
            all_images.append(images)
            ctr += len(images)

    rank_print(f'Collected {len(all_images)} batches')

    # First pass: Get PCA stats from the highest radio1d_size
    rank_print(f'First pass: Computing PCA stats with radio1d_size={args.radio1d_end}')
    pca_stats_list = []

    for images in tqdm(all_images, desc='Computing PCA stats'):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            p_images = preprocessor(images)

            if args.intermediates:
                outputs = model.forward_intermediates(
                    p_images,
                    indices=args.intermediates,
                    return_prefix_tokens=False,
                    norm=False,
                    stop_early=True,
                    output_fmt='NCHW',
                    intermediates_only=True,
                    aggregation=args.intermediate_aggregation,
                    norm_alpha_scheme="none",
                )
                all_feat = outputs
            else:
                kwargs = {}
                if args.neck:
                    kwargs['num_tokens'] = args.radio1d_end
                output = model(p_images, feature_fmt='NCHW', **kwargs)
                if args.adaptor_name:
                    all_feat = [
                        output['backbone'].features,
                        output[args.adaptor_name].features,
                    ]
                else:
                    all_feat = [output[1]]

        all_feat = [rearrange(f, 'b c h w -> b h w c').float() for f in all_feat]
        all_feat = list(zip(*all_feat))

        # Get PCA stats for each image
        for feats in all_feat:
            img_pca_stats = []
            for features in feats:
                _, pca_stats = get_pca_map(
                    features,
                    images.shape[-2:],
                    interpolation=args.interpolation,
                    return_pca_stats=True,
                    skip_components=args.skip
                )
                img_pca_stats.append(pca_stats)
            pca_stats_list.append(img_pca_stats)

    # Second pass: Generate frames for all radio1d_size values
    rank_print(f'Second pass: Generating frames for all radio1d_size values')
    radio1d_sizes = list(range(args.radio1d_start, args.radio1d_end + 1, args.radio1d_step))

    for img_idx, images in enumerate(tqdm(all_images, desc='Processing images')):
        # Storage for animation frames
        viz_frames = defaultdict(list)  # annotation -> list of frames
        sbs_frames = []
        grid_frames = []

        orig = cv2.cvtColor(images[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

        # Save original image
        cv2.imwrite(f'{dirs["orig"]}/vis_{img_idx}.jpg', orig * 255)

        for radio1d_size in tqdm(radio1d_sizes, desc=f'radio1d_size sweep (img {img_idx})', leave=False):
            with torch.autocast(device.type, dtype=torch.bfloat16):
                p_images = preprocessor(images)

                if args.intermediates:
                    outputs = model.forward_intermediates(
                        p_images,
                        indices=args.intermediates,
                        return_prefix_tokens=False,
                        norm=False,
                        stop_early=True,
                        output_fmt='NCHW',
                        intermediates_only=True,
                        aggregation=args.intermediate_aggregation,
                        norm_alpha_scheme="none",
                    )
                    all_feat = outputs
                else:
                    kwargs = {}
                    if args.neck:
                        kwargs['num_tokens'] = radio1d_size
                    output = model(p_images, feature_fmt='NCHW', **kwargs)
                    if args.adaptor_name:
                        all_feat = [
                            output['backbone'].features,
                            output[args.adaptor_name].features,
                        ]
                    else:
                        all_feat = [output[1]]

            all_feat = [rearrange(f, 'b c h w -> b h w c').float() for f in all_feat]
            all_feat = list(zip(*all_feat))

            # Render each feature map with its precomputed PCA basis (computed
            # once at radio1d_end). Reusing the basis keeps colors stable across
            # frames of the radio1d_size sweep.
            feats = all_feat[0]  # First image in batch
            colored = []
            for feat_idx, features in enumerate(feats):
                color = get_pca_map(
                    features,
                    images.shape[-2:],
                    interpolation=args.interpolation,
                    pca_stats=pca_stats_list[img_idx][feat_idx],
                    skip_components=args.skip,
                )
                colored.append(color)

            if args.intermediates:
                annotations = [f'layer_{i}_{args.intermediate_aggregation}' for i in args.intermediates]
            else:
                annotations = ["backbone"]
                if args.adaptor_name is not None:
                    annotations.append(args.adaptor_name)

            for annotation, img in zip(annotations, colored):
                frame = (img * 255).astype(np.uint8)
                frame = add_text_overlay(frame, f"radio1d_size={radio1d_size}")
                viz_frames[annotation].append(frame)

            sbs_images = [orig] + list(colored)
            sbs_frame = np.concatenate(sbs_images, axis=1) * 255
            sbs_frame = sbs_frame.astype(np.uint8)
            sbs_frame = add_text_overlay(sbs_frame, f"radio1d_size={radio1d_size}")
            sbs_frames.append(sbs_frame)

            # Grid frame
            grid_image = create_image_grid_with_annotations(
                images=[[orig] + colored],
                annotations=["Original"] + annotations
            )
            grid_image = (grid_image * 255).astype(np.uint8)
            grid_image = add_text_overlay(grid_image, f"radio1d_size={radio1d_size}", position='bottom')
            grid_frames.append(grid_image)

        # Save animated PNGs
        rank_print(f'Saving animated PNGs for image {img_idx}')

        # Save individual viz animations
        for annotation, frames in viz_frames.items():
            rdir = f'{anim_dirs["viz"]}/{annotation}'
            os.makedirs(rdir, exist_ok=True)
            save_animated_image(frames, f'{rdir}/vis_{img_idx}.png', duration=args.animation_duration)

        # Save side-by-side animation
        save_animated_image(sbs_frames, f'{anim_dirs["sbs"]}/vis_{img_idx}.gif', duration=args.animation_duration, format='GIF')

        # Save grid animation
        save_animated_image(grid_frames, f'{anim_dirs["grid"]}/vis_{img_idx}.png', duration=args.animation_duration)

    rank_print('Animation processing complete!')


def add_text_overlay(image, text, position='top', font=cv2.FONT_HERSHEY_SIMPLEX,
                     font_scale=1.0, font_color=(255, 255, 255), thickness=2,
                     bg_color=(0, 0, 0), padding=10):
    """
    Add text overlay to an image.

    Args:
        image: numpy array (H, W, C) in BGR format with values 0-255
        text: Text to display
        position: 'top' or 'bottom' for text placement
        font: OpenCV font type
        font_scale: Font scale
        font_color: Text color (BGR)
        thickness: Text thickness
        bg_color: Background color for text (BGR)
        padding: Padding around text

    Returns:
        Image with text overlay
    """
    image = image.copy()

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate position
    img_height, img_width = image.shape[:2]

    if position == 'top':
        text_x = padding
        text_y = padding + text_height
        bg_y1 = 0
        bg_y2 = text_height + 2 * padding
    else:  # bottom
        text_x = padding
        text_y = img_height - padding - baseline
        bg_y1 = img_height - text_height - 2 * padding - baseline
        bg_y2 = img_height

    bg_x1 = 0
    bg_x2 = text_width + 2 * padding

    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    alpha = 0.7
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    return image


def save_animated_image(frames, output_path, duration=100, format='PNG'):
    """
    Save a list of numpy arrays as an animated image.

    Args:
        frames: List of numpy arrays (H, W, C) in BGR format with values 0-255
        output_path: Path to save the APNG file
        duration: Duration of each frame in milliseconds
    """
    # Convert BGR frames to RGB PIL Images
    pil_frames = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))

    # Save as animated PNG
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,  # 0 means infinite loop
        format=format
    )


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False, skip: int = 0):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3 + skip, niter=20)[2]
    reduction_mat = reduction_mat[:, skip:]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    img_size,
    interpolation="bicubic",
    return_pca_stats=False,
    pca_stats=None,
    skip_components: int = 0,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1]), skip=skip_components,
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = F.interpolate(
        pca_color.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


def get_scale_map(
    scalar_map: torch.Tensor,
    img_size,
    interpolation="nearest",
):
    """
    scalar_map: (1, h, w, C) is the feature map of a single image.
    """
    if scalar_map.shape[0] != 1:
        scalar_map = scalar_map[None]
    scalar_map = (scalar_map - scalar_map.min()) / (
        scalar_map.max() - scalar_map.min() + 1e-6
    )
    scalar_map = F.interpolate(
        scalar_map.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    # cmap = plt.get_cmap("viridis")
    # scalar_map = cmap(scalar_map)[..., :3]
    # make it 3 channels
    scalar_map = torch.cat([scalar_map] * 3, dim=-1)
    scalar_map = scalar_map.cpu().numpy().squeeze(0)
    return scalar_map


def create_image_grid_with_annotations(images, annotations, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), thickness=2):
    """
    Arrange a nested list of MxN numpy arrays into a single image grid with column annotations.

    Args:
    images (list of list of np.array): Nested list containing the image arrays.
    annotations (list of str): List of annotations for each column.
    font (int): Font type for the annotations.
    font_scale (float): Font scale for the annotations.
    font_color (tuple): Font color for the annotations.
    thickness (int): Thickness of the font.

    Returns:
    np.array: The resulting image grid with annotations.
    """
    # Check if the input is valid
    if not images or not isinstance(images, list) or not all(isinstance(row, list) for row in images):
        raise ValueError("Input must be a nested list of numpy arrays.")

    # Determine the number of rows and columns
    num_rows = len(images)
    num_cols = len(images[0])

    # Check if all rows have the same number of columns
    if not all(len(row) == num_cols for row in images):
        raise ValueError("All rows must have the same number of columns.")

    # Get the dimensions of each image
    first_image_shape = images[0][0].shape
    img_height, img_width = first_image_shape[:2]

    # Check if all images have the same dimensions
    for row in images:
        for img in row:
            if img.shape[:2] != (img_height, img_width):
                raise ValueError("All images must have the same dimensions.")

    # Determine the number of channels
    num_channels = first_image_shape[2] if len(first_image_shape) == 3 else 1

    # Create an empty array for the resulting grid
    annotation_height = 50  # Height reserved for annotations
    grid_height = num_rows * img_height + annotation_height
    grid_width = num_cols * img_width

    if num_channels == 1:
        grid = np.zeros((grid_height, grid_width), dtype=images[0][0].dtype)
    else:
        grid = np.zeros((grid_height, grid_width, num_channels), dtype=images[0][0].dtype)

    # Add annotations at the top of each column
    for col_idx, annotation in enumerate(annotations):
        text_size = cv2.getTextSize(annotation, font, font_scale, thickness)[0]
        text_x = col_idx * img_width + (img_width - text_size[0]) // 2
        text_y = (annotation_height + text_size[1]) // 2
        cv2.putText(grid, annotation, (text_x, text_y), font, font_scale, font_color, thickness)

    # Populate the grid with the images
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            start_y = annotation_height + row_idx * img_height
            start_x = col_idx * img_width
            grid[start_y:start_y+img_height, start_x:start_x+img_width] = img

    return grid


if __name__ == '__main__':
    rank = 0
    world_size = 1

    # if 'WORLD_SIZE' in os.environ:
    #     dist.init_process_group(backend='nccl')
    #     rank = dist.get_rank()
    #     world_size = dist.get_world_size()

    main(rank, world_size)
