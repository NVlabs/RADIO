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
    parser.add_argument('-v', '--model-version', default='radio_v2',
                        help='Which radio model to load.'
    )
    parser.add_argument('-d', '--dataset', default='imagenet-1k',
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
    parser.add_argument('--resize-multiple', type=int, default=16,
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

    args, _ = parser.parse_known_args()

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    rank_print(f'Loading model: "{args.model_version}", ViTDet: {args.vitdet_window_size}, Adaptor: "{args.adaptor_name}", Resolution: {args.resolution}, Max: {args.max_dim}...')
    model, preprocessor, info = load_model(args.model_version, vitdet_window_size=args.vitdet_window_size, adaptor_names=args.adaptor_name,
                                           torchhub_repo=args.torchhub_repo)
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
        dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))
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
                        output_fmt='NLC',
                        intermediates_only=True,
                        aggregation=args.intermediate_aggregation,
                    )
                    assert args.adaptor_name is None
                    all_feat = [o[1] for o in outputs]
                else:
                    output = model(p_images)
                    if args.adaptor_name:
                        all_feat = [
                            output['backbone'].features,
                            output[args.adaptor_name].features,
                        ]
                    else:
                        all_feat = [output[1]]

            if images.shape[-2] != images.shape[-1]:
                num_rows = images.shape[-2] // patch_size
                num_cols = images.shape[-1] // patch_size
            else:
                num_rows = int(round(math.sqrt(all_feat[0].shape[1])))
                num_cols = num_rows

            # m b h w c
            all_feat = [
                rearrange(f, 'b (h w) c -> b h w c', h=num_rows, w=num_cols).float()
                for f in all_feat
            ]
            # all_feat = rearrange(all_feat, 'b m (h w) c -> b m h w c', h=num_rows, w=num_cols).float()

            # b m h w c
            all_feat = list(zip(*all_feat))

            for i, feats in enumerate(all_feat):
                colored = []
                for features in feats:
                    color = get_pca_map(features, images.shape[-2:], interpolation='bilinear')
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

                # Create an image grid with the original image and the colored feature maps.
                grid_image = create_image_grid_with_annotations(
                    images=[[orig] + colored],
                    annotations=["Original"] + annotations)
                if args.intermediates:
                    grid_filename = f'{dirs["grid"]}/vis_{ctr}_{args.intermediate_aggregation}.jpg'
                else:
                    grid_filename = f'{dirs["grid"]}/vis_{ctr}.jpg'
                cv2.imwrite(grid_filename, grid_image * 255)

                op = np.concatenate([orig] + colored, axis=1) * 255

                cv2.imwrite(f'{dirs["sbs"]}/vis_{ctr}.jpg', op)
                ctr += 1


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
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
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1])
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


def get_similarity_map(features: torch.Tensor, img_size=(224, 224)):
    """
    compute the similarity map of the central patch to the rest of the image
    """
    assert len(features.shape) == 4, "features should be (1, C, H, W)"
    H, W, C = features.shape[1:]
    center_patch_feature = features[0, H // 2, W // 2, :]
    center_patch_feature_normalized = center_patch_feature / center_patch_feature.norm()
    center_patch_feature_normalized = center_patch_feature_normalized.unsqueeze(1)
    # Reshape and normalize the entire feature tensor
    features_flat = features.view(-1, C)
    features_normalized = features_flat / features_flat.norm(dim=1, keepdim=True)

    similarity_map_flat = features_normalized @ center_patch_feature_normalized
    # Reshape the flat similarity map back to the spatial dimensions (H, W)
    similarity_map = similarity_map_flat.view(H, W)

    # Normalize the similarity map to be in the range [0, 1] for visualization
    similarity_map = (similarity_map - similarity_map.min()) / (
        similarity_map.max() - similarity_map.min()
    )
    # we don't want the center patch to be the most similar
    similarity_map[H // 2, W // 2] = -1.0
    similarity_map = (
        F.interpolate(
            similarity_map.unsqueeze(0).unsqueeze(0),
            size=img_size,
            mode="bilinear",
        )
        .squeeze(0)
        .squeeze(0)
    )

    similarity_map_np = similarity_map.cpu().numpy()
    negative_mask = similarity_map_np < 0

    colormap = plt.get_cmap("turbo")

    # Apply the colormap directly to the normalized similarity map and multiply by 255 to get RGB values
    similarity_map_rgb = colormap(similarity_map_np)[..., :3]
    similarity_map_rgb[negative_mask] = [1.0, 0.0, 0.0]
    return similarity_map_rgb


def get_cluster_map(
    feature_map: torch.Tensor,
    img_size,
    num_clusters=10,
) -> torch.Tensor:
    kmeans = KMeans(n_clusters=num_clusters, distance=CosineSimilarity, verbose=False)
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    labels = kmeans.fit_predict(
        feature_map.reshape(1, -1, feature_map.shape[-1])
    ).float()
    labels = (
        F.interpolate(
            labels.reshape(1, *feature_map.shape[:-1]), size=img_size, mode="nearest"
        )
        .squeeze()
        .cpu()
        .numpy()
    ).astype(int)
    cmap = plt.get_cmap("rainbow", num_clusters)
    cluster_map = cmap(labels)[..., :3]
    return cluster_map.reshape(img_size[0], img_size[1], 3)


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
