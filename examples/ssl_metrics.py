# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from collections import defaultdict
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

import albumentations as A
from einops import rearrange

from datasets import load_dataset_builder, load_dataset
from datasets.distributed import split_dataset_by_node

from common import rank_print, RandAugment, load_model

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
    parser.add_argument('--use-hf', default=False, action='store_true',
                        help='Use RADIO from HuggingFace Hub'
    )
    parser.add_argument('-v', '--model-version', default='radio_v1',
                        help='Which radio model to load.'
    )
    parser.add_argument('-d', '--dataset', default='imagenet-1k',
                        help='The name of the dataset to classify'
    )
    parser.add_argument('-n', default=1000, type=int, help='The number of samples to load')
    parser.add_argument('-q', default=99, type=int, help='The number of augmented views to generate')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='The batch size')
    parser.add_argument('--resolution', default=336, type=int, help='The square resolution to compute metrics against')
    parser.add_argument('--workers', default=8, type=int, help='Number of loader workers to use')
    parser.add_argument('--vitdet-window-size', default=None, type=int, help='Enable ViTDet at the specific window size')
    parser.add_argument('--log-wandb', default=False, action='store_true', help='Log to wandb')
    parser.add_argument('--wandb-entity', default='adlr', type=str, help='The wandb entity')
    parser.add_argument('--wandb-project', default='lidar_eq', type=str, help='The wandb project')
    parser.add_argument('--wandb-group', default=None, type=str, help='The wandb group')
    parser.add_argument('--wandb-job-type', default=None, type=str, help='The wandb job type')
    parser.add_argument('--wandb-id', default=None, type=str, help='The wandb run id')

    args, _ = parser.parse_known_args()

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    rank_print('Loading model...')
    extra_args = dict()
    if args.vitdet_window_size:
        extra_args['vitdet_window_size'] = args.vitdet_window_size
    model, preprocessor, info = load_model(args.model_version, device=device, **extra_args)
    model.to(device=device).eval()
    preprocessor.to(device=device)
    rank_print('Done')

    tmp = torch.empty(1, 3, args.resolution, args.resolution, dtype=torch.float32, device=device)
    tmp = preprocessor(tmp)
    _, tmp_features = model(tmp)
    downsample = int(round(math.sqrt(args.resolution ** 2 / tmp_features.shape[1])))

    transform_reg = A.Compose([
        A.SmallestMaxSize(args.resolution),
        A.CenterCrop(args.resolution, args.resolution),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    transform_aug = A.Compose([
        RandAugment(num_ops=3, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
        A.RandomResizedCrop(args.resolution, args.resolution, scale=(0.5, 1.0)),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    ds_builder.download_and_prepare()

    n_iter = int(math.ceil(args.n / world_size / args.batch_size))

    dataset = ds_builder.as_dataset(split='train')
    dataset = dataset.to_iterable_dataset(num_shards=world_size * max(args.workers, 1))
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.map(ssl_augment(transform_reg, transform_aug, args.q))
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers)

    rank_me_embeddings = []
    rank_me_features = []
    lidar_embeddings = []

    lidar_eq_cov_b = 0
    lidar_eq_cov_w = 0
    lidar_eq_im_means = []

    num_tokens = 0
    summary_dim = 0
    spatial_dim = 0

    for i, batch in tqdm(enumerate(loader), total=n_iter, desc='Processing', disable=rank > 0, position=0, leave=False):
        if i == n_iter:
            break

        for k, v in batch.items():
            if torch.is_tensor(v):
                v = v.to(device=device, non_blocking=True)
            batch[k] = v

        images = batch['image']
        # Roll the batch and variant dimensions into batch
        flat_images = images.flatten(0, 1)

        summary, features = [], []
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            num_steps = int(math.ceil(flat_images.shape[0] / args.batch_size))
            for c in tqdm(range(0, flat_images.shape[0], args.batch_size), total=num_steps, desc='Model Inference', disable=rank > 0, position=1, leave=False):
                chunk = preprocessor(flat_images[c:c+args.batch_size])
                c_summary, c_features = model(chunk)
                summary.append(c_summary)
                features.append(c_features)

            summary = torch.cat(summary)
            features = torch.cat(features)

        # for _ in tqdm(range(1), desc='LiDAR', disable=rank > 0, position=2):
        if True:
            # Unroll the batch and variant dimensions
            g_summary = summary.float().reshape(*images.shape[:2], *summary.shape[1:])
            g_features = features.float().reshape(*images.shape[:2], *features.shape[1:])

            # RankMe is only computed for the regular images
            rank_me_embeddings.append(g_summary[:, 0])
            rank_me_features.append(g_features[:, 0].flatten(0, 1))
            lidar_embeddings.append(g_summary)

            # eq_cov_b, eq_cov_w, eq_im_means = calc_lidar_eq_pt1(g_features, batch['transforms'], batch['orig_shape'], images.shape[-2:], downsample)
            # lidar_eq_cov_b = lidar_eq_cov_b + eq_cov_b
            # lidar_eq_cov_w = lidar_eq_cov_w + eq_cov_w
            # lidar_eq_im_means.extend(eq_im_means)

            # TODO(mranzinger): Enable this to render composite images
            # if i == 0 and rank == 0:
            #     for k, (image, tx, orig_size) in enumerate(zip(images, batch['transforms'], batch['orig_shape'])):
            #         visualize_augmented_images(image, tx, orig_size, f'_{k}')
            num_tokens = features.shape[-2]
            summary_dim = summary.shape[-1]
            spatial_dim = features.shape[-1]

        del batch
        del images
        del flat_images
        del summary
        del features
        del g_summary
        del g_features
        # del eq_cov_b
        # del eq_cov_w

    del model
    gc.collect()
    torch.cuda.empty_cache()

    def _gather_cat(t):
        return gather_cat(torch.cat(t), rank, world_size, device)

    rank_me_embeddings = _gather_cat(rank_me_embeddings)
    rank_me_features = _gather_cat(rank_me_features)
    lidar_embeddings = _gather_cat(lidar_embeddings)

    # lidar_eq_info = [lidar_eq_cov_b.div_(n_iter), lidar_eq_cov_w.div_(n_iter)]

    # if dist.is_initialized():
    #     for t in lidar_eq_info:
    #         if torch.is_tensor(t):
    #             dist.reduce(t, dst=0, op=dist.ReduceOp.AVG)

    # lidar_eq_info.append(_gather_cat(lidar_eq_im_means))

    if rank > 0:
        return

    rank_me = calc_rank_me(rank_me_embeddings)
    rank_me_features = calc_rank_me(rank_me_features)
    lidar = calc_lidar(lidar_embeddings)
    # lidar_eq = calc_lidar_eq_pt2(*lidar_eq_info)

    print(f'C: {rank_me_embeddings.shape[1]}')
    print(f'RankMe: {rank_me.item():.3f}')
    print(f'RankMeFeat: {rank_me_features.item():.3f}')
    print(f'LiDAR: {lidar.item():.3f}')
    # print(f'LiDAR-EQ: {lidar_eq.item():.3f}')

    if args.log_wandb and rank == 0:
        if wandb is None:
            raise ValueError('WandB not installed!')

        job_type = args.wandb_job_type or info.model_subtype
        job_type += f'_r{args.resolution}'
        if args.vitdet_window_size:
            job_type += f'_w{args.vitdet_window_size}'

        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group or info.model_class,
            job_type=job_type,
            id = args.wandb_id or wandb.util.generate_id(),
            resume='allow',
            config=args,
        )

        wandb.summary['NumTokens'] = num_tokens
        wandb.summary['SummaryDim'] = summary_dim
        wandb.summary['SpatialDim'] = spatial_dim
        wandb.summary['RankMe'] = rank_me.item()
        wandb.summary['RankMeFeat'] = rank_me_features.item()
        wandb.summary['LiDAR'] = lidar.item()
        # wandb.summary['LiDAR-EQ'] = lidar_eq.item()


def album_aug(tx, image, bds) -> Tuple[torch.Tensor, np.ndarray]:
    transformed = tx(image=image, keypoints=bds)

    image = torch.from_numpy(transformed['image']).permute(2, 0, 1).float().div_(255.0)
    bds = torch.tensor(transformed['keypoints'], dtype=torch.float32)

    return image, bds


def ssl_augment(transform_reg, transform_aug, num_variants):
    sample_bounds = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)

    def augment(ex: Dict[str, torch.Tensor]):
        image: Image.Image = ex['image']
        image = image.convert('RGB')

        height, width = image.height, image.width

        image = np.array(image)

        bds = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ]

        im_reg = album_aug(transform_reg, image, bds)
        variants = [album_aug(transform_aug, image, bds) for _ in range(num_variants)]
        variants.insert(0, im_reg)

        images, all_bds = list(zip(*variants))

        images = torch.stack(images)
        # all_bds = torch.stack(all_bds)

        orig_bds = np.array(bds, dtype=np.float32)
        crop_bds = np.array([
            [0, 0],
            [images.shape[-1], 0],
            [images.shape[-1], images.shape[-2]],
            [0, images.shape[-2]],
        ], dtype=np.float32)

        transforms = []
        torch_bds = []

        tx_sample_to_orig = cv2.getPerspectiveTransform(sample_bounds, orig_bds)
        for i, aug_bds in enumerate(all_bds):
            aug_bds = np.array(aug_bds, dtype=np.float32)
            tx_orig_to_aug = cv2.getPerspectiveTransform(orig_bds, aug_bds)
            tx_aug_to_sample = cv2.getPerspectiveTransform(crop_bds, sample_bounds)

            # Maps from [-1, 1] in the source sample space to the augmented view space
            tx_sample_to_aug_sample = np.matmul(tx_aug_to_sample, np.matmul(tx_orig_to_aug, tx_sample_to_orig))
            tx_sample_to_aug_sample = torch.from_numpy(tx_sample_to_aug_sample)

            transforms.append(tx_sample_to_aug_sample)
            torch_bds.append(torch.from_numpy(aug_bds))

        return dict(image=images, bounds=torch.stack(torch_bds), transforms=torch.stack(transforms), orig_shape=torch.tensor([height, width], dtype=torch.int64))
    return augment


def calc_rank_me(embeddings: torch.Tensor):
    # embeddings: N,K

    eig = torch.linalg.svdvals(embeddings.double())

    return smooth_rank(eig)


DELTA = 1e-8


def calc_lidar(embeddings: torch.Tensor, delta: float = DELTA):
    # embeddings: N,V,K
    embeddings = embeddings.double()

    mu_x = embeddings.mean(dim=1)  # N,K
    mu = mu_x.mean(dim=0, keepdim=True)  # 1,K

    ######
    # Compute E_b(e)
    # This is measuring the inter-class covariance. Here, each image is considered to be a different class
    # We want this to not have highly entangled dimensions (meaning that the eigenspectrum is full rank, and uniform valued)
    mu_zc = mu_x - mu
    cov_b = (mu_zc.T @ mu_zc) / (mu_zc.shape[0] - 1)
    ######

    ######
    # Compute E_w(e)
    # This is measuring the intra-class covariance.
    # Assuming a perfectly invariant model, then $e(~x) == mu_x$, which leaves delta*I
    # Basically, we want E_w(e) to be as close to p*I (for some p) as possible
    ex_zc = (embeddings - mu_x.unsqueeze(1)).flatten(0, 1)
    cov_w = (ex_zc.T @ ex_zc) / (ex_zc.shape[0] - 1)
    cov_w.diagonal().add_(delta)
    ######

    return calc_lidar_from_cov(cov_b, cov_w, delta)


def calc_lidar_from_cov(cov_b: torch.Tensor, cov_w: torch.Tensor, delta: float = DELTA):
    ######
    # Compute E_w(e)^-0.5
    ######
    cov_b = cov_b.double()
    cov_w = cov_w.double()

    # # To get the inverse square root of `cov_w`, we can use two numerically stable steps:
    # # 1) Compute the cholesky decomposition of `cov_w`: A = L @ L.T
    # #        L = cholesky(A)
    # #        `L` can be thought of as the square root of A
    # # 2) Compute the inverse of L
    # #        Because `L` is lower triangular, we can use a triangular solver to find this inverse
    # #        `XA = B`
    # #        We want to find X given A and B.
    # #        Set A to L, and B to I. In words, for a matrix A, if XA = I, then X == A^-1
    # #        Thus solving for X finds the inverse of A
    # chol_cov_w = torch.linalg.cholesky(cov_w)
    # B = torch.eye(chol_cov_w.shape[-1], dtype=chol_cov_w.dtype, device=chol_cov_w.device)
    # inv_chol_cov_w = torch.linalg.solve_triangular(chol_cov_w, B, upper=False, left=False)

    # cov_lidar = inv_chol_cov_w @ cov_b @ inv_chol_cov_w.T

    # # `cov_lidar` is square-hermitian, so we can use this method
    # eig = torch.linalg.eigvalsh(cov_lidar)

    # In order to compute a matrix raised to a non-positive-integer power, we need to
    # diagonalize the matrix. Because the covariance matrix is positive definite, then `eigh` is the
    # correct call.
    #
    # Once we have the diagonalized matrix decomposition (A = VEV^T), raising the original matrix to an arbitrary power
    # is equivalent to raising the elements in E to that power, and then recomposing.
    # Meaning A^s = (VEV^T)^s = V(E^s)V^T
    # Raising E^s can be done trivially because it's a diagonal matrix, which is why we raise the elements
    e_cov_w, v_cov_w = torch.linalg.eigh(cov_w)
    e_cov_w.clamp_min_(delta)
    # Sanity check, just to make sure the decomposition succeeded
    # assert torch.allclose(cov_w, v_cov_w @ torch.diag(e_cov_w) @ v_cov_w.T), "Matrix diagonalization failed!"

    rsqrt_e_cov_w = torch.pow(e_cov_w, -0.5)
    rsqrt_cov_w = v_cov_w @ torch.diag(rsqrt_e_cov_w) @ v_cov_w.T
    ######

    cov_lidar = rsqrt_cov_w @ cov_b @ rsqrt_cov_w
    cov_lidar.diagonal().add_(delta)

    eig, _ = torch.linalg.eigh(cov_lidar)

    return smooth_rank(eig)


def calc_lidar_eq_pt1(all_features: torch.Tensor, all_transforms: torch.Tensor, image_sizes: torch.Tensor, input_size: Tuple[int, int], downsample: int, delta: float = DELTA):
    # N: Images
    # V: Augmented views of image
    # T: Spatial vectors for image
    # C: Features of spatial vector

    # all_features: N,V,T,C
    # all_transforms: N,V,3,3
    # image_sizes: N,2

    N, V, T, C = all_features.shape
    options = dict(dtype=all_features.dtype, device=all_features.device)

    grid_downsample = 2

    def _get_warp(features: torch.Tensor, transform: torch.Tensor, image_size):
        features = rearrange(features, '(h w) c -> c h w', h=input_size[0] // downsample, w=input_size[1] // downsample)

        # # Uncomment this to prove that we're measuring feature alignment under affine transform
        # #   Basically, we'd expect the effect rank to go down if the features from different views aren't aligned
        # transform = torch.eye(transform.shape[0], **options)
        return get_warped_features(features, transform, image_size)


    all_cov_b = torch.zeros(C, C, **options)
    all_cov_w = torch.zeros(C, C, **options)

    im_means = []

    for i, (aug_features, transforms, image_size) in enumerate(zip(all_features, all_transforms, image_sizes)):
        ds_image_size = tuple(d // grid_downsample for d in image_size.cpu().tolist())

        # C,H,W
        mean_features_per_px = torch.zeros(aug_features.shape[-1], *ds_image_size, **options)
        # H,W
        counts_per_px = torch.zeros(*ds_image_size, **options)

        for k, (features, transform) in enumerate(zip(aug_features, transforms)):
            warp_features, warp_mask = _get_warp(features, transform, ds_image_size)

            mean_features_per_px.add_(warp_features)
            counts_per_px.add_(warp_mask[0])

            # This process can be memory intensive
            del warp_features
            del warp_mask

        mean_features_per_px /= counts_per_px.clamp_min(1)

        ct_valid_per_px = counts_per_px > 1

        # C
        mean_features_im = mean_features_per_px.sum(dim=(1, 2)) / ct_valid_per_px.sum(dtype=torch.float32)

        im_means.append(mean_features_im[None])

        # C,T
        flat_mean_features = mean_features_per_px.flatten(1)
        flat_ct_valid = ct_valid_per_px.flatten()

        # C,T'
        zc_valid_features = flat_mean_features[:, flat_ct_valid]
        zc_valid_features -= mean_features_im.unsqueeze(1)
        cov_b = (zc_valid_features @ zc_valid_features.T).div_(zc_valid_features.shape[1] - 1)

        cov_w = torch.zeros_like(cov_b)
        cov_w_num_samples = 0
        for k, (features, transform) in enumerate(zip(aug_features, transforms)):
            warp_features, warp_mask = _get_warp(features, transform, ds_image_size)

            zc_warp_features = torch.where(warp_mask > 0, warp_features - mean_features_per_px, 0)
            flat_zc_warp_features = zc_warp_features.flatten(1)

            cov_curr = flat_zc_warp_features @ flat_zc_warp_features.T
            cov_w.add_(cov_curr)
            cov_w_num_samples = cov_w_num_samples + warp_mask.sum()

        cov_w /= (cov_w_num_samples - 1)

        all_cov_b += cov_b
        all_cov_w += cov_w
        pass

    all_cov_b /= N
    all_cov_w /= N
    all_cov_w.diagonal().add_(delta)

    return all_cov_b, all_cov_w, im_means


def calc_lidar_eq_pt2(cov_b: torch.Tensor, cov_w: torch.Tensor, mu_x: torch.Tensor, gamma: float = 0.75, delta: float = DELTA):
    # im_means: N,C

    # 1,C
    mu = mu_x.mean(dim=0, keepdim=True)

    mu_zc = mu_x - mu
    cov_b_g = (mu_zc.T @ mu_zc) / (mu_zc.shape[0] - 1)

    cov_b = gamma * cov_b + (1 - gamma) * cov_b_g

    return calc_lidar_from_cov(cov_b, cov_w, delta)


def smooth_rank(eig: torch.Tensor):
    # Use a formulation of entropy to compute a smooth
    # rank approximator.
    eig = eig[eig > 0]

    # First, we convert the eigenvalues to probabilities
    p = eig / (eig.sum() + 1e-7)

    log_p = p.log()
    p_log_p = p * log_p

    # NOTE: Entropy is maximized when eig is uniform valued, in which case
    # entropy = -log(1 / P) = log(P) with `P` being the number of eigenvalues,
    # and thus e^log(P) = P.
    entropy = -p_log_p.sum()

    return entropy.exp()


def gather_cat(t: torch.Tensor, rank: int, world_size: int, device: torch.device):
    if world_size == 1:
        return t

    orig_device = t.device
    t = t.to(device=device)

    gl = [torch.empty_like(t) for _ in range(world_size)]
    dist.gather(t, gl if rank == 0 else None, dst=0)

    ret = torch.cat(gl) if rank == 0 else t
    ret = ret.to(device=orig_device)
    return ret


def generate_homography_grid(homography: torch.Tensor, size):
    if len(size) == 2:
        size = (1, 1, *size)
    elif len(size) == 3:
        size = (size[0], 1, *size)
    else:
        assert len(size) == 4

    if homography.ndim == 2:
        homography = homography.unsqueeze(0)

    N, C, H, W = size
    base_grid = homography.new(N, H, W, 3)
    linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
    base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
    linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
    base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
    base_grid[:, :, :, 2] = 1

    grid = torch.bmm(base_grid.view(N, H * W, 3), homography.transpose(1, 2))
    grid = grid.view(N, H, W, 3)
    grid[:, :, :, 0] = grid[:, :, :, 0] / grid[:, :, :, 2]
    grid[:, :, :, 1] = grid[:, :, :, 1] / grid[:, :, :, 2]
    grid = grid[:, :, :, :2].float()
    return grid


def visualize_augmented_images(images: torch.Tensor, transforms: torch.Tensor, orig_size: torch.Tensor, suffix: str = ''):
    accum = torch.zeros(3, *orig_size, dtype=images.dtype, device=images.device)
    accum_mask = torch.zeros_like(accum[:1])

    for image, transform in zip(images, transforms):
        warped_image, mask = get_warped_features(image, transform, orig_size)

        accum += warped_image
        accum_mask += mask

    accum /= accum_mask.clamp_min(1)

    def _save_image(im: torch.Tensor, dir: str, path: str):
        os.makedirs(dir, exist_ok=True)

        viz = im.permute(1, 2, 0).mul(255).to(torch.uint8).cpu().numpy()
        viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{dir}/{path}', viz)

    _save_image(accum, 'aug', f'viz_aug{suffix}.jpg')

    im_grid = make_grid(images, nrow=10, padding=4)
    _save_image(im_grid, 'grid', f'viz_grid{suffix}.jpg')
    pass


def get_warped_features(features: torch.Tensor, transform: torch.Tensor, image_size):
    grid = generate_homography_grid(transform, image_size)

    mask = torch.ones_like(features[:1])

    warp_features = F.grid_sample(features[None], grid, mode='bilinear', align_corners=True)[0]
    warp_mask = F.grid_sample(mask[None], grid, mode='bilinear', align_corners=True)[0]

    # Binarize the mask, only keep positions that aren't on the border
    vmask = warp_mask >= 0.99
    warp_mask = torch.where(vmask, warp_mask, 0)
    warp_features = torch.where(vmask, warp_features, 0)

    return warp_features, warp_mask


if __name__ == '__main__':
    rank = 0
    world_size = 1

    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    main(rank, world_size)
