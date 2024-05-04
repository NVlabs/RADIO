# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from collections import defaultdict
import math
import os
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets import load_dataset_builder, load_dataset
from datasets.iterable_dataset import DistributedConfig
from datasets.distributed import split_dataset_by_node

from common import collate, round_up, get_standard_transform, run_rank_0_first, rank_print, load_model


def main(rank: int = 0, world_size: int = 1):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    device = torch.device('cuda', local_rank)

    parser = argparse.ArgumentParser(description='kNN Classification Demo')
    parser.add_argument('--use-hf', default=False, action='store_true',
                        help='Use RADIO from HuggingFace Hub'
    )
    parser.add_argument('-v', '--model-version', default='radio_v2',
                        help='Which radio model to load.'
    )
    parser.add_argument('-a', '--adaptor-name', type=str, default=None, help='Which head to use, if any')
    parser.add_argument('-r', '--resolution', nargs='+', type=int, default=None,
                        help='The input image resolution.'
                        ' If one value is specified, the shortest dimension is resized to this.'
                        ' If two, the image is center cropped.'
                        ' If not specified, center cropped 378px is used.'
    )
    parser.add_argument('-d', '--dataset', default='food101',
                        help='The name of the dataset to classify'
    )
    parser.add_argument('--eval-dataset', default=None, type=str,
                        help='The name of the evaluation dataset, if different than the training one.'
    )
    parser.add_argument('--resize-multiple', type=int, default=16,
                        help='Resize images with dimensions a multiple of this value.'
                        ' This should be equal to the patch size of a ViT (e.g. RADIOv1)'
    )

    parser.add_argument('--train-split', default='train',
                        help='The dataset training split to use'
    )
    parser.add_argument('--eval-split', default='validation',
                        help='The evaluation split to use. If labels are present, accuracy will be computed'
    )
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The batch size. If the input is variable sized, then this argument becomes a maximum.'
    )
    parser.add_argument('-w', '--workers', default=8, type=int,
                        help='The number of data loader workers to use per GPU'
    )
    parser.add_argument('-k', default=20, type=int,
                        help="How many neighbors to use for classification."
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
    parser.add_argument('--amp', default=False, action='store_true', help='Run in mixed precision')
    parser.add_argument('--torchhub-repo',
                        help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )
    parser.add_argument('--use-huggingface', default=False, action='store_true',
                        help='Use the huggingface model')

    args, _ = parser.parse_known_args()

    rank_print('Loading model...')
    model, preprocessor, info = load_model(args.model_version, vitdet_window_size=args.vitdet_window_size,
                                           adaptor_names=args.adaptor_name, force_reload=args.force_reload,
                                           torchhub_repo=args.torchhub_repo, use_huggingface=args.use_huggingface)
    model.to(device=device).eval()
    rank_print('Done')

    if args.resolution is None:
        args.resolution = (model.preferred_resolution.height, model.preferred_resolution.width)

    if args.resize_multiple is None:
        args.resize_multiple = model.min_resolution_step

    transform = get_standard_transform(args.resolution, args.resize_multiple, preprocessor=preprocessor)

    rank_print('Loading dataset...')
    ds_train_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    num_classes = ds_train_builder.info.features['label'].num_classes
    num_train_examples = ds_train_builder.info.splits[args.train_split].num_examples

    # Because data may be downloaded and cached, we want only rank 0 to do that first (to prevent corruption),
    # and then the other ranks will execute once rank 0 is finished
    with run_rank_0_first():
        ds_train_builder.download_and_prepare()

    if args.eval_dataset:
        ds_eval_builder = load_dataset_builder(args.eval_dataset, trust_remote_code=True)
        with run_rank_0_first():
            ds_eval_builder.download_and_prepare()
    else:
        ds_eval_builder = ds_train_builder

    num_eval_examples = ds_eval_builder.info.splits[args.eval_split].num_examples
    assert num_classes == ds_eval_builder.info.features['label'].num_classes, "The number of classes must match between train and eval!"

    num_train_steps = round_up(num_train_examples, args.batch_size * world_size)
    num_eval_steps = round_up(num_eval_examples, args.batch_size * world_size)

    def _get_dataset(builder, split: str):
        dataset = builder.as_dataset(split=split)
        dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, args.workers))
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate,
                            pin_memory=args.workers > 0,
                            drop_last=False,
        )

        return loader

    train_dataset = _get_dataset(ds_train_builder, args.train_split)
    eval_dataset = _get_dataset(ds_eval_builder, args.eval_split)

    rank_print('Loaded dataset!')
    rank_print(f'Description: {ds_train_builder.info.description}')

    # First, create a database of training embeddings with their corresponding labels
    # We only store 1/world_size training embeddings
    rank_print(f'Building {args.train_split} database...')
    train_embeddings, train_labels = _build_database(train_dataset, model, device, num_train_steps, rank, amp=args.amp, adaptor=args.adaptor_name)

    # Gather all of the eval labels onto all of the ranks
    rank_print(f'Building {args.eval_split} database...')
    eval_embeddings, eval_labels = _build_database(eval_dataset, model, device, num_eval_steps, rank, amp=args.amp, adaptor=args.adaptor_name)
    num_valid = eval_labels.shape[0]

    rank_print('Calculating accuracy...')
    knn_top1 = _knn_top1_accuracy(
        train_split_embeddings=train_embeddings,
        train_split_labels=train_labels,
        output=eval_embeddings,
        target=eval_labels,
        distributed=world_size > 1,
        num_classes=num_classes,
        num_valid=num_valid,
        K=args.k,
    )

    total_num_eval = torch.tensor(num_valid, dtype=torch.int64, device=device)
    if world_size > 1:
        dist.reduce(total_num_eval, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(knn_top1, dst=0, op=dist.ReduceOp.SUM)

    accuracy = 100 * knn_top1.item() / total_num_eval.item()

    rank_print(f'Accuracy: {accuracy:.3f}%')


def _get_vote_cls(sim: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    Uses e^`sim` as a weighted vote for the corresponding label in `labels`.

    Returns the label that received the most vote weight.
    """
    weights = torch.exp(
        sim / 0.07
    )  # https://arxiv.org/pdf/1805.01978.pdf, Section 3.4 (Also used by DINO)
    cls_vec = torch.zeros(
        weights.shape[0], num_classes, dtype=weights.dtype, device=weights.device
    )
    cls_vec.scatter_add_(dim=1, index=labels, src=weights)

    # The predicted ID is the the one with the most vote weight
    vote_id = torch.argmax(cls_vec, dim=1)
    return vote_id


def _pad(tensor: torch.Tensor, dim0: int):
    """Utility function to pad a tensor a return a validity mask."""
    valid_mask = torch.ones(dim0, dtype=torch.bool, device=tensor.device)
    valid_mask[tensor.shape[0] :].fill_(False)

    if tensor.shape[0] == dim0:
        # If there is no padding to be done, return the original tensor.
        return tensor, valid_mask

    # Copy valid elements into a new tensor.
    ret = torch.empty(dim0, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    ret[: tensor.shape[0]].copy_(tensor)

    return ret, valid_mask


def _all_to_all(t: torch.Tensor):
    # Unroll the world dim into a list of tensors
    input_tensors = list(t)
    output_tensors = [torch.empty_like(v) for v in input_tensors]

    dist.all_to_all(output_tensors, input_tensors)

    return torch.stack(output_tensors)


def _distributed_topk(
    queries: torch.Tensor,
    keys: torch.Tensor,
    labels: torch.Tensor,
    K: int,
    distributed: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if distributed:
        world_size = dist.get_world_size()
        max_queries = torch.tensor(queries.shape[0], dtype=torch.int64, device=queries.device)
        dist.all_reduce(max_queries, dist.ReduceOp.MAX)

        max_queries = max_queries.item()

        queries, valid_mask = _pad(queries, max_queries)

        all_queries = torch.empty(
            world_size,
            queries.shape[0],
            queries.shape[1],
            dtype=queries.dtype, device=queries.device,
        )
        dist.all_gather_into_tensor(all_queries, queries)
    else:
        all_queries = queries.unsqueeze(0)
        valid_mask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)

    # all_queries: W,Q,C
    # keys: D,C

    # W,Q,D
    similarity = torch.matmul(all_queries, keys.T)

    # W,Q,K
    max_sim, max_idxs = torch.topk(similarity, k=K, dim=2, largest=True, sorted=False)
    max_labels = labels[max_idxs.flatten()].reshape_as(max_idxs)

    if distributed:
        # Queries is the concatenated list of queries for all ranks,
        # which means that max_sim and max_idxs is AllQueries -> KeysForThisRank
        # All to all will then rearrange these to QueriesForThisRank -> AllKeys
        max_sim = _all_to_all(max_sim)
        max_labels = _all_to_all(max_labels)

    # Reduce the per-rank similarities
    # N,K*W
    max_sim = max_sim.permute(1, 2, 0).flatten(1)
    max_labels = max_labels.permute(1, 2, 0).flatten(1)

    if distributed:
        max_sim, max_idxs = torch.topk(max_sim, k=K, dim=1, largest=True, sorted=False)
        max_labels = torch.gather(max_labels, dim=1, index=max_idxs)

    max_sim = max_sim[valid_mask]
    max_labels = max_labels[valid_mask]

    return max_sim, max_labels


def _knn_top1_accuracy(
    train_split_embeddings: torch.Tensor,
    train_split_labels: torch.Tensor,
    output: torch.Tensor,
    target: torch.Tensor,
    distributed: bool,
    num_classes: int,
    num_valid: int,
    K: int = 20,
):
    """Calculate k-NN Top-1 classification accuracy.

    Args:
    * train_split_embeddings: training split embeddings.
    * train_split_labels: training split classes.
    * K: number of top similarities to retain for majority vote.
    * output: embeddings to get nearest neighbors for.
    * target: ground-truth class IDs.
    """

    max_sim, max_labels = _distributed_topk(
        queries=output,
        keys=train_split_embeddings,
        labels=train_split_labels,
        K=K,
        distributed=distributed,
    )

    # Get a weighted vote for each validation sample.
    vote_id = _get_vote_cls(max_sim, max_labels, num_classes=num_classes)

    # Compare against ground truth.
    is_correct = target == vote_id

    is_correct = is_correct[:num_valid]

    # Get total number of correct predictions and calculate accuracy.
    num_correct = is_correct.sum()

    return num_correct


@torch.no_grad()
def _build_database(dataset, model: nn.Module, device: torch.device, num_steps: int, rank: int, amp: bool = True, adaptor: str = None):
    embeddings = []
    db_labels = []
    for batches in tqdm(dataset, total=num_steps, disable=rank > 0):
        for batch in batches:
            images = batch[0].to(device=device, non_blocking=True)
            labels = batch[1].to(device=device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                output = model(images)
                if adaptor is None:
                    summary, features = output
                else:
                    summary, features = output[adaptor]

            summary = F.normalize(summary, p=2, dim=1)

            embeddings.append(summary)
            db_labels.append(labels)

    return torch.cat(embeddings, dim=0), torch.cat(db_labels, dim=0)


if __name__ == '__main__':
    rank = 0
    world_size = 1

    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    main(rank, world_size)
