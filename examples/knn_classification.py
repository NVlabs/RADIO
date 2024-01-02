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

import torchvision.transforms.v2 as transforms

from datasets import load_dataset_builder, load_dataset
from datasets.iterable_dataset import DistributedConfig
from datasets.distributed import split_dataset_by_node


def main(rank: int = 0, world_size: int = 1):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    device = torch.device('cuda', local_rank)

    parser = argparse.ArgumentParser(description='kNN Classification Demo')
    parser.add_argument('--use-hf', default=False, action='store_true',
                        help='Use RADIO from HuggingFace Hub'
    )
    parser.add_argument('-v', '--model-version', default='radio_v1',
                        help='Which radio model to load.'
    )

    parser.add_argument('-r', '--resolution', nargs='+', type=int, default=(378, 378),
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
    parser.add_argument('--resize-multiple', type=int, default=14,
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

    args, _ = parser.parse_known_args()

    def rank_print(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    rank_print('Loading model...')
    if args.use_hf:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(f"nvidia/{args.model_version}", trust_remote_code=True)
    else:
        model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=args.model_version, progress=True)

    model.to(device=device).eval()
    rank_print('Done')

    transform = [
        ResizeTransform(args.resolution, args.resize_multiple),
    ]
    if len(args.resolution) == 2:
        transform.append(transforms.CenterCrop(args.resolution))
    transform.extend([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    transform = transforms.Compose(transform)

    rank_print('Loading dataset...')
    ds_train_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    num_classes = ds_train_builder.info.features['label'].num_classes
    num_train_examples = ds_train_builder.info.splits[args.train_split].num_examples

    def _prepare(builder):
        for i in range(min(2, world_size)):
            if i == rank or (i > 0 and rank > 0):
                builder.download_and_prepare()
            if world_size > 1:
                dist.barrier()
    _prepare(ds_train_builder)

    if args.eval_dataset:
        ds_eval_builder = load_dataset_builder(args.eval_dataset, trust_remote_code=True)
        _prepare(ds_eval_builder)
    else:
        ds_eval_builder = ds_train_builder

    num_eval_examples = ds_eval_builder.info.splits[args.eval_split].num_examples
    assert num_classes == ds_eval_builder.info.features['label'].num_classes, "The number of classes must match between train and eval!"

    num_train_steps = _round_up(num_train_examples, args.batch_size * world_size)
    num_eval_steps = _round_up(num_eval_examples, args.batch_size * world_size)

    def _get_dataset(builder, split: str):
        dataset = builder.as_dataset(split=split)
        dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, args.workers))
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=_collate,
                            pin_memory=args.workers > 0,
                            drop_last=False,
        )

        return loader

    train_dataset = _get_dataset(ds_train_builder, args.train_split)
    eval_dataset = _get_dataset(ds_eval_builder, args.eval_split)

    rank_print('Loaded dataset!')
    rank_print(f'Description: {ds_train_builder.info.description}')

    if not args.resolution:
        args.resolution = (378, 378)

    # First, create a database of training embeddings with their corresponding labels
    # We only store 1/world_size training embeddings
    rank_print(f'Building {args.train_split} database...')
    train_embeddings, train_labels = _build_database(train_dataset, model, device, num_train_steps, rank)

    # Gather all of the eval labels onto all of the ranks
    rank_print(f'Building {args.eval_split} database...')
    eval_embeddings, eval_labels = _build_database(eval_dataset, model, device, num_eval_steps, rank)
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


class ResizeTransform(transforms.Transform):
    def __init__(self, size: Iterable[int], resize_multiple: int = 1):
        super().__init__()

        self.size = size
        self.resize_multiple = resize_multiple

    def _get_nearest(self, value: int):
        return int(round(value / self.resize_multiple) * self.resize_multiple)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = transforms._utils.query_size(flat_inputs)

        if len(self.size) == 1:
            # Shortest-side mode.
            # Resize the short dimension of the image to be the specified size,
            # and the other dimension aspect preserving
            min_sz = min(height, width)
            factor = self.size[0] / min_sz

            rs_height = height * factor
            rs_width = width * factor
            size = (rs_height, rs_width)
        elif len(self.size) == 2:
            # Center-crop mode (the actual crop will be done by subsequent transform)
            in_aspect = height / width
            out_aspect = self.size[0] / self.size[1]

            # Input height varies faster than output
            if in_aspect > out_aspect:
                scale = self.size[1] / width
            else:
                scale = self.size[0] / height

            rs_height = height * scale
            rs_width = width * scale
            size = (rs_height, rs_width)
        else:
            raise ValueError("Unsupported resize mode")

        size = tuple(self._get_nearest(d) for d in size)

        return dict(size=size)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, Image.Image):
            inpt = inpt.convert('RGB')

        size = params['size']

        return transforms.functional.resize(inpt, size=size, interpolation=transforms.InterpolationMode.BICUBIC)


def _round_up(value, multiple: int):
    return int(math.ceil(value / multiple))


def _collate(samples: List[Dict[str, torch.Tensor]]):
    images = [
        s['image']
        for s in samples
    ]
    labels = [
        s['label']
        for s in samples
    ]

    size_groups = defaultdict(lambda: [[],[]])
    for im, lab in zip(images, labels):
        grp = size_groups[im.shape]
        grp[0].append(im)
        grp[1].append(lab)

    ret = [
        (torch.stack(g[0]), torch.stack(g[1]))
        for g in size_groups.values()
    ]
    return ret


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
def _build_database(dataset, model: nn.Module, device: torch.device, num_steps: int, rank: int):
    embeddings = []
    db_labels = []
    for batches in tqdm(dataset, total=num_steps, disable=rank > 0):
        for batch in batches:
            images = batch[0].to(device=device, non_blocking=True)
            labels = batch[1].to(device=device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.bfloat16):
                summary, features = model(images)

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
