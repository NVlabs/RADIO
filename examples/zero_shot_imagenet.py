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
import inspect
import math
import os
from PIL import Image
import sys
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

from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES

BASE_IMAGENET_TEMPLATES = (
    lambda c: f'an image of a {c}.',
)

from common import collate, round_up, get_standard_transform, get_rank, get_world_size, barrier, rank_print, load_model


def main(rank: int = 0, world_size: int = 1):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    device = torch.device('cuda', local_rank)
    parser = argparse.ArgumentParser(description='ZeroShot Classification Demo')

    parser.add_argument('-v', '--model-version', default='radio_v2',
                        help='Which radio model to load.'
    )
    parser.add_argument('-a', '--adaptor-name', default='clip', help='Which head to use')
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
    parser.add_argument('--resize-multiple', type=int, default=None,
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
    parser.add_argument('--use-huggingface', default=False, action='store_true',
                        help='Use the huggingface model')
    parser.add_argument('--csv-out', type=str, default=None,
                        help='Append the zero shot score to the specified csv')
    parser.add_argument('--csv-exp-name', type=str, default=None,
                        help='Add a name as the first column of the csv')
    parser.add_argument('--templates', default='openai', choices=['openai', 'simple', 'base'],
                        help='Which set of templates to use'
    )
    parser.add_argument('--tome-config', type=str, default=None,
                        help='Configure the model with ToMe')

    parser.add_argument('--drop-span', type=int, nargs='+', default=None)
    parser.add_argument('--invalidate-cache', default=False, action='store_true')

    args, _ = parser.parse_known_args()

    rank_print('Loading model...')
    model, preprocessor, info = load_model(args.model_version, adaptor_names=args.adaptor_name, return_spatial_features=False,
                                           vitdet_window_size=args.vitdet_window_size, force_reload=args.force_reload,
                                           torchhub_repo=args.torchhub_repo, use_huggingface=args.use_huggingface,
                                           tome_config=args.tome_config)
    model.to(device=device).eval()
    rank_print('Done')

    if args.drop_span:
        drop_span = args.drop_span
        if len(drop_span) == 1:
            drop_span = (drop_span[0], drop_span[0] + 1)
        for i in range(*drop_span):
            model.blocks[i] = nn.Identity()

    rank_print('Loading dataset...')
    ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    ds_builder.download_and_prepare()
    num_examples = ds_builder.info.splits[args.split].num_examples

    if args.resolution is None:
        args.resolution = (model.preferred_resolution.height, model.preferred_resolution.width)

    if args.resize_multiple is None:
        args.resize_multiple = getattr(model, 'min_resolution_step', model.patch_size)

    transform = get_standard_transform(args.resolution, args.resize_multiple, preprocessor=preprocessor)
    dataset = ds_builder.as_dataset(split=args.split)
    dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, args.workers))
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=args.workers > 0,
                        drop_last=False,
    )
    num_steps = round_up(num_examples, args.batch_size * world_size)
    rank_print('Done')
    rank_print(f'Description: {ds_builder.info.description}')

    rank_print('Building Zero Shot Classifier...')
    adaptor = model.adaptors[args.adaptor_name] if hasattr(model, 'adaptors') else model
    if args.templates == 'openai':
        templates = OPENAI_IMAGENET_TEMPLATES
    elif args.templates == 'simple':
        templates = SIMPLE_IMAGENET_TEMPLATES
    elif args.templates == 'base':
        templates = BASE_IMAGENET_TEMPLATES
    else:
        raise ValueError(f'Unknown template set: {args.templates}')
    classifier = get_clip_classifier(
        model=adaptor, tokenizer=adaptor.tokenizer, model_key=args.model_version, adaptor_key=args.adaptor_name, device=device,
        templates=templates, invalidate_cache=args.invalidate_cache,
    ).float()
    rank_print('Done')

    rank_print('Classifying...')
    topks = {
        k: torch.tensor(0.0, dtype=torch.float32, device=device)
        for k in (1, 5)
    }
    num_processed = 0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    postproc = getattr(model, 'zero_shot_postproc', lambda x: x)

    total_tokens = torch.tensor(0, device=device, dtype=torch.float64)
    with torch.inference_mode(), tqdm(total=num_examples, disable=rank > 0) as t:
        for i, batches in enumerate(loader):
            if i == 1:
                if world_size > 1:
                    dist.barrier()
                start_time.record()
            for images, targets in batches:
                images = images.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)

                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                    output = model(images)
                    ada_output = output[args.adaptor_name]
                    summary = ada_output.summary
                    summary = F.normalize(summary, dim=-1)

                    features = ada_output.features
                    total_tokens += features.shape[0] * features.shape[1]

                    logits = summary.to(classifier.dtype) @ classifier
                    logits = postproc(logits)

                    accs = accuracy(logits, targets, topk=topks.keys())
                    for k, acc in zip(topks.keys(), accs):
                        topks[k].add_(acc * images.shape[0])
                    num_processed += images.shape[0]

            t.set_postfix({'Rank': '0', **{f'Top-{k}': f'{v.item() / num_processed:.03f}' for k, v in topks.items()}})
            t.update(world_size * args.batch_size)

    if world_size > 1:
        rank_print('\tWaiting for all ranks to complete...')
        num_processed = torch.tensor(num_processed, device=device)
        dist.reduce(num_processed, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_tokens, dst=0, op=dist.ReduceOp.SUM)

        for k, acc in topks.items():
            dist.reduce(acc, dst=0, op=dist.ReduceOp.SUM)
        rank_print('\tDone')
    rank_print('Done')

    avg_tokens = (total_tokens / num_processed).item()
    rank_print(f'Average Tokens per Image: {avg_tokens:.2f}')

    end_time.record()
    end_time.synchronize()
    elapsed_time_sec = start_time.elapsed_time(end_time) / 1000.0

    rank_print(f'Resolution: {args.resolution}')
    rank_print('Accuracy:')
    for k, acc in topks.items():
        acc = (acc / num_processed).item()

        rank_print(f'\tTop {k}: {acc:.3f}')

        if rank == 0 and k == 1 and args.csv_out:
            with open(args.csv_out, 'a') as fd:
                if args.csv_exp_name:
                    fd.write(f'{args.csv_exp_name},')
                fd.write(f'{" ".join(str(r) for r in args.resolution)},{acc:.4f},{elapsed_time_sec:.3f},{avg_tokens:.3f}\n')
                # drop_span = args.drop_span
                # if not drop_span:
                #     drop_span = ['baseline', '']
                # elif len(drop_span) == 1:
                #     drop_span = [drop_span[0], drop_span[0] + 1]
                # fd.write(f'{drop_span[0]},{drop_span[1]},{acc:.4f}\n')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


@torch.no_grad()
def get_clip_classifier(model, tokenizer,
                        model_key: str,
                        adaptor_key: str,
                        device: torch.device,
                        dist_group = None,
                        normalize_intermediate: bool = True,
                        templates: List[str] = OPENAI_IMAGENET_TEMPLATES,
                        classnames: List[str] = IMAGENET_CLASSNAMES,
                        invalidate_cache: bool = False,
):
    """
    Build zero-shot classifier weights.

    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        device: Device to use
        dist_group: Optional - torch.distributed group to spread work across
    """
    # NOTE(mranzinger): This comes from open_clip.build_zero_shot_classifier
    # where the only difference is that computation is spread across all ranks
    # in the distributed group, as opposed to the open_clip implementation which
    # computes everything for every rank.
    model.to(device=device)

    temp_flat = '_'.join([template('') for template in templates])
    cls_flat = '_'.join(classnames)
    norm_key = "_norm" if normalize_intermediate else ""
    key_str = (model_key + adaptor_key + temp_flat + cls_flat + norm_key).encode('utf-8')

    cache_hash = sha256(key_str).hexdigest()[:16]

    cache_dir = os.path.join(hub.get_dir(), 'NVlabs_RADIO_main_clip/classifier')
    os.makedirs(cache_dir, exist_ok=True)

    barrier()

    cache_file = os.path.join(cache_dir, f'{cache_hash}.pt')
    if not invalidate_cache and os.path.exists(cache_file):
        cache = torch.load(cache_file, map_location=device)
        return cache

    use_format = isinstance(templates[0], str)

    rank = get_rank(dist_group)
    world_size = get_world_size(dist_group)

    classes_per_rank = round_up(len(classnames), world_size)
    num_even_classes = classes_per_rank * world_size

    # This will make the number of classes processed per-rank the same
    # for all ranks
    even_classnames = list(classnames) + ['' for _ in range(num_even_classes - len(classnames))]

    my_classnames = even_classnames[rank * classes_per_rank : (rank + 1) * classes_per_rank]

    texts = []
    for classname in my_classnames:
        for template in templates:
            text = template.format(classname) if use_format else template(classname)
            texts.append(text)

    all_embeddings = []

    has_raw_text_arg = 'raw_text' in inspect.signature(model.encode_text).parameters

    batch_size = 1024
    for i in tqdm(range(0, len(texts), batch_size), desc="batch", disable=rank > 0):
        curr_texts = texts[i : i + batch_size]
        tokens = tokenizer(curr_texts).to(device)
        text_args = dict()
        if has_raw_text_arg:
            text_args['raw_text'] = curr_texts

        embeddings = model.encode_text(tokens, normalize=normalize_intermediate, **text_args)
        all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = all_embeddings.reshape(len(my_classnames), len(templates), -1)

    class_embeddings = all_embeddings.mean(dim=1)  # Average over templates
    class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)

    if world_size > 1:
        world_class_embeddings = torch.empty(
            len(even_classnames), class_embeddings.shape[-1],
            dtype=class_embeddings.dtype, device=device,
        )
        dist.all_gather_into_tensor(world_class_embeddings, class_embeddings, group=dist_group)
        class_embeddings = world_class_embeddings

    class_embeddings = class_embeddings[:len(classnames)].T

    if get_rank() == 0:
        torch.save(class_embeddings, cache_file)
    barrier()

    return class_embeddings


if __name__ == '__main__':
    rank = 0
    world_size = 1

    if 'WORLD_SIZE' in os.environ and '--help' not in sys.argv:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    main(rank, world_size)
