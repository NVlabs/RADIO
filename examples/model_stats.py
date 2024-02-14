# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from timm.models.vision_transformer import Attention, Mlp


LAYER_STATS = dict()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Model Stats Demo')
    parser.add_argument('--use-hf', default=False, action='store_true',
                        help='Use RADIO from HuggingFace Hub'
    )
    parser.add_argument('-v', '--model-version', default='radio_v2',
                        help='Which radio model to load.'
    )
    parser.add_argument('-d', '--dataset', default='food101',
                        help='The name of the dataset to classify'
    )

    args, _ = parser.parse_known_args()

    print('Loading model...')
    if args.use_hf:
        from transformers import AutoModel
        model: nn.Module = AutoModel.from_pretrained(f"nvidia/{args.model_version}", trust_remote_code=True)
    else:
        model: nn.Module = torch.hub.load('NVlabs/RADIO', 'radio_model', version=args.model_version, progress=True)

    model.cuda().eval()
    print('Done')

    spect_norms = []
    names = []

    for prm_name, prm in model.named_parameters():
        if prm.ndim < 2 or 'pos_embed' in prm_name:
            continue

        def _add_norm(n, p):
            w = p.flatten(1)
            _, s, _ = torch.linalg.svd(w)

            spect_norms.append(s[0].clone())
            names.append(n)

        if prm_name.endswith('qkv.weight'):
            for n, p in zip(['q', 'k', 'v'], prm.chunk(3)):
                _add_norm(prm_name.replace('qkv', n), p)
        else:
            _add_norm(prm_name, prm)

    spect_norms = torch.stack(spect_norms)

    max_norm, max_norm_idx = spect_norms.max(dim=0)
    med_norm = spect_norms.median().item()
    avg_norm = spect_norms.mean().item()

    print(f'Spectral - Max: {max_norm.item()}, Median: {med_norm}, Mean: {avg_norm}')
    print(f'Max Parameter: {names[max_norm_idx.item()]}')
    sorted_norms, sorted_idxs = torch.sort(spect_norms, descending=True)
    print(f'Norms: {sorted_norms.cpu()}')

    for n, mod in model.named_modules():
        if isinstance(mod, Attention):
            mod.register_forward_hook(_attn_hook(n))
        elif isinstance(mod, Mlp):
            mod.register_forward_hook(_mlp_hook(n))

    transform = transforms.Compose([
        transforms.Resize([378, 378]),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    ds_builder.download_and_prepare()

    dataset = ds_builder.as_dataset(split='train').to_iterable_dataset(num_shards=8)
    dataset = dataset.map(lambda ex: dict(image=transform(ex['image'])))
    loader = DataLoader(dataset, batch_size=16, pin_memory=True, num_workers=8)

    n_iter = 100
    for i, batch in tqdm(enumerate(loader), total=n_iter):
        if i == n_iter:
            break

        images = batch['image'].cuda()

        model(images)

    _normalize_stats(n_iter)

    for layer_name, stats in LAYER_STATS.items():
        print(layer_name)
        for metric_name, metric in stats.items():
            print(f'\t{metric_name} - {metric}')



def _attn_hook(name: str):
    def _hook(module: Attention, input: Tuple[torch.Tensor], output: torch.Tensor):
        input = input[0]
        B, N, C = input.shape
        q, k, v = module.qkv(input).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        q, k = module.q_norm(q), module.k_norm(k)
        q = q * module.scale
        attn = q @ k.transpose(-2, -1)

        logit_stats = _get_stats(attn)

        log_attn = torch.log_softmax(attn, dim=-1, dtype=torch.float64)
        attn = log_attn.exp()
        prod = attn * log_attn
        prod = torch.where(attn < 1e-20, 0, prod)

        entropy = -torch.sum(prod, dim=-1)

        entropy_stats = _get_stats(entropy)

        input_stats = _get_stats(input)
        output_stats = _get_stats(output)

        _add_stats(name, input=input_stats, output=output_stats, attn_logits=logit_stats, attn_entropy=entropy_stats)

    return _hook

def _mlp_hook(name: str):
    def _hook(module: Mlp, input: torch.Tensor, output: torch.Tensor):
        _add_stats(
            name,
            input=_get_stats(input[0]),
            output=_get_stats(output),
        )
    return _hook


def _get_stats(t: torch.Tensor):
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    # Everything but the last dimension is treated as independent
    t = t.reshape(-1, t.shape[-1])

    max_val = t.amax()
    min_val = t.amin()
    mean_val = t.mean()
    median_val = t.median()
    norm = t.norm(dim=1).mean()

    return dict(max_val=max_val, min_val=min_val, mean_val=mean_val, median_val=median_val, norm=norm)


def _add_stats(name: str, **kwargs):
    global LAYER_STATS

    record = {**kwargs}
    if name not in LAYER_STATS:
        LAYER_STATS[name] = record
    else:
        layer = LAYER_STATS[name]
        for n, v in record.items():
            e = layer[n]
            for k in v.keys():
                if 'max' in k:
                    e[k] = torch.maximum(e[k], v[k])
                elif 'min' in k:
                    e[k] = torch.minimum(e[k], v[k])
                else:
                    e[k].add_(v[k])


def _normalize_stats(n_iter: int):
    for v in LAYER_STATS.values():
        for metric in v.values():
            for n in metric.keys():
                stat = metric[n]
                if 'max' not in n and 'min' not in n:
                    stat.div_(n_iter)
                metric[n] = stat.item()



if __name__ == '__main__':
    main()
