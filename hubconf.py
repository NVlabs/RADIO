# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

dependencies = ['torch', 'timm', 'einops']

from argparse import Namespace
from typing import Union, Dict, Any

import torch
from torch import nn
from torch.hub import load_state_dict_from_url

from timm.models import clean_state_dict, VisionTransformer

from radio.model import create_model_from_args
from radio.input_conditioner import get_default_conditioner, InputConditioner


resource_map = {
    'radio_v1': 'https://huggingface.co/nvidia/RADIO/resolve/main/radio_v1.pth.tar?download=true'
}

_DEFAULT_VERSION = 'radio_v1'


class RADIOModel(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 input_conditioner: InputConditioner,
                 return_summary: bool,
                 return_spatial_features: bool,
    ):
        super().__init__()

        self.model = model
        self.input_conditioner = input_conditioner
        self.return_summary = return_summary
        self.return_spatial_features = return_spatial_features

    def forward(self, x: torch.Tensor):
        x = self.input_conditioner(x)

        y = self.model.forward_features(x)

        if isinstance(y, (list, tuple)):
            summary, all_feat = y
        elif isinstance(self.model, VisionTransformer):
            patch_gen = getattr(self.model, 'patch_generator', None)
            if patch_gen is not None:
                summary = y[:, :patch_gen.num_cls_tokens].flatten(1)
                all_feat = y[:, patch_gen.num_skip:]
            elif self.model.global_pool == 'avg':
                summary = y[:, self.model.num_prefix_tokens:].mean(dim=1)
                all_feat = y
            else:
                summary = y[:, 0]
                all_feat = y[:, 1:]
        else:
            raise ValueError("Unsupported model type")

        if self.return_summary and self.return_spatial_features:
            return summary, all_feat
        elif self.return_summary:
            return summary
        return all_feat


def radio_model(version: str = '',
                       progress: bool = True,
                       return_summary: bool = True,
                       return_spatial_features: bool = True,
                       **kwargs,
) -> RADIOModel:
    if not version:
        version = _DEFAULT_VERSION

    chk = load_state_dict_from_url(resource_map[version], progress=progress, map_location='cpu')

    mod = create_model_from_args(chk['args'])

    state_dict = chk['state_dict']
    state_dict = clean_state_dict(state_dict)

    mod.load_state_dict(_get_prefix_state_dict(state_dict, 'base_model.'), strict=False)

    conditioner = get_default_conditioner()
    conditioner.load_state_dict(_get_prefix_state_dict(state_dict, 'input_conditioner.'))

    return RADIOModel(mod, conditioner, return_summary=return_summary, return_spatial_features=return_spatial_features)


def _get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    mod_state_dict = {
        k[len(prefix):]: v
        for k, v
        in state_dict.items()
        if k.startswith(prefix)
    }
    return mod_state_dict
