# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

dependencies = ["torch", "timm", "einops"]

from dataclasses import dataclass
import os
from typing import Dict, Any, Optional, Union, List
import warnings

import torch
from torch.hub import load_state_dict_from_url

from timm.models import clean_state_dict

from radio.adaptors import adaptor_registry
from radio.radio_model import RADIOModel, create_model_from_args, Resolution
from radio.input_conditioner import get_default_conditioner
from radio.vitdet import apply_vitdet_arch, VitDetArgs


@dataclass
class RadioResource:
    url: str
    patch_size: int
    max_resolution: int
    preferred_resolution: Resolution


resource_map = {
    "radio_v1": RadioResource("https://huggingface.co/nvidia/RADIO/resolve/main/radio_v1.pth.tar?download=true",
                              patch_size=14, max_resolution=1050,
                              preferred_resolution=Resolution(378, 378),
    ),
    "radio_v2": RadioResource("https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.pth.tar?download=true",
                              patch_size=16, max_resolution=2048,
                              preferred_resolution=Resolution(432, 432),
    ),
}

_DEFAULT_VERSION = "radio_v2"


def radio_model(
    version: str = "",
    progress: bool = True,
    return_summary: bool = True,
    return_spatial_features: bool = True,
    adaptor_names: Union[str, List[str]] = None,
    vitdet_window_size: Optional[int] = None,
    **kwargs,
) -> RADIOModel:
    if not version:
        version = _DEFAULT_VERSION

    if os.path.isfile(version):
        chk = torch.load(version, map_location="cpu")
        resource = RadioResource(version, patch_size=None, max_resolution=None, preferred_resolution=None)
    else:
        resource = resource_map[version]
        chk = load_state_dict_from_url(
            resource.url, progress=progress, map_location="cpu"
        )

    mod = create_model_from_args(chk["args"])

    if "state_dict_ema" in chk:
        state_dict = chk["state_dict_ema"]
    else:
        state_dict = chk["state_dict"]
    state_dict = clean_state_dict(state_dict)

    key_warn = mod.load_state_dict(get_prefix_state_dict(state_dict, "base_model."), strict=False)
    if key_warn.missing_keys:
        warnings.warn(f'Missing keys in state dict: {key_warn.missing_keys}')
    if key_warn.unexpected_keys:
        warnings.warn(f'Unexpected keys in state dict: {key_warn.unexpected_keys}')

    conditioner = get_default_conditioner()
    conditioner.load_state_dict(get_prefix_state_dict(state_dict, "input_conditioner."))

    summary_idxs = torch.tensor([
        i
        for i, t in enumerate(chk["args"].teachers)
        if t.get("use_summary", True)
    ], dtype=torch.int64)

    if vitdet_window_size is not None:
        apply_vitdet_arch(mod, VitDetArgs(vitdet_window_size, radio.num_summary_tokens))

    if adaptor_names is None:
        adaptor_names = []
    elif isinstance(adaptor_names, str):
        adaptor_names = [adaptor_names]

    teachers = chk["args"].teachers
    adaptors = dict()
    for adaptor_name in adaptor_names:
        for tidx, tconf in enumerate(teachers):
            if tconf["name"] == adaptor_name:
                break
        else:
            raise ValueError(f'Unable to find the specified adaptor name. Known names: {list(t["name"] for t in teachers)}')

        ttype = tconf["type"]

        pf_head = f'_heads.{tidx}'
        pf_feat = f'_feature_projections.{tidx}'

        adaptor_state = dict()
        for k, v in state_dict.items():
            if k.startswith(pf_head):
                adaptor_state['summary' + k[len(pf_head):]] = v
            elif k.startswith(pf_feat):
                adaptor_state['feature' + k[len(pf_feat):]] = v

        adaptor = adaptor_registry.create_adaptor(ttype, chk["args"], tconf, adaptor_state)
        adaptor.head_idx = tidx
        adaptors[adaptor_name] = adaptor

    radio = RADIOModel(
        mod,
        conditioner,
        summary_idxs=summary_idxs,
        patch_size=resource.patch_size,
        max_resolution=resource.max_resolution,
        window_size=vitdet_window_size,
        preferred_resolution=resource.preferred_resolution,
        adaptors=adaptors,
    )

    return radio


def get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    mod_state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    return mod_state_dict
