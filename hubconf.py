# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

dependencies = ["torch", "timm", "einops"]

from typing import Dict, Any

from torch.hub import load_state_dict_from_url

from timm.models import clean_state_dict

from radio.model import RADIOModel, create_model_from_args
from radio.input_conditioner import get_default_conditioner


resource_map = {
    "radio_v1": "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v1.pth.tar?download=true"
}

_DEFAULT_VERSION = "radio_v1"


def radio_model(
    version: str = "",
    progress: bool = True,
    return_summary: bool = True,
    return_spatial_features: bool = True,
    **kwargs,
) -> RADIOModel:
    if not version:
        version = _DEFAULT_VERSION

    chk = load_state_dict_from_url(
        resource_map[version], progress=progress, map_location="cpu"
    )

    mod = create_model_from_args(chk["args"])

    state_dict = chk["state_dict"]
    state_dict = clean_state_dict(state_dict)

    mod.load_state_dict(get_prefix_state_dict(state_dict, "base_model."), strict=False)

    conditioner = get_default_conditioner()
    conditioner.load_state_dict(get_prefix_state_dict(state_dict, "input_conditioner."))

    return RADIOModel(
        mod,
        conditioner,
        return_summary=return_summary,
        return_spatial_features=return_spatial_features,
    )


def get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    mod_state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    return mod_state_dict
