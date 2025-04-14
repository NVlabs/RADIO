# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

dependencies = ["torch", "timm", "einops"]

import os
from typing import Dict, Any, Optional, Union, List
import warnings

import torch
from torch.hub import load_state_dict_from_url

from timm.models import clean_state_dict

from radio.adaptor_registry import adaptor_registry
from radio.common import DEFAULT_VERSION, RadioResource, RESOURCE_MAP
from radio.enable_damp import configure_damp_from_args
from radio.enable_spectral_reparam import disable_spectral_reparam, configure_spectral_reparam_from_args
from radio.feature_normalizer import FeatureNormalizer, IntermediateFeatureNormalizer
from radio.radio_model import RADIOModel, create_model_from_args
from radio.input_conditioner import get_default_conditioner
from radio.vitdet import apply_vitdet_arch, VitDetArgs


def radio_model(
    version: str = "",
    progress: bool = True,
    adaptor_names: Union[str, List[str]] = None,
    vitdet_window_size: Optional[int] = None,
    return_checkpoint: bool = False,
    **kwargs,
) -> RADIOModel:
    if not version:
        version = DEFAULT_VERSION

    if os.path.isfile(version):
        chk = torch.load(version, map_location="cpu", weights_only=False)
        resource = RadioResource(version, patch_size=None, max_resolution=None, preferred_resolution=None)
    else:
        resource = RESOURCE_MAP[version]
        chk = load_state_dict_from_url(
            resource.url, progress=progress, map_location="cpu", weights_only=False,
        )

    if "state_dict_ema" in chk:
        state_dict = chk["state_dict_ema"]
        chk['args'].spectral_reparam = False
    else:
        state_dict = chk["state_dict"]

    args = chk["args"]
    mod = create_model_from_args(args)

    mod_state_dict = get_prefix_state_dict(state_dict, "base_model.")

    if args.spectral_reparam:
        configure_spectral_reparam_from_args(mod, args, state_dict_guidance=mod_state_dict)

    if getattr(args, 'damp', None):
        configure_damp_from_args(mod, args)

    state_dict = clean_state_dict(state_dict)

    key_warn = mod.load_state_dict(mod_state_dict, strict=False)
    if key_warn.missing_keys:
        warnings.warn(f'Missing keys in state dict: {key_warn.missing_keys}')
    if key_warn.unexpected_keys:
        warnings.warn(f'Unexpected keys in state dict: {key_warn.unexpected_keys}')

    if chk['args'].spectral_reparam:
        # Spectral reparametrization uses PyTorch's "parametrizations" API. The idea behind
        # the method is that instead of there being a `weight` tensor for certain Linear layers
        # in the model, we make it a dynamically computed function. During training, this
        # helps stabilize the model. However, for downstream use cases, it shouldn't be necessary.
        # Disabling it in this context means that instead of having `w' = f(w)`, we just compute `w' = f(w)`
        # once, during this function call, and replace the parametrization with the realized weights.
        # This makes the model run faster, and also use less memory.
        disable_spectral_reparam(mod)
        chk['args'].spectral_reparam = False

    conditioner = get_default_conditioner()
    conditioner.load_state_dict(get_prefix_state_dict(state_dict, "input_conditioner."))

    dtype = getattr(chk['args'], 'dtype', torch.float32)
    mod.to(dtype=dtype)
    conditioner.dtype = dtype

    cls_token_per_teacher = getattr(chk['args'], 'cls_token_per_teacher', True)
    if cls_token_per_teacher:
        name_to_idx_map = dict()
        for i, t in enumerate(chk['args'].teachers):
            if t.get('use_summary', True):
                name = t['name']
                if name not in name_to_idx_map:
                    name_to_idx_map[name] = i
        summary_idxs = torch.tensor(sorted(name_to_idx_map.values()), dtype=torch.int64)
    else:
        summary_idxs = torch.tensor([0], dtype=torch.int64)

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

        pf_idx_head = f'_heads.{tidx}'
        pf_name_head = f'_heads.{adaptor_name}'
        pf_idx_feat = f'_feature_projections.{tidx}'
        pf_name_feat = f'_feature_projections.{adaptor_name}'

        adaptor_state = dict()
        for k, v in state_dict.items():
            if k.startswith(pf_idx_head):
                adaptor_state['summary' + k[len(pf_idx_head):]] = v
            elif k.startswith(pf_name_head):
                adaptor_state['summary' + k[len(pf_name_head):]] = v
            elif k.startswith(pf_idx_feat):
                adaptor_state['feature' + k[len(pf_idx_feat):]] = v
            elif k.startswith(pf_name_feat):
                adaptor_state['feature' + k[len(pf_name_feat):]] = v

        adaptor = adaptor_registry.create_adaptor(ttype, chk["args"], tconf, adaptor_state)
        adaptor.head_idx = tidx if cls_token_per_teacher else 0
        adaptors[adaptor_name] = adaptor

    feat_norm_sd = get_prefix_state_dict(state_dict, '_feature_normalizer.')
    feature_normalizer = None
    if feat_norm_sd:
        feature_normalizer = FeatureNormalizer(feat_norm_sd['mean'].shape[0], dtype=dtype)
        feature_normalizer.load_state_dict(feat_norm_sd)

    inter_feat_norm_sd = get_prefix_state_dict(state_dict, '_intermediate_feature_normalizer.')
    inter_feature_normalizer = None
    if inter_feat_norm_sd:
        inter_feature_normalizer = IntermediateFeatureNormalizer(
            *inter_feat_norm_sd['means'].shape[:2],
            rot_per_layer=inter_feat_norm_sd['rotation'].ndim == 3,
            dtype=dtype
        )
        inter_feature_normalizer.load_state_dict(inter_feat_norm_sd)

    radio = RADIOModel(
        mod,
        conditioner,
        summary_idxs=summary_idxs,
        patch_size=resource.patch_size,
        max_resolution=resource.max_resolution,
        window_size=vitdet_window_size,
        preferred_resolution=resource.preferred_resolution,
        adaptors=adaptors,
        feature_normalizer=feature_normalizer,
        inter_feature_normalizer=inter_feature_normalizer,
    )

    if vitdet_window_size is not None:
        apply_vitdet_arch(
            mod,
            VitDetArgs(
                vitdet_window_size,
                radio.num_summary_tokens,
                num_windowed=resource.vitdet_num_windowed,
                num_global=resource.vitdet_num_global,
            ),
        )

    if return_checkpoint:
        return radio, chk
    return radio


def get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    mod_state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    return mod_state_dict
