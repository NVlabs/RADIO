# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from argparse import Namespace
import inspect

import torch
from torch import nn
import torch.nn.functional as F

from .adaptor_base import AdaptorBase, AdaptorInput, RadioOutput
from .adaptor_module_factory import create_mlp_from_state, create_mlp_from_config


class GenericAdaptor(AdaptorBase):
    def __init__(self, main_config: Namespace, adaptor_config, state, mlp_config=None):
        super().__init__()

        summary_mlp_version = main_config.mlp_version
        feature_mlp_version = getattr(main_config, 'spatial_mlp_version', None) or summary_mlp_version

        summary_extra_args = dict()
        spatial_extra_args = dict()
        ups = None
        ups_rank = None
        if adaptor_config is not None:
            ups = adaptor_config.get('fd_upsample_factor', None)
            ups_rank = adaptor_config.get('fd_upsample_rank', None)
            summary_mlp_version = adaptor_config.get('summary_mlp_version', summary_mlp_version)
            feature_mlp_version = adaptor_config.get('spatial_mlp_version', feature_mlp_version)
        elif mlp_config is not None:
            ups = mlp_config["feature"].get('upsample_factor', None)
            ups_rank = mlp_config["feature"].get('upsample_rank', None)
        if isinstance(summary_mlp_version, dict):
            d = summary_mlp_version.copy()
            new_type = d.pop('type')
            summary_mlp_version = new_type
            summary_extra_args.update(d)
        if ups is not None:
            spatial_extra_args['upsample_factor'] = ups
            spatial_extra_args['upsample_rank'] = ups_rank

        if state is not None:
            spectral_heads = getattr(main_config, 'spectral_heads', False)
            self.head_mlp = create_mlp_from_state(summary_mlp_version, state, 'summary.', spectral_weights=spectral_heads, is_summary=True, **summary_extra_args)
            if not self.head_mlp.handles_summary_and_spatial:
                self.feat_mlp = create_mlp_from_state(feature_mlp_version, state, 'feature.', spectral_weights=spectral_heads, is_summary=False, **spatial_extra_args)
            else:
                self.feat_mlp = None
        else:
            assert mlp_config is not None, "Config must not be None if state is None"

            self.head_mlp =  create_mlp_from_config(
                summary_mlp_version,
                mlp_config["summary"]["input_dim"],
                mlp_config["summary"]["hidden_dim"],
                mlp_config["summary"]["output_dim"],
                mlp_config["summary"]["num_inner"],
                is_summary=True,
                **summary_extra_args,
            )
            if not self.head_mlp.handles_summary_and_spatial:
                self.feat_mlp = create_mlp_from_config(
                    feature_mlp_version,
                    mlp_config["feature"]["input_dim"],
                    mlp_config["feature"]["hidden_dim"],
                    mlp_config["feature"]["output_dim"],
                    mlp_config["feature"]["num_inner"],
                    is_summary=False,
                    **spatial_extra_args,
                )
            else:
                self.feat_mlp = None

    def forward(self, input: AdaptorInput) -> RadioOutput:
        # Convert input'd type to the type of the first parameter of the adaptor.
        first_param = next(self.parameters())

        # Build extra_args for head_mlp based on its signature
        head_mlp_sig = inspect.signature(self.head_mlp.forward)
        extra_summary_args = {}
        if 'grid_sizes' in head_mlp_sig.parameters:
            extra_summary_args['grid_sizes'] = [input.patch_shape]

        if self.head_mlp.handles_summary_and_spatial:
            summary, feat = self.head_mlp(
                input.summary.to(dtype=first_param.dtype),
                input.features.to(dtype=first_param.dtype),
                **extra_summary_args,
            )
        else:
            summary = self.head_mlp(input.summary.to(dtype=first_param.dtype), **extra_summary_args).to(dtype=input.summary.dtype)
            assert self.feat_mlp is not None
            feat = self.feat_mlp(input.features.to(dtype=first_param.dtype), images=input.images, patch_size=input.patch_size).to(dtype=input.features.dtype)

        if input.feature_fmt == 'NCHW':
            feat = (feat.reshape(feat.shape[0], input.patch_shape[0] * self.feat_mlp.upsample_factor, input.patch_shape[1] * self.feat_mlp.upsample_factor, feat.shape[2])
                        .permute(0, 3, 1, 2)
            )

        return RadioOutput(summary, feat)
