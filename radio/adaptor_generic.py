# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from argparse import Namespace

import torch
from torch import nn
import torch.nn.functional as F

from .adaptor_base import AdaptorBase, AdaptorInput, RadioOutput
from .adaptor_mlp import create_mlp_from_state, create_mlp_from_config


class GenericAdaptor(AdaptorBase):
    def __init__(self, main_config: Namespace, adaptor_config, state, mlp_config=None):
        super().__init__()

        extra_args = dict()
        ups = None
        ups_rank = None
        if adaptor_config is not None:
            ups = adaptor_config.get('fd_upsample_factor', None)
            ups_rank = adaptor_config.get('fd_upsample_rank', None)
        elif mlp_config is not None:
            ups = mlp_config["feature"].get('upsample_factor', None)
            ups_rank = mlp_config["feature"].get('upsample_rank', None)
        if ups is not None:
            extra_args['upsample_factor'] = ups
            extra_args['upsample_rank'] = ups_rank

        if state is not None:
            spectral_heads = getattr(main_config, 'spectral_heads', False)
            self.head_mlp = create_mlp_from_state(main_config.mlp_version, state, 'summary.', spectral_weights=spectral_heads)
            self.feat_mlp = create_mlp_from_state(main_config.mlp_version, state, 'feature.', spectral_weights=spectral_heads, **extra_args)
        else:
            assert mlp_config is not None, "Config must not be None if state is None"

            self.head_mlp =  create_mlp_from_config(
                main_config.mlp_version,
                mlp_config["summary"]["input_dim"],
                mlp_config["summary"]["hidden_dim"],
                mlp_config["summary"]["output_dim"],
                mlp_config["summary"]["num_inner"],
            )
            self.feat_mlp = create_mlp_from_config(
                main_config.mlp_version,
                mlp_config["feature"]["input_dim"],
                mlp_config["feature"]["hidden_dim"],
                mlp_config["feature"]["output_dim"],
                mlp_config["feature"]["num_inner"],
                **extra_args
            )

    def forward(self, input: AdaptorInput) -> RadioOutput:
        # Convert input'd type to the type of the first parameter of the adaptor.
        first_param = next(self.parameters())
        summary = self.head_mlp(input.summary.to(dtype=first_param.dtype)).to(dtype=input.summary.dtype)
        feat = self.feat_mlp(input.features.to(dtype=first_param.dtype), images=input.images, patch_size=input.patch_size).to(dtype=input.features.dtype)

        if input.feature_fmt == 'NCHW':
            feat = (feat.reshape(feat.shape[0], input.images.shape[-2] // input.patch_size * self.feat_mlp.upsample_factor, input.images.shape[-1] // input.patch_size * self.feat_mlp.upsample_factor, feat.shape[2])
                        .permute(0, 3, 1, 2)
            )

        return RadioOutput(summary, feat)
