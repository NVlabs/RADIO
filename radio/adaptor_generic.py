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

        if state is not None:
            self.head_mlp = create_mlp_from_state(main_config.mlp_version, state, 'summary.')
            self.feat_mlp = create_mlp_from_state(main_config.mlp_version, state, 'feature.')
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
            )

    def forward(self, input: AdaptorInput) -> RadioOutput:
        summary = self.head_mlp(input.summary)
        feat = self.feat_mlp(input.features)

        return RadioOutput(summary, feat)
