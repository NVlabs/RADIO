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
from .adaptor_mlp import create_mlp_from_state


class GenericAdaptor(AdaptorBase):
    def __init__(self, main_config: Namespace, adaptor_config, state):
        super().__init__()

        self.head_mlp = create_mlp_from_state(main_config.mlp_version, state, 'summary.')
        self.feat_mlp = create_mlp_from_state(main_config.mlp_version, state, 'feature.')

    def forward(self, input: AdaptorInput) -> RadioOutput:
        summary = self.head_mlp(input.summary)
        feat = self.feat_mlp(input.features)

        return RadioOutput(summary, feat)
