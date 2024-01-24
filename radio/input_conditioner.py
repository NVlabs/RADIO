# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Union, Tuple

import torch
from torch import nn


norm_t = Union[Tuple[float, float, float], torch.Tensor]

class InputConditioner(nn.Module):
    def __init__(self,
                 input_scale: float,
                 norm_mean: norm_t,
                 norm_std: norm_t,
                 dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.dtype = dtype

        # self.input_scale = input_scale
        self.register_buffer("norm_mean", _to_tensor(norm_mean) / input_scale)
        self.register_buffer("norm_std", _to_tensor(norm_std) / input_scale)

    def forward(self, x: torch.Tensor):
        # x = x * self.input_scale
        y = (x - self.norm_mean) / self.norm_std
        return y.to(self.dtype)


def get_default_conditioner():
    from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

    return InputConditioner(
        input_scale=1.0,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
    )


def _to_tensor(v: norm_t):
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1)
