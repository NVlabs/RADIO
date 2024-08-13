# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from typing import Optional
import torch
from torch import nn


class FeatureNormalizer(nn.Module):
    def __init__(self, embed_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        self.register_buffer('mean', torch.zeros(embed_dim, dtype=dtype))
        self.register_buffer('tx', torch.eye(embed_dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x @ self.tx.T
        return x
