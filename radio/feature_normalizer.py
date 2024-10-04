# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from collections import namedtuple
from typing import NamedTuple, Optional, Tuple
import torch
from torch import nn


def _run_kernel(x: torch.Tensor, mean: torch.Tensor, tx: torch.Tensor):
    if x.ndim <= 3:
        x = x - mean
        x = x @ tx.T
    elif x.ndim == 4:
        x = x - mean.reshape(1, -1, 1, 1)
        kernel = tx.reshape(*tx.shape, 1, 1)
        x = torch.nn.functional.conv2d(x, weight=kernel, bias=None, stride=1, padding=0)
    else:
        raise ValueError(f'Unsupported input dimension: {x.ndim}, shape: {x.shape}')
    return x


class FeatureNormalizer(nn.Module):
    def __init__(self, embed_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        self.register_buffer('mean', torch.zeros(embed_dim, dtype=dtype))
        self.register_buffer('tx', torch.eye(embed_dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _run_kernel(x, self.mean, self.tx)
        return x


class InterFeatState(NamedTuple):
    y: torch.Tensor
    alpha: torch.Tensor


class IntermediateFeatureNormalizerBase(nn.Module):
    def forward(self, x: torch.Tensor, index: int, skip: Optional[int] = None) -> InterFeatState:
        raise NotImplementedError()


class IntermediateFeatureNormalizer(IntermediateFeatureNormalizerBase):
    def __init__(self, num_intermediates: int, embed_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer('alphas', torch.ones(num_intermediates, dtype=dtype))
        self.register_buffer('rotation', torch.eye(embed_dim, dtype=dtype))
        self.register_buffer('means', torch.zeros(num_intermediates, embed_dim, dtype=dtype))

    def forward(self, x: torch.Tensor, index: int, skip: Optional[int] = None) -> InterFeatState:
        if skip:
            prefix, x = x[:, :skip], x[:, skip:]

        y = _run_kernel(x, self.means[index], self.rotation)

        alpha = self.alphas[index]
        if skip:
            alpha = torch.cat([
                torch.ones(skip, dtype=alpha.dtype, device=alpha.device),
                alpha[None].expand(y.shape[1]),
            ]).reshape(1, -1, 1)
            y = torch.cat([prefix, y], dim=1)
        else:
            if x.ndim == 3:
                alpha = alpha.reshape(1, 1, 1).expand(1, y.shape[1], 1)
            elif x.ndim == 4:
                alpha = alpha.reshape(1, 1, 1, 1).expand(1, 1, *y.shape[2:])
            else:
                raise ValueError(f'Unsupported input dimension: {x.ndim}')

        return InterFeatState(y, alpha)


class NullIntermediateFeatureNormalizer(IntermediateFeatureNormalizerBase):
    instance = None

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(1, dtype=dtype, device=device))

    @staticmethod
    def get_instance(dtype: torch.dtype, device: torch.device):
        if NullIntermediateFeatureNormalizer.instance is None:
            NullIntermediateFeatureNormalizer.instance = NullIntermediateFeatureNormalizer(dtype, device)
        return NullIntermediateFeatureNormalizer.instance

    def forward(self, x: torch.Tensor, index: int, skip: Optional[int] = None) -> InterFeatState:
        return InterFeatState(x, self.alpha)
