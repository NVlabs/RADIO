# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import List, Optional, Set, Tuple, Union
from types import MethodType

import torch
from torch import nn

from timm.models import VisionTransformer, checkpoint_seq

from .vit_patch_generator import ViTPatchGenerator


def _forward_cpe(self: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_generator(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return x


def _take_indices(
        num_blocks: int,
        n: Optional[Union[int, List[int], Tuple[int]]],
) -> Tuple[Set[int], int]:
    if isinstance(n, int):
        assert n >= 0
        take_indices = {x for x in range(num_blocks - n, num_blocks)}
    else:
        take_indices = {num_blocks + idx if idx < 0 else idx for idx in n}
    return take_indices, max(take_indices)


def _forward_intermediates_cpe(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int], Tuple[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHW',
        intermediates_only: bool = False,
        aggregation: Optional[str] = "sparse",
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """ Forward features that returns intermediates.

    The Dense layer aggregation method is inspired from the paper: "Dense Connector for MLLMs"
    by Yao, Huanjin et al. (2024). arXiv preprint arXiv:2405.13800}

    Args:
        x: Input image tensor
        indices: Take last n blocks if int, select matching indices if sequence
        return_prefix_tokens: Return both prefix and spatial intermediate tokens
        norm: Apply norm layer to all intermediates
        stop_early: Stop iterating over blocks when last desired intermediate hit
        output_fmt: Shape of intermediate feature outputs
        intermediates_only: Only return intermediate features
        aggregation: intermediate layer aggregation method (sparse or dense)
    Returns:
    """
    assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
    assert aggregation in ('sparse', 'dense'), 'Aggregation must be one of sparse or dense.'
    reshape = output_fmt == 'NCHW'
    intermediates = []
    take_indices, max_index = _take_indices(len(self.blocks), indices)
    # forward pass
    B, _, height, width = x.shape
    x = self.patch_generator(x)

    if not stop_early:  # can't slice blocks in torchscript
        blocks = self.blocks
    else:
        blocks = self.blocks[:max_index + 1]

    accumulator = 0
    num_accumulated = 0

    for i, blk in enumerate(blocks):
        x = blk(x)
        if aggregation == "dense":
            accumulator = accumulator + x
            num_accumulated += 1
        if i in take_indices:
            if aggregation == "dense":
                x_ = accumulator / num_accumulated
                num_accumulated = 0
                accumulator = 0
            else:
                 x_ = x
            # normalize intermediates with final norm layer if enabled
            intermediates.append(self.norm(x_) if norm else x_)

    # process intermediates

    # split prefix (e.g. class, distill) and spatial feature tokens
    prefix_tokens = [y[:, 0:self.patch_generator.num_cls_tokens] for y in intermediates]
    intermediates = [y[:, self.patch_generator.num_skip:] for y in intermediates]

    if reshape:
        # reshape to BCHW output format
        H = height // self.patch_generator.patch_size
        W = width // self.patch_generator.patch_size
        intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
    if not torch.jit.is_scripting() and return_prefix_tokens:
        # return_prefix not support in torchscript due to poor type handling
        intermediates = list(zip(intermediates, prefix_tokens))
    if intermediates_only:
        return intermediates
    x = self.norm(x)
    return x, intermediates

def enable_cpe(model: nn.Module,
               max_img_size: Union[int, Tuple[int, int]] = 1024,
               num_cls_tokens: int = 1,
               pos_dropout: float = 0.1,
               register_multiple: int = 0,
):
    if not isinstance(model, VisionTransformer):
        raise ValueError("CPE only support for VisionTransformer models!")

    patch_size = model.patch_embed.patch_size[0]
    embed_dim = model.embed_dim
    input_dims = model.patch_embed.img_size
    normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
    cls_token = model.cls_token is not None

    max_img_size = int(round(max_img_size / patch_size) * patch_size)

    patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
    )

    model.patch_generator = patch_generator
    model.patch_embed = None
    model.cls_token = None
    model.pos_embed = None
    model.pos_drop = None
    model.num_cls_tokens = num_cls_tokens
    model.num_registers = patch_generator.num_registers

    model.forward_features = MethodType(_forward_cpe, model)
    model.forward_intermediates = MethodType(_forward_intermediates_cpe, model)
