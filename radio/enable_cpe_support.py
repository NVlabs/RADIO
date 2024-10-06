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

from radio.feature_normalizer import IntermediateFeatureNormalizerBase, NullIntermediateFeatureNormalizer

from .extra_models import DinoWrapper
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
        inter_feature_normalizer: Optional[IntermediateFeatureNormalizerBase] = None,
        norm_alpha_scheme = "post-alpha",
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
        norm_alpha_scheme: apply alpha before ("pre-alpha") or after accumulation ("post-alpha")
    Returns:
    """
    assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
    assert aggregation in ('sparse', 'dense'), 'Aggregation must be one of sparse or dense.'
    reshape = output_fmt == 'NCHW'
    intermediates = []
    take_indices, max_index = _take_indices(len(self.blocks), indices)
    take_indices = sorted(take_indices)
    # forward pass
    B, _, height, width = x.shape
    x = self.patch_generator(x)

    if not stop_early:  # can't slice blocks in torchscript
        blocks = self.blocks
    else:
        blocks = self.blocks[:max_index + 1]

    if inter_feature_normalizer is None or norm_alpha_scheme == 'none':
        inter_feature_normalizer = NullIntermediateFeatureNormalizer.get_instance(x.dtype, x.device)

    assert norm_alpha_scheme in ('none', 'pre-alpha', 'post-alpha'), f'Unsupported alpha scheme: {norm_alpha_scheme}'
    post_alpha_scheme = norm_alpha_scheme == 'post-alpha'

    accumulator = 0
    alpha_sum = 0
    num_accumulated = 0
    num_skip = self.patch_generator.num_skip

    take_off = 0

    for i, blk in enumerate(blocks):
        x = blk(x)
        if aggregation == "dense":
            # Arbitrarily use the rotation matrix from the final layer in the dense group
            y, alpha = inter_feature_normalizer(x, i, rot_index=take_indices[take_off], skip=num_skip)
            if post_alpha_scheme:
                accumulator = accumulator + y
                alpha_sum = alpha_sum + alpha
            else:
                accumulator = accumulator + (alpha * y)
                alpha_sum += 1
            num_accumulated += 1
        if i == take_indices[take_off]:
            if aggregation == "dense":
                alpha = alpha_sum / num_accumulated
                x_ = alpha * accumulator / num_accumulated
                num_accumulated = 0
                accumulator = 0
                alpha_sum = 0
            else:
                 y, alpha = inter_feature_normalizer(x, i, skip=num_skip)
                 x_ = alpha * y
            # normalize intermediates with final norm layer if enabled
            intermediates.append(self.norm(x_) if norm else x_)
            take_off = min(take_off + 1, len(take_indices) - 1)

    # process intermediates

    # split prefix (e.g. class, distill) and spatial feature tokens
    prefix_tokens = [y[:, :self.patch_generator.num_cls_tokens] for y in intermediates]
    intermediates = [y[:, num_skip:] for y in intermediates]

    if reshape:
        # reshape to BCHW output format
        H = height // self.patch_generator.patch_size
        W = width // self.patch_generator.patch_size
        intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
    if not torch.jit.is_scripting() and return_prefix_tokens:
        # return_prefix not support in torchscript due to poor type handling
        intermediates = list(zip(prefix_tokens, intermediates))
    if intermediates_only:
        return intermediates
    x = self.norm(x)
    return x, intermediates


def _forward_cpe_dinov2(self: DinoWrapper, x: torch.Tensor) -> torch.Tensor:
    y = _forward_cpe(self.inner, x)

    return y[:, 0], y[:, self.num_summary_tokens:]


def _forward_intermediates_cpe_dinov2(self: DinoWrapper, *args, **kwargs):
    return _forward_intermediates_cpe(self.inner, *args, **kwargs)


def _enable_cpe_for_timm_vit(model: VisionTransformer,
                             max_img_size: Union[int, Tuple[int, int]] = 1024,
                             num_cls_tokens: int = 1,
                             pos_dropout: float = 0.1,
                             register_multiple: int = Optional[None],
                             num_registers: int = Optional[None],
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
        num_registers=num_registers,
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


def _enable_cpe_for_dv2_reg_vit(model: DinoWrapper,
                                max_img_size: Union[int, Tuple[int, int]] = 1024,
                                num_cls_tokens: int = 1,
                                pos_dropout: float = 0.1,
                                register_multiple: int = Optional[None],
                                num_registers: int = Optional[None],
):
    patch_size = model.patch_size
    embed_dim = model.embed_dim
    input_dims = model.inner.patch_embed.patches_resolution
    normalize_patches = not isinstance(model.inner.patch_embed.norm, nn.Identity)
    cls_token = True

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
        num_registers=num_registers,
        init_from=model,
        patch_bias=True,
    )

    inner = model.inner
    inner.patch_generator = patch_generator
    inner.patch_embed = None
    inner.cls_token = None
    inner.pos_embed = None
    inner.register_tokens = None

    model.forward_features = MethodType(_forward_cpe_dinov2, model)
    model.forward_intermediates = MethodType(_forward_intermediates_cpe_dinov2, model)


def enable_cpe(model: nn.Module,
               *args,
               **kwargs,
):
    if isinstance(model, VisionTransformer):
        _enable_cpe_for_timm_vit(model, *args, **kwargs)
    elif isinstance(model, DinoWrapper):
        _enable_cpe_for_dv2_reg_vit(model, *args, **kwargs)
    else:
        raise ValueError(f'CPE not supported for this model type: {type(model)}')
