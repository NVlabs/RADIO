# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from timm.models import VisionTransformer, checkpoint_seq

from .extra_models import DinoWrapper
from .vit_patch_generator import ViTPatchGenerator
from .forward_intermediates import forward_intermediates
from .dual_hybrid_vit import HybridModel


# --------------------------------------------------------------------------- #
# CPE method mixins
#
# These are class-level methods (not bound to a specific instance), so they
# resolve through the MRO each call. That is the key property that lets a
# `nn.DataParallel` replica — which is a shallow copy with rewritten
# `_parameters` / `_buffers` — dispatch to its own `self.patch_generator`
# instead of the original module's. The previous `MethodType(_forward_cpe,
# model)` approach captured the construction-time `model` instance inside the
# bound method, so a replica would walk the original module's submodules and
# hit cross-device tensor errors.
# --------------------------------------------------------------------------- #


class _CPEViTMixin:
    """CPE-enabled forward methods for a timm `VisionTransformer` subclass.

    Expects `self.patch_generator`, `self.blocks`, `self.norm` to be present
    (attached by `_enable_cpe_for_timm_vit`).
    """

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_generator(x)
        if getattr(self, 'grad_checkpointing', False) and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_intermediates(self, x: torch.Tensor, norm: bool = False, **kwargs):
        return forward_intermediates(
            self,
            patch_extractor=self.patch_generator,
            num_summary_tokens=self.patch_generator.num_skip,
            num_cls_tokens=self.patch_generator.num_cls_tokens,
            norm=self.norm if norm else lambda y: y,
            x=x,
            **kwargs,
        )

    @contextmanager
    def cpe_video_mode(self, t: int):
        original_num_frames = self.patch_generator.num_video_frames
        self.patch_generator.num_video_frames = t
        try:
            yield
        finally:
            self.patch_generator.num_video_frames = original_num_frames


class _CPEDinoWrapperMixin:
    """CPE-enabled forward methods for a `DinoWrapper` subclass.

    Delegates to `self.inner`, which must itself have been reclassed with
    `_CPEViTMixin` so the inner's CPE forward is available.
    """

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inner.forward_features(x)
        return y[:, 0], y[:, self.num_summary_tokens:]

    def forward_intermediates(self, *args, **kwargs):
        return self.inner.forward_intermediates(*args, **kwargs)

    @contextmanager
    def cpe_video_mode(self, t: int):
        with self.inner.cpe_video_mode(t):
            yield


class _CPEHybridMixin:
    """CPE plumbing for `HybridModel`. Only `cpe_video_mode` needs to delegate
    to the inner ViT — the rest of HybridModel's forward path already routes
    through `self.vit.forward_features` (now CPE-enabled by reclass).
    """

    @contextmanager
    def cpe_video_mode(self, t: int):
        with self.vit.cpe_video_mode(t):
            yield


# --------------------------------------------------------------------------- #
# Dynamic-subclass factory
# --------------------------------------------------------------------------- #


_CPE_CLASS_CACHE: Dict[Tuple[Type, Type], Type] = {}


def _make_cpe_subclass(base: Type, mixin: Type) -> Type:
    """Return a cached subclass of `base` that mixes in `mixin`.

    The cache key is `(base, mixin)`, so reclassing many instances of the
    same model type into the same CPE flavor reuses one class object.
    """
    key = (base, mixin)
    cls = _CPE_CLASS_CACHE.get(key)
    if cls is not None:
        return cls

    name = f'CPE_{base.__name__}'
    cls = type(name, (mixin, base), {})
    cls.__module__ = __name__
    cls.__qualname__ = name
    _CPE_CLASS_CACHE[key] = cls
    return cls


# --------------------------------------------------------------------------- #
# Enablers
# --------------------------------------------------------------------------- #


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
    model.patch_size = patch_size
    model.num_cls_tokens = num_cls_tokens
    model.num_registers = patch_generator.num_registers

    model.__class__ = _make_cpe_subclass(type(model), _CPEViTMixin)


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
        patch_bias=True,
    )

    inner = model.inner
    inner.patch_generator = patch_generator
    inner.patch_embed = None
    inner.cls_token = None
    inner.pos_embed = None
    inner.register_tokens = None
    inner.patch_size = patch_size

    inner.__class__ = _make_cpe_subclass(type(inner), _CPEViTMixin)
    model.__class__ = _make_cpe_subclass(type(model), _CPEDinoWrapperMixin)


def enable_cpe(model: nn.Module,
               *args,
               **kwargs,
):
    if isinstance(model, VisionTransformer):
        _enable_cpe_for_timm_vit(model, *args, **kwargs)
    elif isinstance(model, DinoWrapper):
        _enable_cpe_for_dv2_reg_vit(model, *args, **kwargs)
    elif isinstance(model, HybridModel):
        _enable_cpe_for_timm_vit(model.vit, *args, **kwargs)
        model.__class__ = _make_cpe_subclass(type(model), _CPEHybridMixin)
    else:
        raise ValueError(f'CPE not supported for this model type: {type(model)}')
