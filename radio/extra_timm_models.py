# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from torch import nn

from timm.models import register_model
from timm.models.vision_transformer import (
    VisionTransformer,
    _create_vision_transformer as _timm_create_vision_transformer,
    Mlp,
    Block,
    LayerScale as TIMMLayerScale,
)

from . import dinov2_arch


@register_model
def vit_tiny_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=14, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=14, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Base (ViT-B/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_args = dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16)
    if pretrained:
        # There is no pretrained version of ViT-H/16, but we can adapt a ViT-H/14 for this purpose
        model = _create_vision_transformer('vit_huge_patch14_224', pretrained=True, **dict(model_args, **kwargs))
    else:
        model = _create_vision_transformer('vit_huge_patch16_224', pretrained=False, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch16_224_mlpnorm(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model = vit_huge_patch16_224(pretrained=pretrained, **kwargs)

    for m in model.modules():
        if isinstance(m, Mlp) and not isinstance(m.norm, nn.LayerNorm):
            m.norm = nn.LayerNorm(m.fc1.out_features)

    return model


@register_model
def vit_bigG_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1664, depth=48, num_heads=16, init_values=1e-6)
    model = _create_vision_transformer('vit_bigG_patch14', pretrained=False, **dict(model_args, **kwargs))
    return model


def _create_vision_transformer(*args, **kwargs):
    model = _timm_create_vision_transformer(*args, **kwargs)
    _patch_layer_scale(model)
    return model


def _patch_layer_scale(model: VisionTransformer):
    def replace_ls(old_ls: TIMMLayerScale):
        new_ls = dinov2_arch.LayerScale(old_ls.gamma.shape[0], inplace=old_ls.inplace)
        new_ls.load_state_dict(old_ls.state_dict())
        return new_ls

    # Monkey patch: Replace TIMM's LayerScale with our modified DINOv2 one, that uses a param name
    # other than gamma, so that HFHub doesn't mess with it!
    for mod in model.modules():
        if isinstance(mod, Block):
            if isinstance(mod.ls1, TIMMLayerScale):
                mod.ls1 = replace_ls(mod.ls1)
            if isinstance(mod.ls2, TIMMLayerScale):
                mod.ls2 = replace_ls(mod.ls2)
    pass
