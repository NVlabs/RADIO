# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from timm.models import register_model
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer


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
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_args = dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16)
    model = _create_vision_transformer('vit_huge_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
