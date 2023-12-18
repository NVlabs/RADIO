# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from torch import nn

from timm.models import create_model, VisionTransformer

from .enable_cpe_support import enable_cpe
from .input_conditioner import InputConditioner


class RADIOModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_conditioner: InputConditioner,
        return_summary: bool,
        return_spatial_features: bool,
    ):
        super().__init__()

        self.model = model
        self.input_conditioner = input_conditioner
        self.return_summary = return_summary
        self.return_spatial_features = return_spatial_features

    def forward(self, x: torch.Tensor):
        x = self.input_conditioner(x)

        y = self.model.forward_features(x)

        if isinstance(y, (list, tuple)):
            summary, all_feat = y
        elif isinstance(self.model, VisionTransformer):
            patch_gen = getattr(self.model, "patch_generator", None)
            if patch_gen is not None:
                summary = y[:, : patch_gen.num_cls_tokens].flatten(1)
                all_feat = y[:, patch_gen.num_skip :]
            elif self.model.global_pool == "avg":
                summary = y[:, self.model.num_prefix_tokens :].mean(dim=1)
                all_feat = y
            else:
                summary = y[:, 0]
                all_feat = y[:, 1:]
        else:
            raise ValueError("Unsupported model type")

        if self.return_summary and self.return_spatial_features:
            return summary, all_feat
        elif self.return_summary:
            return summary
        return all_feat


def create_model_from_args(args) -> nn.Module:
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # Skip weight initialization unless it's explicitly requested.
    weight_init = args.model_kwargs.pop("weight_init", "skip")

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        weight_init=weight_init,
        **args.model_kwargs,
    )

    assert (
        not args.cls_token_per_teacher or args.cpe_max_size is not None
    ), "CPE must be enabled for multiple CLS tokens!"

    if args.cpe_max_size is not None:
        enable_cpe(
            model,
            args.cpe_max_size,
            num_cls_tokens=len(args.teachers) if args.cls_token_per_teacher else 1,
            register_multiple=args.register_multiple,
        )

    return model
