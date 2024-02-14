# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from typing import Optional, Callable, Union, Tuple, Any, Dict, NamedTuple

import torch
from torch import nn

from timm.models import create_model, VisionTransformer

from .enable_cpe_support import enable_cpe
from .input_conditioner import InputConditioner
# Register extra models
from . import extra_timm_models


class Resolution(NamedTuple):
    height: int
    width: int


class AdaptorInput(NamedTuple):
    images: torch.Tensor
    summary: torch.Tensor
    features: torch.Tensor


class RadioOutput(NamedTuple):
    summary: torch.Tensor
    features: torch.Tensor


class RADIOModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_conditioner: InputConditioner,
        patch_size: int,
        max_resolution: int,
        preferred_resolution: Resolution,
        summary_idxs: Optional[torch.Tensor] = None,
        window_size: int = None,
        adaptors: Dict[str, nn.Module] = None,
    ):
        super().__init__()

        self.model = model
        self.input_conditioner = input_conditioner
        if summary_idxs is not None:
            self.register_buffer('summary_idxs', summary_idxs)
        else:
            self.summary_idxs = None

        self._preferred_resolution = preferred_resolution
        self._patch_size = patch_size
        self._max_resolution = max_resolution
        self._window_size = window_size

        adaptors = adaptors or dict()
        self.adaptors = nn.ModuleDict(adaptors)

    @property
    def num_summary_tokens(self) -> int:
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.num_skip
        elif self.model.global_pool == 'avg':
            return 0
        return 1

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def max_resolution(self) -> int:
        return self._max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self._preferred_resolution

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def min_resolution_step(self) -> int:
        res = self.patch_size
        if self.window_size is not None:
            res *= self.window_size
        return res

    def make_preprocessor_external(self) -> Callable[[torch.Tensor], torch.Tensor]:
        ret = self.input_conditioner
        self.input_conditioner = nn.Identity()
        return ret

    def get_nearest_supported_resolution(self, height: int, width: int) -> Resolution:
        height = int(round(height / self.min_resolution_step) * self.min_resolution_step)
        width = int(round(width / self.min_resolution_step) * self.min_resolution_step)

        height = max(height, self.min_resolution_step)
        width = max(width, self.min_resolution_step)

        return Resolution(height=height, width=width)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.input_conditioner(x)

        y = self.model.forward_features(x)

        if isinstance(y, (list, tuple)):
            all_summary, all_feat = y
            bb_summary = all_summary
        elif isinstance(self.model, VisionTransformer):
            patch_gen = getattr(self.model, "patch_generator", None)
            if patch_gen is not None:
                all_summary = y[:, : patch_gen.num_cls_tokens]
                if self.summary_idxs is not None:
                    bb_summary = all_summary[:, self.summary_idxs]
                else:
                    bb_summary = all_summary
                all_feat = y[:, patch_gen.num_skip :]
            elif self.model.global_pool == "avg":
                all_summary = y[:, self.model.num_prefix_tokens :].mean(dim=1)
                bb_summary = all_summary
                all_feat = y
            else:
                all_summary = y[:, 0]
                bb_summary = all_summary
                all_feat = y[:, 1:]
        else:
            raise ValueError("Unsupported model type")

        ret = RadioOutput(bb_summary.flatten(1), all_feat)
        if self.adaptors:
            ret = dict(backbone=ret)
            for name, adaptor in self.adaptors.items():
                summary = all_summary[:, adaptor.head_idx]
                ada_input = AdaptorInput(images=x, summary=summary, features=all_feat)
                v = adaptor(ada_input)
                ret[name] = v

        return ret


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
