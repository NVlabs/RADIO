# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from enum import Enum
import sys
import warnings
import debugpy
from typing import Tuple, Union

from einops import rearrange
import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_, xavier_uniform_
from timm.models.convnext import convnext_pico, ConvNeXt
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from .tome_merge import bipartite_soft_matching, merge_source, merge_wavg
from .tome_utils import parse_r
from PIL import Image, ImageDraw, ImageFont


class ToMeMode(Enum):
    DISABLED = 0
    CONSTANT = 1
    DYNAMIC = 2


class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(self.ls1(x_attn))

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn_mask = None
        if size is not None:
            attn_mask = size.log()[:, None, None, :, 0]
        x = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
            attn_mask=attn_mask,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def _tx_pre_hook(model: VisionTransformer, mode: Union[ToMeMode, str] = ToMeMode.DISABLED, **mode_args):
    if isinstance(mode, str):
        mode = ToMeMode[mode.upper()]

    inf_idx = [0]

    def blocks_forward_pre_hook(module, input):
        """
        Forward pre-hook to initialize ToMe info before processing blocks.
        This replaces the functionality that was in make_tome_class.
        """
        model._tome_info["num_input_tokens"] = num_tokens = input[0].shape[1]

        def _get_r(r_pct: float | None = None, final_r: int | None = None, per_r: int | None = None, num_final: int | None = None):
            r = None
            num_tokens = input[0].shape[1] - model._tome_info['class_token']
            if r_pct is not None:
                final_r = int(r_pct * num_tokens)

            if num_final is not None:
                final_r = num_tokens - num_final

            if final_r is not None:
                all_r = [0]
                tail = [int(num_tokens - final_r)]
                for _ in range(len(model.blocks) - 1):
                    tail.append(tail[-1] * 2)
                tail.reverse()

                fp_per_r = final_r / (len(model.blocks) - 1)

                leftover = 0
                first_lo_idx = len(model.blocks) - 1
                for i in range(1, len(model.blocks)):
                    fp_curr = int(i * fp_per_r)
                    fp_prev = int((i - 1) * fp_per_r)

                    num_curr = fp_curr - fp_prev
                    num_possible = tail[i]
                    if num_curr > num_possible:
                        curr_leftover = num_curr - num_possible
                        leftover += curr_leftover
                        num_curr = num_possible
                        first_lo_idx = min(first_lo_idx, i)
                    all_r.append(num_curr)

                placed = True
                while leftover > 0 and placed:
                    fp_per_r = leftover / (first_lo_idx - 1)
                    new_leftover = 0
                    for i in range(1, first_lo_idx):
                        fp_curr = int(i * fp_per_r)
                        fp_prev = int((i - 1) * fp_per_r)

                        num_curr = fp_curr - fp_prev
                        num_possible = tail[i] - all_r[i]
                        if num_curr > num_possible:
                            curr_leftover = num_curr - num_possible
                            new_leftover += curr_leftover
                            num_curr = num_possible
                            first_lo_idx = min(first_lo_idx, i)
                        all_r[i] += num_curr
                    placed = new_leftover < leftover
                    leftover = new_leftover

                r = all_r

            if r is None:
                if per_r is None:
                    raise ValueError("For constant inference r, provide either r_pct, final_r, or r.")
                r = [0] + [per_r] * (len(model.blocks) - 1)

            return r

        curr_idx = inf_idx[0]
        inf_idx[0] += input[0].shape[0]

        if mode == ToMeMode.DISABLED:
            r = 0
        elif mode == ToMeMode.CONSTANT:
            r_pct = mode_args.get("r_pct", None)
            final_r = mode_args.get("final_r", None)
            per_r = mode_args.get("r", None)
            num_final = mode_args.get("num_final", None)
            r = _get_r(r_pct=r_pct, final_r=final_r, per_r=per_r, num_final=num_final)
        elif mode == ToMeMode.DYNAMIC:
            with torch.no_grad():
                pred_r: torch.Tensor = model.r_predictor(model._tome_info["input_img"])
            loss_threshold = mode_args.get("loss_threshold", None)
            if loss_threshold is None:
                raise ValueError("For dynamic inference r, provide loss_threshold.")

            r_pct = (pred_r < loss_threshold).sum(dim=1, dtype=torch.float32) / pred_r.shape[1]
            r_pct.clamp_max_(0.95)

            if model._tome_info["verbose"]:
                for i in range(pred_r.shape[0]):
                    # print(f"Dynamic ToMe [{curr_idx}] - pred-loss {i} - r-pct: {r_pct[i].item():.3f}, Losses: {', '.join(f'{v:.3f}' for v in pred_r[i].tolist())}")
                    print(f"Dynamic ToMe [{curr_idx + i}] - pred-loss: {pred_r[i, 0].item():.3f} - r-pct: {r_pct[i].item():.3f}")

            if r_pct.shape[0] > 1:
                warnings.warn("Dynamic ToMe works best when the batch size is 1. When larger, the minimum r_pct over the batch is used.")
                r_pct = r_pct.amin(dim=0).item()
            else:
                r_pct = r_pct[0].item()

            r = _get_r(r_pct=r_pct)
        else:
            raise ValueError(f"Unknown ToMe mode {mode}.")

        model._tome_info["r"] = parse_r(len(model.blocks), r)
        model._tome_info["final_r"] = sum(model._tome_info["r"])

        if model._tome_info["verbose"]:
            # print(f'ToMe [{curr_idx}] - num-tokens: {num_tokens}, r-per-block: {model._tome_info["r"]}, final-r: {model._tome_info["final_r"]}')
            print(f'ToMe [{curr_idx}] - num-tokens: {num_tokens}, final-r: {model._tome_info["final_r"]}, leftover: {num_tokens - model._tome_info["final_r"]}')

        model._tome_info["size"] = None
        model._tome_info["source"] = None
    return blocks_forward_pre_hook


def _input_pre_hook(model: VisionTransformer):
    def patches_forward_pre_hook(module, input):
        """
        Forward pre-hook to initialize ToMe info before processing patches.
        """
        model._tome_info["input_img"] = input[0]

    return patches_forward_pre_hook


class ConstantBeta(nn.Module):
    def __init__(self, beta: float, num_betas: int = 1):
        super().__init__()
        self.register_buffer("ct", torch.zeros(num_betas, dtype=torch.float32))
        self.register_buffer("beta", torch.full((num_betas,), beta, dtype=torch.float32), persistent=False)

    def forward(self, increments: torch.Tensor):
        self.ct += increments
        return torch.where(self.ct != 1, self.beta, 0)


class PowerBeta(nn.Module):
    def __init__(self, std_rel: float = 0.1, num_betas: int = 1):
        super().__init__()
        self.gamma = np.roots([1, 7, 16 - std_rel**-2, 12 - std_rel**-2]).real.max()
        self.register_buffer("ct", torch.zeros(num_betas, dtype=torch.float32))

    def forward(self, increments: torch.Tensor):
        self.ct += increments
        beta = (1 - 1 / self.ct) ** (self.gamma + 1)
        return torch.where(self.ct > 0, beta, 1)


class PredictorNet(nn.Module):
    def __init__(self, num_output_slots: int = 100, debug: bool = False):
        super().__init__()

        self.base: ConvNeXt = convnext_pico(pretrained=True)

        self.pos = nn.Parameter(torch.randn(1, num_output_slots, self.base.num_features) * 0.02)

        self.register_buffer('causal_mask', nn.Transformer.generate_square_subsequent_mask(num_output_slots))

        self.tx = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.base.num_features, nhead=8, dim_feedforward=self.base.num_features * 4, dropout=0.0,
                                       batch_first=True, norm_first=True),
            num_layers=2,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(self.base.num_features),
            nn.Linear(self.base.num_features, 1),
        )

        for n, p in self.named_parameters():
            if p.dim() > 1 and not n.startswith('base'):
                xavier_uniform_(p)

        self.pos_buffer = nn.Parameter(torch.zeros(1, self.base.num_features, 16, 16))
        self.mom_beta = ConstantBeta(beta=0.95, num_betas=num_output_slots)
        self.register_buffer('expected_loss', torch.zeros(num_output_slots, dtype=torch.float32))
        self.step_ct = 0

        self.debug = debug
        if debug:
            self._debug_info = []

    def forward(self, x):
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        y = self.base.forward_features(x) + self.pos_buffer
        y = rearrange(y, 'b c h w -> b (h w) c')

        pe_tgt = self.pos.expand(y.shape[0], -1, -1)  # (batch_size, seq_len, 256)

        y = self.tx(pe_tgt, y, tgt_mask=self.causal_mask, tgt_is_causal=True)
        y = self.proj(y)[..., 0]
        y = F.softplus(y)  # Ensure positive values for gradient norms
        y = torch.cumsum(y, dim=1)  # Enforce monotonic increase (more tokens dropped = higher loss)

        if self.debug:
            for y_l, x_l in zip(y, x):
                self._debug_info.append((y_l[-1].item(), x_l.cpu()))

        return y

    def finish_debug(self, normalizer):
        if not self.debug:
            return

        import torchvision.transforms.functional as TF

        self._debug_info.sort(key=lambda v: v[0])

        images = torch.stack(list(v[1] for v in self._debug_info))
        images = images * normalizer.norm_std.cpu() + normalizer.norm_mean.cpu()

        # Try to use a default font, fallback to default if not available
        try:
            font = ImageFont.truetype("OpenSans-VariableFont_wdth,wght.ttf", 42)
        except:
            font = ImageFont.load_default()

        # Convert to PIL images and add text overlay
        annotated_images = []
        for i, (pred_loss, _) in enumerate(self._debug_info):
            # Convert tensor to PIL Image
            img_tensor = images[i].clamp(0, 1)
            pil_img = TF.to_pil_image(img_tensor)

            # Create a drawing context
            draw = ImageDraw.Draw(pil_img)

            # Add text with background for better visibility
            text = f"{pred_loss:.3f}"
            bbox = draw.textbbox((7, 7), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw background rectangle
            draw.rectangle(bbox, fill='black')
            # Draw text
            draw.text((7, 7), text, fill='white', font=font)

            # Convert back to tensor
            annotated_tensor = TF.to_tensor(pil_img)
            annotated_images.append(annotated_tensor)

        # Stack annotated images
        annotated_images = torch.stack(annotated_images)

        print(f'Pred Losses: {", ".join(f"{v[0]:.3f}" for v in self._debug_info)}')

        save_image(annotated_images, 'pnet_order.jpg', nrow=8)

def _tx_post_hook(model: VisionTransformer):
    def blocks_forward_post_hook(module, input, output):
        """
        Un-merges the tokens, so that the spatial correspondence is preserved.
        """
        expand = model._tome_info["expand"]
        if not expand:
            return output
        # Source is a [B, O, I] mapping tensor, so multiplying by the transpose gives us back the input dimension
        source = model._tome_info["source"]
        if source is None:
            return output

        with torch.autocast('cuda', enabled=False):
            expanded = torch.matmul(source.mT.float(), output.float())
        return expanded
    return blocks_forward_post_hook


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True,
    mode: ToMeMode = ToMeMode.DYNAMIC, disable_dynamic: bool = False, **mode_args,
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    # rank = dist.get_rank() if dist.is_initialized() else 0
    rank = 0
    rng = np.random.default_rng(0xBAD5EED + rank)

    num_prefix = getattr(model, "num_prefix_tokens", None)
    if num_prefix is None:
        num_prefix = 1 if model.cls_token is not None else 0

    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": num_prefix,
        "distill_token": False,
        "expand": bool(mode_args.get("expand", False)),
        "rng": rng,
        "verbose": bool(mode_args.get("verbose", False)),
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    if not disable_dynamic:
        model.r_predictor = PredictorNet(debug=bool(mode_args.get("debug", False)))

    model.patch_generator.register_forward_pre_hook(_input_pre_hook(model))
    # Register the pre-hook on the blocks module
    model.blocks.register_forward_pre_hook(_tx_pre_hook(model, mode=mode, **mode_args))
    model.blocks.register_forward_hook(_tx_post_hook(model))

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
