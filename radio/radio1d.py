# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RADIO1D: Vision Transformer with Variable-Length 1D Token Compression

This module implements RADIO1D, a Vision Transformer variant that compresses spatial tokens
into a variable-length 1D sequence of "global tokens" during encoding, then reconstructs
the full spatial resolution via a decoder.

Architecture Overview:
======================

    Input Image (B, 3, H_img, W_img)  # Any size divisible by patch_size
            │
            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                        ENCODER                                 │
    ├───────────────────────────────────────────────────────────────┤
    │  Patch Embedding → [cls, registers, patches]                  │
    │         (B, num_prefix + H*W, embed_dim)                      │
    │                       │                                        │
    │                       ▼                                        │
    │  Transformer Blocks (before downscale)                         │
    │         (B, num_prefix + H*W, embed_dim)                      │
    │                       │                                        │
    │                       ▼                                        │
    │  ┌─────────────────────────────────────┐                      │
    │  │  PatchMerging (at downscale_levels) │                      │
    │  │  - Halves H, W: (H, W) → (H/2, W/2) │                      │
    │  │  - Doubles embed_dim: C → 2C        │                      │
    │  └─────────────────────────────────────┘                      │
    │                       │                                        │
    │                       ▼                                        │
    │  Transformer Blocks (after downscale)                          │
    │         (B, num_prefix + H/2*W/2, 2*embed_dim)                │
    │                       │                                        │
    │                       ▼                                        │
    │  Final Norm                                                    │
    └───────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    TOKEN SLICING                               │
    ├───────────────────────────────────────────────────────────────┤
    │  Sample num_tokens per sample from mode distribution           │
    │  (clamped to available spatial tokens H*W)                    │
    │                                                                │
    │  slice_1d_tokens():                                            │
    │    - prefix_tokens: (B, num_prefix, C)                        │
    │    - global_tokens: (B, max_tokens, C) - sliced patches       │
    │    - global_token_mask: (B, max_tokens) - validity mask       │
    └───────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                        DECODER                                 │
    ├───────────────────────────────────────────────────────────────┤
    │  1. Pad global_tokens with filler_tokens to H*W               │
    │     (filler_tokens interpolated if size differs from ref)     │
    │                                                                │
    │  2. Use decoder's own learnable prefix_tokens (NOT encoder's) │
    │     This ensures decoder only reconstructs from 1D tokens     │
    │                                                                │
    │  3. Concatenate: [prefix_tokens, padded_patches]              │
    │                                                                │
    │  4. Decoder Blocks (before upscale)                            │
    │                                                                │
    │  5. ┌─────────────────────────────────────┐                   │
    │     │  PatchSplitting (at upscale_levels) │                   │
    │     │  - Doubles H, W: (H, W) → (2H, 2W)  │                   │
    │     │  - Halves embed_dim: C → C/2        │                   │
    │     └─────────────────────────────────────┘                   │
    │                                                                │
    │  6. Decoder Blocks (after upscale)                             │
    │                                                                │
    │  7. Final Norm                                                 │
    │         (B, num_prefix + H_out*W_out, target_embed_dim)       │
    └───────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                        OUTPUT                                  │
    ├───────────────────────────────────────────────────────────────┤
    │  {                                                             │
    │    "encoder": (B, num_prefix + max_tokens, encoder_C)         │
    │    "decoder": (B, num_prefix + H_out*W_out, target_C)         │
    │  }                                                             │
    │                                                                │
    │  Sizes depend on input image size, not fixed!                  │
    └───────────────────────────────────────────────────────────────┘

Key Components:
===============

1. PatchMerging (Encoder Downscaling)
   - Merges 2x2 patches into 1: (H, W) -> (H/2, W/2)
   - Doubles embedding dimension: C -> 2C
   - Applied at specified downscale_levels (e.g., before block 19)

2. PatchSplitting (Decoder Upscaling)
   - Splits 1 patch into 2x2: (H, W) -> (2H, 2W)
   - Halves embedding dimension: C -> C/2
   - Inverse of PatchMerging

3. Token Slicing (slice_1d_tokens)
   - Samples variable num_tokens per sample from Gaussian mixture distribution
   - Clamps num_tokens to available spatial tokens (important for smaller images)
   - Returns separate prefix tokens (cls + registers) and global tokens (spatial)
   - Pads to max tokens with validity mask

4. RADIO1D_Decoder
   - Uses its own learnable prefix_tokens (NOT the encoder's, to avoid information leak)
   - Uses learnable filler_tokens to pad sliced tokens back to full sequence
   - Applies transformer blocks with PatchSplitting for upscaling
   - Reconstructs original spatial resolution and embedding dimension

Key Parameters:
===============

- downscale_levels: Block indices for encoder downscaling, e.g., [19]
- modes: Token count modes for sampling, e.g., [64, 128, 196]
- mode_weights: Probability weights for modes, e.g., [0.33, 0.34, 0.33]
- decoder_depth: Number of decoder blocks, e.g., 6
- decoder_upscale_levels: Block indices for decoder upscaling, e.g., [3]

Training vs Inference:
======================

- Training: Samples different num_tokens per sample from mode distribution.
  The encoder output is padded to the max sampled value in the micro-batch.
- Inference: Uses provided num_tokens or defaults to max(modes).

Return Format:
==============

Both outputs are full sequences [cls, registers, patches/global_tokens]:
- output["encoder"]: (B, num_prefix + max_tokens, encoder_C) - compressed representation
- output["decoder"]: (B, num_prefix + output_patches, target_C) - reconstructed full resolution

This format is compatible with the RADIO1D framework's expected return structure.

Variable Image Size Support:
============================

The model supports any image size that is a multiple of the patch size in each dimension.
The decoder uses bilinear interpolation of filler_tokens to adapt to different input sizes.

Example sizes (with patch_size=16, downscale_levels=[19]):
- 224x224 -> 14x14 patches -> 7x7 after downscale -> 14x14 after decode = 196 output patches
- 448x448 -> 28x28 patches -> 14x14 after downscale -> 28x28 after decode = 784 output patches
- 320x384 -> 20x24 patches -> 10x12 after downscale -> 20x24 after decode = 480 output patches
"""

from abc import ABC
from copy import deepcopy
from functools import partial
from logging import getLogger
import math
from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import ModuleList
from torch.nn import functional as F
from torch.distributed.nn.functional import all_reduce as all_reduce_with_gradients
from torch.utils.checkpoint import checkpoint
from timm.models import register_model, build_model_with_cfg
from timm.models.vision_transformer import (
    VisionTransformer,
    Mlp,
    Attention,
    Block,
)
from timm.layers import (
    AttentionPoolLatent,
    PatchEmbed,
    PatchDropout,
    LayerNorm,
    trunc_normal_,
    get_norm_layer,
    get_act_layer,
    to_2tuple,
)

from .vit_patch_generator import ViTPatchGenerator
from .utils import get_rank

logger = getLogger(__name__)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for the rounding operation."""
    x_hat = x.detach().round()
    return x + (x_hat - x).detach()


def const_ste(x: torch.Tensor, c: float) -> torch.Tensor:
    """Straight-through estimator that returns a constant `c` with the same shape as `x`,
    while routing gradients through `x` (identity)."""
    return c - x.detach() + x


# Type definitions
LayerType = Union[str, Callable, Type[nn.Module]]


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Downsample features by merging 2x2 neighboring patches.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            size: Union[int, Tuple[int, int]] = 2,
            device=None,
            dtype=None,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.size: Tuple[int, int] = to_2tuple(size)
        self.downscale = math.prod(self.size)
        self.norm = norm_layer(self.downscale * dim, **dd)
        self.reduction = nn.Linear(self.downscale * dim, self.out_dim, bias=False, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features with shape (B, H, W, C).

        Returns:
            Output features with shape (B, H//2, W//2, out_dim).
        """
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, W % self.size[1], 0, H % self.size[0])
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // self.size[0], self.size[0], W // self.size[1], self.size[1], C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


def sample_multinomial_batch(
    modes: List[int],
    weights: List[float],
    batch_size: int,
    sigma: float = 30.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample token counts for each sample in a batch.

    Uses torch.multinomial for sampling, ensuring reproducibility with torch.manual_seed().

    Args:
        modes: List of mode values (e.g., [128, 256, 512])
        weights: List of weights for each mode
        batch_size: Number of samples to generate
        sigma: Standard deviation for Gaussian mixture

    Returns:
        Tensor of shape (batch_size,) with sampled token counts
    """
    min_val = min(modes)
    max_val = max(modes)

    # Create values tensor
    values = torch.arange(min_val, max_val + 1, dtype=torch.long)

    # Compute the probability density using a mixture of Gaussians
    modes_t = torch.tensor(modes, dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)

    # values: (num_values,), modes_t: (num_modes,) -> broadcast to (num_values, num_modes)
    diff = values.unsqueeze(1).float() - modes_t.unsqueeze(0)  # (num_values, num_modes)
    gaussian = torch.exp(-diff.pow(2) / (2 * sigma ** 2))  # (num_values, num_modes)
    probs = (gaussian * weights_t.unsqueeze(0)).sum(dim=1)  # (num_values,)
    probs = probs / probs.sum()

    # Sample indices using torch.multinomial
    sampled_indices = torch.multinomial(probs, batch_size, replacement=True, generator=generator)

    # Map indices to actual token counts
    sampled_values = values[sampled_indices]
    return sampled_values


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lambda_: Union[float, torch.Tensor] = 1.0):
        lambda_ = torch.as_tensor(lambda_, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        lambda_, = ctx.saved_tensors
        return lambda_ * grad_output, None


def slice_1d_tokens(
    x: torch.Tensor,
    num_tokens: torch.Tensor,
    num_prefix_tokens: int,
    max_tokens: Optional[int] = None,
    use_last_tokens: bool = False,
    dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice variable numbers of 1D tokens per sample.

    Args:
        x: Input tensor of shape (B, N, C) where N = num_prefix + num_spatial
        num_tokens: Tensor of shape (B,) with number of tokens to keep per sample
        num_prefix_tokens: Number of prefix tokens (cls, registers) to always keep
        max_tokens: Maximum number of global tokens (for padding). If None, uses max(num_tokens)
        use_last_tokens: If True, take the last num_tokens instead of the first

    Returns:
        Tuple of (prefix_tokens, global_tokens, global_token_mask):
            - prefix_tokens: (B, num_prefix, C) prefix tokens (cls, registers)
            - global_tokens: (B, max_tokens, C) padded 1D global tokens
            - global_token_mask: (B, max_tokens) boolean mask (True = valid token)
    """
    B, N, C = x.shape
    device = x.device
    num_spatial = N - num_prefix_tokens

    # Ensure num_tokens is on the right device and clamp to available spatial tokens
    num_tokens = num_tokens.to(device)
    num_tokens = num_tokens.clamp(max=num_spatial)

    if max_tokens is None:
        max_tokens = int(num_tokens.max().item())

    # Separate prefix and global tokens
    prefix = x[:, :num_prefix_tokens]  # (B, num_prefix, C)
    global_feats = x[:, num_prefix_tokens:]  # (B, num_spatial, C)

    # Create output tensor with padding for global tokens
    global_tokens = torch.zeros(B, max_tokens, C, device=device, dtype=x.dtype)

    # Create global token mask
    token_indices = torch.arange(global_feats.shape[1], device=device).unsqueeze(0)  # (1, max_tokens)
    global_token_mask = token_indices < num_tokens.unsqueeze(1)  # (B, max_tokens)

    if dynamic:
        zero_ste = const_ste(num_tokens, 0.0)
        one_ste = const_ste(num_tokens, 1.0)
        where_ste = torch.where(global_token_mask, one_ste.unsqueeze(1), zero_ste.unsqueeze(1))
        # Reducing the gradient magnitude through this gate application stabilizes training.
        # Since it's multiplicatively applied to every surviving token, then `1 / num_tokens` means
        # that the signal is consistent across different token counts.
        where_ste = GradScale.apply(where_ste, 1 / num_tokens.clamp_min(1).unsqueeze(-1))
        global_feats = global_feats * where_ste.unsqueeze(-1)

    cpu_num_tokens = num_tokens.tolist()

    # Copy tokens for each sample (clamped to available)
    for i in range(B):
        n = int(cpu_num_tokens[i])
        if use_last_tokens:
            # Take the last n tokens from the spatial sequence
            global_tokens[i, :n] = global_feats[i, -n:]
        else:
            # Take the first n tokens from the spatial sequence
            global_tokens[i, :n] = global_feats[i, :n]

    return prefix, global_tokens, global_token_mask[:, :max_tokens]


class PatchSplitting(nn.Module):
    """Patch Splitting Layer - Inverse of PatchMerging.

    Upsample features by splitting each patch into 2x2 neighboring patches.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or dim // 2 if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim // 2
        # Expand channels to 4x output dim, then reshape to 2x2 spatial
        self.expansion = nn.Linear(dim, 4 * self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features with shape (B, H, W, C).

        Returns:
            Output features with shape (B, 2*H, 2*W, out_dim).
        """
        B, H, W, C = x.shape

        # Expand channels: (B, H, W, C) -> (B, H, W, 4 * out_dim)
        x = self.expansion(x)

        # Reshape to split each token into 2x2 neighbors
        # (B, H, W, 4 * out_dim) -> (B, H, W, 2, 2, out_dim) -> (B, 2*H, 2*W, out_dim)
        x = x.reshape(B, H, W, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, 2 * H, 2 * W, self.out_dim)

        x = self.norm(x)
        return x


class RADIO1D_Decoder(nn.Module):
    """Decoder for RADIO1D that reconstructs the original sequence length and embedding dimension.

    Takes compressed global tokens from the encoder and reconstructs the full spatial resolution
    by applying inverse patch merging (splitting) operations.
    """

    def __init__(
            self,
            input_embed_dim: int,
            target_embed_dim: int,
            ref_spatial_size: Tuple[int, int],
            num_prefix_tokens: int,
            depth: int,
            upscale_levels: List[int],
            num_heads: int = 16,
            mlp_ratio: float = 4.0,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        """
        Args:
            input_embed_dim: Embedding dimension of input (after encoder downscaling).
            target_embed_dim: Target embedding dimension (original encoder dimension).
            ref_spatial_size: Reference spatial size (H, W) for filler token initialization.
                This is the expected spatial dimensions at the model's nominal image size
                (e.g., (7, 7) for 224x224 with patch_size=16 and one downscale).
                At runtime, filler tokens are interpolated to match actual input size.
            num_prefix_tokens: Number of prefix tokens (cls + registers).
            depth: Number of decoder blocks.
            upscale_levels: List of block indices where upscaling should happen.
            num_heads: Number of attention heads.
            mlp_ratio: MLP ratio for transformer blocks.
            norm_layer: Normalization layer.
        """
        super().__init__()

        self.input_embed_dim = input_embed_dim
        self.target_embed_dim = target_embed_dim
        self.ref_H, self.ref_W = ref_spatial_size
        self.num_prefix_tokens = num_prefix_tokens
        self.upscale_levels = set(upscale_levels) if upscale_levels else set()

        # Learnable filler tokens - initialized at reference size, interpolated at runtime if needed
        ref_num_patches = self.ref_H * self.ref_W
        scale = input_embed_dim ** -0.5
        self.filler_tokens = nn.Parameter(torch.randn(ref_num_patches, input_embed_dim) * scale)

        # Learnable prefix tokens for the decoder (independent from encoder's prefix tokens)
        # This ensures the decoder only reconstructs from 1D global tokens, not encoder prefix info
        self.prefix_tokens = nn.Parameter(torch.randn(num_prefix_tokens, input_embed_dim) * scale)

        # Build blocks and upscale layers
        embed_dim = input_embed_dim
        blocks = []
        upscale_blocks = []
        prefix_proj_blocks = []

        for i in range(depth):
            if upscale_levels is not None and i in upscale_levels:
                upscale_block = PatchSplitting(embed_dim)
                # Projection for prefix tokens to match new (reduced) embed_dim
                prefix_proj = nn.Linear(embed_dim, upscale_block.out_dim, bias=False)
                num_heads = max(1, num_heads * upscale_block.out_dim // embed_dim)
                embed_dim = upscale_block.out_dim
                upscale_blocks.append(upscale_block)
                prefix_proj_blocks.append(prefix_proj)

            blocks.append(Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            ))

        self.blocks = nn.ModuleList(blocks)
        self.upscale_blocks = nn.ModuleList(upscale_blocks)
        self.prefix_proj_blocks = nn.ModuleList(prefix_proj_blocks)

        # Final norm
        self.norm = norm_layer(embed_dim)

        # Verify output dimension matches target
        assert embed_dim == target_embed_dim, \
            f"Decoder output dim {embed_dim} doesn't match target {target_embed_dim}"

    def _apply_upscale(
            self,
            x: torch.Tensor,
            upscale_idx: int,
            H: int,
            W: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """Apply patch splitting upscale operation.

        Args:
            x: Input tensor of shape (B, N, C) where N = num_prefix_tokens + H*W
            upscale_idx: Index into self.upscale_blocks and self.prefix_proj_blocks
            H: Current spatial height (in patches)
            W: Current spatial width (in patches)

        Returns:
            Tuple of (upscaled tensor, new H, new W)
        """
        B, N, C = x.shape

        # Separate prefix tokens from patch tokens
        prefix_tokens = x[:, :self.num_prefix_tokens]  # (B, num_prefix, C)
        patch_tokens = x[:, self.num_prefix_tokens:]   # (B, H*W, C)

        # Reshape patch tokens to spatial format for PatchSplitting
        patch_tokens = patch_tokens.reshape(B, H, W, C)

        # Apply patch splitting (spatial upsampling)
        patch_tokens = self.upscale_blocks[upscale_idx](patch_tokens)  # (B, 2H, 2W, C')

        # Get new dimensions
        _, H_new, W_new, C_new = patch_tokens.shape

        # Reshape back to sequence format
        patch_tokens = patch_tokens.reshape(B, H_new * W_new, C_new)

        # Project prefix tokens to match new channel dimension
        prefix_tokens = self.prefix_proj_blocks[upscale_idx](prefix_tokens)  # (B, num_prefix, C')

        # Concatenate prefix and patch tokens
        x = torch.cat([prefix_tokens, patch_tokens], dim=1)

        return x, H_new, W_new

    def _get_filler_tokens(self, H: int, W: int, B: int, device: torch.device) -> torch.Tensor:
        """Get filler tokens interpolated to the required spatial size.

        Args:
            H: Target height in patches
            W: Target width in patches
            B: Batch size
            device: Target device

        Returns:
            Filler tokens of shape (B, H*W, C)
        """
        if H == self.ref_H and W == self.ref_W:
            # No interpolation needed - matches reference size
            filler = self.filler_tokens.unsqueeze(0).expand(B, -1, -1)
        else:
            # Interpolate filler tokens to match the required size
            # Reshape to 2D grid, interpolate, then flatten
            filler_2d = self.filler_tokens.reshape(self.ref_H, self.ref_W, -1).permute(2, 0, 1).unsqueeze(0)
            # Interpolate: (1, C, ref_H, ref_W) -> (1, C, H, W)
            filler_2d = nn.functional.interpolate(filler_2d, size=(H, W), mode='bilinear', align_corners=False)
            # Reshape back: (1, C, H, W) -> (1, H*W, C)
            filler = filler_2d.squeeze(0).permute(1, 2, 0).reshape(1, H * W, -1)
            filler = filler.expand(B, -1, -1)

        return filler.to(device)

    def forward(
            self,
            global_tokens: torch.Tensor,
            global_token_mask: torch.Tensor,
            input_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, int, int]:
        """Forward pass through decoder.

        Args:
            global_tokens: Global tokens from encoder (B, num_tokens, C_in), possibly padded
            global_token_mask: Boolean mask for valid global tokens (B, num_tokens)
            input_size: Tuple of (H, W) spatial dimensions of the downscaled patches

        Returns:
            Tuple of (features, H, W):
                - features: Reconstructed features (B, num_prefix + H*W, target_embed_dim)
                - H: Output spatial height
                - W: Output spatial width
        """
        B = global_tokens.shape[0]
        H, W = input_size
        device = global_tokens.device

        # Get filler tokens (interpolated if needed for variable image sizes)
        filler = self._get_filler_tokens(H, W, B, device)

        # Combine global tokens with filler tokens
        # Valid global tokens replace the corresponding filler tokens
        patch_tokens = filler.clone()
        # Use the mask to place valid global tokens
        for i in range(B):
            n_valid = global_token_mask[i].sum().int().item()
            patch_tokens[i, :n_valid] = global_tokens[i, :n_valid]

        # Use decoder's own learnable prefix tokens (not encoder's, to avoid information leak)
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(B, -1, -1)

        # Concatenate prefix tokens and patch tokens
        x = torch.cat([prefix_tokens, patch_tokens], dim=1)  # (B, num_prefix + H*W, C)

        # Apply decoder blocks with optional upscaling
        upscale_idx = 0
        for i, blk in enumerate(self.blocks):
            # Apply upscale before this block if specified
            if i in self.upscale_levels:
                x, H, W = self._apply_upscale(x, upscale_idx, H, W)
                upscale_idx += 1

            # Apply transformer block
            x = blk(x)

        x = self.norm(x)

        return x, H, W


class KSampleDistribution(ABC):
    def __init__(self, synchronized: bool = False):
        self.synchronized = synchronized
        g = None
        if synchronized:
            g = torch.Generator(device='cuda')
            g.manual_seed(42)
        self.generator = g

    def set_curr_step(self, step: int):
        if self.generator is not None:
            self.generator.manual_seed(step)

    def get_max_tokens(self, outside_max: int) -> int:
        return outside_max

    def get_expected_tokens(self, outside_max: int) -> int:
        return self.get_max_tokens(outside_max)

    def _sample(self, batch_size: int, max_tokens: int) -> torch.Tensor:
        ...

    def sample(self, batch_size: int, max_tokens: int) -> torch.Tensor:
        inner_bs = 1 if self.synchronized else batch_size
        inner_sample = self._sample(inner_bs, max_tokens)
        if self.synchronized:
            inner_sample = inner_sample.expand(batch_size)
        return inner_sample


class MultiModeGaussSampleDistribution(KSampleDistribution):
    def __init__(self, modes: List[int], mode_weights: List[float], max_tokens: Optional[int] = None, synchronized: bool = False):
        super().__init__(synchronized=synchronized)
        if len(modes) != len(mode_weights):
            raise ValueError("modes and mode_weights must have the same length")
        assert all(mode > 0 for mode in modes)
        assert all(weight >= 0 for weight in mode_weights)
        assert sum(mode_weights) == 1.0
        self.modes = modes
        self.mode_weights = mode_weights
        self._max_tokens = max_tokens

    def get_max_tokens(self, outside_max: int) -> int:
        my_max = self._max_tokens or max(self.modes)
        return min(my_max, outside_max)

    def _sample(self, batch_size: int, max_tokens: int) -> torch.Tensor:
        num_tokens_per_sample = sample_multinomial_batch(self.modes, self.mode_weights, batch_size, generator=self.generator)
        torch.clamp_max_(num_tokens_per_sample, max_tokens)
        return num_tokens_per_sample


class UniformKSampleDistribution(KSampleDistribution):
    def _sample(self, batch_size: int, max_tokens: int) -> torch.Tensor:
        f = torch.rand(batch_size, dtype=torch.float32, device='cuda', generator=self.generator)
        f = torch.round(f * max_tokens)
        f = torch.clamp(f, min=1.0, max=max_tokens)
        return f.long()

    def get_expected_tokens(self, outside_max: int) -> int:
        return outside_max // 2

class BetaKSampleDistribution(KSampleDistribution):
    def __init__(self, target_pct: float = 0.25, synchronized: bool = False):
        super().__init__(synchronized=synchronized)
        self.target_pct = target_pct

        # This is one particular solution where the mode of the beta distribution is equal to target_pct
        # `mode = (alpha - 1) / (alpha + beta - 2)`
        # and we add the additional constraint that
        # `alpha + beta = 2 / rate`
        rate = torch.as_tensor(target_pct, dtype=torch.float32, device='cuda')
        alpha = 3 - (2 * rate)
        beta = (2 / rate) - 3 + (2 * rate)

        self.alpha = alpha
        self.beta = beta
        self.beta_dist = torch.distributions.Beta(alpha, beta)

    def _sample(self, batch_size: int, max_tokens: int) -> torch.Tensor:
        f = torch._sample_dirichlet(
            self.beta_dist._dirichlet.concentration,
            generator=self.generator,
        ).select(-1, 0)
        f = torch.round(f * max_tokens)
        f = torch.clamp(f, min=1.0, max=max_tokens)
        return f.long()

    def get_expected_tokens(self, outside_max: int) -> int:
        beta_mean = self.alpha / (self.alpha + self.beta)
        return int(beta_mean * outside_max)


class TriangleKSampleDistribution(KSampleDistribution):
    '''
    Triangle distribution, defined as p(x) = 2 - 2x for x in [0, 1]
    with expected value 1/3.
    '''

    def _sample(self, batch_size: int, max_tokens: int) -> torch.Tensor:
        u = torch.rand(batch_size, dtype=torch.float32, device='cuda', generator=self.generator)
        # Use inverse transform sampling
        f = 1 - torch.sqrt(u.clamp_min_(1e-8))
        f = torch.round(f * (max_tokens - 1)) + 1
        f = torch.clamp(f, min=1.0, max=max_tokens)
        return f.long()


    def get_expected_tokens(self, outside_max: int) -> int:
        return outside_max // 3


class InterpolateKSampleDistributions(KSampleDistribution):
    def __init__(self, dist_a: Union[KSampleDistribution, dict, str], dist_b: Union[KSampleDistribution, dict, str], num_steps: int, synchronized: bool = False):
        super().__init__(synchronized=synchronized)
        self.dist_a = self._instantiate(dist_a)
        self.dist_b = self._instantiate(dist_b)
        self.num_steps = num_steps
        self.curr_step = 0

    def set_curr_step(self, step: int):
        super().set_curr_step(step)
        self.curr_step = step
        self.dist_a.set_curr_step(step)
        self.dist_b.set_curr_step(step)

    def get_max_tokens(self, outside_max: int) -> int:
        return max(self.dist_a.get_max_tokens(outside_max), self.dist_b.get_max_tokens(outside_max))

    def get_expected_tokens(self, outside_max: int) -> int:
        ea = self.dist_a.get_expected_tokens(outside_max)
        eb = self.dist_b.get_expected_tokens(outside_max)
        alpha = max(0, min(1, self.curr_step / self.num_steps))
        return int((1 - alpha) * ea + alpha * eb)

    def _sample(self, batch_size: int, max_tokens: int) -> torch.Tensor:
        sample_a = self.dist_a.sample(batch_size, max_tokens).float()
        sample_b = self.dist_b.sample(batch_size, max_tokens).float()
        alpha = max(0, min(1, self.curr_step / self.num_steps))
        f = (1 - alpha) * sample_a + alpha * sample_b
        f = torch.round(f).clamp(min=1.0, max=max_tokens)
        return f.long()

    def _instantiate(self, dist: Union[KSampleDistribution, dict, str]) -> KSampleDistribution:
        if isinstance(dist, KSampleDistribution):
            return dist
        elif isinstance(dist, dict):
            dist_type = dist.pop('type')
            if dist_type not in _K_SAMPLER_FACTORY:
                raise ValueError(f"Unknown KSampleDistribution type: {dist_type}")
            return _K_SAMPLER_FACTORY[dist_type](**dist)
        elif isinstance(dist, str):
            if dist not in _K_SAMPLER_FACTORY:
                raise ValueError(f"Unknown KSampleDistribution type: {dist}")
            return _K_SAMPLER_FACTORY[dist]()
        else:
            raise ValueError("dist must be a KSampleDistribution instance, a dict, or a str")


_K_SAMPLER_FACTORY = {
    'multimode_gaussian': MultiModeGaussSampleDistribution,
    'uniform': UniformKSampleDistribution,
    'beta': BetaKSampleDistribution,
    'triangle': TriangleKSampleDistribution,
    'interpolate': InterpolateKSampleDistributions,
}


class RADIO1D(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]
    _iter_count: torch.Tensor
    dynamic_rate_vec: Optional[torch.Tensor]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            # scale_attn_norm: bool = False,
            # scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            pool_include_prefix: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            embed_norm_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            num_cls_tokens: Optional[int] = None,
            cpe_max_size: Optional[int] = None,
            num_registers: Optional[int] = None,
            register_multiple: Optional[int] = None,
            downscale_levels: Optional[List[int]] = None,
            k_sample_config: Optional[dict] = None,
            decoder_depth: int = 6,
            decoder_upscale_levels: Optional[List[int]] = None,
            dynamic_rate: bool = False,
            dynamic_temperature: float = 1.0,
            progressive_reduction: bool = False,
            cka_weight: float = 0.0,
            cka_weight_final: Optional[float] = None,
            uniform_k: bool = False,
            grad_checkpointing: Union[bool, int] = False,
            decoder_grad_checkpointing: Union[bool, int] = False,
            downscale_expansion_factor: float = 2.0,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            num_cls_tokens: Number of class tokens.
            cpe_max_size: Maximum size of the input image.
            num_registers: Number of registers.
            register_multiple: Register multiple.
            downscale_levels: Downscale levels.
            modes: Modes for the input image size.
            mode_weights: Weights for the modes.
            decoder_depth: Number of decoder blocks.
            decoder_upscale_levels: Block indices in decoder where upscaling should happen.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or LayerNorm
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.dynamic_rate = dynamic_rate
        self.dynamic_temperature = dynamic_temperature
        self.progressive_reduction = progressive_reduction
        self.cka_weight = cka_weight
        self.cka_weight_final = cka_weight_final or cka_weight

        if dynamic_rate:
            if num_registers is None or num_registers == 0:
                raise ValueError("dynamic_rate requires at least one register token")
            self.register_buffer('dynamic_rate_vec', torch.randn(embed_dim))
            self.dynamic_rate_projector = nn.Linear(embed_dim, 1)

        if k_sample_config is None:
            self.k_sampler = UniformKSampleDistribution(synchronized=dynamic_rate or uniform_k)
        else:
            k_sample_config = deepcopy(k_sample_config)
            sampler_type = k_sample_config.pop('type')
            self.k_sampler = _K_SAMPLER_FACTORY[sampler_type](**k_sample_config, synchronized=dynamic_rate or uniform_k)

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        if cpe_max_size is not None:
            # Replace patch embed with CPE patch generator
            input_dims = img_size
            max_img_size = int(round(cpe_max_size / patch_size) * patch_size)
            self.patch_generator = ViTPatchGenerator(
                patch_size=patch_size,
                embed_dim=embed_dim,
                input_dims=input_dims,
                normalize_patches=pre_norm,
                cls_token=self.has_class_token,
                max_input_dims=max_img_size,
                pos_dropout=pos_drop_rate,
                num_cls_tokens=num_cls_tokens,
                register_multiple=register_multiple,
                num_registers=num_registers,
                #init_from=self,
                #adaptive_patch_tokenizer_config=None,
            )
            self.patch_embed = None
            self.cls_token = None
            self.pos_embed = None
            self.pos_drop = None
            self.num_cls_tokens = num_cls_tokens
            self.num_registers = num_registers
            self.num_prefix_tokens = self.patch_generator.num_cls_patches
        else:
            self.patch_generator = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Save original dimensions for decoder creation
        original_embed_dim = embed_dim
        original_num_patches = num_patches
        original_num_heads = num_heads

        downscale_blocks = []
        prefix_proj_blocks = []  # Projection layers for prefix tokens during downscaling
        blocks = []
        feature_info = []
        for i in range(depth):
            if downscale_levels is not None and i in downscale_levels:
                downscale_block = PatchMerging(embed_dim)
                # Projection for prefix tokens to match new embed_dim
                prefix_proj = nn.Linear(embed_dim, downscale_block.out_dim, bias=False)
                num_heads = int(num_heads * downscale_block.out_dim // embed_dim)
                embed_dim = downscale_block.out_dim
                downscale_blocks.append(downscale_block)
                prefix_proj_blocks.append(prefix_proj)

            blocks.append(block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                # scale_attn_norm=scale_attn_norm,
                # scale_mlp_norm=scale_mlp_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            ))
            feature_info.append(dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction))

        self.blocks = ModuleList(blocks)
        self.downscale_blocks = ModuleList(downscale_blocks)
        self.prefix_proj_blocks = ModuleList(prefix_proj_blocks)
        self.downscale_levels = set(downscale_levels) if downscale_levels else set()
        self.feature_info = feature_info
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        if isinstance(grad_checkpointing, bool):
            self.grad_checkpointing = len(blocks) if grad_checkpointing else 0
        else:
            self.grad_checkpointing = min(grad_checkpointing, len(blocks))

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Create decoder (always needed to reconstruct from sliced global tokens)
        # Compute dimensions after encoder (with or without downscaling)
        if downscale_levels:
            num_downscales = len(downscale_levels)
            encoder_num_patches = original_num_patches // (4 ** num_downscales)
            encoder_embed_dim = embed_dim  # embed_dim after all downscales
            # Default upscale_levels: upscale at the midpoint of decoder
            if decoder_upscale_levels is None:
                decoder_upscale_levels = [decoder_depth // 2] * num_downscales
        else:
            encoder_num_patches = original_num_patches
            encoder_embed_dim = original_embed_dim
            decoder_upscale_levels = []  # No upscaling needed

        # Compute reference spatial size for decoder filler tokens
        ref_H = ref_W = int(encoder_num_patches ** 0.5)
        self.decoder = RADIO1D_Decoder(
            input_embed_dim=encoder_embed_dim,
            target_embed_dim=original_embed_dim,
            ref_spatial_size=(ref_H, ref_W),  # Reference size for filler token init
            num_prefix_tokens=self.num_prefix_tokens,
            depth=decoder_depth,
            upscale_levels=decoder_upscale_levels,
            num_heads=original_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        # Iteration counter for logging (not a parameter, won't be saved in state_dict)
        self.register_buffer('_iter_count', torch.tensor(0, dtype=torch.long))

        self.register_buffer('_total_num_tokens', torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer('_total_num_samples', torch.tensor(0, dtype=torch.long), persistent=False)

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        ret = super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.k_sampler.set_curr_step(int(self._iter_count.item()))
        return ret

    def _apply_downscale(self, x: torch.Tensor, downscale_idx: int, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """Apply patch merging downscale operation.

        Args:
            x: Input tensor of shape (B, N, C) where N = num_prefix_tokens + H*W
            downscale_idx: Index into self.downscale_blocks and self.prefix_proj_blocks
            H: Current spatial height (in patches)
            W: Current spatial width (in patches)

        Returns:
            Tuple of (downscaled tensor, new H, new W)
        """
        B, N, C = x.shape
        num_prefix = self.num_prefix_tokens

        # Separate prefix tokens (cls, registers) from patch tokens
        prefix_tokens = x[:, :num_prefix]  # (B, num_prefix, C)
        patch_tokens = x[:, num_prefix:]   # (B, H*W, C)

        # Reshape patch tokens to spatial format for PatchMerging
        patch_tokens = patch_tokens.reshape(B, H, W, C)

        # Apply patch merging (spatial downsampling)
        patch_tokens = self.downscale_blocks[downscale_idx](patch_tokens)  # (B, H', W', C')

        # Get new dimensions
        _, H_new, W_new, C_new = patch_tokens.shape

        # Reshape back to sequence format
        patch_tokens = patch_tokens.reshape(B, H_new * W_new, C_new)

        # Project prefix tokens to match new channel dimension
        prefix_tokens = self.prefix_proj_blocks[downscale_idx](prefix_tokens)  # (B, num_prefix, C')

        # Concatenate prefix and patch tokens
        x = torch.cat([prefix_tokens, patch_tokens], dim=1)

        return x, H_new, W_new

    def forward_encoder(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
        use_last_tokens: bool = False,
    ) -> dict:
        """Forward pass through encoder only (embeddings, transformer blocks, token slicing).

        Args:
            x: Input image tensor of shape (B, C, H, W)
            attn_mask: Optional attention mask
            num_tokens: Number of 1D tokens to output per sample.
                       If None during training: samples per-sample from mode distribution
                       If None during inference: uses expected number of tokens
                       If negative, uses dynamic rate prediction
            use_last_tokens: If True, take the last num_tokens instead of the first

        Returns:
            Dict with keys:
                - "encoder": (B, num_prefix + max_tokens, C) - prefix tokens + 1D global tokens
                - "global_tokens": (B, max_tokens, C) - sliced global tokens (for decoder input)
                - "global_token_mask": (B, max_tokens) - validity mask for global tokens
                - "encoder_spatial_size": (H, W) - spatial dimensions after encoding
                - "original_spatial_size": (H, W) - original spatial dimensions before padding
        """
        B = x.shape[0]

        # Infer spatial dimensions from input image before patch embedding
        _, _, H_img, W_img = x.shape
        if self.patch_embed is not None:
            patch_size = self.patch_embed.patch_size[0]
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
        else:
            images = x  # Save for visualization
            patch_size = self.patch_generator.patch_size
            x = self.patch_generator(x)

        # Compute spatial dimensions (in patches) for downscaling
        H = H_img // patch_size
        W = W_img // patch_size
        # Save original dimensions before any padding during downscaling
        original_H, original_W = H, W

        # Sample num_tokens per sample, clamped to available spatial tokens
        num_spatial_tokens = H * W
        total_downscale = math.prod(ds.downscale for ds in self.downscale_blocks)
        num_spatial_tokens //= total_downscale

        if num_tokens is not None:
            num_tokens = min(num_tokens, num_spatial_tokens)
            num_tokens_per_sample = torch.full((B,), num_tokens, dtype=torch.long, device=x.device)
        else:
            if self.training:
                num_tokens_per_sample = self.k_sampler.sample(B, max_tokens=num_spatial_tokens)
            else:
                # In eval mode, return all available tokens if num_tokens not specified
                num_tokens = num_spatial_tokens
                num_tokens_per_sample = torch.full((B,), num_tokens, dtype=torch.long, device=x.device)

        is_dynamic = False
        if self.dynamic_rate and (num_tokens is None or num_tokens < 0):
            if num_tokens is not None:
                num_tokens = -num_tokens
            target_rate_pct = num_tokens_per_sample.float() / num_spatial_tokens
            rate_vec = (target_rate_pct * 2 - 1).unsqueeze(1) * self.dynamic_rate_vec.unsqueeze(0)
            x0 = x[:, :self.num_cls_tokens]
            x1 = x[:, self.num_cls_tokens + 1:]
            # Replace the first register instead with this dynamic rate vector, allowing
            # the model to know what the target rate will be
            x = torch.cat([x0, rate_vec.unsqueeze(1), x1], dim=1)
            is_dynamic = True

        # Apply transformer blocks with optional downscaling
        downscale_idx = 0
        use_checkpoint = self.grad_checkpointing and not torch.jit.is_scripting()

        first_downscale_level = min(self.downscale_levels)
        last_downscale_level = max(self.downscale_levels)

        curr_num_tokens = None
        total_num_to_drop = None

        if attn_mask is not None:
            raise NotImplementedError("attn_mask not currently supported at input.")

        for i, blk in enumerate(self.blocks):
            # Apply downscale before this block if specified
            if i in self.downscale_levels:
                if i == first_downscale_level and is_dynamic:
                    dyn_token = x[:, self.num_cls_tokens]
                    dyn_pred_logits = self.dynamic_rate_projector(dyn_token).squeeze(1)
                    if self.training:
                        gumbels = torch.rand(*dyn_pred_logits.shape, 2, dtype=dyn_pred_logits.dtype, device=dyn_pred_logits.device).clamp_min_(1e-8)
                        gumbels = -(-gumbels.log()).log()  # Sample from Gumbel(0,1)
                        gumbels[..., 1].neg_()
                        gumbels = gumbels.sum(dim=-1)
                        dyn_pred = dyn_pred_logits + gumbels / self.dynamic_temperature
                    else:
                        dyn_pred = dyn_pred_logits
                    dyn_pred = F.sigmoid(dyn_pred)
                    dyn_keep = 1 + dyn_pred * (num_spatial_tokens - 1)
                    dyn_keep = round_ste(dyn_keep)
                    target_num_tokens_per_sample = num_tokens_per_sample
                    num_tokens_per_sample = dyn_keep

                x, H, W = self._apply_downscale(x, downscale_idx, H, W)
                downscale_idx += 1

            if self.progressive_reduction:
                if i == last_downscale_level:
                    assert attn_mask is None
                    attn_mask = torch.ones(B, x.shape[1], dtype=torch.bool, device=x.device)
                    curr_num_tokens = torch.full((B,), x.shape[1] - self.num_prefix_tokens, dtype=torch.int64, device=x.device)
                    ct_seq = torch.arange(0, x.shape[1], dtype=torch.int64, device=x.device).unsqueeze(0).expand(B, -1)
                elif i > last_downscale_level:
                    assert curr_num_tokens is not None
                    num_remaining_blocks = len(self.blocks) - i
                    curr_num_to_drop = (curr_num_tokens - num_tokens_per_sample.long()) // num_remaining_blocks
                    max_keep = curr_num_tokens + self.num_prefix_tokens - curr_num_to_drop
                    curr_attn_mask = ct_seq[:, :x.shape[1]] < max_keep.unsqueeze(1)
                    attn_mask = attn_mask & curr_attn_mask
                    curr_num_tokens -= curr_num_to_drop

                    # We only need to keep up to the longest valid sequence, so this allows us to progressively
                    # truncate x, saving on compute
                    is_valid_for_any = torch.any(attn_mask, dim=0)
                    curr_valid_length = torch.count_nonzero(is_valid_for_any, dim=0).item()
                    x = x[:, :curr_valid_length]
                    attn_mask = attn_mask[:, :curr_valid_length]

            my_use_checkpoint = use_checkpoint and i < self.grad_checkpointing

            fwd_fn = partial(checkpoint, blk, use_reentrant=False) if my_use_checkpoint else blk

            # Apply transformer block
            if attn_mask is not None:
                if attn_mask.ndim == 4:
                    my_mask = attn_mask
                elif attn_mask.ndim == 2:
                    my_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            else:
                my_mask = None
            # x = fwd_fn(x, attn_mask=my_mask)
            x = fwd_fn(x)

        x = self.norm(x)

        #if self.patch_generator is not None:
        #    x = self.patch_generator.broadcast_masks(x, apt_masks, pos_enc=pos_enc)
        #     self.patch_generator.maybe_visualize(images, x, apt_masks, self)

        self._total_num_tokens += num_tokens_per_sample.sum().long()
        self._total_num_samples += B

        # Log token counts every 100 iterations
        self._iter_count += 1
        if self._iter_count % 100 == 0 and get_rank() == 0:
            # min_tokens = num_tokens_per_sample.min().item()
            # max_tokens = num_tokens_per_sample.max().item()
            # mean_tokens = num_tokens_per_sample.float().mean().item()
            # print(f"[RADIO1D iter {self._iter_count.item()}] 1D tokens: min={min_tokens}, max={max_tokens}, mean={mean_tokens:.1f}, use_last_tokens={use_last_tokens}")
            mean_tokens = self._total_num_tokens.float() / self._total_num_samples.float()
            print(f"[RADIO1D iter {self._iter_count.item()}] avg 1D tokens: {mean_tokens.item():.2f}")

        # Slice tokens with per-sample counts, padded to max in this micro-batch
        prefix_tokens, global_tokens, global_token_mask = slice_1d_tokens(
            x,
            num_tokens_per_sample,
            num_prefix_tokens=self.num_prefix_tokens,
            use_last_tokens=use_last_tokens,
            dynamic=self.dynamic_rate,
        )

        # Return encoder output with metadata needed for decoder
        encoder_output = torch.cat([prefix_tokens, global_tokens], dim=1)
        ret = {
            "encoder": encoder_output,
            "global_tokens": global_tokens,
            "global_token_mask": global_token_mask,
            "encoder_spatial_size": (H, W),
            "original_spatial_size": (original_H, original_W),
        }
        if is_dynamic:
            ret["dynamic_rate"] = dict(
                pred_rate=num_tokens_per_sample,
                pred_pct=dyn_pred,
                target_rate=target_num_tokens_per_sample,
                target_pct=target_num_tokens_per_sample / num_spatial_tokens,
                pred_logits=dyn_pred_logits,
                num_spatial_tokens=num_spatial_tokens,
            )

        self.apply_aux_losses(ret)

        return ret

    def forward_decoder(
        self,
        global_tokens: torch.Tensor,
        global_token_mask: torch.Tensor,
        encoder_spatial_size: Tuple[int, int],
        original_spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Forward pass through decoder only.

        Args:
            global_tokens: Global tokens from encoder (B, max_tokens, C)
            global_token_mask: Boolean mask for valid global tokens (B, max_tokens)
            encoder_spatial_size: Tuple of (H, W) spatial dimensions after encoding
            original_spatial_size: Tuple of (H, W) original spatial dimensions before padding

        Returns:
            Decoded features (B, num_prefix + H*W, target_embed_dim)
        """
        B = global_tokens.shape[0]
        H, W = encoder_spatial_size
        original_H, original_W = original_spatial_size

        # Decode back to original resolution (decoder uses its own prefix tokens)
        decoded, decoded_H, decoded_W = self.decoder(
            global_tokens=global_tokens,
            global_token_mask=global_token_mask,
            input_size=(H, W),
        )

        # Crop to original dimensions if needed (handles odd H/W that were padded during downscaling)
        if decoded_H != original_H or decoded_W != original_W:
            prefix = decoded[:, :self.num_prefix_tokens]
            patches = decoded[:, self.num_prefix_tokens:]  # (B, decoded_H*decoded_W, C)
            patches = patches.reshape(B, decoded_H, decoded_W, -1)
            patches = patches[:, :original_H, :original_W, :].reshape(B, original_H * original_W, -1)
            decoded = torch.cat([prefix, patches], dim=1)

        return decoded

    def apply_aux_losses(self, encoder_result: dict):
        if not self.training or not self.dynamic_rate:
            return

        dyn_dict = encoder_result['dynamic_rate']
        pred_rate = dyn_dict['pred_rate']
        pred_pct = dyn_dict['pred_pct']
        target_rate = dyn_dict['target_rate']
        target_pct = dyn_dict['target_pct']
        pred_logits = dyn_dict['pred_logits']
        num_spatial_tokens = dyn_dict['num_spatial_tokens']

        pred_local_num_tokens = pred_rate.sum()
        pred_global_num_tokens = pred_local_num_tokens.clone()
        local_num_tokens = torch.tensor(num_spatial_tokens * pred_rate.shape[0], dtype=torch.float32, device=pred_rate.device)
        global_num_tokens = local_num_tokens.clone()
        if dist.is_initialized():
            dist.all_reduce(global_num_tokens, op=dist.ReduceOp.SUM)
            pred_global_num_tokens = all_reduce_with_gradients(pred_global_num_tokens, op=dist.ReduceOp.SUM)

        global_pred_pct = pred_global_num_tokens / global_num_tokens

        loss_rate = F.mse_loss(global_pred_pct, target_pct[0])

        aux_losses: Dict[str, torch.Tensor] = getattr(self, 'auxiliary_losses', dict())
        self.auxiliary_losses = aux_losses
        aux_losses['dynamic_rate_mse'] = 1.0 * loss_rate.mean()

        quantile = 0.98
        quantile_sym = (1.0 - quantile) / 2 + quantile
        log_q = math.log(quantile_sym / (1 - quantile_sym))
        logit_threshold = log_q / self.dynamic_temperature
        logit_excess = F.relu(torch.abs(pred_logits) - logit_threshold).pow(2)
        aux_losses['dynamic_rate_logit_penalty'] = 0.1 * logit_excess.mean()

        aux_losses['dynamic_rate_abs_diff'] = (global_pred_pct - target_pct[0]).abs().detach()

        # caps = ', '.join(f'{v * 100:.1f}%' for v in pred_pct[:4].tolist())
        # viz_caption = f"Dynamic Rate Pred: Target: {target_pct[0].item() * 100:.1f}%, Achieved: {global_pred_pct.item() * 100:.1f}%, Pred: [{caps}]"
        # FeatureDistillationLoss.VIZ_CAPTION = viz_caption
        pass

    def forward_features(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
        use_last_tokens: bool = False,
    ) -> dict:
        """Forward pass through encoder and decoder (full feature extraction).

        Args:
            x: Input image tensor of shape (B, C, H, W)
            attn_mask: Optional attention mask
            num_tokens: Number of 1D tokens to output per sample.
                       If None during training: samples per-sample from mode distribution
                       If None during inference: uses max(modes)
            use_last_tokens: If True, take the last num_tokens instead of the first

        Returns:
            Dict with keys:
                - "encoder": (B, num_prefix + max_tokens, C) - prefix tokens + 1D global tokens
                - "decoder": (B, num_prefix + H*W, target_embed_dim) - reconstructed full sequence
                  where H, W are the spatial dims after encoding (and after any upscaling in decoder)
        """
        # Run encoder
        encoder_result = self.forward_encoder(x, attn_mask=attn_mask, num_tokens=num_tokens, use_last_tokens=use_last_tokens)

        # Run decoder
        decoded = self.forward_decoder(
            global_tokens=encoder_result["global_tokens"],
            global_token_mask=encoder_result["global_token_mask"],
            encoder_spatial_size=encoder_result["encoder_spatial_size"],
            original_spatial_size=encoder_result["original_spatial_size"],
        )

        # Return format: both encoder and decoder are full sequences
        # encoder: [prefix_tokens (cls + registers), global_tokens]
        # decoder: [prefix_tokens, decoded_patches] (already concatenated by decoder)
        return {"encoder": encoder_result["encoder"], "decoder": decoded}

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHW',
        intermediates_only: bool = False,
        output_dict: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], dict]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs ('NCHW' or 'NLC')
            intermediates_only: Only return intermediate features
            output_dict: Return outputs as a dictionary
            attn_mask: Optional attention mask

        Returns:
            Depends on flags:
            - intermediates_only=True: List of intermediate features
            - output_dict=True: Dict with 'image_features' and 'image_intermediates'
            - Otherwise: Tuple of (final_features, intermediates)
        """
        assert output_fmt in ('NCHW', 'NLC'), f"Invalid output_fmt: {output_fmt}"

        # Determine which block indices to collect
        num_blocks = len(self.blocks)
        if indices is None:
            take_indices = list(range(num_blocks))
        elif isinstance(indices, int):
            # Take last n blocks
            take_indices = list(range(max(0, num_blocks - indices), num_blocks))
        else:
            take_indices = list(indices)

        max_index = max(take_indices) if take_indices else num_blocks - 1

        # Infer spatial dimensions from input image before patch embedding
        B, _, H_img, W_img = x.shape
        if self.patch_embed is not None:
            patch_size = self.patch_embed.patch_size[0]
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            apt_masks = None
            pos_enc = None
        else:
            images = x
            patch_size = self.patch_generator.patch_size
            x = self.patch_generator(x)

            #if apt_attn_mask is not None:
            #    attn_mask = apt_attn_mask

        # Compute spatial dimensions (in patches) for downscaling
        H = H_img // patch_size
        W = W_img // patch_size

        # Collect intermediate activations
        intermediates = []
        intermediates_prefix = [] if return_prefix_tokens else None
        downscale_idx = 0

        for i, blk in enumerate(self.blocks):
            # Apply downscale before this block if specified
            if i in self.downscale_levels:
                x, H, W = self._apply_downscale(x, downscale_idx, H, W)
                downscale_idx += 1

            # Apply transformer block
            if attn_mask is not None:
                x = blk(x, attn_mask=attn_mask)
            else:
                x = blk(x)

            # Collect intermediate if this index is requested
            if i in take_indices:
                # Get spatial tokens (excluding prefix tokens)
                num_prefix = self.num_prefix_tokens
                feat = x[:, num_prefix:]  # (B, H*W, C)

                if norm:
                    feat = self.norm(feat)

                # Reshape to output format
                if output_fmt == 'NCHW':
                    C = feat.shape[-1]
                    feat = feat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                # else 'NLC' - keep as is

                intermediates.append(feat)

                if return_prefix_tokens:
                    prefix = x[:, :num_prefix]
                    if norm:
                        prefix = self.norm(prefix)
                    intermediates_prefix.append(prefix)

            # Stop early if we've collected all needed intermediates
            if stop_early and i >= max_index:
                break

        # Compute final features if needed
        if not intermediates_only:
            # Continue from where we left off if we stopped early
            if stop_early and max_index < num_blocks - 1:
                for i in range(max_index + 1, num_blocks):
                    if i in self.downscale_levels:
                        x, H, W = self._apply_downscale(x, downscale_idx, H, W)
                        downscale_idx += 1
                    if attn_mask is not None:
                        x = blk(x, attn_mask=attn_mask)
                    else:
                        x = self.blocks[i](x)

            x = self.norm(x)

            if self.patch_generator is not None:
                x = self.patch_generator.broadcast_masks(x, apt_masks, pos_enc=pos_enc)

        # Return in requested format
        if output_dict:
            result = {
                'image_intermediates': intermediates,
            }
            if not intermediates_only:
                result['image_features'] = x
            if return_prefix_tokens:
                result['image_intermediates_prefix'] = intermediates_prefix
            return result
        elif intermediates_only:
            return intermediates
        else:
            return x, intermediates

    def get_first_downscale_block_idx(self) -> Optional[int]:
        """Return the index of the first downscaling block, or None if no downscaling."""
        if not self.downscale_levels:
            return None
        return min(self.downscale_levels)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            num_tokens: Optional[int] = None,
            use_last_tokens: bool = False,
    ) -> dict:
        """Forward pass through encoder only.

        Args:
            x: Input image tensor of shape (B, C, H, W)
            attn_mask: Optional attention mask
            num_tokens: Number of 1D tokens to use per sample (for slicing)
            use_last_tokens: If True, take the last num_tokens instead of the first

        Returns:
            Dict with keys:
                - "encoder": (B, num_prefix + max_tokens, C) - prefix tokens + 1D global tokens
                - "global_tokens": (B, max_tokens, C) - sliced global tokens (for decoder input)
                - "global_token_mask": (B, max_tokens) - validity mask for global tokens
                - "encoder_spatial_size": (H, W) - spatial dimensions after encoding
                - "original_spatial_size": (H, W) - original spatial dimensions before padding
        """
        return self.forward_encoder(x, attn_mask=attn_mask, num_tokens=num_tokens, use_last_tokens=use_last_tokens)


@register_model
def radio1d_large_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    if pretrained:
        raise ValueError('There is no pretrained weights for radio1d_large_patch16_224')

    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('radio1d_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def radio1d_so400m_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT model matching the architecture of the So400M model from
    "Scaling Vision Transformers to 400 Million Parameters" (https://arxiv.org/abs/2302.05442).
    """
    if pretrained:
        raise ValueError('There is no pretrained weights for vit_so400m_patch16_224')
    mlp_ratio = 4304 / 1152

    model_args = dict(patch_size=16, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=mlp_ratio)
    model = _create_vision_transformer('radio1d_so400m_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))

    return model


@register_model
def radio1d_huge_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    if pretrained:
        raise ValueError('There is no pretrained weights for radio1d_huge_patch16_224')

    model_args = dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16)
    model = _create_vision_transformer('radio1d_huge_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def magneto_init(model: VisionTransformer, num_blocks: int = None):
    '''
    Initialization following [Magneto](http://arxiv.org/abs/2210.06423)
    '''
    # model.cuda()

    attention_modules = [m for m in model.modules() if isinstance(m, Attention)]
    mlp_modules = [m for m in model.modules() if isinstance(m, Mlp)]

    if num_blocks is None:
        num_blocks = len(model.blocks)
    gamma = math.sqrt(math.log(2 * num_blocks))

    for m in attention_modules:
        qkv = m.qkv
        q, k, v = qkv.weight.data.chunk(3, dim=0)
        xavier_normal_(q, gain=1)
        xavier_normal_(k, gain=1)
        xavier_normal_(v, gain=gamma)
        xavier_normal_(m.proj.weight.data, gain=gamma)

    for m in mlp_modules:
        xavier_normal_(m.fc1.weight.data, gain=gamma)
        xavier_normal_(m.fc2.weight.data, gain=gamma)


def _init_layerscale(model: VisionTransformer):
    # https://proceedings.neurips.cc/paper_files/paper/2022/file/ae0cba715b60c4052359b3d52a2cff7f-Paper-Conference.pdf
    for i, block in enumerate(model.blocks):
        if isinstance(block, Block):
            ls = 1 / math.sqrt(i + 1)
            block.ls1.gamma.data.fill_(ls)
            block.ls2.gamma.data.fill_(ls)
        elif isinstance(block, PatchMerging):
            ls = 1 / math.sqrt(i + 1)
            block.reduction.weight.data.fill_(ls)


def _create_vision_transformer(name, *args, pretrained=False, **kwargs):
    #model = RADIO1D(*args, **kwargs)
    model = build_model_with_cfg(RADIO1D, name, pretrained=pretrained, **dict(args, **kwargs))
    if not pretrained:
        magneto_init(model)
        if kwargs.get('init_values', None) == -1234:
            _init_layerscale(model)
    return model
