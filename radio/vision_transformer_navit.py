# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""
***********************************************************************
WARNING: This file is auto-generated. Do not make edits directly to it.
***********************************************************************

NaViT (Native Resolution Vision Transformer) with RoPE2D position encoding — single-file vendorable build.

This module implements a Vision Transformer that:
- Accepts variable-size images (NaViT style)
- Uses 2D Rotary Position Embeddings (RoPE2D) for position information
- Leverages Flash Attention with variable-length sequences for efficiency
- Supports packing multiple images into a single batch without padding

The implementation follows the NaViT paper: https://arxiv.org/abs/2307.06304
and uses RoPE2D for better resolution generalization.

All local layer dependencies (RoPE2D, LayerScale, VarlenRoPEAttention,
VarlenRoPEXAttn, NaViTBlock, NaViTMLP, NormMethod, VarlenAttnPool) are
inlined so the file is fully self-contained.
"""

from dataclasses import dataclass
from enum import Enum
from functools import partial
import inspect
from logging import getLogger
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from torch import Tensor

from timm.models import register_model

_LOGGER = getLogger(__name__)


###############################################################################
# Fused rotary embedding kernel (optional)
###############################################################################

_FUSED_ROTARY_AVAILABLE = False
_apply_rotary_emb_fused: Optional[Callable[..., Tensor]] = None

try:
    from flash_attn.layers.rotary import apply_rotary_emb as _apply_rotary_emb_fused  # type: ignore
    _FUSED_ROTARY_AVAILABLE = True
    _LOGGER.info('RoPE2D: Fused rotary embedding kernel available (flash_attn.layers.rotary)')
except ImportError:
    _LOGGER.info('RoPE2D: Fused rotary kernel not available, using manual rotate_half fallback')


###############################################################################
# Flash Attention imports with fallback
###############################################################################

_FA_VARLEN_AVAILABLE = False
_FA_VERSION: Optional[int] = None
flash_attn_varlen_func: Optional[Callable[..., Tensor]] = None

try:
    # Try Flash Attention 3 first
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen  # type: ignore
    flash_attn_varlen_func = _fa3_varlen
    _FA_VARLEN_AVAILABLE = True
    _FA_VERSION = 3
    _LOGGER.info('VarlenRoPEAttention: Using Flash Attention 3 with variable length support')
except ImportError:
    try:
        # Fall back to Flash Attention 1/2
        from flash_attn import flash_attn_varlen_func as _fa2_varlen  # type: ignore
        flash_attn_varlen_func = _fa2_varlen
        _FA_VARLEN_AVAILABLE = True
        _FA_VERSION = 2
        _LOGGER.info('VarlenRoPEAttention: Using Flash Attention 2 with variable length support')
    except ImportError:
        _LOGGER.warning('VarlenRoPEAttention: Flash Attention not available, varlen mode will not work')


def is_flash_attn_available() -> bool:
    """Check if FlashAttention with variable-length support is available."""
    return _FA_VARLEN_AVAILABLE


_NATTEN_AVAILABLE = False
na2d_func: Optional[Callable[..., Tensor]] = None

try:
    from natten.functional import na2d  # type: ignore
    na2d_func = na2d
    _NATTEN_AVAILABLE = True
    _LOGGER.info('VarlenRoPEAttention: NATTEN na2d available for neighborhood attention')
except ImportError:
    _LOGGER.info('VarlenRoPEAttention: NATTEN not available, neighborhood attention will not work')


###############################################################################
# RoPE2D — 2D Rotary Position Embedding
###############################################################################

def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding.

    Splits the embedding dimension into two halves: one for row positions
    and one for column positions. Each half uses standard 1D RoPE.

    Supports p-RoPE which truncates the lowest frequencies of RoPE to create
    robust semantic channels. When rope_fraction=1.0, this is standard RoPE.
    When rope_fraction=0.0, this is equivalent to NoPE (no positional encoding).

    Args:
        dim: Total dimension for position encoding (typically head_dim).
        base: Base for computing inverse frequencies.
        rope_fraction: Fraction of dimensions to apply RoPE to (p in p-RoPE).
            1.0 = full RoPE (default), 0.0 = no RoPE (NoPE).
            Values like 0.75 remove the lowest 25% of frequencies.
            The highest frequencies (most positionally sensitive) are kept.
    """
    inv_freq: Tensor
    rope_mask: Tensor

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        rope_fraction: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.rope_fraction = rope_fraction

        if dim % 4 != 0:
            raise ValueError(f"RoPE2D requires dim divisible by 4, got {dim}")

        if not 0.0 <= rope_fraction <= 1.0:
            raise ValueError(f"rope_fraction must be in [0, 1], got {rope_fraction}")

        # We split dim into 4 parts: sin/cos for row, sin/cos for col
        # Each spatial dimension gets dim/2, with sin and cos interleaved
        quarter_dim = dim // 4

        # For p-RoPE: only apply RoPE to the highest frequencies
        # The lowest frequencies (corresponding to largest indices) are zeroed out
        rope_quarter_dim = int(rope_fraction * quarter_dim)
        nope_quarter_dim = quarter_dim - rope_quarter_dim

        # Compute inverse frequencies for the RoPE portion
        # Following the paper: use the highest frequencies (smallest indices)
        if rope_quarter_dim > 0:
            inv_freq = 1.0 / (base ** (torch.arange(0, rope_quarter_dim, dtype=torch.float32) / quarter_dim))
            # Pad with zeros for the NoPE portion (lowest frequencies)
            if nope_quarter_dim > 0:
                inv_freq = torch.cat([inv_freq, torch.zeros(nope_quarter_dim, dtype=torch.float32)])
        else:
            # Pure NoPE: all zeros
            inv_freq = torch.zeros(quarter_dim, dtype=torch.float32)

        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Create mask for which dimensions have rotation applied
        # True where RoPE is applied, False where it's NoPE (no rotation)
        rope_mask = torch.ones(quarter_dim, dtype=torch.bool)
        if nope_quarter_dim > 0:
            rope_mask[-nope_quarter_dim:] = False
        # Expand to full dimension: (quarter_dim,) -> (dim,) accounting for row/col and sin/cos
        # The pattern is: [row_freqs, col_freqs] each repeated twice for cos/sin structure
        rope_mask = rope_mask.repeat(4)  # Repeat for row sin/cos and col sin/cos
        self.register_buffer('rope_mask', rope_mask, persistent=False)

        # Store dimensions info for inspection
        self._rope_dims = rope_quarter_dim * 4  # Total dims with rotation
        self._nope_dims = nope_quarter_dim * 4  # Total dims without rotation

    @property
    def num_rope_dims(self) -> int:
        """Number of dimensions that have RoPE applied."""
        return self._rope_dims

    @property
    def num_nope_dims(self) -> int:
        """Number of dimensions that don't have RoPE (p-RoPE semantic channels)."""
        return self._nope_dims

    def _compute_cos_sin_half(
        self,
        position_ids: Tensor,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Compute cos and sin at half-dim (before tiling).

        Args:
            position_ids: (..., seq_len, 2) or (seq_len, 2) tensor of (row, col) positions.
            dtype: Output dtype.

        Returns:
            Tuple of (cos, sin) tensors of shape (..., seq_len, dim//2).
        """
        freqs = position_ids.float().unsqueeze(-1) @ self.inv_freq.unsqueeze(0)  # (..., seq_len, 2, D//4)
        freqs = freqs.flatten(-2)  # (..., seq_len, D//2)

        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        return cos, sin

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_position_ids: Tensor,
        k_position_ids: Optional[Tensor] = None,
        attn_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply 2D RoPE to queries and keys.

        Args:
            q: Query tensor of shape (..., num_heads, seq_len, dim) or (..., seq_len, dim).
            k: Key tensor of shape (..., num_heads, seq_len, dim) or (..., seq_len, dim).
            q_position_ids: (seq_len_q, 2) or (..., seq_len_q, 2) tensor of (row, col) positions for queries.
            k_position_ids: (seq_len_k, 2) or (..., seq_len_k, 2) tensor of (row, col) positions for keys.
                           If None, uses q_position_ids (for self-attention).
            attn_info: Optional dictionary for caching cos/sin values.

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        if k_position_ids is None:
            k_position_ids = q_position_ids

        dtype = q.dtype
        same_positions = k_position_ids is q_position_ids

        # Get or compute half-dim cos/sin (cached across blocks via attn_info)
        q_cos_sin = attn_info.get('rope_q_cache', None) if attn_info is not None else None
        if q_cos_sin is not None:
            q_cos_half, q_sin_half = q_cos_sin
        else:
            q_cos_half, q_sin_half = self._compute_cos_sin_half(q_position_ids, dtype)
            if attn_info is not None:
                attn_info['rope_q_cache'] = (q_cos_half, q_sin_half)

        if same_positions:
            k_cos_half, k_sin_half = q_cos_half, q_sin_half
        else:
            k_cos_sin = attn_info.get('rope_k_cache', None) if attn_info is not None else None
            if k_cos_sin is not None:
                k_cos_half, k_sin_half = k_cos_sin
            else:
                k_cos_half, k_sin_half = self._compute_cos_sin_half(k_position_ids, dtype)
                if attn_info is not None:
                    attn_info['rope_k_cache'] = (k_cos_half, k_sin_half)

        if _FUSED_ROTARY_AVAILABLE:
            return self._apply_fused(q, k, q_cos_half, q_sin_half, k_cos_half, k_sin_half, dtype)
        else:
            return self._apply_manual(q, k, q_cos_half, q_sin_half, k_cos_half, k_sin_half, dtype)

    def _apply_fused(
        self,
        q: Tensor, k: Tensor,
        q_cos: Tensor, q_sin: Tensor,
        k_cos: Tensor, k_sin: Tensor,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Apply RoPE using the fused flash_attn kernel.

        apply_rotary_emb expects:
          x:   (batch, seq_len, num_heads, head_dim)
          cos: (seq_len, rotary_dim)  — i.e. 2D
        Our inputs are either:
          - 4D: (B, num_heads, seq_len, head_dim)  [batched path]
          - 3D: (num_heads, seq_len, head_dim)       [varlen path]
        cos/sin are either:
          - 2D: (seq_len, dim//2)                     [varlen path]
          - 3D: (B, seq_len, dim//2)                  [batched path]
        """
        assert _apply_rotary_emb_fused is not None

        squeezed = q.dim() == 3
        if squeezed:
            # (H, S, D) -> (1, H, S, D)
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)

        # Swap heads <-> seqlen: (B, H, S, D) -> (B, S, H, D)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)

        # Fused kernel expects cos/sin as 2D (seqlen, rotary_dim).
        # For the batched path they are (B, S, D//2) — since every
        # element in the batch has the same positions we can just take [0].
        if q_cos.dim() == 3:
            q_cos = q_cos[0]
            q_sin = q_sin[0]
        if k_cos.dim() == 3:
            k_cos = k_cos[0]
            k_sin = k_sin[0]

        q = _apply_rotary_emb_fused(q, q_cos, q_sin, interleaved=False)
        k = _apply_rotary_emb_fused(k, k_cos, k_sin, interleaved=False)

        # Swap back: (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)

        if squeezed:
            q = q.squeeze(0)
            k = k.squeeze(0)

        return q.to(dtype=dtype), k.to(dtype=dtype)

    def _apply_manual(
        self,
        q: Tensor, k: Tensor,
        q_cos_half: Tensor, q_sin_half: Tensor,
        k_cos_half: Tensor, k_sin_half: Tensor,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Apply RoPE using the manual rotate_half fallback."""
        # Tile half-dim cos/sin to full dim for the rotate_half convention
        q_cos = q_cos_half.tile(2)
        q_sin = q_sin_half.tile(2)
        k_cos = k_cos_half.tile(2)
        k_sin = k_sin_half.tile(2)

        # Broadcast cos/sin properly for the input shape
        while q_cos.dim() < q.dim():
            q_cos = q_cos.unsqueeze(-3)
            q_sin = q_sin.unsqueeze(-3)
        while k_cos.dim() < k.dim():
            k_cos = k_cos.unsqueeze(-3)
            k_sin = k_sin.unsqueeze(-3)

        q_rot = q * q_cos + rotate_half(q) * q_sin
        k_rot = k * k_cos + rotate_half(k) * k_sin

        return q_rot.to(dtype=dtype), k_rot.to(dtype=dtype)


###############################################################################
# LayerScale
###############################################################################

class LayerScale(nn.Module):
    """Layer Scale from CaiT paper (https://arxiv.org/abs/2103.17239).

    Applies a learnable per-channel scaling factor after attention/MLP.
    """

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


###############################################################################
# VarlenRoPEAttention — self-attention with RoPE + Flash Attention varlen
###############################################################################

class VarlenRoPEAttention(nn.Module):
    """Multi-head self-attention with 2D RoPE and FlashAttention variable-length support.

    This attention module is designed to work with variable-length sequences
    packed together, using cumulative sequence lengths for flash attention.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension of each head. If None, uses dim // num_heads.
        qkv_bias: Whether to use bias in QKV projection.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
        rope: RoPE2D module for 2D rotary position embeddings.
        norm_layer: Normalization layer for Q/K.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: RoPE2D,
        head_dim: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        self_attn_kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop_p = attn_drop
        self.self_attn_kernel_size = self_attn_kernel_size

        self.qkv = nn.Linear(dim, 3 * num_heads * self.head_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(
        self,
        x: Tensor,
        attn_info: Dict,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor. Shape depends on mode:
               - Flash varlen: (total_tokens, dim)
               - Padded: (batch_size, seq_len, dim)
            position_ids: (total_tokens, 2) or (batch_size, seq_len, 2) tensor of (row, col) positions.
            attn_info: Dict with 'cu_seqlens' and 'max_seqlen' for variable-length flash attention.

        Returns:
            Output tensor with same shape as input.
        """
        if self.self_attn_kernel_size is not None:
            if 'grid_size' not in attn_info:
                raise ValueError('`grid_size` must be present in `attn_info` for neighborhood attention to work!')
            # Neighborhood attention (na2d) for spatial tokens, skip prefix tokens
            return self._forward_na2d(x, position_ids, attn_info)
        elif _FA_VARLEN_AVAILABLE and 'cu_seqlens' in attn_info and position_ids is not None:
            # Flash attention with variable length sequences
            return self._forward_flash_varlen(x, position_ids, attn_info)
        else:
            # Standard batched attention (with optional padding mask)
            return self._forward_batched(x, position_ids, attn_info)

    def _forward_flash_varlen(
        self,
        x: Tensor,
        position_ids: Tensor,
        attn_info: Dict,
    ) -> Tensor:
        """Forward pass using Flash Attention with variable length sequences."""
        assert flash_attn_varlen_func is not None, "FlashAttention not available"
        assert self.attn_drop_p == 0, "Flash attention with variable length does not support dropout"

        total_tokens = x.shape[0]

        # Compute QKV
        qkv = self.qkv(x)  # (total_tokens, 3 * num_heads * head_dim)
        qkv = qkv.reshape(total_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (total_tokens, num_heads, head_dim)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply 2D RoPE
        if position_ids is not None:
            # Reshape for RoPE: (total_tokens, num_heads, head_dim) -> (num_heads, total_tokens, head_dim)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            q, k = self.rope(q, k, position_ids, attn_info=attn_info)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)

        # Flash attention
        cu_seqlens = attn_info['cu_seqlens']
        max_seqlen = attn_info['max_seqlen']

        out = flash_attn_varlen_func(
            q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scale,
        )

        # Reshape and project
        out = out.reshape(total_tokens, -1)  # (total_tokens, num_heads * head_dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

    def _forward_batched(
        self,
        x: Tensor,
        position_ids: Optional[Tensor],
        attn_info: Optional[Dict],
    ) -> Tensor:
        """Forward pass using standard batched attention."""
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, N, num_heads, head_dim)

        # Transpose to (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply 2D RoPE if position_ids provided
        if position_ids is not None:
            # position_ids: (B, N, 2) -> need to apply per-batch
            q, k = self.rope(q, k, position_ids, attn_info=attn_info)

        # Get attention mask if available
        attn_mask = attn_info.get('attn_mask') if attn_info else None

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16),
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

    def _forward_na2d(
        self,
        x: Tensor,
        position_ids: Optional[Tensor],
        attn_info: Dict,
    ) -> Tensor:
        """Forward pass using NATTEN neighborhood attention for spatial tokens.

        Prefix tokens (e.g. homography, summary) are skipped (zero self-attention
        residual) and only spatial tokens participate in neighborhood attention.

        Requires 'grid_size' (H, W) and 'num_prefix' (int) in attn_info.
        """
        assert na2d_func is not None, "NATTEN not available"
        assert self.self_attn_kernel_size is not None

        grid_h, grid_w = attn_info['grid_size']
        num_prefix: int = attn_info.get('num_prefix', 0)
        is_varlen = 'cu_seqlens' in attn_info

        if is_varlen:
            raise NotImplementedError("na2d neighborhood attention is not yet supported in varlen mode")

        B = x.shape[0]
        spatial_len = grid_h * grid_w

        # Compute QKV for spatial tokens only
        x_spatial = x[:, num_prefix:]  # (B, H*W, dim)
        pos_spatial = position_ids[:, num_prefix:] if position_ids is not None else None

        qkv = self.qkv(x_spatial).reshape(B, spatial_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, H*W, num_heads, head_dim)

        # Transpose to (B, num_heads, H*W, head_dim) for norm and RoPE
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply 2D RoPE
        if pos_spatial is not None:
            q, k = self.rope(q, k, pos_spatial, attn_info=attn_info)

        # Transpose back to (B, H*W, num_heads, head_dim) then reshape to (B, H, W, num_heads, head_dim)
        q = q.transpose(1, 2).reshape(B, grid_h, grid_w, self.num_heads, self.head_dim)
        k = k.transpose(1, 2).reshape(B, grid_h, grid_w, self.num_heads, self.head_dim)
        v = v.reshape(B, grid_h, grid_w, self.num_heads, self.head_dim)

        # NATTEN neighborhood attention
        out = na2d_func(
            q.to(dtype=torch.bfloat16),
            k.to(dtype=torch.bfloat16),
            v.to(dtype=torch.bfloat16),
            kernel_size=self.self_attn_kernel_size,
            scale=self.scale,
        )  # (B, H, W, num_heads, head_dim)

        # Reshape back to (B, H*W, dim)
        out = out.reshape(B, spatial_len, -1)

        # Project spatial output
        out = self.proj(out)
        out = self.proj_drop(out)

        # Prefix tokens get zero self-attention residual
        if num_prefix > 0:
            prefix_out = out.new_zeros(B, num_prefix, out.shape[-1])
            out = torch.cat([prefix_out, out], dim=1)

        return out


###############################################################################
# VarlenRoPEXAttn — cross-attention with RoPE + Flash Attention varlen
###############################################################################

class VarlenRoPEXAttn(nn.Module):
    """Cross-attention with 2D RoPE and FlashAttention variable-length support.

    This module performs cross-attention where queries come from one sequence
    and keys/values come from another (potentially different length) sequence.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension of each head. If None, uses dim // num_heads.
        qkv_bias: Whether to use bias in projections.
        proj_drop: Projection dropout rate.
        rope: RoPE2D module for 2D rotary position embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: RoPE2D,
        head_dim: Optional[int] = None,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection (from target)
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
        # Key/Value projections (from memory)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)

        self.out_proj = nn.Linear(num_heads * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        self.rope = rope

    def _forward_varlen(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_position_ids: Tensor,
        mem_position_ids: Tensor,
        attn_info: Dict,
    ) -> Tensor:
        """Forward pass using FlashAttention with variable-length sequences."""
        assert flash_attn_varlen_func is not None, "FlashAttention not available"

        # Project Q, K, V
        q = self.q_proj(tgt).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(memory).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(memory).reshape(-1, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q_t = q.transpose(0, 1)  # (H, N_q, D)
        k_t = k.transpose(0, 1)  # (H, N_kv, D)
        q_t, k_t = self.rope(q_t, k_t, tgt_position_ids, mem_position_ids, attn_info=attn_info)
        q = q_t.transpose(0, 1)
        k = k_t.transpose(0, 1)

        # Flash attention
        out = flash_attn_varlen_func(
            q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16),
            cu_seqlens_q=attn_info['cu_seqlens_q'],
            cu_seqlens_k=attn_info['cu_seqlens_kv'],
            max_seqlen_q=attn_info['max_seqlen_q'],
            max_seqlen_k=attn_info['max_seqlen_kv'],
            softmax_scale=self.scale,
        )

        # Reshape and project
        out = out.reshape(-1, self.dim)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

    def _forward_batched(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_position_ids: Tensor,
        mem_position_ids: Tensor,
        attn_info: Dict,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass using standard batched attention."""
        B, N_q, D = tgt.shape
        N_kv = memory.shape[1]

        # Project Q, K, V
        q = self.q_proj(tgt).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = self.rope(q, k, tgt_position_ids, mem_position_ids, attn_info=attn_info)

        # Create attention mask from padding mask
        attn_mask = None
        if memory_key_padding_mask is not None:
            attn_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_kv)
            attn_mask = attn_mask.expand(-1, self.num_heads, N_q, -1)
            attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16),
            attn_mask=attn_mask, scale=self.scale)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_position_ids: Tensor,
        mem_position_ids: Tensor,
        attn_info: Dict,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Unified forward that dispatches to varlen or batched based on arguments."""
        if 'cu_seqlens_q' in attn_info:
            assert 'cu_seqlens_kv' in attn_info and 'max_seqlen_q' in attn_info and 'max_seqlen_kv' in attn_info
            return self._forward_varlen(
                tgt, memory, tgt_position_ids, mem_position_ids, attn_info=attn_info,
            )
        else:
            return self._forward_batched(
                tgt, memory, tgt_position_ids, mem_position_ids, attn_info=attn_info, memory_key_padding_mask=memory_key_padding_mask
            )


###############################################################################
# NaViTMLP / NaViTBlock — transformer building blocks
###############################################################################

class NormMethod(Enum):
    PRE = 0
    POST = 1
    PERI = 2


def _create_pre_norm(norm_layer: Callable[..., nn.Module], dim: int, norm_method: NormMethod) -> nn.Module:
    if norm_method in (NormMethod.PRE, NormMethod.PERI):
        return norm_layer(dim)
    return nn.Identity()


def _create_post_norm(norm_layer: Callable[..., nn.Module], dim: int, norm_method: NormMethod) -> nn.Module:
    if norm_method in (NormMethod.POST, NormMethod.PERI):
        return norm_layer(dim)
    return nn.Identity()


class NaViTMLP(nn.Module):
    """MLP block for NaViT."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        self.grad_checkpointing = grad_checkpointing

        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        fn = partial(checkpoint, self._inner_forward, use_reentrant=False) if self.grad_checkpointing else self._inner_forward
        return fn(x)

    def _inner_forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NaViTBlock(nn.Module):
    """Transformer block for NaViT with RoPE attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: RoPE2D,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        qk_norm_layer: Callable[..., nn.Module] = nn.Identity,
        init_values: Optional[float] = None,
        grad_checkpointing: bool = False,
        norm_method: NormMethod = NormMethod.PRE,
    ):
        super().__init__()
        self.norm_attn_pre = _create_pre_norm(norm_layer, dim, norm_method)
        self.norm_attn_post = _create_post_norm(norm_layer, dim, norm_method)
        self.attn = VarlenRoPEAttention(
            dim=dim,
            num_heads=num_heads,
            rope=rope,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=qk_norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm_mlp_pre = _create_pre_norm(norm_layer, dim, norm_method)
        self.norm_mlp_post = _create_post_norm(norm_layer, dim, norm_method)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = NaViTMLP(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
            grad_checkpointing=grad_checkpointing,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(
        self,
        x: Tensor,
        position_ids: Optional[Tensor] = None,
        attn_info: Optional[Dict] = None,
    ) -> Tensor:
        x = x + self.ls1(self.norm_attn_post(self.attn(self.norm_attn_pre(x), position_ids=position_ids, attn_info=attn_info)))
        x = x + self.ls2(self.norm_mlp_post(self.mlp(self.norm_mlp_pre(x))))
        return x

    # Legacy name mapping for checkpoint backward compatibility.
    _LEGACY_KEY_MAP = {
        "norm1.": "norm_attn_pre.",
        "norm2.": "norm_mlp_pre.",
    }

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for old_prefix, new_prefix in self._LEGACY_KEY_MAP.items():
            old_full = prefix + old_prefix
            new_full = prefix + new_prefix
            keys_to_rename = [k for k in state_dict if k.startswith(old_full)]
            for k in keys_to_rename:
                state_dict[new_full + k[len(old_full):]] = state_dict.pop(k)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


###############################################################################
# VarlenAttnPool — attention pooling for summary tokens
###############################################################################

class VarlenAttnPool(nn.Module):
    positions: Tensor

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_tokens: int,
        rope: RoPE2D,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.scale = self.head_dim ** -0.5
        mlp_dim = int(dim * mlp_ratio)

        self.tgt_norm = nn.LayerNorm(dim)
        self.mem_norm = nn.LayerNorm(dim)

        self.cross_attn = VarlenRoPEXAttn(
            dim=dim,
            num_heads=num_heads,
            rope=rope,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        self.tgt = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.register_buffer('positions', torch.zeros(1, num_tokens, 2, dtype=torch.float32), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        tgt_key = prefix + 'tgt'
        if tgt_key in state_dict:
            ckpt_tgt = state_dict[tgt_key]
            ckpt_num_tokens = ckpt_tgt.shape[1]
            if ckpt_num_tokens != self.num_tokens:
                if ckpt_num_tokens > self.num_tokens:
                    raise ValueError(
                        f"Checkpoint has {ckpt_num_tokens} attention pool tokens, "
                        f"but the model only has {self.num_tokens}. "
                        f"The model's num_tokens must be >= the checkpoint's."
                    )
                # Pad with the initialized values for the extra tokens
                new_tgt = self.tgt.data.clone()
                new_tgt[:, :ckpt_num_tokens, :] = ckpt_tgt
                state_dict[tgt_key] = new_tgt

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(
        self,
        x: Tensor,
        mem_position_ids: Tensor,
        cross_attn_info: Dict[str, Any],
    ) -> Tensor:
        if 'cu_seqlens_kv' in cross_attn_info:
            return self._forward_varlen(x, mem_position_ids, cross_attn_info)
        else:
            return self._forward_batched(x, mem_position_ids, cross_attn_info)

    def _forward_varlen(
        self,
        x: Tensor,
        mem_position_ids: Tensor,
        cross_attn_info: Dict[str, Any],
    ) -> Tensor:
        B = cross_attn_info['cu_seqlens_kv'].shape[0] - 1

        tgt = self.tgt.expand(B, -1, -1).flatten(0, 1)
        tgt = self.tgt_norm(tgt)

        tgt_cu_seqlens = torch.arange(0, (B + 1) * self.num_tokens, self.num_tokens, dtype=torch.int32, device=x.device)
        cross_attn_info['cu_seqlens_q'] = tgt_cu_seqlens
        cross_attn_info['max_seqlen_q'] = self.num_tokens

        mem = self.mem_norm(x)

        tgt_pos = self.positions.expand(B, -1, -1).flatten(0, 1)

        tgt = self.cross_attn(
            tgt=tgt,
            memory=mem,
            tgt_position_ids=tgt_pos,
            mem_position_ids=mem_position_ids,
            attn_info=cross_attn_info,
        )

        tgt = tgt + self.mlp(self.mlp_norm(tgt))

        tgt = tgt.unflatten(0, (B, self.num_tokens))

        return tgt

    def _forward_batched(
        self,
        x: Tensor,
        mem_position_ids: Tensor,
        cross_attn_info: Dict[str, Any],
    ) -> Tensor:
        tgt = self.tgt.expand(len(x), -1, -1)
        tgt = self.tgt_norm(tgt)

        mem = self.mem_norm(x)
        tgt_pos = self.positions.expand(len(x), -1, -1)

        tgt = self.cross_attn(
            tgt=tgt,
            memory=mem,
            tgt_position_ids=tgt_pos,
            mem_position_ids=mem_position_ids,
            attn_info=cross_attn_info,
        )

        tgt = tgt + self.mlp(self.mlp_norm(tgt))

        return tgt


###############################################################################
# NaViT — main model
###############################################################################

@dataclass
class NaViTOutput:
    """Output from NaViT forward pass.

    Attributes:
        features: Packed feature tensor of shape (total_tokens, embed_dim) or
                  (batch_size, max_tokens, embed_dim) if padded.
        cu_seqlens: Cumulative sequence lengths for flash attention.
        max_seqlen: Maximum sequence length in the batch.
        image_ids: Tensor mapping each token to its source image index.
        position_ids: (N, 2) tensor of (row, col) positions for each token.
        cls_tokens: Optional class tokens if num_prefix_tokens > 0, shape (num_images, num_prefix, embed_dim).
    """
    features: Tensor
    cu_seqlens: Optional[Tensor] = None
    max_seqlen: int = 0
    image_ids: Optional[Tensor] = None
    position_ids: Optional[Tensor] = None
    cls_tokens: Optional[Tensor] = None


@dataclass
class ViTDetConfig:
    window_size: int
    global_attn_indices: Set[int]


class NaViTPatchEmbed(nn.Module):
    """Patch embedding for NaViT that handles variable-size images.

    Args:
        patch_size: Size of each patch.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        bias: Whether to use bias in convolution.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=bias)

    def forward(self, x: Union[Tensor, List[Tensor]]) -> Tuple[Tensor, Union[List[Tuple[int, int]], Tuple[int, int]]]:
        """
        Args:
            x: Image tensor of shape (C, H, W) or (B, C, H, W).

        Returns:
            Tuple of (patches, grid_size) where:
            - patches: (num_patches, embed_dim) or (B, num_patches, embed_dim)
            - grid_size: (grid_h, grid_w) tuple
        """
        if isinstance(x, list):
            return self.forward_list(x)
        return self.forward_batched(x)

    def forward_list(self, x: List[Tensor]) -> Tuple[Tensor, List[Tuple[int, int]]]:
        """Forward pass for a list of variable-size images.

        Args:
            x: List of image tensors, each of shape (C, H, W).
        """
        flat_tensors = []
        grid_sizes = []
        for img in x:
            grid_sizes.append(self._get_grid(img))
            flat_tensors.append(self._rearrange(img))

        flat_tensors = torch.cat(flat_tensors, dim=0)
        flat_tensors = self.proj(flat_tensors)

        return flat_tensors, grid_sizes

    def forward_batched(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        grid = self._get_grid(x)

        x = self._rearrange(x)
        x = self.proj(x)

        return x, grid

    def _get_grid(self, x: Tensor) -> Tuple[int, int]:
        H, W = x.shape[-2:]
        assert H % self.patch_size == 0, f"Height {H} not divisible by patch size {self.patch_size}"
        assert W % self.patch_size == 0, f"Width {W} not divisible by patch size {self.patch_size}"

        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        return grid_h, grid_w

    def _rearrange(self, x: Tensor) -> Tensor:
        flatten = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            flatten = True

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        if flatten:
            x = x.squeeze(0)
        return x


_NORM_FACTORY = {
    'layernorm': nn.LayerNorm,
    'rmsnorm': nn.RMSNorm,
}


class NaViT(nn.Module):
    """Native Resolution Vision Transformer with 2D RoPE.

    This model processes variable-size images efficiently by:
    1. Packing all patches from different images into a single sequence
    2. Using cumulative sequence lengths for Flash Attention
    3. Applying 2D RoPE for position information

    Args:
        patch_size: Size of each patch.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embed dim.
        qkv_bias: Whether to use bias in QKV projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer constructor or string identifier.
        act_layer: Activation layer constructor.
        num_prefix_tokens: Number of prefix tokens (e.g., CLS tokens) per image.
        rope_base: Base frequency for RoPE.
        rope_fraction: Fraction of dimensions to apply RoPE to (p in p-RoPE).
            1.0 = full RoPE (default), values < 1.0 remove the lowest frequencies.
            See "Round and Round We Go!" paper for details.
        init_values: Initial value for LayerScale. If None, LayerScale is disabled.
    """
    blocks: nn.ModuleList  # List of NaViTBlock
    prefix_pos: Optional[Tensor]

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Union[Callable[..., nn.Module], str] = 'layernorm',
        act_layer: Callable[..., nn.Module] = nn.GELU,
        num_cls_tokens: int = 0,
        num_registers: int = 0,
        rope_base: float = 1000.0,
        rope_fraction: float = 1.0,
        init_values: Optional[float] = None,
        grad_checkpointing: Union[bool, int] = False,
        pool_type: str = 'cls_token',
        vitdet_config: Union[dict, ViTDetConfig, None] = None,
        norm_method: Union[NormMethod, str] = NormMethod.PRE,
        matryoshka: bool = False,
        post_norm_affine: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pool_type = pool_type

        if isinstance(norm_method, str):
            norm_method = NormMethod[norm_method.upper()]
        if isinstance(norm_layer, str):
            norm_layer = _NORM_FACTORY[norm_layer.lower()]

        if pool_type.startswith('attn'):
            qk_norm = pool_type == 'attn_qknorm'
            self.attn_pool = VarlenAttnPool(
                dim=embed_dim,
                num_heads=num_heads,
                num_tokens=num_cls_tokens,
                rope=RoPE2D(embed_dim // num_heads, base=rope_base, rope_fraction=rope_fraction),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            )
            num_cls_tokens = 0  # CLS tokens are now implemented as part of the attention pool
            self.pool_type = 'attn'
        elif pool_type == 'cls_token':
            pass
        else:
            raise ValueError(f'Unsupported pool type: {pool_type}')

        self.num_cls_tokens = num_cls_tokens
        self.num_register_tokens = num_registers
        num_prefix_tokens = num_cls_tokens + num_registers
        self.num_prefix_tokens = num_prefix_tokens
        self.num_heads = num_heads
        self.depth = depth
        self.rope_fraction = rope_fraction
        self.matryoshka = matryoshka

        # Matryoshka: compute granularity dims (powers of 2 up to embed_dim)
        if matryoshka:
            dims = []
            weights = []
            d = 128
            w = 1
            while d < embed_dim:
                dims.append(d)
                weights.append(w)
                d *= 2
                w *= 2
            dims.append(embed_dim)
            weights.append(w)
            self.matryoshka_dims = np.array(dims, dtype=np.int32)
            self.matryoshka_weights = np.array(weights, dtype=np.float32)
        else:
            self.matryoshka_dims = np.array([embed_dim], dtype=np.int32)
            self.matryoshka_weights = np.array([1.0], dtype=np.float32)

        self.matryoshka_weights = self.matryoshka_weights / self.matryoshka_weights.sum()

        if isinstance(vitdet_config, dict):
            global_attn_indices = vitdet_config.get('global_attn_indices', None)
            if global_attn_indices is None:
                num_global_attn = vitdet_config.get('num_global', 4)
                if num_global_attn <= 0:
                    raise ValueError(f"Either 'global_attn_indices' or 'num_global_attn' must be specified in vitdet_config")
                cycle_length = (depth + num_global_attn - 1) // num_global_attn
                global_attn_indices = list(range(cycle_length - 1, depth, cycle_length))
                # Always end with global attention
                if global_attn_indices[-1] != (depth - 1):
                    global_attn_indices.append(depth - 1)
            vitdet_config = ViTDetConfig(vitdet_config['window_size'], set(global_attn_indices))
        self.vitdet_config = vitdet_config

        # Patch embedding
        self.patch_embed = NaViTPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Optional CLS tokens (one per image)
        if num_prefix_tokens > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, num_prefix_tokens, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            self.register_buffer('prefix_pos', torch.full((num_prefix_tokens, 2), -1.0, dtype=torch.float32), persistent=False)
        else:
            self.cls_token = None

        # Pre-normalization
        self.norm_pre = nn.Identity()  # norm_layer(embed_dim)

        # Shared RoPE instance for all blocks
        head_dim = embed_dim // num_heads
        self.rope = RoPE2D(head_dim, base=rope_base, rope_fraction=rope_fraction)

        if isinstance(grad_checkpointing, bool):
            grad_checkpointing = depth
        self.grad_checkpointing = grad_checkpointing

        def norm_ctor(depth: int):
            extra = dict(eps=1e-6)
            def _fn(*args, **kwargs):
                return norm_layer(*args, **kwargs, **extra)
            return _fn

        # Transformer blocks
        self.blocks = nn.ModuleList([
            NaViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                rope=self.rope,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_ctor(i),
                init_values=init_values,
                grad_checkpointing=i < grad_checkpointing,
                norm_method=norm_method,
            )
            for i in range(depth)
        ])

        self.norm_post = norm_layer(embed_dim, eps=1e-6, elementwise_affine=post_norm_affine)

    def _sample_matryoshka_dim(self) -> int:
        """Sample a random matryoshka dimension during training, or return full dim."""
        if self.matryoshka and self.training:
            ret = np.random.choice(self.matryoshka_dims, p=self.matryoshka_weights)
            return ret
        return self.embed_dim

    def _matryoshka_truncate(self, x: Tensor, dim: int) -> Tensor:
        """Truncate to `dim` dimensions for matryoshka training."""
        if dim < self.embed_dim:
            zeros = torch.zeros_like(x[..., dim:])
            sl = x[..., :dim]
            ret = torch.cat([sl, zeros], dim=-1)
            return ret
        return x

    def forward(
        self,
        x: Union[Tensor, List[Tensor]],
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        """Forward pass.

        Args:
            x: Either a batched tensor (B, C, H, W) for standard ViT mode,
               or a list of tensors [(C, H1, W1), (C, H2, W2), ...] for NaViT mode.

        Returns:
            For NaViT mode (list input): Tuple of (Optional[Tensor], List[Tensor])
            For standard mode (tensor input): Tuple of (summary, features) where summary is (B, embed_dim) or (B, N, embed_dim)
        """
        if isinstance(x, (list, tuple)):
            return self._forward_packed(list(x))
        else:
            # Standard batched forward for uniform-size images
            return self._forward_batched(x)

    def forward_features(
        self,
        x: Union[Tensor, List[Tensor]],
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        """Alias for forward to get features."""
        return self.forward(x)

    def _forward_packed(
        self,
        images: List[Tensor],
    ) -> tuple[Tensor, List[Tensor]]:
        """Forward pass with a list of variable-size images.

        This is the primary interface for NaViT. Images are packed together
        without padding for maximum efficiency.

        Args:
            images: List of image tensors, each with shape (C, H, W).
                    Images can have different sizes but H and W must be
                    divisible by patch_size.

        Returns:
            NaViTOutput with packed features and metadata.
        """
        device = images[0].device
        num_images = len(images)

        x, grid_sizes = self.patch_embed(images)

        re_packed = []
        re_offset = 0

        cu_seqlens, vitdet_cu_seqlens = [0], [torch.tensor(0, dtype=torch.int32, device=device)]
        max_seqlen, vitdet_max_seqlen = 0, self.num_prefix_tokens
        if self.vitdet_config is not None:
            vitdet_full_patch_order = []
            cls_patch_order = torch.arange(0, self.num_prefix_tokens, dtype=torch.int64, device=device)
            t_prefix_len = torch.tensor(self.num_prefix_tokens, dtype=torch.int32, device=device)

        for gs in grid_sizes:
            num_patches = gs[0] * gs[1]
            curr_slice = x[re_offset : re_offset + num_patches]

            if self.num_prefix_tokens > 0:
                assert self.cls_token is not None
                re_packed.append(self.cls_token[0])  # (num_prefix, embed_dim)

                if self.vitdet_config is not None:
                    vitdet_full_patch_order.append(cls_patch_order + vitdet_cu_seqlens[-1])
                    # During the windowed portion, the prefix tokens attend to each other, and not the windows
                    vitdet_cu_seqlens.append(vitdet_cu_seqlens[-1] + t_prefix_len)

            if self.vitdet_config is not None:
                curr_slice, patch_order, wnd_cu_seqlens, max_window_ct = self._reorder_vitdet_packed(curr_slice, gs)
                vitdet_max_seqlen = max(vitdet_max_seqlen, max_window_ct)
                vitdet_full_patch_order.append(patch_order + vitdet_cu_seqlens[-1])
                vitdet_cu_seqlens.extend(vitdet_cu_seqlens[-1] + wnd_cu_seqlens)

            re_packed.append(curr_slice)
            re_offset += num_patches

            max_seqlen = max(max_seqlen, num_patches + self.num_prefix_tokens)
            cu_seqlens.append(cu_seqlens[-1] + num_patches + self.num_prefix_tokens)

        if re_packed:
            x = torch.cat(re_packed, dim=0)

        # Create position IDs
        position_ids = self._create_position_ids(grid_sizes, device)

        if self.vitdet_config is not None:
            vitdet_cu_seqlens_tensor = torch.stack(vitdet_cu_seqlens).int()
            vitdet_full_patch_order = torch.cat(vitdet_full_patch_order)
            # The positions reorder in vitdet mode, so that we don't have to do any data movement between modes
            position_ids = position_ids[vitdet_full_patch_order]

        # Apply pre-normalization
        x = self.norm_pre(x)

        # Prepare attention info for Flash Attention
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

        attn_info = {
            'cu_seqlens': cu_seqlens_tensor,
            'max_seqlen': max_seqlen,
        }
        if self.vitdet_config is not None:
            attn_info_vitdet = {
                'cu_seqlens': vitdet_cu_seqlens_tensor,
                'max_seqlen': vitdet_max_seqlen,
            }

        # Forward through transformer blocks
        for i, block in enumerate(self.blocks):
            if self.vitdet_config is not None and i not in self.vitdet_config.global_attn_indices:
                x = block(x, position_ids=position_ids, attn_info=attn_info_vitdet)
            else:
                x = block(x, position_ids=position_ids, attn_info=attn_info)

        x = self.norm_post(x)
        mat_dim = self._sample_matryoshka_dim()
        x = self._matryoshka_truncate(x, mat_dim)

        # Shuffle tokens back to original order
        if self.vitdet_config is not None:
            y = torch.zeros_like(x)
            x = torch.scatter(y, dim=0, index=vitdet_full_patch_order.reshape(-1, 1).expand(-1, x.shape[1]), src=x)

        cls_tokens_list = []
        features_list = []

        for i in range(num_images):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]

            curr_slice = x[start:end]
            cls_tokens = curr_slice[:self.num_cls_tokens] if self.num_cls_tokens > 0 else None
            features = curr_slice[self.num_prefix_tokens:]

            if cls_tokens is not None:
                cls_tokens_list.append(cls_tokens)
            features_list.append(features)

        if self.pool_type == 'attn':
            assert not cls_tokens_list
            pool_attn_info = {
                'cu_seqlens_kv': cu_seqlens_tensor,
                'max_seqlen_kv': max_seqlen,
            }
            if self.vitdet_config is not None:
                tmp_pos = torch.zeros_like(position_ids)
                position_ids = torch.scatter(tmp_pos, dim=0, index=vitdet_full_patch_order.reshape(-1, 1).expand(-1, position_ids.shape[1]), src=position_ids)

            summary = self.attn_pool(x, mem_position_ids=position_ids, cross_attn_info=pool_attn_info)
        elif self.pool_type == 'cls_token':
            assert cls_tokens_list
            summary = torch.stack(cls_tokens_list, dim=0)  # (num_images, num_cls_tokens, embed_dim)
        else:
            raise ValueError(f'Unsupported pool type: {self.pool_type}')

        return summary, features_list

    def _forward_batched(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for batched uniform-size images."""
        if self.vitdet_config is not None:
            summary, feats = self._forward_packed(list(x))
            return summary, torch.stack(feats, dim=0)

        B, C, H, W = x.shape

        # Embed patches
        patches, grid_size = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Create position IDs (same for all images in batch)
        grid_h, grid_w = grid_size
        rows = torch.linspace(0, grid_h - 1, grid_h, device=x.device, dtype=torch.float32)
        cols = torch.linspace(0, grid_w - 1, grid_w, device=x.device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
        position_ids = torch.stack([row_grid.flatten(), col_grid.flatten()], dim=-1)
        position_ids = position_ids.unsqueeze(0).expand(B, -1, -1)  # (B, num_patches, 2)

        # Add CLS tokens if present
        if self.cls_token is not None:
            assert self.prefix_pos is not None
            cls_tokens = self.cls_token.expand(B, -1, -1)
            patches = torch.cat([cls_tokens, patches], dim=1)
            prefix_pos = self.prefix_pos.unsqueeze(0).expand(B, -1, -1)
            position_ids = torch.cat([prefix_pos, position_ids], dim=1)

        # Apply pre-normalization
        x = self.norm_pre(patches)

        # This allows RoPE to cache stuff
        attn_info = dict()

        # Forward through transformer blocks (batched mode, no attn_info needed)
        for block in self.blocks:
            x = block(x, position_ids=position_ids, attn_info=attn_info)

        x = self.norm_post(x)
        mat_dim = self._sample_matryoshka_dim()
        x = self._matryoshka_truncate(x, mat_dim)

        if self.pool_type == 'attn':
            pool_attn_info = {}
            summary = self.attn_pool(x, mem_position_ids=position_ids, cross_attn_info=pool_attn_info)
        elif self.pool_type == 'cls_token':
            summary = x[:, :self.num_cls_tokens]
        else:
            raise ValueError(f'Unsupported pool type: {self.pool_type}')

        features = x[:, self.num_prefix_tokens:]

        return summary, features

    @torch.no_grad()
    def _create_position_ids(
        self,
        grid_sizes: List[Tuple[int, int]],
        device: torch.device,
    ) -> Tensor:
        """Create 2D position IDs for all patches from multiple images.

        Args:
            grid_sizes: List of (grid_h, grid_w) for each image.
            device: Device for the output tensor.

        Returns:
            (total_patches, 2) tensor of (row, col) positions.
        """
        position_ids = []
        for grid_h, grid_w in grid_sizes:
            if self.num_prefix_tokens > 0:
                assert self.prefix_pos is not None
                # Add position IDs for prefix tokens
                position_ids.append(self.prefix_pos)

            rows = torch.linspace(0, grid_h - 1, grid_h, device=device, dtype=torch.float32)
            cols = torch.linspace(0, grid_w - 1, grid_w, device=device, dtype=torch.float32)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
            pos = torch.stack([row_grid.flatten(), col_grid.flatten()], dim=-1)

            position_ids.append(pos)

        return torch.cat(position_ids, dim=0)

    def _reorder_vitdet_packed(self, tokens: Tensor, grid_size: Tuple[int, int]) -> Tuple[Tensor, Tensor, Tensor, int]:
        """Reorder tokens for ViTDet-style windowed attention.

        Args:
            tokens: (num_patches, embed_dim) tensor of patch tokens.
            grid_size: (grid_h, grid_w) tuple.

        Returns:
            [0] Reordered tokens of shape (num_patches, embed_dim)
            [1] patch_order: (num_patches,) tensor of original indices for each token
            [2] cu_seqlens: Cumulative sequence lengths for each window (for flash attention)
            [3] the maximum number of tokens in any window
        """
        assert self.vitdet_config is not None

        grid_h, grid_w = grid_size
        num_patches = grid_h * grid_w
        assert tokens.shape[0] == num_patches

        window_size = self.vitdet_config.window_size
        assert window_size is not None and window_size > 0

        # Calculate number of windows in each dimension
        num_windows_h = math.ceil(grid_h / window_size)
        num_windows_w = math.ceil(grid_w / window_size)

        # Create a mapping from (row, col) to window index
        row_indices = torch.arange(grid_h, device=tokens.device)
        col_indices = torch.arange(grid_w, device=tokens.device)
        row_grid, col_grid = torch.meshgrid(row_indices, col_indices, indexing='ij')
        window_row_idx = row_grid // window_size
        window_col_idx = col_grid // window_size
        window_idx = window_row_idx * num_windows_w + window_col_idx  # (grid_h, grid_w)

        # Flatten and sort by window index
        flat_window_idx = window_idx.flatten()
        patch_order = torch.argsort(flat_window_idx)

        reordered_tokens = tokens[patch_order]

        seq_lens = torch.bincount(flat_window_idx, minlength=num_windows_h * num_windows_w)
        cu_seqlens = seq_lens.cumsum(dim=0, dtype=torch.int32)

        max_window_ct = min(grid_h, window_size) * min(grid_w, window_size)

        return reordered_tokens, patch_order, cu_seqlens, max_window_ct

###############################################################################
# Initialization helpers
###############################################################################

def _magneto_init(model: NaViT, num_blocks: Optional[int] = None):
    '''
    Initialization following [Magneto](http://arxiv.org/abs/2210.06423)
    '''
    attention_modules = [m for m in model.modules() if isinstance(m, VarlenRoPEAttention)]
    mlp_modules = [m for m in model.modules() if isinstance(m, NaViTMLP)]

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


def _init_layerscale(model: NaViT):
    # https://proceedings.neurips.cc/paper_files/paper/2022/file/ae0cba715b60c4052359b3d52a2cff7f-Paper-Conference.pdf
    for i, block in enumerate(model.blocks):
        ls = 1 / math.sqrt(i + 1)
        assert isinstance(block.ls1, LayerScale) and isinstance(block.ls2, LayerScale)
        block.ls1.gamma.data.fill_(ls)
        block.ls2.gamma.data.fill_(ls)


###############################################################################
# Factory functions
###############################################################################

def create_navit(
    variant: str = 'base',
    pretrained: bool = False,
    **kwargs,
) -> NaViT:
    """Create a NaViT model.

    Args:
        variant: Model variant ('tiny', 'small', 'base', 'large', 'huge').
        pretrained: Whether to load pretrained weights (not yet supported).
        **kwargs: Additional arguments passed to NaViT constructor, including:
            - rope_fraction: Fraction of dimensions to apply RoPE to (p in p-RoPE).
              1.0 = full RoPE (default), 0.75 = 75% RoPE / 25% NoPE.
              See "Round and Round We Go!" paper for details.
            - rope_base: Base frequency for RoPE (default 10000.0).

    Returns:
        NaViT model instance.
    """
    configs = {
        'tiny': dict(embed_dim=192, depth=12, num_heads=3),
        'small': dict(embed_dim=384, depth=12, num_heads=6),
        'base': dict(embed_dim=768, depth=12, num_heads=12),
        'large': dict(embed_dim=1024, depth=24, num_heads=16),
        'so400m': dict(embed_dim=1152, depth=27, num_heads=16),
        'huge': dict(embed_dim=1280, depth=32, num_heads=16),
        'giant': dict(embed_dim=1408, depth=40, num_heads=16),
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(configs.keys())}")

    config = configs[variant]
    config.update(kwargs)

    model = NaViT(**config)  # type: ignore[arg-type]

    if pretrained:
        raise NotImplementedError("Pretrained weights not yet available for NaViT")

    _magneto_init(model)
    if kwargs.get('init_values', None) == -1234:
        _init_layerscale(model)

    return model


# Convenience factory functions
@register_model
def navit_tiny(**kwargs) -> NaViT:
    return create_navit('tiny', **kwargs)

@register_model
def navit_small(**kwargs) -> NaViT:
    return create_navit('small', **kwargs)

@register_model
def navit_base(**kwargs) -> NaViT:
    return create_navit('base', **kwargs)

@register_model
def navit_large(**kwargs) -> NaViT:
    return create_navit('large', **kwargs)

@register_model
def navit_so400m(**kwargs) -> NaViT:
    return create_navit('so400m', **kwargs)

@register_model
def navit_huge(**kwargs) -> NaViT:
    return create_navit('huge', **kwargs)

@register_model
def navit_giant(**kwargs) -> NaViT:
    return create_navit('giant', **kwargs)
