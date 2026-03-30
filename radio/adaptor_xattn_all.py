"""Cross-attention adaptor head with RoPE2D and FlashAttention support — single-file vendorable build.

This module provides XAttnAllHead, a cross-attention based adaptor head that:
- Uses 2D Rotary Position Embeddings (RoPE2D)
- Supports variable-length sequences via FlashAttention
- Projects features from student to teacher resolution

All local layer dependencies (MLPBase, VarlenRoPEXBlock) are inlined.
Shared modules (RoPE2D, VarlenRoPEAttention, VarlenRoPEXAttn, LayerScale,
is_flash_attn_available) are imported from vision_transformer_navit_single.
"""

from logging import getLogger
import math
from typing import Any, Callable, Dict, Optional, Union, List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from einops import rearrange

from .vision_transformer_navit import (
    RoPE2D,
    VarlenRoPEAttention,
    VarlenRoPEXAttn,
    LayerScale,
    is_flash_attn_available,
)
from .adaptor_base import AdaptorModuleBase

_LOGGER = getLogger(__name__)
_DEBUG = False


BASE_GRID_CACHE = dict()
def generate_homography_grid(homography: torch.Tensor, size):
    global BASE_GRID_CACHE

    N, (H, W) = size[0], size[-2:]
    if size not in BASE_GRID_CACHE:
        base_grid = homography.new(N, H, W, 3)
        linear_points = torch.linspace(-1, 1, W, device=base_grid.device) if W > 1 else torch.Tensor([-1], device=base_grid.device)
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H, device=base_grid.device), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H, device=base_grid.device) if H > 1 else torch.Tensor([-1], device=base_grid.device)
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W, device=base_grid.device)).expand_as(base_grid[:, :, :, 1])
        base_grid[:, :, :, 2] = 1
        BASE_GRID_CACHE[size] = base_grid
    else:
        base_grid = BASE_GRID_CACHE[size]

    grid = torch.bmm(base_grid.view(N, H * W, 3), homography.transpose(1, 2))
    grid = grid.view(N, H, W, 3)
    grid[:, :, :, 0] = grid[:, :, :, 0] / grid[:, :, :, 2]
    grid[:, :, :, 1] = grid[:, :, :, 1] / grid[:, :, :, 2]
    grid = grid[:, :, :, :2].float()
    return grid


def bulk_process(fn: Callable, *args, num_test: int = 1):
    if any(isinstance(args[i], (list, tuple)) for i in range(min(num_test, len(args)))):
        argl = zip(*args)
        ret = [fn(*a) for a in argl]
    else:
        ret = fn(*args)
    return ret


###############################################################################
# VarlenRoPEXBlock — inlined from models/layers/varlen_rope_xblock.py
###############################################################################

class VarlenRoPEXBlock(nn.Module):
    """Transformer decoder block with self-attention, cross-attention, and MLP.

    Supports both variable-length sequences (via FlashAttention) and batched operation.

    The block performs:
    1. Self-attention on target with RoPE
    2. Cross-attention from target to memory with RoPE
    3. MLP

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to input dim.
        qkv_bias: Whether to use bias in attention projections.
        proj_drop: Projection dropout rate.
        self_attn_rope: RoPE2D module for self-attention.
        cross_attn_rope: RoPE2D module for cross-attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        self_attn_rope: RoPE2D,
        cross_attn_rope: RoPE2D,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        init_values: Optional[float] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        mlp_dim = int(dim * mlp_ratio)

        # Self-attention on target (uses VarlenRoPEAttention)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = VarlenRoPEAttention(
            dim=dim,
            num_heads=num_heads,
            rope=self_attn_rope,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
        )

        # Cross-attention
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.cross_attn_mem_norm = nn.LayerNorm(dim)
        self.cross_attn = VarlenRoPEXAttn(
            dim=dim,
            num_heads=num_heads,
            rope=cross_attn_rope,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
        )

        # MLP
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        self.self_attn_ls = LayerScale(dim, init_values) if init_values is not None else nn.Identity()
        self.cross_attn_ls = LayerScale(dim, init_values) if init_values is not None else nn.Identity()
        self.mlp_ls = LayerScale(dim, init_values) if init_values is not None else nn.Identity()

        self.dropout = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_position_ids: Tensor,
        mem_position_ids: Tensor,
        self_attn_info: Dict[str, Any],
        cross_attn_info: Dict[str, Any],
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Self-attention on target
        tgt_normed = self.self_attn_norm(tgt)
        self_attn_out = self.self_attn(tgt_normed, position_ids=tgt_position_ids, attn_info=self_attn_info)
        tgt = tgt + self.dropout(self.self_attn_ls(self_attn_out))

        # Cross-attention
        tgt_normed = self.cross_attn_norm(tgt)
        mem_normed = self.cross_attn_mem_norm(memory)

        cross_attn_out = self.cross_attn(
            tgt=tgt_normed,
            memory=mem_normed,
            tgt_position_ids=tgt_position_ids,
            mem_position_ids=mem_position_ids,
            attn_info=cross_attn_info,
        )
        tgt = tgt + self.dropout(self.cross_attn_ls(cross_attn_out))

        # MLP
        tgt = tgt + self.dropout(self.mlp_ls(self.mlp(self.mlp_norm(tgt))))

        return tgt


###############################################################################
# XAttnAllAdaptor
###############################################################################

class XAttnAllAdaptor(AdaptorModuleBase):
    """Cross-attention head that uses RoPE2D and supports variable-length sequences with FlashAttention."""
    layers: nn.ModuleList  # List of VarlenRoPEXBlock

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        teacher_input_size: Optional[Tuple[int, int]] = None,
        teacher_patch_size: Optional[int] = None,
        num_inner: int = 0,
        pre_norm: bool = False,
        device: Optional[torch.device] = None,
        upsample_factor: int = 1,
        upsample_rank: int = 0,
        num_heads: int = 16,
        num_layers: int = 4,
        rope_base: float = 10000.0,
        rope_fraction: float = 0.75,
        feature_dim: int | None = None,
        init_values: Optional[float] = -1234,
        student_depth: int | None = None,
        **kwargs  # Ignore kwargs that might be to other "mlp" versions, e.g. teacher_summary_idxs
    ) -> None:
        super().__init__(requires_summary_and_spatial=True, handles_summary_and_spatial=True)

        if teacher_input_size is not None and teacher_patch_size is not None:
            t_patches: Optional[Tuple[int, int]] = (teacher_input_size[0] // teacher_patch_size, teacher_input_size[1] // teacher_patch_size)
        else:
            t_patches = None

        self.t_patches: Optional[Tuple[int, int]] = t_patches
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        self.scale = self.head_dim ** -0.5

        feature_dim = feature_dim or output_size

        # Separate RoPE2D instances for self-attention and cross-attention (for caching)
        self.self_attn_rope = RoPE2D(self.head_dim, base=rope_base, rope_fraction=rope_fraction)
        self.cross_attn_rope = RoPE2D(self.head_dim, base=rope_base, rope_fraction=rope_fraction)

        # Cross-attention layers using shared RoPE instances
        self.layers = nn.ModuleList([
            VarlenRoPEXBlock(
                dim=input_size,
                num_heads=num_heads,
                self_attn_rope=self.self_attn_rope,
                cross_attn_rope=self.cross_attn_rope,
                mlp_ratio=4.0,
                qkv_bias=True,
                proj_drop=0.0,
                init_values=init_values,
            )
            for _ in range(num_layers)
        ])

        self.summary_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, feature_dim),
        )

        self.homography_map = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size),
        )

        self._init_weights(student_depth)
        if init_values == -1234:
            self._init_layerscale(student_depth)

    def _init_weights(self, student_depth: int | None):
        '''
        Initialization following [Magneto](http://arxiv.org/abs/2210.06423)
        '''
        attention_modules = [m for m in self.modules() if isinstance(m, VarlenRoPEAttention)]
        xattn_modules = [m for m in self.modules() if isinstance(m, VarlenRoPEXAttn)]
        mlp_modules: List[nn.Sequential] = [m for n, m in self.named_modules() if n.endswith('.mlp')]

        encode_depth = student_depth or 32  # Some default
        decode_depth = len(self.layers)

        gamma = math.sqrt(math.log(3 * decode_depth))

        for m in attention_modules:
            qkv = m.qkv
            q, k, v = qkv.weight.data.chunk(3, dim=0)
            xavier_normal_(q, gain=1)
            xavier_normal_(k, gain=1)
            xavier_normal_(v, gain=gamma)
            xavier_normal_(m.proj.weight.data, gain=gamma)

        for m in xattn_modules:
            xavier_normal_(m.q_proj.weight.data, gain=1)
            xavier_normal_(m.k_proj.weight.data, gain=1)
            xavier_normal_(m.v_proj.weight.data, gain=gamma)
            xavier_normal_(m.out_proj.weight.data, gain=gamma)

        for m in mlp_modules:
            xavier_normal_(m[0].weight.data, gain=gamma)
            xavier_normal_(m[2].weight.data, gain=gamma)

    def _init_layerscale(self, student_depth: int | None):
        student_depth = student_depth or 32

        # https://proceedings.neurips.cc/paper_files/paper/2022/file/ae0cba715b60c4052359b3d52a2cff7f-Paper-Conference.pdf
        for i, block in enumerate(self.layers):
            ls = 1 / math.sqrt(student_depth + i + 1)
            assert isinstance(block.self_attn_ls, LayerScale) and isinstance(block.cross_attn_ls, LayerScale) and isinstance(block.mlp_ls, LayerScale)
            block.self_attn_ls.gamma.data.fill_(ls)
            block.cross_attn_ls.gamma.data.fill_(ls)
            block.mlp_ls.gamma.data.fill_(ls)

    def forward(
        self,
        summary: torch.Tensor,
        spatial: Union[torch.Tensor, List[torch.Tensor]],
        homography: Optional[torch.Tensor] = None,
        grid_sizes: Optional[List[Tuple[int, int]]] = None,
        target_images: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, List[torch.Tensor]]]:
        tgt, tgt_position_ids = self._create_tgt_buffer(summary, spatial, homography, grid_sizes, target_images=target_images)

        if isinstance(spatial, list):
            return self.forward_list(spatial, tgt, tgt_position_ids, student_grid_sizes=grid_sizes, **kwargs)
        return self.forward_batched(spatial, tgt, tgt_position_ids, student_grid_sizes=grid_sizes, **kwargs)

    def forward_list(
        self,
        spatial: List[torch.Tensor],
        tgt: torch.Tensor,
        tgt_position_ids: torch.Tensor,
        student_grid_sizes: Optional[List[Tuple[int, int]]],
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with variable-length spatial sequences using FlashAttention.

        Args:
            spatial: List of (N_i, D) spatial features for each image
            tgt: (B, N, D) target tokens for cross-attention
            tgt_position_ids: (B, N, 2) position ids for target tokens
            student_grid_sizes: Optional list of (H, W) grid sizes for each image.

        Returns:
            Tuple of (summary, features) where summary is (B, D) and
            features is a (B, total_patches, D) tensor.
        """
        assert is_flash_attn_available(), "FlashAttention is required for forward_list"
        assert student_grid_sizes is not None, "student_grid_sizes must be provided for forward_list"

        device = tgt.device
        batch_size = len(spatial)

        # Build memory (key/value): concatenate summary + spatial for each image
        memory_list = []
        memory_cu_seqlens = [0]
        memory_max_seqlen = 0
        for i in range(batch_size):
            memory_list.append(spatial[i])

            seqlen = spatial[i].shape[0]
            memory_cu_seqlens.append(memory_cu_seqlens[-1] + seqlen)
            memory_max_seqlen = max(memory_max_seqlen, seqlen)

        memory = torch.cat(memory_list, dim=0)  # (total_memory, D)

        mem_position_ids = self._create_position_ids(
            student_grid_sizes,
            num_prefix=0,
            device=device,
            dtype=torch.float32,
        )

        tgt_seq_len = tgt.shape[1]  # homography_token + summary + spatial patches
        tgt = tgt.flatten(0, 1)

        tgt_cu_seqlens = [0]
        for i in range(batch_size):
            tgt_cu_seqlens.append(tgt_cu_seqlens[-1] + tgt_seq_len)

        tgt_position_ids = tgt_position_ids.flatten(0, 1)

        device_memory_cu_seqlens = torch.tensor(memory_cu_seqlens, device=device, dtype=torch.int32)
        device_tgt_cu_seqlens = torch.tensor(tgt_cu_seqlens, device=device, dtype=torch.int32)

        cross_attn_info = {
            'cu_seqlens_q': device_tgt_cu_seqlens,
            'cu_seqlens_kv': device_memory_cu_seqlens,
            'max_seqlen_q': tgt_seq_len,
            'max_seqlen_kv': memory_max_seqlen,
        }
        self_attn_info = {
            'cu_seqlens': device_tgt_cu_seqlens,
            'max_seqlen': tgt_seq_len,
        }

        for layer in self.layers:  # type: ignore[assignment]
            tgt = layer(
                tgt=tgt,
                memory=memory,
                tgt_position_ids=tgt_position_ids,
                mem_position_ids=mem_position_ids,
                self_attn_info=self_attn_info,
                cross_attn_info=cross_attn_info,
            )

        tgt = tgt.reshape(batch_size, tgt_seq_len, -1)

        # The homography token is at index 0, skip it
        summary = self.summary_proj(tgt[:, 1])
        features = self.feature_proj(tgt[:, 2:])

        return summary, features

    def forward_batched(
        self,
        spatial: torch.Tensor,
        tgt: torch.Tensor,
        tgt_position_ids: torch.Tensor,
        student_grid_sizes: Optional[List[Tuple[int, int]]],
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for batched inputs with uniform sequence lengths.

        Args:
            spatial: (B, N, D) spatial features
            tgt: (B, N, D) target tokens for cross-attention
            tgt_position_ids: (B, N, 2) position ids for target tokens
            student_grid_sizes: Optional list of (H, W) grid sizes for each image.
                               If None, assumes square grid.
            memory_key_padding_mask: Optional (B, 1+N) mask for memory padding

        Returns:
            Tuple of (summary, features) where summary is (B, D) and
            features is (B, H*W, D).
        """
        B = tgt.shape[0]
        device = tgt.device
        N = spatial.shape[1]

        memory = spatial  # (B, N, D)

        # Infer student grid sizes if not provided
        if student_grid_sizes is None:
            side = int(N ** 0.5)
            if side * side != N:
                raise ValueError("Spatial features must form a square grid if student_grid_sizes is not provided.")
            student_grid_sizes = [(side, side) for _ in range(B)]
        assert all(student_grid_sizes[0] == sgs for sgs in student_grid_sizes), "All student grid sizes must be the same for forward_batched, got: {}".format(student_grid_sizes)

        # Create memory position IDs
        mem_position_ids = self._create_position_ids(
            [student_grid_sizes[0]],
            num_prefix=0,
            device=device,
            dtype=torch.float32,
        )
        # Expand for batch dimension
        mem_position_ids = mem_position_ids.unsqueeze(0).expand(B, -1, -1)

        self_attn_info, cross_attn_info = dict(), dict()

        # Forward through layers
        for layer in self.layers:  # type: ignore[assignment]
            tgt = layer(
                tgt=tgt,
                memory=memory,
                tgt_position_ids=tgt_position_ids,
                mem_position_ids=mem_position_ids,
                self_attn_info=self_attn_info,
                cross_attn_info=cross_attn_info,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # The homography token is at index 0, skip it
        summary_out = self.summary_proj(tgt[:, 1])
        features = self.feature_proj(tgt[:, 2:])

        return summary_out, features

    @torch.no_grad()
    def _create_position_ids(
        self,
        grid_sizes: List[Tuple[int, ...]],
        num_prefix: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Create 2D position IDs for target patches.

        Args:
            grid_sizes: List of (grid_h, grid_w) for each image.
            device: Device for the output tensor.

        Returns:
            (total_patches, 2) tensor of (row, col) positions.
        """
        position_ids = []

        if num_prefix > 0:
            prefix_pos = torch.full((num_prefix, 2), -1.0, device=device, dtype=dtype)

        for grid_h, grid_w in grid_sizes:
            if num_prefix > 0:
                position_ids.append(prefix_pos)

            rows = torch.linspace(0, grid_h - 1, grid_h, device=device, dtype=torch.float32)
            cols = torch.linspace(0, grid_w - 1, grid_w, device=device, dtype=torch.float32)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
            pos = torch.stack([row_grid.flatten(), col_grid.flatten()], dim=-1)
            position_ids.append(pos)

        return torch.cat(position_ids, dim=0)

    def _create_tgt_buffer(
        self,
        summary: Tensor,
        spatial: Union[Tensor, List[Tensor]],
        homography: Optional[Tensor],
        grid_sizes: Optional[List[Tuple[int, int]]],
        target_images: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Create target buffer for decoder input.

        Args:
            spatial: Either (B, N, D) tensor or list of (N_i, D) tensors.
            homography: (B, D) homography conditioning tokens.
        Returns:
            (B, seq_len, D) target buffer tensor.
        """
        global _DEBUG

        if grid_sizes is None:
            if isinstance(spatial, list):
                raise ValueError("grid_sizes must be provided when spatial is a list.")
            side = int(spatial.shape[1] ** 0.5)
            if side * side != spatial.shape[1]:
                raise ValueError("Spatial features must form a square grid if grid_sizes is not provided.")
            grid_sizes = [(side, side)]

        if homography is None:
            homography = torch.eye(3, device='cuda').unsqueeze(0).expand(len(spatial), -1, -1)

        def process(feats: Tensor, grid_i: Tensor, f_size: Tuple[int, int]) -> Tensor:
            if feats.ndim == 2:
                feats = rearrange(feats, '(h w) d -> 1 d h w', h=f_size[0], w=f_size[1])
                grid_i = grid_i.unsqueeze(0)
                squeeze = True
            else:
                feats = rearrange(feats, 'b (h w) d -> b d h w', h=f_size[0][0], w=f_size[0][1])
                squeeze = False

            tgt = F.grid_sample(feats, grid_i, align_corners=True, mode='bilinear')
            tgt = rearrange(tgt, 'b d h w -> b (h w) d')
            if squeeze:
                tgt = tgt.squeeze(0)
            return tgt

        t_patches = self.t_patches if self.t_patches is not None else grid_sizes[0]

        grid = generate_homography_grid(homography, (len(spatial), 1) + t_patches)  # (B, H, W, <xy>)

        # Range [0, 1]
        norm_grid = grid.add(1).div_(2.0)

        sizes_tensor = torch.tensor(grid_sizes, device=norm_grid.device, dtype=norm_grid.dtype)#.sub_(1)
        # sizes_tensor is (H, W) = (y, x); flip to (W, H) = (x, y) to match norm_grid's (x, y) format
        sizes_tensor = sizes_tensor.flip(-1)
        if sizes_tensor.ndim == 1:
            sizes_tensor = sizes_tensor.unsqueeze(0).expand(len(spatial), -1)
        orig_pos_ids = norm_grid * sizes_tensor[:, None, None, :]
        # orig_pos_ids is in (x, y) format, and we need (y, x)
        orig_pos_ids = torch.flip(orig_pos_ids, dims=[-1])

        prefix_pos_ids = torch.full((len(spatial), 2, 2), -1.0, device=orig_pos_ids.device, dtype=orig_pos_ids.dtype)

        tgt_position_ids = torch.cat([
            prefix_pos_ids,
            orig_pos_ids.flatten(1, 2),
        ], dim=1)

        tgt: Tensor = bulk_process(process, spatial, grid, grid_sizes)
        if isinstance(tgt, list):
            tgt = torch.stack(tgt, dim=0)

        homography_token = self.homography_map(homography.flatten(1))

        tgt = torch.cat([
            homography_token.unsqueeze(1),  # (B, 1, D)
            summary.unsqueeze(1),  # (B, 1, D)
            tgt,  # (B, N, D)
        ], dim=1)

        return tgt, tgt_position_ids
