import math
from typing import Final, List, Optional, Tuple, Union


from einops import rearrange
from timm.models import register_model
import torch
from torch import Type, nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_, zeros_

from .forward_intermediates import forward_intermediates


def _get_init_scale(num_encoder_layers: int, num_decoder_layers: int, is_encoder: bool):
    if num_encoder_layers > 0 and num_decoder_layers == 0:
        return math.sqrt(math.log(2 * num_encoder_layers))
    if num_decoder_layers > 0 and num_encoder_layers == 0:
        return math.sqrt(math.log(2 * num_decoder_layers))
    if is_encoder:
        # Both encoders and decoders
        return math.sqrt(
            0.33 * math.log(3 * num_decoder_layers) * math.log(2 * num_encoder_layers)
        )

    return math.sqrt(math.log(3 * num_decoder_layers))


# [1,2]    [1,1,2,2]
# [3,4] -> [3,3,4,4]
# [5,6]    [5,5,6,6]
def duplicate_interleave(m):
    return m.view(-1, 1).repeat(1, 2).view(m.shape[0], -1)

# 0,1,2,3,4,5,6,7 -> -1,0,-3,2,-5,4,-7,6
def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


class XPosEmbedding2D(torch.nn.Module):
    """Implementation of xPos based on RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """

    def __init__(
        self,
        head_dim: int,
        base=50000,
        scale_base=512
    ):
        super().__init__()
        half_dim = head_dim // 2
        self.half_dim = half_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.token_shape_cached = None
        self.batch_size_cached = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None
        self.scale_cached: torch.Tensor | None = None
        self.scale_base = scale_base
        self.register_buffer("scale",
                             (torch.arange(0, half_dim, 2) + 0.4 * half_dim) / (1.4 * half_dim))

    def cos_sin(
        self,
        token_shape: Tuple[int, int],
        device="cuda",
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if token_shape != self.token_shape_cached:
            self.token_shape_cached = token_shape
            y = torch.arange(token_shape[0], device=device, dtype=self.inv_freq.dtype)
            x = torch.arange(token_shape[1], device=device, dtype=self.inv_freq.dtype)
            x, y = torch.meshgrid(x, y, indexing='xy')

            y_freqs = torch.einsum("i,j->ij", y.flatten(), self.inv_freq)
            x_freqs = torch.einsum("i,j->ij", x.flatten(), self.inv_freq)

            y_scales = self.scale ** y.flatten().div(self.scale_base)[:, None]
            x_scales = self.scale ** x.flatten().div(self.scale_base)[:, None]

            freqs = torch.cat([y_freqs, x_freqs], dim=-1)
            emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

            scales = torch.cat([y_scales, x_scales], dim=-1)
            scales = torch.repeat_interleave(scales, repeats=2, dim=-1)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]
            self.scale_cached = scales[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)
            self.scale_cached = self.scale_cached.type(dtype)

        return self.cos_cached, self.sin_cached, self.scale_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, token_shape: Tuple[int, int]):
        batch, seq_len, head_dim = q.shape
        cos, sin, scale = self.cos_sin(token_shape, q.device, q.dtype)
        # scale = self.scale**torch.arange(seq_len).to(self.scale).div(self.scale_base)[:, None]
        # scale = torch.repeat_interleave(scale, 2, dim=-1).to(q.device)
        # scale = torch.cat([scale, scale], dim=-1)
        # scale = 1
        return (
            (q * cos * scale) + (rotate_every_two(q) * sin * scale),
            (k * cos * (1 / scale)) + (rotate_every_two(k) * sin * (1 / scale)),
        )


class MagnetoAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, pos_emb: XPosEmbedding2D):
        super().__init__()
        self.num_heads = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.pos_emb = pos_emb

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, num_prefix_tokens: int, patch_shape: Tuple[int, int]) -> torch.Tensor:
        B, N, C = x.shape
        x = self.norm0(x)

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        q_pref = q[:, :num_prefix_tokens]
        q_patch = q[:, num_prefix_tokens:]

        k_pref = k[:, :num_prefix_tokens]
        k_patch = k[:, num_prefix_tokens:]

        q_patch, k_patch = self.pos_emb(q_patch, k_patch, patch_shape)

        q = torch.cat([q_pref, q_patch], dim=1)
        k = torch.cat([k_pref, k_patch], dim=1)

        def head_reshape(t: torch.Tensor):
            return t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = head_reshape(q)
        k = head_reshape(k)
        v = head_reshape(v)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm1(x)
        x = self.proj(x)
        return x

    def _reset_parameters(self):
        xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            zeros_(self.qkv.bias)
        xavier_normal_(self.proj.weight)
        zeros_(self.proj.bias)


class MagnetoTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, pos_emb: XPosEmbedding2D,
                 num_encoder_layers: int, num_decoder_layers: int = 0,
                 dim_mhsa: int = 0,
                 dim_feedforward: int = 2048,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True):
        super().__init__()

        if dim_mhsa == 0:
            dim_mhsa = d_model

        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers

        self.attn = MagnetoAttention(d_model, nhead, pos_emb)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear2 = nn.Linear(d_model, dim_feedforward)
        self.norm3 = nn.LayerNorm(dim_feedforward, eps=layer_norm_eps)
        self.linear3 = nn.Linear(dim_feedforward, d_model)

    def initialize(self):
        gamma = _get_init_scale(self._num_encoder_layers, self._num_decoder_layers, is_encoder=True)

        # Magneto Initialization
        for mod in self.children():
            if isinstance(mod, nn.Linear):
                xavier_normal_(mod.weight.data, gamma)
            elif isinstance(mod, MagnetoAttention):
                mod._reset_parameters()

    def forward(self, x: torch.Tensor, num_prefix_tokens: int, patch_shape: Tuple[int, int]) -> torch.Tensor:
        x = x + self._sa_block(x, num_prefix_tokens, patch_shape)
        x = x + self._ff_block(x)
        return x

    def _sa_block(self, x: torch.Tensor, num_prefix_tokens: int, patch_shape: Tuple[int, int]) -> torch.Tensor:
        x = self.attn(x, num_prefix_tokens, patch_shape)
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm2(x)
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.norm3(x)
        x = self.linear3(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            num_cls_tokens: int = 1,
            num_reg_tokens: int = 0,
    ) -> None:
        """
        Args:
            patch_size: Patch size.
            in_chans: Number of image input channels.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            num_cls_tokens: Number of cls tokens
            num_reg_tokens: Number of register tokens.
            block_fn: Transformer block layer.
        """
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_cls_tokens = num_cls_tokens
        self.num_reg_tokens = num_reg_tokens

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.prefix_buffer = nn.Parameter(torch.randn(1, self.num_prefix_tokens, embed_dim) * .02)

        pos_emb = XPosEmbedding2D(embed_dim)

        self.blocks = nn.ModuleList([
            MagnetoTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                num_encoder_layers=depth,
                num_decoder_layers=0,
                dim_feedforward=int(embed_dim * mlp_ratio),
                pos_emb=pos_emb,
            )
            for _ in range(depth)
        ])

        for block in self.blocks:
            block.initialize()

    @property
    def num_prefix_tokens(self):
        return self.num_cls_tokens + self.num_reg_tokens

    @property
    def num_summary_tokens(self):
        return self.num_prefix_tokens

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, patch_shape = self._patchify(x)

        for block in self.blocks:
            x = block(x, self.num_prefix_tokens, patch_shape)

        summary = x[:, :self.num_cls_tokens]
        features = x[:, self.num_prefix_tokens:]

        return summary, features

    def forward_intermediates(self, x: torch.Tensor, norm: bool = False, **kwargs):
        patch_shape = tuple(d // self.patch_size for d in x.shape[-2:])

        def patch_extractor(x: torch.Tensor):
            x, _ = self._patchify(x)
            return x

        return forward_intermediates(
            self,
            patch_extractor=patch_extractor,
            num_summary_tokens=self.num_prefix_tokens,
            num_cls_tokens=self.num_cls_tokens,
            norm=lambda y: y,
            x=x,
            block_kwargs=dict(num_prefix_tokens=self.num_prefix_tokens, patch_shape=patch_shape),
            **kwargs,
        )

    def _patchify(self, x: torch.Tensor):
        x = self.patch_embed(x)
        patch_shape = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')

        prefix = self.prefix_buffer.expand(x.shape[0], -1, -1)

        x = torch.cat([prefix, x], dim=1)
        return x, patch_shape


@register_model
def vit_base_patch16_xpos(num_cls_tokens: int = 1, num_reg_tokens: int = 0, **kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                             num_cls_tokens=num_cls_tokens, num_reg_tokens=num_reg_tokens)


@register_model
def vit_large_patch16_xpos(num_cls_tokens: int = 1, num_reg_tokens: int = 0, **kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                             num_cls_tokens=num_cls_tokens, num_reg_tokens=num_reg_tokens)


@register_model
def vit_huge_patch16_xpos(num_cls_tokens: int = 1, num_reg_tokens: int = 0, **kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16,
                             num_cls_tokens=num_cls_tokens, num_reg_tokens=num_reg_tokens)


@register_model
def vit_giant_patch16_xpos(num_cls_tokens: int = 1, num_reg_tokens: int = 0, **kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=16, embed_dim=1408, depth=40, num_heads=16,
                             num_cls_tokens=num_cls_tokens, num_reg_tokens=num_reg_tokens)


@register_model
def vit_bigG_patch16_xpos(num_cls_tokens: int = 1, num_reg_tokens: int = 0, **kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=16, embed_dim=1664, depth=48, num_heads=16,
                             num_cls_tokens=num_cls_tokens, num_reg_tokens=num_reg_tokens)