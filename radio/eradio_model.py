#!/usr/bin/env python3

# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Created by Pavlo Molchanov, LPR - DL Efficiency Research team
# based on Fastervit1 from LPR

import torch
import torch.nn as nn
from timm.models.registry import register_model

from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
import numpy as np
import torch.nn.functional as F
from .block import C2f

TRT = False  # should help for TRT

import pickle

global bias_indx
bias_indx = -1
DEBUG = False


def pixel_unshuffle(data, factor=2):
    # performs nn.PixelShuffle(factor) in reverse, torch has some bug for ONNX and TRT, so doing it manually
    B, C, H, W = data.shape
    return (
        data.view(B, C, factor, H // factor, factor, W // factor)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, -1, H // factor, W // factor)
    )


class SwiGLU(nn.Module):
    # should be more advanced, but doesnt improve results so far
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
    Returns:
        windows - local window features (num_windows*B, window_size*window_size, C)
        (Hp, Wp) -  the size of the padded image
    """
    B, C, H, W = x.shape

    if window_size == 0 or (window_size == H and window_size == W):
        windows = x.flatten(2).transpose(1, 2)
        Hp, Wp = H, W
    else:
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0, 0, 0))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
        windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)

    return windows, (Hp, Wp)


class Conv2d_BN(nn.Module):
    """
    Conv2d + BN layer with folding capability to speed up inference
    """

    def __init__(
        self,
        a,
        b,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        bias=False,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            a, b, kernel_size, stride, padding, dilation, groups, bias=False
        )
        if 1:
            self.bn = torch.nn.BatchNorm2d(b)
            torch.nn.init.constant_(self.bn.weight, bn_weight_init)
            torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):

        # return 1
        if not isinstance(self.bn, nn.Identity):
            c, bn = self.conv, self.bn
            w = bn.weight / (bn.running_var + bn.eps) ** 0.5
            w = c.weight * w[:, None, None, None]
            b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
            self.conv.weight.data.copy_(w)
            self.conv.bias = nn.Parameter(b)
            self.bn = nn.Identity()


def window_reverse(windows, window_size, H, W, pad_hw):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
        pad_w - a tuple of image passing used in windowing step
    Returns:
        x: (B, C, H, W)

    """
    # print(f"window_reverse, windows.shape {windows.shape}")
    Hp, Wp = pad_hw
    if window_size == 0 or (window_size == H and window_size == W):
        B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
        x = windows.transpose(1, 2).view(B, -1, H, W)
    else:
        B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
        x = windows.view(
            B, Hp // window_size, Wp // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], Hp, Wp)

        if Hp > H or Wp > W:
            x = x[:, :, :H, :W,].contiguous()

    return x


class PosEmbMLPSwinv2D(nn.Module):
    def __init__(
        self, window_size, pretrained_window_size, num_heads, seq_length, no_log=False
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / np.log2(8)
            )

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.grid_exists = False

        self.deploy = False

        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.seq_length = seq_length
        self.register_buffer("relative_bias", relative_bias)  # for EMA

    def switch_to_deploy(self):
        self.deploy = True
        self.grid_exists = True

    def forward(self, input_tensor):
        # for efficiency, we want this forward to be folded into a single operation (sum)
        # if resolution stays the same, then we dont need to recompute MLP layers
        #
        # to dynamically adjust patch size over the step
        # if not (input_tensor.shape[1:] == self.relative_bias.shape[1:]):
        #     self.grid_exists = False

        if self.training:
            self.grid_exists = False

        if self.deploy and self.grid_exists:
            input_tensor += self.relative_bias
            return input_tensor

        if not self.grid_exists:
            self.grid_exists = True

            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table
            ).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH

            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

            self.relative_bias = relative_position_bias.unsqueeze(0)

        input_tensor += self.relative_bias
        return input_tensor


class GRAAttentionBlock(nn.Module):
    def __init__(
        self,
        window_size,
        dim_in,
        dim_out,
        num_heads,
        drop_path=0.0,
        qk_scale=None,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        use_swiglu=True,
        subsample_ratio=1,
        dim_ratio=1,
        conv_base=False,
        do_windowing=True,
        multi_query=False,
    ) -> None:
        super().__init__()

        dim = dim_in
        # conv_base = True
        SHUFFLE = True
        SHUFFLE = False
        self.do_windowing = do_windowing

        if do_windowing:
            if SHUFFLE:
                self.downsample_op = (
                    torch.nn.PixelUnshuffle(subsample_ratio)
                    if subsample_ratio > 1
                    else torch.nn.Identity()
                )
                self.downsample_mixer = (
                    nn.Conv2d(
                        dim_in * (subsample_ratio * subsample_ratio),
                        dim_in * (dim_ratio),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                    if dim * dim_ratio != dim * subsample_ratio * subsample_ratio
                    else torch.nn.Identity()
                )
            else:
                if conv_base:
                    self.downsample_op = (
                        nn.Conv2d(
                            dim_in,
                            dim_out,
                            kernel_size=subsample_ratio,
                            stride=subsample_ratio,
                        )
                        if subsample_ratio > 1
                        else nn.Identity()
                    )
                    self.downsample_mixer = nn.Identity()
                else:
                    self.downsample_op = (
                        nn.AvgPool2d(
                            kernel_size=subsample_ratio, stride=subsample_ratio
                        )
                        if subsample_ratio > 1
                        else nn.Identity()
                    )
                    self.downsample_mixer = (
                        Conv2d_BN(dim_in, dim_out, kernel_size=1, stride=1)
                        if subsample_ratio > 1
                        else nn.Identity()
                    )

        if do_windowing:
            if SHUFFLE:
                self.upsample_mixer = (
                    nn.Conv2d(
                        dim_in * dim_ratio,
                        dim_in * (subsample_ratio * subsample_ratio),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                    if dim * dim_ratio != dim * subsample_ratio * subsample_ratio
                    else torch.nn.Identity()
                )
                self.upsample_op = (
                    torch.nn.PixelShuffle(subsample_ratio)
                    if subsample_ratio > 1
                    else torch.nn.Identity()
                )
            else:
                if conv_base:
                    self.upsample_mixer = nn.Identity()
                    self.upsample_op = (
                        nn.ConvTranspose2d(
                            dim_in,
                            dim_out,
                            kernel_size=subsample_ratio,
                            stride=subsample_ratio,
                        )
                        if subsample_ratio > 1
                        else nn.Identity()
                    )
                else:
                    self.upsample_mixer = (
                        nn.Upsample(scale_factor=subsample_ratio, mode="nearest")
                        if subsample_ratio > 1
                        else nn.Identity()
                    )
                    self.upsample_op = (
                        Conv2d_BN(
                            dim_in,
                            dim_out,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        )
                        if subsample_ratio > 1
                        else nn.Identity()
                    )

        self.window_size = window_size

        self.norm1 = norm_layer(dim_in)
        if DEBUG:
            print(
                f"GRAAttentionBlock: input_resolution: , window_size: {window_size}, dim_in: {dim_in}, dim_out: {dim_out}, num_heads: {num_heads}, drop_path: {drop_path}, qk_scale: {qk_scale}, qkv_bias: {qkv_bias}, layer_scale: {layer_scale}"
            )

        self.attn = WindowAttention(
            dim_in,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            resolution=window_size,
            seq_length=window_size ** 2,
            dim_out=dim_in,
            multi_query=multi_query,
        )
        if DEBUG:
            print(
                f"Attention: dim_in: {dim_in}, num_heads: {num_heads}, qkv_bias: {qkv_bias}, qk_scale: {qk_scale}, resolution: {window_size}, seq_length: {window_size**2}, dim_out: {dim_in}"
            )
            print(f"drop_path: {drop_path}, layer_scale: {layer_scale}")

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma1 = (
            nn.Parameter(layer_scale * torch.ones(dim_in)) if use_layer_scale else 1
        )

        ### mlp layer
        mlp_ratio = 4
        self.norm2 = norm_layer(dim_in)
        mlp_hidden_dim = int(dim_in * mlp_ratio)

        activation = nn.GELU if not use_swiglu else SwiGLU
        mlp_hidden_dim = (
            int((4 * dim_in * 1 / 2) / 64) * 64 if use_swiglu else mlp_hidden_dim
        )

        self.mlp = Mlp(
            in_features=dim_in,
            hidden_features=mlp_hidden_dim,
            act_layer=activation,
            use_swiglu=use_swiglu,
        )

        self.gamma2 = (
            nn.Parameter(layer_scale * torch.ones(dim_in)) if layer_scale else 1
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if DEBUG:
            print(
                f"MLP layer: dim_in: {dim_in}, dim_out: {dim_in}, mlp_hidden_dim: {mlp_hidden_dim}"
            )
            print(f"drop_path: {drop_path}, layer_scale: {layer_scale}")

    def forward(self, x):
        skip_connection = x

        if self.do_windowing:
            # performing windowing if required
            x = self.downsample_op(x)
            x = self.downsample_mixer(x)

            if self.window_size > 0:
                H, W = x.shape[2], x.shape[3]

            x, pad_hw = window_partition(x, self.window_size)

        # window attention
        x = x + self.drop_path1(self.gamma1 * self.attn(self.norm1(x)))
        # mlp layer
        x = x + self.drop_path2(self.gamma2 * self.mlp(self.norm2(x)))

        if self.do_windowing:
            if self.window_size > 0:
                x = window_reverse(x, self.window_size, H, W, pad_hw)

            x = self.upsample_mixer(x)
            x = self.upsample_op(x)

            if (
                x.shape[2] != skip_connection.shape[2]
                or x.shape[3] != skip_connection.shape[3]
            ):
                x = torch.nn.functional.pad(
                    x,
                    (
                        0,
                        -x.shape[3] + skip_connection.shape[3],
                        0,
                        -x.shape[2] + skip_connection.shape[2],
                    ),
                )
        # need to add skip connection because downsampling and upsampling will break residual connection
        # 0.5 is needed to make sure that the skip connection is not too strong
        # in case of no downsample / upsample we can show that 0.5 compensates for the residual connection
        x = 0.5 * x + 0.5 * skip_connection

        return x


class MultiResolutionAttention(nn.Module):
    """
    MultiResolutionAttention (MRA) module
    The idea is to use multiple attention blocks with different resolution
    Feature maps are downsampled / upsampled for each attention block on different blocks
    Every attention block supports

    """

    def __init__(
        self,
        window_size,
        sr_ratio,
        dim,
        dim_ratio,
        num_heads,
        do_windowing=True,
        layer_scale=1e-5,
        norm_layer=nn.LayerNorm,
        drop_path=0,
        qkv_bias=False,
        qk_scale=1.0,
        use_swiglu=True,
        multi_query=False,
        conv_base=False,
    ) -> None:
        """
        Args:
            input_resolution: input image resolution
            window_size: window size
            compression_ratio: compression ratio
            max_depth: maximum depth of the GRA module
        """
        super().__init__()

        depth = len(sr_ratio)

        self.attention_blocks = nn.ModuleList()

        for i in range(depth):
            subsample_ratio = sr_ratio[i]
            if len(window_size) > i:
                window_size_local = window_size[i]
            else:
                window_size_local = window_size[0]

            self.attention_blocks.append(
                GRAAttentionBlock(
                    window_size=window_size_local,
                    dim_in=dim,
                    dim_out=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    drop_path=drop_path,
                    use_swiglu=use_swiglu,
                    subsample_ratio=subsample_ratio,
                    dim_ratio=dim_ratio,
                    do_windowing=do_windowing,
                    multi_query=multi_query,
                    conv_base=conv_base,
                ),
            )

    def forward(self, x):

        for attention_block in self.attention_blocks:
            x = attention_block(x)

        return x


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        use_swiglu=True,
        drop=0.0,
    ):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(
            in_features, hidden_features * (2 if use_swiglu else 1), bias=False
        )
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        # self.drop = GaussianDropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        x = x.view(x_size)
        return x


class Downsample(nn.Module):
    """
    Down-sampling block

    Pixel Unshuffle is used for down-sampling, works great accuracy - wise but takes 10% more TRT time
    """

    def __init__(
        self, dim, shuffle=False,
    ):
        """
        Args:
            dim: feature size dimension.
            shuffle: idea with
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        dim_out = 2 * dim

        if shuffle:
            self.norm = lambda x: pixel_unshuffle(x, factor=2)
            self.reduction = Conv2d_BN(dim * 4, dim_out, 1, 1, 0, bias=False)
        else:
            # removed layer norm for better, in this formulation we are getting 10% better speed
            # LayerNorm for high resolution inputs will be a pain as it pools over the entire spatial dimension
            self.norm = nn.Identity()
            self.reduction = Conv2d_BN(dim, dim_out, 3, 2, 1, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96, shuffle_down=False):
        """
        Args:
            in_chans: number of input channels.
            in_dim: intermediate feature size dimension to speed up stem.
            dim: final stem channel number
            shuffle_down: use PixelUnshuffle for down-sampling, effectively increases the receptive field
        """

        super().__init__()
        # shuffle_down = False
        if not shuffle_down:
            self.proj = nn.Identity()
            self.conv_down = nn.Sequential(
                Conv2d_BN(in_chans, in_dim, 3, 2, 1, bias=False),
                nn.ReLU(),
                Conv2d_BN(in_dim, dim, 3, 2, 1, bias=False),
                nn.ReLU(),
            )
        else:
            self.proj = lambda x: pixel_unshuffle(x, factor=4)

            # self.conv_down = nn.Sequential(Conv2d_BN(in_chans*16, in_dim, 3, 1, 1),
            #                                nn.SiLU(),
            #                                Conv2d_BN(in_dim, dim, 3, 1, 1),
            #                                nn.SiLU(),
            #                                )
            self.conv_down = nn.Sequential(
                Conv2d_BN(in_chans * 16, dim, 3, 1, 1), nn.ReLU(),
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional block, used in first couple of stages
    Experimented with plan resnet-18 like modules, they are the best in terms of throughput
    Experimented with RepVGG, dont see significant improvement in accuracy
    Finally, YOLOv8 idea seem to work fine (resnet-18 like block with squeezed feature dimension, and feature concatendation at the end)
    """

    def __init__(
        self, dim, drop_path=0.0, layer_scale=None, kernel_size=3, rep_vgg=False
    ):
        super().__init__()
        self.rep_vgg = rep_vgg
        if not rep_vgg:
            self.conv1 = Conv2d_BN(
                dim, dim, kernel_size=kernel_size, stride=1, padding=1
            )
            self.act1 = nn.GELU()
        else:
            self.conv1 = RepVGGBlock(
                dim, dim, kernel_size=kernel_size, stride=1, padding=1, groups=1
            )

        if not rep_vgg:
            self.conv2 = Conv2d_BN(
                dim, dim, kernel_size=kernel_size, stride=1, padding=1
            )
        else:
            self.conv2 = RepVGGBlock(
                dim, dim, kernel_size=kernel_size, stride=1, padding=1, groups=1
            )

        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        if not self.rep_vgg:
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class WindowAttention(nn.Module):
    # Windowed Attention from SwinV2
    # use a MLP trick to deal with various input image resolutions, then fold it to improve speed
    # tested multi-querry attention, but it is not as good as full attention:
    # look into palm: https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py
    # single kv attention, mlp in parallel (didnt improve speed)

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        resolution=0,
        seq_length=0,
        dim_out=None,
        multi_query=False,
    ):
        # taken from EdgeViT and tweaked with attention bias.
        super().__init__()
        if not dim_out:
            dim_out = dim
        self.multi_query = multi_query
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads

        self.dim_internal = dim

        self.scale = qk_scale or head_dim ** -0.5
        if not multi_query:
            if TRT:
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.k = nn.Linear(dim, dim, bias=qkv_bias)
                self.v = nn.Linear(dim, dim, bias=qkv_bias)
            else:
                self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim + 2 * self.head_dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim_out, bias=False)
        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[resolution, resolution],
            pretrained_window_size=[resolution, resolution],
            num_heads=num_heads,
            seq_length=seq_length,
        )

        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape

        if not self.multi_query:
            if TRT:
                q = (
                    self.q(x)
                    .reshape(B, -1, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
                )
                k = (
                    self.k(x)
                    .reshape(B, -1, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
                )
                v = (
                    self.v(x)
                    .reshape(B, -1, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
                )
            else:
                qkv = (
                    self.qkv(x)
                    .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = self.qkv(x)
            (q, k, v) = qkv.split(
                [self.dim_internal, self.head_dim, self.head_dim], dim=2
            )

            q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            k = k.reshape(B, -1, 1, C // self.num_heads).permute(0, 2, 1, 3)
            v = v.reshape(B, -1, 1, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.pos_emb_funct(attn)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        return x


class FasterViTLayer(nn.Module):
    """
    fastervitlayer
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        conv=False,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        drop_path=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        sr_dim_ratio=1,
        sr_ratio=1,
        multi_query=False,
        use_swiglu=True,
        rep_vgg=False,
        yolo_arch=False,
        downsample_shuffle=False,
        conv_base=False,
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            input_resolution: input image resolution.
            window_size: window size in each stage.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """

        super().__init__()
        self.conv = conv
        self.yolo_arch = False
        if conv:
            if not yolo_arch:
                self.blocks = nn.ModuleList(
                    [
                        ConvBlock(
                            dim=dim,
                            drop_path=drop_path[i]
                            if isinstance(drop_path, list)
                            else drop_path,
                            layer_scale=layer_scale_conv,
                            rep_vgg=rep_vgg,
                        )
                        for i in range(depth)
                    ]
                )
            else:
                self.blocks = C2f(dim, dim, n=depth, shortcut=True, e=0.5)
                self.yolo_arch = True
        else:
            if not isinstance(window_size, list):
                window_size = [window_size]
            self.window_size = window_size[0]
            self.do_single_windowing = True
            if not isinstance(sr_ratio, list):
                sr_ratio = [sr_ratio]
            if any([sr != 1 for sr in sr_ratio]) or len(set(window_size)) > 1:
                self.do_single_windowing = False
                do_windowing = True
            else:
                self.do_single_windowing = True
                do_windowing = False

            self.blocks = nn.ModuleList()
            for i in range(depth):

                self.blocks.append(
                    MultiResolutionAttention(
                        window_size=window_size,
                        sr_ratio=sr_ratio,
                        dim=dim,
                        dim_ratio=sr_dim_ratio,
                        num_heads=num_heads,
                        norm_layer=norm_layer,
                        drop_path=drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path,
                        layer_scale=layer_scale,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        use_swiglu=use_swiglu,
                        do_windowing=do_windowing,
                        multi_query=multi_query,
                        conv_base=conv_base,
                    )
                )

        self.transformer = not conv

        self.downsample = (
            None if not downsample else Downsample(dim=dim, shuffle=downsample_shuffle)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        if self.transformer and self.do_single_windowing:
            H, W = x.shape[2], x.shape[3]
            x, pad_hw = window_partition(x, self.window_size)

        if not self.yolo_arch:
            for bn, blk in enumerate(self.blocks):
                x = blk(x)
        else:
            x = self.blocks(x)

        if self.transformer and self.do_single_windowing:
            x = window_reverse(x, self.window_size, H, W, pad_hw)

        if self.downsample is None:
            return x, x

        return self.downsample(x), x  # changing to output pre downsampled features


class FasterViT(nn.Module):
    """
    FasterViT
    """

    def __init__(
        self,
        dim,
        in_dim,
        depths,
        window_size,
        mlp_ratio,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=False,
        qk_scale=None,
        layer_scale=None,
        layer_scale_conv=None,
        layer_norm_last=False,
        sr_ratio=[1, 1, 1, 1],
        max_depth=-1,
        conv_base=False,
        use_swiglu=False,
        multi_query=False,
        norm_layer=nn.LayerNorm,
        rep_vgg=False,
        drop_uniform=False,
        yolo_arch=False,
        shuffle_down=False,
        downsample_shuffle=False,
        return_full_features=False,
        full_features_head_dim=128,
        neck_start_stage=1,
        use_neck=False,
        **kwargs,
    ):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            return_full_features: output dense features as well as logits
            full_features_head_dim: number of channels in the dense features head
            neck_start_stage: a stage id to start full feature neck. Model has 4 stages, indix starts with 0
                                for 224 resolution, the output of the stage before downsample:
                                stage 0: 56x56, stage 1: 28x28, stage 2: 14x14, stage 3: 7x7
            use_neck: even for summarization embedding use neck
        """
        super().__init__()

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(
            in_chans=in_chans, in_dim=in_dim, dim=dim, shuffle_down=shuffle_down
        )
        # set return_full_features true if we want to return full features from all stages
        self.return_full_features = return_full_features
        self.use_neck = use_neck

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if drop_uniform:
            dpr = [drop_path_rate for x in range(sum(depths))]

        if not isinstance(max_depth, list):
            max_depth = [max_depth] * len(depths)

        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False

            level = FasterViTLayer(
                dim=int(dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=(i < 3),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                sr_ratio=sr_ratio[i],
                use_swiglu=use_swiglu,
                multi_query=multi_query,
                norm_layer=norm_layer,
                rep_vgg=rep_vgg,
                yolo_arch=yolo_arch,
                downsample_shuffle=downsample_shuffle,
                conv_base=conv_base,
            )

            self.levels.append(level)

        if self.return_full_features or self.use_neck:
            # create feature projection layers for segmentation output
            self.neck_features_proj = nn.ModuleList()
            self.neck_start_stage = neck_start_stage
            upsample_ratio = 1
            for i in range(len(depths)):
                level_n_features_output = int(dim * 2 ** i)

                if self.neck_start_stage > i:
                    continue

                if (
                    upsample_ratio > 1
                ) or full_features_head_dim != level_n_features_output:
                    feature_projection = nn.Sequential()
                    # feature_projection.add_module("norm",LayerNorm2d(level_n_features_output)) #slow, but better

                    if 0:
                        # Train: 0 [1900/10009 ( 19%)]  Loss: 6.113 (6.57)  Time: 0.548s,  233.40/s  (0.549s,  233.04/s)  LR: 1.000e-05  Data: 0.015 (0.013)
                        feature_projection.add_module(
                            "norm", nn.BatchNorm2d(level_n_features_output)
                        )  # fast, but worse
                        feature_projection.add_module(
                            "dconv",
                            nn.ConvTranspose2d(
                                level_n_features_output,
                                full_features_head_dim,
                                kernel_size=upsample_ratio,
                                stride=upsample_ratio,
                            ),
                        )
                    else:
                        # pixel shuffle based upsampling
                        # Train: 0 [1950/10009 ( 19%)]  Loss: 6.190 (6.55)  Time: 0.540s,  236.85/s  (0.548s,  233.38/s)  LR: 1.000e-05  Data: 0.015 (0.013)
                        feature_projection.add_module(
                            "norm", nn.BatchNorm2d(level_n_features_output)
                        )  # fast, but worse
                        feature_projection.add_module(
                            "conv",
                            nn.Conv2d(
                                level_n_features_output,
                                full_features_head_dim
                                * upsample_ratio
                                * upsample_ratio,
                                kernel_size=1,
                                stride=1,
                            ),
                        )
                        feature_projection.add_module(
                            "upsample_pixelshuffle", nn.PixelShuffle(upsample_ratio)
                        )

                else:
                    feature_projection = nn.Sequential()
                    feature_projection.add_module(
                        "norm", nn.BatchNorm2d(level_n_features_output)
                    )

                self.neck_features_proj.append(feature_projection)

                if i > 0 and self.levels[i - 1].downsample is not None:
                    upsample_ratio *= 2

        num_features = (
            full_features_head_dim
            if (self.return_full_features or self.use_neck)
            else num_features
        )

        self.num_features = num_features

        self.norm = (
            LayerNorm2d(num_features)
            if layer_norm_last
            else nn.BatchNorm2d(num_features)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = (
            nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)
        # pass

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        full_features = None
        for il, level in enumerate(self.levels):
            x, pre_downsample_x = level(x)

            if self.return_full_features or self.use_neck:
                if self.neck_start_stage > il:
                    continue
                if full_features is None:
                    full_features = self.neck_features_proj[il - self.neck_start_stage](
                        pre_downsample_x
                    )
                else:
                    # upsample torch tensor x to match full_features size, and add to full_features
                    feature_projection = self.neck_features_proj[
                        il - self.neck_start_stage
                    ](pre_downsample_x)
                    if (
                        feature_projection.shape[2] != full_features.shape[2]
                        or feature_projection.shape[3] != full_features.shape[3]
                    ):
                        feature_projection = torch.nn.functional.pad(
                            feature_projection,
                            (
                                0,
                                -feature_projection.shape[3] + full_features.shape[3],
                                0,
                                -feature_projection.shape[2] + full_features.shape[2],
                            ),
                        )
                    full_features += feature_projection

        # x = self.norm(full_features if (self.return_full_features or self.use_neck) else x)
        x = self.norm(x)  # new version for
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if not self.return_full_features:
            return x, None

        return x, full_features

    def forward(self, x):
        x, full_features = self.forward_features(x)
        x = self.head(x)
        if full_features is not None:
            return x, full_features
        return x

    def switch_to_deploy(self):
        """
        A method to perform model self-compression
        merges BN into conv layers
        converts MLP relative positional bias into precomputed buffers
        """
        for level in [self.patch_embed, self.levels, self.head]:
            for module in level.modules():
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy()


@register_model
def fastervit2_small(pretrained=False, **kwargs):  # ,
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=96,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [1, 2], 1],
        use_swiglu=False,
        downsample_shuffle=False,
        yolo_arch=True,
        shuffle_down=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_tiny(pretrained=False, **kwargs):  # ,
    model = FasterViT(
        depths=[1, 3, 4, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=80,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        downsample_shuffle=False,
        yolo_arch=True,
        shuffle_down=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_base(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_base_fullres1(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=1024,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_base_fullres2(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=512,
        neck_start_stage=1,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_base_fullres3(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=256,
        neck_start_stage=1,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_base_fullres4(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=256,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_base_fullres5(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=512,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


# pyt: 1934, 4202 TRT
@register_model
def fastervit2_large(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128 + 64,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_large_fullres(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[None, None, [7, 7], 7],
        dim=192,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=1536,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_large_fullres_ws8(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[None, None, [8, 8], 8],
        dim=192,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=1536,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_large_fullres_ws16(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[None, None, [16, 16], 16],
        dim=192,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=1536,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_large_fullres_ws32(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[None, None, [32, 32], 32],
        dim=192,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=1536,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


# pyt: 897
@register_model
def fastervit2_xlarge(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128 + 128 + 64,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


# pyt:
@register_model
def fastervit2_huge(pretrained=False, **kwargs):
    model = FasterViT(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=128 + 128 + 128 + 64,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_xtiny(pretrained=False, **kwargs):  # ,
    model = FasterViT(
        depths=[1, 3, 4, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=64,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.1,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        downsample_shuffle=False,
        yolo_arch=True,
        shuffle_down=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_xxtiny_5(pretrained=False, **kwargs):  # ,
    model = FasterViT(
        depths=[1, 3, 4, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=48,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.05,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        downsample_shuffle=False,
        yolo_arch=True,
        shuffle_down=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def fastervit2_xxxtiny(pretrained=False, **kwargs):  # ,
    model = FasterViT(
        depths=[1, 3, 4, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, [7, 7], 7],
        dim=32,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        downsample_shuffle=False,
        yolo_arch=True,
        shuffle_down=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def eradio(pretrained=False, **kwargs):
    return fastervit2_large_fullres_ws16(pretrained=pretrained, **kwargs)
