# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Convolution modules
"""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = ('Conv', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# Pavlo's implementation with switch to deploy
class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, a, b, kernel_size=1, stride=1, padding=None, g=1, dilation=1, bn_weight_init=1, bias=False, act=True):
        super().__init__()

        self.conv = torch.nn.Conv2d(a, b, kernel_size, stride, autopad(kernel_size, padding, dilation), dilation, g, bias=False)
        if 1:
            self.bn = torch.nn.BatchNorm2d(b)
            torch.nn.init.constant_(self.bn.weight, bn_weight_init)
            torch.nn.init.constant_(self.bn.bias, 0)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        if not isinstance(self.bn, nn.Identity):
            # return 1
            c, bn = self.conv, self.bn
            w = bn.weight / (bn.running_var + bn.eps) ** 0.5
            w = c.weight * w[:, None, None, None]
            b = bn.bias - bn.running_mean * bn.weight / \
                (bn.running_var + bn.eps)**0.5
            # m = torch.nn.Conv2d(w.size(1) * c.groups,
            #                     w.size(0),
            #                     w.shape[2:],
            #                     stride=c.stride,
            #                     padding=c.padding,
            #                     dilation=c.dilation,
            #                     groups=c.groups)
            self.conv.weight.data.copy_(w)
            self.conv.bias = nn.Parameter(b)
            # self.conv.bias.data.copy_(b)
            # self.conv = m.to(c.weight.device)
            self.bn = nn.Identity()
