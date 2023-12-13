# From:
# https://github.com/facebookresearch/dinov2/tree/main/dinov2/eval/segmentation/models/decode_heads

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict


@HEADS.register_module()
class BNHead(BaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        kwargs.pop("pool_scales", None)
        super().__init__(**kwargs)
        assert self.in_channels == self.channels, f"{self.in_channels} != {self.channels}"
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

    def init_weights(self):
        # Initialize weights from pretrained checkpoint.
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            new_dict = dict()
            for key, value in state_dict.items():
                # Remove the "decode_head." prefix from all keys.
                if key.startswith("decode_head."):
                    new_dict[key[len("decode_head."):]] = value
            state_dict = new_dict

            load_state_dict(self, state_dict, strict=False, logger=None)
        else:
            super().init_weights()

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            #print("before", *(x.shape for x in inputs), "to", inputs[0].shape[2:])
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            assert all([x.shape[2:] == inputs[0].shape[2:] for x in inputs]), [x.shape for x in inputs]
            #upsampled_inputs = [
            #    resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
            #    for x in inputs
            #]
            upsampled_inputs = inputs
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        #print("linear head input", [input.shape for input in inputs], [input.mean().item() for input in inputs], [input.std().item() for input in inputs])
        output = self._forward_feature(inputs)
        #print("linear head features", output.shape, output.mean(), output.std())
        output = self.cls_seg(output)
        return output
