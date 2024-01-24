# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from argparse import Namespace

import torch
from torch import nn
import torch.nn.functional as F

from radio.radio_model import RADIOModel
from radio.adaptors.adaptor_registry import adaptor_registry, dict_t, state_t
from radio.adaptors.mlp import create_mlp_from_state


class OpenCLIP_RADIO(nn.Module):
    def __init__(self, base_model: RADIOModel, main_config: Namespace, adaptor_config: dict_t, state: state_t):
        super().__init__()

        self.base_model = base_model
        self.head_mlp = create_mlp_from_state(main_config.mlp_version, state, 'summary.')
        self.feat_mlp = create_mlp_from_state(main_config.mlp_version, state, 'feature.')

        import open_clip

        self.oc_model = open_clip.create_model_from_pretrained(
            model_name=adaptor_config['model'],
            pretrained=adaptor_config['pretrained'],
            return_transform=False,
        )
        # Unload these parameters
        self.oc_model.visual = None

        self.tokenizer = open_clip.get_tokenizer(model_name=adaptor_config['model'])

    @property
    def return_summary(self):
        return self.base_model.return_summary

    @return_summary.setter
    def return_summary(self, v: bool):
        self.base_model.return_summary = v

    @property
    def return_spatial_features(self):
        return self.base_model.return_spatial_features

    @return_spatial_features.setter
    def return_spatial_features(self, v: bool):
        self.base_model.return_spatial_features = v

    @property
    def return_both(self):
        return self.base_model.return_both

    def forward(self, x: torch.Tensor):
        ret = self.base_model(x)

        if self.return_both:
            return (self.head_mlp(ret[0]), self.feat_mlp(ret[1]))
        elif self.return_summary:
            return self.head_mlp(ret)
        elif self.return_spatial_features:
            return self.feat_mlp(ret)
        raise ValueError("Unsupported return mode!")

    def encode_image(self, image, normalize: bool = False):
        if not self.return_summary:
            raise ValueError(f'`return_summary` must be set to True!')

        vision_summary = self(image)
        if self.return_both:
            vision_summary = vision_summary[0]

        return F.normalize(vision_summary, dim=-1) if normalize else vision_summary

    def encode_text(self, text, normalize: bool = False):
        return self.oc_model.encode_text(text, normalize=normalize)

@adaptor_registry.register_adaptor("open_clip")
def create_open_clip_adaptor(base_model: RADIOModel, main_config: Namespace, adaptor_config: dict_t, state: state_t):
    return OpenCLIP_RADIO(base_model, main_config, adaptor_config, state)
