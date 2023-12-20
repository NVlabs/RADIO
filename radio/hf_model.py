# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from collections import namedtuple
from typing import Optional

from timm.models import VisionTransformer
import torch
from transformers import PretrainedConfig, PreTrainedModel


from .eradio_model import eradio
from .radio_model import create_model_from_args
from .radio_model import RADIOModel as RADIOModelBase
from .input_conditioner import get_default_conditioner, InputConditioner


class RADIOConfig(PretrainedConfig):
    """Pretrained Hugging Face configuration for RADIO models."""

    def __init__(
        self,
        args: Optional[dict] = None,
        version: Optional[str] = "v1",
        return_summary: Optional[bool] = True,
        return_spatial_features: Optional[bool] = True,
        **kwargs,
    ):
        self.args = args
        self.version = version
        self.return_summary = return_summary
        self.return_spatial_features = return_spatial_features
        super().__init__(**kwargs)


class RADIOModel(PreTrainedModel):
    """Pretrained Hugging Face model for RADIO.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = RADIOConfig

    def __init__(self, config):
        super().__init__(config)

        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)
        self.config = config
        model = create_model_from_args(args)
        input_conditioner: InputConditioner = get_default_conditioner()

        self.radio_model = RADIOModelBase(
            model,
            input_conditioner,
            config.return_summary,
            config.return_spatial_features,
        )

    @property
    def model(self) -> VisionTransformer:
        return self.radio_model.model

    @property
    def input_conditioner(self) -> InputConditioner:
        return self.radio_model.input_conditioner

    def forward(self, x: torch.Tensor):
        return self.radio_model.forward(x)


class ERADIOConfig(PretrainedConfig):
    """Pretrained Hugging Face configuration for ERADIO models."""

    def __init__(
        self,
        args: Optional[dict] = None,
        version: Optional[str] = "v1",
        return_summary: Optional[bool] = True,
        return_spatial_features: Optional[bool] = True,
        **kwargs,
    ):
        self.args = args
        self.version = version
        self.return_summary = return_summary
        self.return_spatial_features = return_spatial_features
        super().__init__(**kwargs)


class ERADIOModel(PreTrainedModel):
    """Pretrained Hugging Face model for ERADIO.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = ERADIOConfig

    def __init__(self, config):
        super().__init__(config)

        config.args["in_chans"] = 3
        config.args["num_classes"] = 0
        config.args["return_full_features"] = config.return_spatial_features

        self.config = config
        model = eradio(**config.args)
        self.input_conditioner: InputConditioner = get_default_conditioner()
        self.return_summary = config.return_summary
        self.return_spatial_features = config.return_spatial_features
        self.model = model

    def forward(self, x: torch.Tensor):
        x = self.input_conditioner(x)
        y = self.model.forward_features(x)
        summary, features = self.model.forward_features(x)

        if isinstance(y, tuple):
            summary, features = y
        else:
            summary = y
            features = None

        if self.return_summary and self.return_spatial_features:
            return summary, features
        elif self.return_summary:
            return summary
        return features
