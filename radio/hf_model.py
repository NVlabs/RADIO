# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Callable, Optional, List, Union

from timm.models import VisionTransformer
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


from .common import RESOURCE_MAP, DEFAULT_VERSION

# Import all required modules.
from .adaptor_base import AdaptorBase, RadioOutput, AdaptorInput
from .adaptor_generic import GenericAdaptor, AdaptorBase
from .adaptor_mlp import create_mlp_from_state
from .adaptor_registry import adaptor_registry
from .cls_token import ClsToken
from .enable_cpe_support import enable_cpe
from .enable_spectral_reparam import configure_spectral_reparam_from_args
from .eradio_model import eradio
from .radio_model import create_model_from_args
from .radio_model import RADIOModel as RADIOModelBase, Resolution
from .input_conditioner import get_default_conditioner, InputConditioner
from .open_clip_adaptor import OpenCLIP_RADIO
from .vit_patch_generator import ViTPatchGenerator
from .vitdet import apply_vitdet_arch, VitDetArgs

# Register extra models
from .extra_timm_models import *


class RADIOConfig(PretrainedConfig):
    """Pretrained Hugging Face configuration for RADIO models."""

    def __init__(
        self,
        args: Optional[dict] = None,
        version: Optional[str] = DEFAULT_VERSION,
        patch_size: Optional[int] = None,
        max_resolution: Optional[int] = None,
        preferred_resolution: Optional[Resolution] = None,
        adaptor_names: Union[str, List[str]] = None,
        vitdet_window_size: Optional[int] = None,
        **kwargs,
    ):
        self.args = args
        for field in ["dtype", "amp_dtype"]:
            if self.args is not None and field in self.args:
                # Convert to a string in order to make it serializable.
                # For example for torch.float32 we will store "float32",
                # for "bfloat16" we will store "bfloat16".
                self.args[field] = str(args[field]).split(".")[-1]
        self.version = version
        resource = RESOURCE_MAP[version]
        self.patch_size = patch_size or resource.patch_size
        self.max_resolution = max_resolution or resource.max_resolution
        self.preferred_resolution = (
            preferred_resolution or resource.preferred_resolution
        )
        self.adaptor_names = adaptor_names
        self.vitdet_window_size = vitdet_window_size
        super().__init__(**kwargs)


class RADIOModel(PreTrainedModel):
    """Pretrained Hugging Face model for RADIO.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = RADIOConfig

    def __init__(self, config: RADIOConfig):
        super().__init__(config)

        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)
        self.config = config

        model = create_model_from_args(args)
        input_conditioner: InputConditioner = get_default_conditioner()

        dtype = getattr(args, "dtype", torch.float32)
        if isinstance(dtype, str):
            # Convert the dtype's string representation back to a dtype.
            dtype = getattr(torch, dtype)
        model.to(dtype=dtype)
        input_conditioner.dtype = dtype

        summary_idxs = torch.tensor(
            [i for i, t in enumerate(args.teachers) if t.get("use_summary", True)],
            dtype=torch.int64,
        )

        adaptor_names = config.adaptor_names
        if adaptor_names is not None:
            raise NotImplementedError(
                f"Adaptors are not yet supported in Hugging Face models. Adaptor names: {adaptor_names}"
            )

        adaptors = dict()

        self.radio_model = RADIOModelBase(
            model,
            input_conditioner,
            summary_idxs=summary_idxs,
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
            window_size=config.vitdet_window_size,
            preferred_resolution=config.preferred_resolution,
            adaptors=adaptors,
        )

    @property
    def adaptors(self) -> nn.ModuleDict:
        return self.radio_model.adaptors

    @property
    def model(self) -> VisionTransformer:
        return self.radio_model.model

    @property
    def input_conditioner(self) -> InputConditioner:
        return self.radio_model.input_conditioner

    @property
    def num_summary_tokens(self) -> int:
        return self.radio_model.num_summary_tokens

    @property
    def patch_size(self) -> int:
        return self.radio_model.patch_size

    @property
    def max_resolution(self) -> int:
        return self.radio_model.max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self.radio_model.preferred_resolution

    @property
    def window_size(self) -> int:
        return self.radio_model.window_size

    @property
    def min_resolution_step(self) -> int:
        return self.radio_model.min_resolution_step

    def make_preprocessor_external(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.radio_model.make_preprocessor_external()

    def get_nearest_supported_resolution(self, height: int, width: int) -> Resolution:
        return self.radio_model.get_nearest_supported_resolution(height, width)

    def switch_to_deploy(self):
        return self.radio_model.switch_to_deploy()

    def forward(self, x: torch.Tensor):
        return self.radio_model.forward(x)
