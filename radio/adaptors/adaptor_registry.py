# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from argparse import Namespace
from typing import Dict, Any

import torch

from ..radio_model import RADIOModel

dict_t = Dict[str, Any]
state_t = Dict[str, torch.Tensor]


class AdaptorRegistry:
    def __init__(self):
        self._registry = {}

    def register_adaptor(self, name):
        def decorator(factory_function):
            if name in self._registry:
                raise ValueError(f"Model '{name}' already registered")
            self._registry[name] = factory_function
            return factory_function
        return decorator

    def create_adaptor(self, name, base_model: RADIOModel, main_config: Namespace, adaptor_config: dict_t, state: state_t):
        if name not in self._registry:
            raise ValueError(f"Adaptor '{name}' not found in registry")
        return self._registry[name](base_model, main_config, adaptor_config, state)

# Creating an instance of the registry
adaptor_registry = AdaptorRegistry()
