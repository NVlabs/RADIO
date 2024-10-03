# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Register the adaptors
from .adaptor_registry import adaptor_registry
from . import open_clip_adaptor
from .adaptor_base import AdaptorInput, RadioOutput, AdaptorBase

# Enable support for other model types via the timm register_model mechanism
from . import extra_timm_models
from . import extra_models
