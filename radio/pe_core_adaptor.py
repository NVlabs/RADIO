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

from .adaptor_registry import adaptor_registry, dict_t, state_t
from .adaptor_generic import GenericAdaptor
from .utils import rank_gate


class PECoreAdaptor(GenericAdaptor):
    """Text-aligned adaptor for Meta's Perception Encoder (PE-Core).

    Loads the text tower from `core.vision_encoder.pe.CLIP` (the
    `facebookresearch/perception_models` package) and routes `encode_text`
    through it. The visual tower from the PE-Core CLIP is discarded — RADIO's
    own backbone provides the image features that the adaptor's summary/feature
    heads (loaded from the RADIO checkpoint state) are aligned to.
    """

    def __init__(self, main_config: Namespace, adaptor_config: dict_t, state: state_t):
        super().__init__(main_config, adaptor_config, state)

        version = adaptor_config['model']  # e.g. 'PE-Core-G14-448'

        try:
            from core.vision_encoder import pe as pe_mod
            from core.vision_encoder.transforms import get_text_tokenizer
        except ImportError as e:
            raise ImportError(
                "PE-Core adaptor requires the `perception_models` package. "
                "Install with: pip install git+https://github.com/facebookresearch/perception_models"
            ) from e

        available = pe_mod.CLIP.available_configs()
        if version not in available:
            raise ValueError(
                f"Unknown PE-Core variant '{version}'. Available: {available}"
            )

        with rank_gate():
            inner = pe_mod.CLIP.from_config(version, pretrained=True)

        # We only need the text tower; the visual tower is provided by RADIO.
        inner.visual = None

        text_cfg = pe_mod.PE_TEXT_CONFIG[version]
        self.tokenizer = get_text_tokenizer(text_cfg.context_length)
        self.inner = inner

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self.inner.encode_text(text, normalize=normalize)


@adaptor_registry.register_adaptor("pe_core")
def create_pe_core_adaptor(main_config: Namespace, adaptor_config: dict_t, state: state_t):
    return PECoreAdaptor(main_config, adaptor_config, state)
