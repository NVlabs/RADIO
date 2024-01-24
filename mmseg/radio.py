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
import math
from typing import List, Optional
import warnings

from mmengine.model import BaseModule
from mmseg.models.builder import BACKBONES
from timm.models.vision_transformer import VisionTransformer
import torch
import torch.distributed as dist

from transformers import AutoModel

# A student model to learn the teacher's features.
@BACKBONES.register_module()
class RADIO(BaseModule):
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        # Instantiate a RADIO model.
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Pull the model on rank 0 first.
            model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=token)
        if dist.is_initialized():
            dist.barrier()
            if dist.get_rank() > 0:
                # Now pull the model from cache on other ranks.
                model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=token)

        model.eval()

        self.base_model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass."""
        _, _, H, W = x.shape

        # Scale inputs to the range [0, 1].
        x = x / 255.0

        _, features = self.base_model(x)

        if isinstance(self.base_model.model, VisionTransformer):
            # Reshape
            B, _, C = features.shape

            if hasattr(self.base_model.model, "patch_generator"):
                # Cropped Positional Embedding (CPE) case.
                patch_height = patch_width = self.base_model.model.patch_generator.patch_size
            else:
                # Standard ViT case.
                patch_height, patch_width = self.base_model.model.patch_embed.patch_size
            features = features.reshape(B, math.ceil(H/patch_height), math.ceil(W/patch_width),  C).permute(0, 3, 1, 2).contiguous()

        # IMPORTANT: prevent gradients from flowing back towards the backbone.
        features = features.detach()

        return [features]

    def train(self, mode=True):
        """Intercept call."""
        # Drop a warning if mode is True.
        if mode:
            warnings.warn("RADIO is always in eval mode.")
        pass

    def init_weights(self):
        # This is a no-op as the model weights are loaded during instantiation.
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            pass
        else:
            raise ValueError(f"Unhandled case: {self.init_cfg}")
