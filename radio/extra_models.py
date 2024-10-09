from distutils.version import LooseVersion
from typing import List, Optional, Tuple, Union
import warnings

import torch
from torch import nn

from timm.models.registry import register_model

from .forward_intermediates import forward_intermediates


class PaliGemmaWrapper(nn.Module):
    def __init__(self, vis_model: nn.Module, embed_dim: int):
        super().__init__()

        self.vis_model = vis_model
        self.embed_dim = embed_dim

    @property
    def patch_size(self):
        return self.vis_model.embeddings.patch_size

    @property
    def blocks(self):
        return self.vis_model.encoder.layers

    @property
    def embed_dim(self):
        return self.vis_model.embeddings.embed_dim

    def forward(self, x: torch.Tensor):
        outputs = self.vis_model(
            x,
            return_dict=False,
            interpolate_pos_encoding=True,
        )

        features = outputs[0].to(torch.float32)

        summary = features.mean(dim=1)

        return summary, features

    def forward_features(self, x: torch.Tensor):
        return self(x)


def _get_paligemma_model(repo: str, embed_dim: int = None, dtype: torch.dtype = torch.bfloat16):
    from transformers import PaliGemmaForConditionalGeneration, __version__ as tx_version

    if LooseVersion(tx_version) > LooseVersion('4.44.2'):
        warnings.warn(f'Your transformers version "{tx_version}" is higher than 4.44.2, and for whatever reason, PaliGemma might be broken.')

    extra_args = dict()

    if dtype is not None:
        extra_args['torch_dtype'] = dtype
        rev = str(dtype).split('.')[-1]
        extra_args['revision'] = rev

    model = PaliGemmaForConditionalGeneration.from_pretrained(repo, **extra_args)

    vis_model = model.vision_tower.vision_model

    vis_model = PaliGemmaWrapper(vis_model, embed_dim)

    return vis_model

@register_model
def paligemma_896_student(**kwargs):
    model = _get_paligemma_model('google/paligemma-3b-pt-896', embed_dim=1152, dtype=None)

    return model


def _load_dino_v2(dino_v2_model, cache_dir: Optional[str] = None, pretrained=True, **kwargs):
    if cache_dir:
        torch.hub.set_dir(cache_dir)
    model = torch.hub.load(
        'facebookresearch/dinov2',
        dino_v2_model,
        pretrained=pretrained,
        # **kwargs,
    )
    return model


class DinoWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module):
        super().__init__()

        self.inner = dino_model
        dino_model.blocks = nn.Sequential(*dino_model.blocks)

    @property
    def embed_dim(self):
        return self.inner.embed_dim

    @property
    def patch_size(self):
        return self.inner.patch_size

    @property
    def num_cls_tokens(self):
        return getattr(self.inner, 'num_tokens', 1)

    @property
    def num_registers(self):
        return getattr(self.inner, 'num_register_tokens', 0)

    @property
    def num_summary_tokens(self):
        return self.num_cls_tokens + self.num_registers

    @property
    def blocks(self):
        return self.inner.blocks

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = self.inner.forward_features(*args, **kwargs)

        cls_token = parts['x_norm_clstoken']
        features = parts['x_norm_patchtokens']

        return cls_token, features

    def forward_features(self, x: torch.Tensor):
        x = self.inner.prepare_tokens_with_masks(x)
        x = self.inner.blocks(x)
        x_norm = self.inner.norm(x)

        return x_norm[:, 0], x_norm[:, self.num_summary_tokens:]

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.prepare_tokens_with_masks(x)

    def forward_intermediates(self,
        x: torch.Tensor,
        norm: bool = False,
        **kwargs,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        return forward_intermediates(
            self,
            patch_extractor=self.inner.prepare_tokens_with_masks,
            num_summary_tokens=self.num_summary_tokens,
            num_cls_tokens=self.num_cls_tokens,
            norm=self.inner.norm if norm else lambda y: y,
            x=x,
            **kwargs,
        )


@register_model
def dino_v2_g_student(**kwargs):
    model = _load_dino_v2('dinov2_vitg14_reg', pretrained=False)
    model = DinoWrapper(model)

    return model
