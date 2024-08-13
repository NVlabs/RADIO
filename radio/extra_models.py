from distutils.version import LooseVersion
import warnings

import torch
from torch import nn

from timm.models.registry import register_model


class PaliGemmaWrapper(nn.Module):
    def __init__(self, vis_model: nn.Module, embed_dim: int):
        super().__init__()

        self.vis_model = vis_model
        self.embed_dim = embed_dim

    @property
    def patch_size(self):
        return 14

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
