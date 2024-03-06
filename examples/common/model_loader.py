from dataclasses import dataclass
import os

import torch
from torch import nn
from torch.nn import functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from radio.radio_model import RadioOutput

class DinoWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module):
        super().__init__()
        self.inner = dino_model

    @property
    def patch_size(self):
        return self.inner.patch_size

    def forward(self, *args, **kwargs):
        parts = self.inner.forward_features(*args, **kwargs)

        cls_token = parts['x_norm_clstoken']
        features = parts['x_norm_patchtokens']

        return cls_token, features


class CLIPWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module, tokenizer, adaptor_name: str, clip_mode: bool = False):
        super().__init__()
        self.inner = clip_model
        clip_model.visual.output_tokens = True
        self.tokenizer = tokenizer
        self.adaptor_name = adaptor_name

        if not clip_mode and hasattr(clip_model.visual, 'proj'):
            visual = clip_model.visual
            proj = visual.proj
            I = torch.eye(proj.shape[0], dtype=proj.dtype, device=proj.device)
            visual.proj = nn.Parameter(I)

    @property
    def patch_size(self):
        return self.inner.visual.patch_size[0]

    def forward(self, *args, **kwargs):
        enc = self.inner.visual(*args, **kwargs)

        if isinstance(enc, (tuple, list)):
            token, features = enc
        else:
            token, features = enc, None

        op = RadioOutput(token, features)

        if self.adaptor_name:
            return {
                'backbone': op,
                self.adaptor_name: op,
            }
        return op

    def encode_image(self, image, normalize: bool = False):
        token, _ = self(image)

        if normalize:
            token = F.normalize(token, dim=-1)

        return token

    def encode_text(self, text, normalize: bool = False):
        return self.inner.encode_text(text, normalize=normalize)


@dataclass
class ModelInfo:
    model_class: str
    model_subtype: str


def load_model(version: str, adaptor_names: str = None, use_huggingface: bool = False, use_local_lib: bool = True,
               device: torch.device = None, return_spatial_features: bool = True, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if os.path.isfile(version) or version.startswith('radio'):
        if use_huggingface:
            from transformers import AutoModel
            model: nn.Module = AutoModel.from_pretrained(f"nvidia/{version}", trust_remote_code=True, **kwargs)
        elif use_local_lib:
            from hubconf import radio_model
            model = radio_model(version=version, progress=True, adaptor_names=adaptor_names, **kwargs)
        else:
            model: nn.Module = torch.hub.load('NVlabs/RADIO', 'radio_model', version=version, progress=True,
                                              adaptor_names=adaptor_names, return_spatial_features=return_spatial_features,
                                              **kwargs,
            )

        preprocessor = model.make_preprocessor_external()
        info = ModelInfo(model_class='RADIO', model_subtype=version.replace('/', '_'))
    elif version.startswith('dinov2'):
        model = torch.hub.load('facebookresearch/dinov2', version, pretrained=True, **kwargs)
        model = DinoWrapper(model)

        from radio.input_conditioner import InputConditioner
        preprocessor = InputConditioner(1.0, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        info = ModelInfo(model_class='DINOv2', model_subtype=version.replace('dinov2_', ''))
    elif version.startswith('open_clip'):
        import open_clip
        _, model_arch, pretrained = version.split(',')
        model = open_clip.create_model(model_arch, pretrained, device=device)
        viz_model = model.visual

        from radio.input_conditioner import InputConditioner
        preprocessor = InputConditioner(1.0,
            getattr(viz_model, 'image_mean', open_clip.OPENAI_DATASET_MEAN),
            getattr(viz_model, 'image_std', open_clip.OPENAI_DATASET_STD),
        )

        tokenizer = open_clip.get_tokenizer(model_arch)

        model = CLIPWrapper(model, tokenizer, adaptor_names, clip_mode='clip' in adaptor_names if adaptor_names else False)
        info = ModelInfo(model_class='open_clip', model_subtype=pretrained)
    else:
        raise ValueError(f'Unsupported model version: {version}')

    if device is not None:
        model.to(device=device)

    return model, preprocessor, info
