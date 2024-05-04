from dataclasses import dataclass
import os
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from radio.adaptor_base import RadioOutput
from radio.input_conditioner import InputConditioner

class DinoWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module):
        super().__init__()
        self.inner = dino_model

    @property
    def patch_size(self):
        return self.inner.patch_size

    @property
    def vision_encoder(self):
        return self.inner

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

    @property
    def vision_encoder(self):
        return self.inner.visual

    def forward(self, *args, **kwargs):
        enc = self.inner.visual(*args, **kwargs)

        if isinstance(enc, (tuple, list)):
            token, features = enc
        else:
            token, features = enc, None

        return self._wrap_output(token, features)

    def _wrap_output(self, token, features):
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
        try:
            return self.inner.encode_text(text, normalize=normalize)
        except TypeError:
            ret = self.inner.encode_text(text)
            if normalize:
                ret = F.normalize(ret, dim=-1)
            return ret


class SigLIPWrapper(CLIPWrapper):
    def forward(self, *args, **kwargs):
        features = self.inner.visual.trunk.forward_features(*args, **kwargs)
        token = self.inner.visual.trunk.attn_pool(features)
        return self._wrap_output(token, features)


class SAMWrapper(nn.Module):
    def __init__(self, sam_encoder: nn.Module):
        super().__init__()
        self.inner = sam_encoder

    @property
    def embed_dim(self):
        return self.inner.patch_embed.proj.out_channels

    @property
    def patch_size(self):
        return self.inner.patch_embed.proj.kernel_size[0]

    @property
    def vision_encoder(self):
        return self.inner

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.inner.patch_embed(x)
        if self.inner.pos_embed is not None:
            x = x + self.inner.pos_embed

        for blk in self.inner.blocks:
            x = blk(x)

        features = x.flatten(1, 2)

        summary = features.mean(dim=1)

        return summary, features


class InternViTWrapper(nn.Module):
    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.inner = model

        if tokenizer is not None:
            self.tokenizer = lambda texts: tokenizer(texts, return_tensors='pt', max_length=80,
                                                     truncation=True, padding='max_length').input_ids
        else:
            self.tokenizer = None

    @property
    def embed_dim(self):
        return 3200

    @property
    def patch_size(self):
        return self.inner.embeddings.patch_size

    @property
    def vision_encoder(self):
        return self.inner

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is not None:
            y = self.inner.encode_image(x, mode='InternVL-C')
            ret = RadioOutput(y.float(), None)
            return dict(backbone=ret, clip=ret)

        z = self.inner(x)
        y = z.last_hidden_state.float()

        summary = y[:, 0]
        features = y[:, 1:]

        return RadioOutput(summary, features)

    def encode_image(self, image, normalize: bool = False):
        token, _ = self(image)
        token = self.inner.clip_projector(token)

        if normalize:
            token = F.normalize(token, dim=-1)

        return token

    def encode_text(self, text, normalize: bool = False):
        token = self.inner.encode_text(text)

        if normalize:
            token = F.normalize(token, dim=-1)

        return token


class OpenAI_CLIP_VisionAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_resolution = model.input_resolution
        self.output_dim = model.output_dim
        self.conv1 = model.conv1

        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre

        self.transformer = model.transformer

        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def patch_size(self):
        return self.conv1.kernel_size

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        feats = x[:, 1:]

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, feats

@dataclass
class ModelInfo:
    model_class: str
    model_subtype: str


def load_model(version: str, adaptor_names: str = None, use_huggingface: bool = False, use_local_lib: bool = True,
               device: torch.device = None, return_spatial_features: bool = True, force_reload: bool = False,
               torchhub_repo="NVlabs/RADIO", **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if os.path.isfile(version) or 'radio' in version:
        if use_huggingface:
            from transformers import AutoModel, AutoConfig
            hf_repo = 'E-RADIO' if 'eradio' in version else 'RADIO'
            hf_repo = f"nvidia/{hf_repo}"
            config = AutoConfig.from_pretrained(
                hf_repo,
                trust_remote_code=True,
                version=version,
                adaptor_names=adaptor_names,
                **kwargs,
            )
            model: nn.Module = AutoModel.from_pretrained(hf_repo, config=config, trust_remote_code=True, **kwargs)
        elif use_local_lib:
            from hubconf import radio_model
            model = radio_model(version=version, progress=True, adaptor_names=adaptor_names, **kwargs)
        else:
            model: nn.Module = torch.hub.load(torchhub_repo, 'radio_model', version=version, progress=True,
                                              adaptor_names=adaptor_names, return_spatial_features=return_spatial_features,
                                              force_reload=force_reload, **kwargs,
            )

        preprocessor = model.make_preprocessor_external()
        info = ModelInfo(model_class='RADIO', model_subtype=version.replace('/', '_'))
    elif version.startswith('dinov2'):
        model = torch.hub.load('facebookresearch/dinov2', version, pretrained=True, force_reload=force_reload, **kwargs)
        model = DinoWrapper(model)

        preprocessor = InputConditioner(1.0, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        info = ModelInfo(model_class='DINOv2', model_subtype=version.replace('dinov2_', ''))
    elif version.startswith('open_clip'):
        import open_clip
        _, model_arch, pretrained = version.split(',')
        model = open_clip.create_model(model_arch, pretrained, device=device)
        viz_model = model.visual

        preprocessor = InputConditioner(1.0,
            getattr(viz_model, 'image_mean', open_clip.OPENAI_DATASET_MEAN),
            getattr(viz_model, 'image_std', open_clip.OPENAI_DATASET_STD),
        )

        tokenizer = open_clip.get_tokenizer(model_arch)

        factory = CLIPWrapper
        if model_arch == 'ViT-SO400M-14-SigLIP-384':
            factory = SigLIPWrapper

        model = factory(model, tokenizer, adaptor_names, clip_mode='clip' in adaptor_names if adaptor_names else False)
        info = ModelInfo(model_class='open_clip', model_subtype=pretrained)
    elif version.startswith('openai_clip'):
        import clip as openai_clip

        _, model_name = version.split(',')
        model, preprocess = openai_clip.load(
            model_name,
            device=device,
            jit=False,
        )

        model.visual = OpenAI_CLIP_VisionAdapter(model.visual).to(device)
        norm = preprocess.transforms[-1]
        preprocessor = InputConditioner(
            input_scale=1.0,
            norm_mean=norm.mean,
            norm_std=norm.std,
            dtype=torch.float16,
        )

        model = CLIPWrapper(model, tokenizer=openai_clip.tokenize, adaptor_name=adaptor_names, clip_mode='clip' in adaptor_names if adaptor_names else False)
        info = ModelInfo(model_class='openai_clip', model_subtype=model_name)
    elif version.startswith('sam'):
        from segment_anything.build_sam import sam_model_registry, ImageEncoderViT, Sam
        _, chk_path = version.split(',')
        fname = os.path.basename(chk_path)
        prefix = 'sam_vit_'
        assert fname.startswith(prefix) and fname[len(prefix)] in ('h', 'l', 'b'), "Invalid checkpoint file"
        model_name = fname[4:9]
        model = sam_model_registry[model_name](checkpoint=chk_path)

        preprocessor = InputConditioner(
            input_scale=255.0,
            norm_mean=model.pixel_mean,
            norm_std=model.pixel_std,
        )

        img_encoder = model.image_encoder
        model = SAMWrapper(img_encoder)
        info = ModelInfo(model_class='SAM', model_subtype=model_name)
    elif version.startswith('InternV'):
        from transformers import AutoModel, AutoConfig, CLIPImageProcessor, AutoTokenizer

        hfhub_name = f'OpenGVLab/{version}'
        model = AutoModel.from_pretrained(
            hfhub_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)

        if version.startswith('InternVL'):
            tokenizer = AutoTokenizer.from_pretrained(hfhub_name, use_fast=False, add_eos_token=True, trust_remote_code=True)
            tokenizer.pad_token_id = 0
        else:
            tokenizer = None

        preprocessor = CLIPImageProcessor.from_pretrained(hfhub_name)

        preprocessor = InputConditioner(1.0,
            norm_mean=preprocessor.image_mean,
            norm_std=preprocessor.image_std,
            dtype=torch.bfloat16,
        )

        model = InternViTWrapper(model, tokenizer)
        info = ModelInfo(model_class='InternViT', model_subtype=version[10:])
    else:
        raise ValueError(f'Unsupported model version: {version}')

    if device is not None:
        model.to(device=device)

    return model, preprocessor, info
