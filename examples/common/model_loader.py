from dataclasses import dataclass
import os
from types import MethodType
from typing import List, Optional, Tuple

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from common.utils import rank_gate
from radio.adaptor_base import RadioOutput
from radio.input_conditioner import InputConditioner
from radio.siglip2_adaptor import SigLIP2WrappedTokenizer


def dv2_sdpa(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]
    x = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=False,
        dropout_p=self.attn_drop.p if self.training else 0.,
        scale=self.scale,
    )
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class DinoWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module, patch_attn: bool = True):
        super().__init__()
        self.inner = dino_model
        if patch_attn:
            for n, m in self.inner.named_modules():
                if n.endswith('.attn'):
                    m.old_forward = m.forward
                    m.forward = MethodType(dv2_sdpa, m)

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
        if hasattr(clip_model, 'visual'):
            clip_model.visual.output_tokens = True
        self.tokenizer = tokenizer
        self.adaptor_name = adaptor_name

        if not clip_mode and hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'proj'):
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

    def encode_text(self, text, normalize: bool = False, **kwargs):
        try:
            return self.inner.encode_text(text, normalize=normalize)
        except TypeError:
            ret = self.inner.encode_text(text)
            if normalize:
                ret = F.normalize(ret, dim=-1)
            return ret


class SigLIPWrapper(CLIPWrapper):
    def __init__(self, clip_model, tokenizer, adaptor_name, clip_mode = False):
        super().__init__(clip_model, tokenizer, adaptor_name, clip_mode)
        self.vision_encoder.trunk.patch_embed.img_size = (378, 378)

    @property
    def patch_size(self):
        return 14

    def forward(self, *args, **kwargs):
        features = self.inner.visual.trunk.forward_features(*args, **kwargs)
        token = self.inner.visual.trunk.attn_pool(features)
        return self._wrap_output(token, features)

class SigLIP2Wrapper(CLIPWrapper):
    def __init__(self, clip_model, tokenizer, proc, adaptor_name, clip_mode = False, patch_size: int = 16, is_dynamic: bool = True):
        super().__init__(clip_model, tokenizer, adaptor_name, clip_mode)
        self._patch_size = patch_size
        self._proc = proc
        self._is_dynamic = is_dynamic

        self.register_buffer('mask', torch.ones(1, 1, dtype=torch.int32))

    @property
    def patch_size(self):
        return self._patch_size

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out_h = x.shape[-2] // self._patch_size
        out_w = x.shape[-1] // self._patch_size

        extra = dict()

        if self._is_dynamic:
            pixel_values = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                     p1=self._patch_size, p2=self._patch_size,
                                     h=out_h, w=out_w)
            mask = self.mask.expand(*pixel_values.shape[:2])
            shapes = torch.tensor([(out_h, out_w)] * pixel_values.shape[0], dtype=torch.int64, device=x.device)

            extra = dict(attention_mask=mask, spatial_shapes=shapes)
        else:
            pixel_values = x

        output = self.inner.vision_model(pixel_values=pixel_values, return_dict=True, **extra)

        summary = output.pooler_output
        features = output.last_hidden_state

        if kwargs.get('feature_fmt', None) == 'NCHW':
            features = rearrange(features, 'b (h w) c -> b c h w', h=out_h, w=out_w)

        return self._wrap_output(summary, features)

    def encode_text(self, text, normalize: bool = False):
        output = self.inner.text_model(**text, return_dict=True)
        token = output.pooler_output

        if normalize:
            token = F.normalize(token, dim=-1)

        return token

    def zero_shot_postproc(self, logits: torch.Tensor):
        logit_scale, logit_bias = self.inner.logit_scale.to(logits.device), self.inner.logit_bias.to(logits.device)
        logits = logits * logit_scale.exp() + logit_bias
        return logits


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
    checkpoint: Optional[dict] = None


# Because many of these load functions might download a model, `rank_gate` will first allow rank 0 to execute (thus downloading when applicable),
# and once it completes, it allows all other ranks to execute, using the now cached weights.
@rank_gate
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
            model, chk = radio_model(version=version, progress=True, adaptor_names=adaptor_names, return_checkpoint=True, **kwargs)
        else:
            model, chk = torch.hub.load(torchhub_repo, 'radio_model', version=version, progress=True,
                                              adaptor_names=adaptor_names, return_spatial_features=return_spatial_features,
                                              force_reload=force_reload,
                                              return_checkpoint=True, **kwargs,
            )

        preprocessor = model.make_preprocessor_external()
        info = ModelInfo(model_class='RADIO', model_subtype=version.replace('/', '_'), checkpoint=chk)
    elif version.startswith('dinov2'):
        model = torch.hub.load('facebookresearch/dinov2', version, pretrained=True, force_reload=force_reload, **kwargs)
        model = DinoWrapper(model)

        preprocessor = InputConditioner(1.0, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        info = ModelInfo(model_class='DINOv2', model_subtype=version.replace('dinov2_', ''))
    elif version.startswith('dinov3'):
        _, chk_path = version.split(',')
        fname = os.path.basename(chk_path)
        model_version = fname[:fname.index('_pretrain')]
        model: nn.Module = torch.hub.load('facebookresearch/dinov3', model_version, pretrained=False)

        chk = torch.load(chk_path, map_location='cpu')
        model.load_state_dict(chk, strict=True)
        model = DinoWrapper(model, patch_attn=False)

        preprocessor = InputConditioner(1.0, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        info = ModelInfo(model_class='DINOv3', model_subtype=model_version.replace('dinov3_', ''))
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
    elif version.startswith('siglip2'):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        from transformers.image_utils import load_image
        version_map = {
            'siglip2-so400m-512': ('google/siglip2-so400m-patch16-512', False),
            'siglip2-so400m': ('google/siglip2-so400m-patch16-naflex', True),
            'siglip2-g': ('google/siglip2-giant-opt-patch16-384', False),
        }
        version_map['siglip2'] = version_map['siglip2-so400m']

        version, is_dynamic = version_map[version]

        model = AutoModel.from_pretrained(version, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(version, trust_remote_code=True)

        img_proc = proc.image_processor
        preprocessor = InputConditioner(1.0,
            norm_mean=img_proc.image_mean,
            norm_std=img_proc.image_std,
        )

        tokenizer = SigLIP2WrappedTokenizer(proc)

        model = SigLIP2Wrapper(model, tokenizer, proc, adaptor_names, clip_mode='clip' in adaptor_names if adaptor_names else False, is_dynamic=is_dynamic)
        info = ModelInfo(model_class='SigLIP2', model_subtype=version)
    else:
        raise ValueError(f'Unsupported model version: {version}')

    if device is not None:
        model.to(device=device)

    return model, preprocessor, info
