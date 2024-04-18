from logging import getLogger
import math
import os
from typing import Union, Tuple
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _SpectralNorm

from timm.models.vision_transformer import Attention, Mlp

_EPS = 1e-5


class _SNReweight(_SpectralNorm):
    def __init__(self, weight: torch.Tensor, *args, init_norm_to_current: bool = False, alpha: float = 0.05, version: int = 2, **kwargs):
        super().__init__(weight, *args, **kwargs)

        self.alpha = alpha
        self.version = version
        self.register_buffer('_sn_version', torch.tensor(version))

        if init_norm_to_current:
            # This will set the numerator to match the denominator, which should preserve the original values
            init_scale = self._get_sigma(weight).item()
        else:
            init_scale = 1.0

        if version == 1:
            init_value = init_scale
        elif version == 2:
            t = init_scale - alpha
            if t < _EPS:
                getLogger("spectral_reparam").warn(f'The initialized spectral norm {init_scale} is too small to be represented. Setting to {_EPS} instead.')
                t = _EPS

            init_value = math.log(math.exp(t) - 1)
        else:
            raise ValueError(f'Unsupported version: {version}')

        # Make 2D so that weight decay gets applied
        self.scale = nn.Parameter(torch.tensor([[init_value]], dtype=torch.float32, device=weight.device))

    # Re-implementing this because we need to make division by sigma safe
    def _get_sigma(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            sigma = weight.norm()
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))

        return sigma + self.eps

    def forward(self, weight: torch.Tensor, *args, **kwargs):
        dtype = weight.dtype
        sigma = self._get_sigma(weight, *args, **kwargs)

        if self.version == 1:
            scale = self.scale
        elif self.version == 2:
            scale = F.softplus(self.scale) + self.alpha
        else:
            raise ValueError(f'Unsupported version: {self.version}')

        scale = scale.float() / sigma.float()

        y = weight * scale

        if dtype in (torch.float16, torch.bfloat16):
            y = y.to(dtype)
        return y

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version_key = f'{prefix}_sn_version'
        if version_key not in state_dict:
            self.version = 1
            state_dict[version_key] = torch.tensor(1)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class _AttnSNReweight(nn.Module):
    def __init__(self, weight: torch.Tensor, *args, init_norm_to_current: bool = False, renorm_values: bool = False, **kwargs):
        super().__init__()

        parts = weight.split(weight.shape[0] // 3, dim=0)

        ct = 2 if not renorm_values else 3

        self.parts = nn.ModuleList([
            _SNReweight(p, *args, init_norm_to_current=init_norm_to_current, **kwargs) if i < ct else nn.Identity()
            for i, p in enumerate(parts)
        ])

    def forward(self, weight: torch.Tensor, *args, **kwargs):
        parts = weight.split(weight.shape[0] // 3, dim=0)

        parts = [
            fn(p)
            for fn, p in zip(self.parts, parts)
        ]

        return torch.cat(parts, dim=0)


def enable_spectral_reparam(model: nn.Module,
                            n_power_iterations: int = 1,
                            eps: float = 1e-6,
                            init_norm_to_current: bool = False,
                            renorm_values: bool = True,
                            renorm_mlp: bool = True):
    # print('Enabling spectral reparametrization')
    for mod in model.modules():
        if isinstance(mod, Attention):
            parametrize.register_parametrization(
                mod.qkv,
                'weight',
                _AttnSNReweight(mod.qkv.weight, n_power_iterations, dim=0, eps=eps, init_norm_to_current=init_norm_to_current, renorm_values=renorm_values),
            )
            pass
        elif isinstance(mod, Mlp) and renorm_mlp:
            parametrize.register_parametrization(
                mod.fc1,
                'weight',
                _SNReweight(mod.fc1.weight, n_power_iterations, dim=0, eps=eps, init_norm_to_current=init_norm_to_current),
            )
            parametrize.register_parametrization(
                mod.fc2,
                'weight',
                _SNReweight(mod.fc2.weight, n_power_iterations, dim=0, eps=eps, init_norm_to_current=init_norm_to_current),
            )
            pass


def configure_spectral_reparam_from_args(model: nn.Module, args):
    spectral_reparam = getattr(args, 'spectral_reparam', False)
    if isinstance(spectral_reparam, bool) and spectral_reparam:
        enable_spectral_reparam(model, init_norm_to_current=args.pretrained)
    elif isinstance(spectral_reparam, dict):
        enable_spectral_reparam(
            model,
            n_power_iterations=spectral_reparam.get('n_power_iterations', 1),
            eps=spectral_reparam.get('eps', 1e-12),
            init_norm_to_current=args.pretrained,
        )


def disable_spectral_reparam(model: nn.Module):
    for mod in model.modules():
        if isinstance(mod, Attention):
            parametrize.remove_parametrizations(mod.qkv, 'weight')
            pass
        elif isinstance(mod, Mlp):
            parametrize.remove_parametrizations(mod.fc1, 'weight')
            parametrize.remove_parametrizations(mod.fc2, 'weight')
            pass


if __name__ == '__main__':
    import argparse
    from . import radio_model as create_model

    parser = argparse.ArgumentParser(description='Remove parametrization from state dict')
    parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to load')
    parser.add_argument('--output', type=str, default='', help='Where to store the checkpoint')
    parser.add_argument('--release', default=False, action='store_true', help='Prune extraneous checkpoint fields')
    parser.add_argument('--strict', default=False, action='store_true', help='Strictly load the state dict')

    args = parser.parse_args()

    if not args.output:
        chk_dir, chk_name = os.path.split(args.checkpoint)
        args.output = os.path.join(chk_dir, f'clean_{chk_name}')
        print(f'Set output to "{args.output}"')

    chk = torch.load(args.checkpoint, map_location='cpu', mmap=True)

    model = create_model.create_model_from_args(chk['args'])

    key = 'base_model.'
    mod_state = dict()
    extra_state = dict()
    for k, v in chk['state_dict'].items():
        if k.startswith(key):
            mod_state[k[len(key):]] = v
        else:
            extra_state[k] = v

    chk_load_info = model.load_state_dict(mod_state, strict=args.strict)
    if chk_load_info.unexpected_keys or chk_load_info.missing_keys:
        print(chk_load_info)

    if chk['args'].spectral_reparam:
        disable_spectral_reparam(model)

    if hasattr(chk['args'], 'dtype'):
        model.to(dtype=chk['args'].dtype)

    mod_state = model.state_dict()
    final_state = dict()
    final_state.update({f'{key}{k}': v for k, v in mod_state.items()})
    final_state.update(extra_state)

    chk['state_dict'] = final_state
    chk['args'].spectral_reparam = False

    if args.release:
        chk = {
            'arch': chk['arch'],
            'epoch': chk['epoch'],
            'state_dict': chk['state_dict'],
            'args': chk['args'],
        }

    torch.save(chk, args.output)
    pass
