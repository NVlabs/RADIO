"""End-to-end smoke test mimicking MLoRE's RADIO wrapper call pattern.

MLoRE calls (under autocast bf16):
    intermediates = radio_model.forward_intermediates(
        x, indices=select_list, intermediates_only=True,
    )
    intermediates = torch.cat(intermediates, dim=1)

This test builds a synthetic RADIOModel(NaViT, ...), runs the same call, and
verifies the chain works and produces the cat-able output MLoRE expects.
"""
import pytest
import torch

from radio.radio_model import RADIOModel, Resolution
from radio.input_conditioner import InputConditioner
from radio.vision_transformer_navit import create_navit


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NaViT requires CUDA"
)


def _build_radio_with_navit(**navit_kwargs) -> RADIOModel:
    torch.manual_seed(0)
    navit = create_navit(
        variant='tiny',
        patch_size=14,
        num_cls_tokens=1,
        pool_type='cls_token',
        init_values=1e-5,
        **navit_kwargs,
    )
    conditioner = InputConditioner(input_scale=1.0, norm_mean=0.0, norm_std=1.0)
    return RADIOModel(
        model=navit,
        input_conditioner=conditioner,
        patch_size=navit.patch_size,
        max_resolution=14 * 32,
        preferred_resolution=Resolution(14 * 8, 14 * 8),
    ).cuda().eval()


@cuda_only
def test_mlore_call_pattern_intermediates_only():
    """Replay MLoRE's exact call shape: indices list + intermediates_only=True + cat dim=1."""
    radio = _build_radio_with_navit()
    num_blocks = len(radio.model.blocks)
    skip = num_blocks // 4
    select_list = list(range(skip - 1, num_blocks, skip))

    B, H, W = 2, 14 * 8, 14 * 8
    x = torch.randn(B, 3, H, W, device='cuda')

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        intermediates = radio.forward_intermediates(
            x, indices=select_list, intermediates_only=True,
        )

    assert isinstance(intermediates, list), f"expected list, got {type(intermediates)}"
    assert len(intermediates) == len(select_list)

    for i, t in enumerate(intermediates):
        assert torch.is_tensor(t), f"intermediate {i} is {type(t)}, expected Tensor"
        assert t.ndim == 4, f"intermediate {i} ndim {t.ndim}, expected 4 (NCHW)"
        assert t.shape[0] == B
        assert t.shape[2:] == (H // 14, W // 14), \
            f"intermediate {i} spatial {tuple(t.shape[2:])} != {(H // 14, W // 14)}"

    # MLoRE's downstream op:
    cat = torch.cat(intermediates, dim=1)
    assert cat.shape == (B, radio.model.embed_dim * len(select_list), H // 14, W // 14)


@cuda_only
def test_mlore_call_pattern_with_vitdet():
    """Same call pattern with a vitdet-enabled NaViT (the path that routes
    main forward through _forward_packed too)."""
    radio = _build_radio_with_navit(vitdet_config={'window_size': 4, 'num_global': 4})
    num_blocks = len(radio.model.blocks)
    skip = num_blocks // 4
    select_list = list(range(skip - 1, num_blocks, skip))

    B, H, W = 2, 14 * 8, 14 * 8
    x = torch.randn(B, 3, H, W, device='cuda')

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        intermediates = radio.forward_intermediates(
            x, indices=select_list, intermediates_only=True,
        )
        cat = torch.cat(intermediates, dim=1)

    assert cat.shape == (B, radio.model.embed_dim * len(select_list), H // 14, W // 14)


@cuda_only
def test_full_return_with_radio_extract_final():
    """Verify intermediates_only=False also works: RADIOModel runs _extract_final on the
    returned (summary, features) tuple."""
    radio = _build_radio_with_navit()
    x = torch.randn(2, 3, 14 * 4, 14 * 4, device='cuda')

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        final, intermediates = radio.forward_intermediates(
            x, indices=[5, 11], output_fmt='NCHW',
        )

    # final is whatever _extract_final returns — a RadioOutput (summary, features)
    assert hasattr(final, 'summary') and hasattr(final, 'features')
    assert final.summary.ndim == 2  # (B, C_summary) after flatten
    assert final.features.ndim == 4  # (B, C, H, W)
    assert len(intermediates) == 2


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
