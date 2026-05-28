"""Unit tests for NaViT.forward_intermediates.

The core correctness check uses forward hooks on every block: we run the
standard `forward()` path, capture each block's output, then run
`forward_intermediates(...)` and verify it returns exactly those captured
values at the requested layer indices (after applying the documented
post-processing: optional `norm_post`, prefix-token strip, output_fmt reshape).
This proves the intermediates the method exposes are the *same* tensors that
flow through the main forward path, not a recomputation.
"""
from typing import List, Optional, Tuple

import pytest
import torch

from radio.vision_transformer_navit import create_navit, NaViT


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NaViT requires CUDA (flash attention / autocast)"
)


def _autocast():
    return torch.amp.autocast('cuda', dtype=torch.bfloat16)


def _build(
    num_cls_tokens: int = 1,
    pool_type: str = 'cls_token',
    vitdet: bool = False,
    patch_size: int = 14,
    **kw,
) -> NaViT:
    torch.manual_seed(0)
    cfg = dict(
        num_cls_tokens=num_cls_tokens,
        pool_type=pool_type,
        patch_size=patch_size,
        init_values=1e-5,
    )
    cfg.update(kw)
    if vitdet:
        cfg['vitdet_config'] = {'window_size': 4, 'num_global': 4}
    return create_navit(variant='tiny', **cfg).cuda().eval()


def _capture_block_outputs(model: NaViT, run):
    """Run `run()` while hooking every block; return per-block outputs and run() return."""
    captures: List[Optional[torch.Tensor]] = [None] * len(model.blocks)
    handles = []

    def make_hook(i):
        def hook(_mod, _inp, out):
            captures[i] = out
        return hook

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_hook(i)))
    try:
        result = run()
    finally:
        for h in handles:
            h.remove()
    return captures, result


def _close(a: torch.Tensor, b: torch.Tensor, atol: float = 5e-3, rtol: float = 5e-3) -> bool:
    return torch.allclose(a, b, atol=atol, rtol=rtol)


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _post_process_batched_hook(
    hook_out: torch.Tensor,
    model: NaViT,
    norm: bool,
    output_fmt: str,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Convert a `(B, N+prefix, C)` block output into the format `forward_intermediates`
    is expected to return for the given output_fmt."""
    x = model.norm_post(hook_out) if norm else hook_out
    x = x[:, model.num_prefix_tokens:]
    if output_fmt == 'NCHW':
        B, _, C = x.shape
        x = x.reshape(B, grid_h, grid_w, C).permute(0, 3, 1, 2).contiguous()
    return x


def _post_process_packed_hook(
    hook_out: torch.Tensor,
    model: NaViT,
    norm: bool,
    output_fmt: str,
    grid_sizes: List[Tuple[int, int]],
    cu_seqlens: List[int],
) -> List[torch.Tensor]:
    """Convert a flat `(total_tokens, C)` block output into the per-image list that
    `forward_intermediates` is expected to return."""
    x = model.norm_post(hook_out) if norm else hook_out
    out: List[torch.Tensor] = []
    for i, (gh, gw) in enumerate(grid_sizes):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        curr = x[start + model.num_prefix_tokens : end]
        if output_fmt == 'NCHW':
            curr = curr.reshape(gh, gw, -1).permute(2, 0, 1).contiguous()
        out.append(curr)
    return out


def _packed_cu_seqlens(model: NaViT, grid_sizes: List[Tuple[int, int]]) -> List[int]:
    cu = [0]
    for gh, gw in grid_sizes:
        cu.append(cu[-1] + gh * gw + model.num_prefix_tokens)
    return cu


# ----- hook-based intermediate verification -----------------------------------

@cuda_only
@pytest.mark.parametrize('pool_type,num_cls_tokens', [('cls_token', 1), ('attn_qknorm', 4)])
@pytest.mark.parametrize('output_fmt', ['NLC', 'NCHW'])
@pytest.mark.parametrize('norm', [False, True])
def test_intermediates_equal_main_path_batched(pool_type, num_cls_tokens, output_fmt, norm):
    model = _build(num_cls_tokens=num_cls_tokens, pool_type=pool_type)
    H = W = 14 * 4
    x = torch.randn(2, 3, H, W, device='cuda')

    def run_main():
        with torch.inference_mode(), _autocast():
            return model(x)

    captures, _ = _capture_block_outputs(model, run_main)
    assert all(c is not None for c in captures)
    assert all(c.ndim == 3 for c in captures), "batched main path should yield 3D block outputs"

    indices = [2, 5, 11]
    with torch.inference_mode(), _autocast():
        intermediates = model.forward_intermediates(
            x, indices=indices, norm=norm, output_fmt=output_fmt, intermediates_only=True,
        )

    assert len(intermediates) == len(indices)
    for layer_idx, block_i in enumerate(indices):
        expected = _post_process_batched_hook(captures[block_i], model, norm, output_fmt, 4, 4)
        actual = intermediates[layer_idx]
        assert expected.shape == actual.shape, \
            f"layer {block_i}: shape {tuple(expected.shape)} vs {tuple(actual.shape)}"
        assert _close(expected, actual), \
            f"layer {block_i} ({pool_type}, fmt={output_fmt}, norm={norm}): max diff {_max_diff(expected, actual):.4e}"


@cuda_only
@pytest.mark.parametrize('output_fmt', ['NLC', 'NCHW'])
@pytest.mark.parametrize('norm', [False, True])
def test_intermediates_equal_main_path_packed(output_fmt, norm):
    model = _build(num_cls_tokens=1, pool_type='cls_token')
    images = [
        torch.randn(3, 14 * 3, 14 * 5, device='cuda'),
        torch.randn(3, 14 * 4, 14 * 4, device='cuda'),
        torch.randn(3, 14 * 2, 14 * 6, device='cuda'),
    ]
    grid_sizes = [(3, 5), (4, 4), (2, 6)]
    cu_seqlens = _packed_cu_seqlens(model, grid_sizes)

    def run_main():
        with torch.inference_mode(), _autocast():
            return model(images)

    captures, _ = _capture_block_outputs(model, run_main)
    assert all(c is not None for c in captures)
    assert all(c.ndim == 2 for c in captures), "packed main path should yield 2D flat block outputs"

    indices = [0, 4, 8, 11]
    with torch.inference_mode(), _autocast():
        intermediates = model.forward_intermediates(
            images, indices=indices, norm=norm, output_fmt=output_fmt, intermediates_only=True,
        )

    assert len(intermediates) == len(indices)
    for layer_idx, block_i in enumerate(indices):
        expected_per_image = _post_process_packed_hook(
            captures[block_i], model, norm, output_fmt, grid_sizes, cu_seqlens,
        )
        actual_per_image = intermediates[layer_idx]
        assert len(expected_per_image) == len(actual_per_image)
        for img_idx, (e, a) in enumerate(zip(expected_per_image, actual_per_image)):
            assert e.shape == a.shape, \
                f"layer {block_i} img {img_idx}: shape {tuple(e.shape)} vs {tuple(a.shape)}"
            assert _close(e, a), \
                f"layer {block_i} img {img_idx} (fmt={output_fmt}, norm={norm}): max diff {_max_diff(e, a):.4e}"


# ----- final output parity with forward() -------------------------------------

@cuda_only
@pytest.mark.parametrize('pool_type,num_cls_tokens', [('cls_token', 1), ('attn_qknorm', 4)])
@pytest.mark.parametrize('vitdet', [False, True])
def test_final_matches_forward_batched(pool_type, num_cls_tokens, vitdet):
    model = _build(num_cls_tokens=num_cls_tokens, pool_type=pool_type, vitdet=vitdet)
    side = 14 * 8 if vitdet else 14 * 4
    x = torch.randn(2, 3, side, side, device='cuda')
    with torch.inference_mode(), _autocast():
        ref_summary, ref_features = model(x)
        (final, _) = model.forward_intermediates(x, indices=[11], output_fmt='NLC')
        summary, features = final

    assert summary.shape == ref_summary.shape
    assert features.shape == ref_features.shape
    assert _close(summary, ref_summary), f"summary max diff {_max_diff(summary, ref_summary):.4e}"
    assert _close(features, ref_features), f"features max diff {_max_diff(features, ref_features):.4e}"


@cuda_only
@pytest.mark.parametrize('vitdet', [False, True])
def test_final_matches_forward_packed(vitdet):
    model = _build(num_cls_tokens=1, pool_type='cls_token', vitdet=vitdet)
    images = [
        torch.randn(3, 14 * 4, 14 * 8, device='cuda'),
        torch.randn(3, 14 * 8, 14 * 4, device='cuda'),
    ]
    with torch.inference_mode(), _autocast():
        ref_summary, ref_feats_list = model(images)
        (final, _) = model.forward_intermediates(images, indices=[11], output_fmt='NLC')
        summary, features_list = final

    assert _close(summary, ref_summary), f"summary max diff {_max_diff(summary, ref_summary):.4e}"
    assert len(features_list) == len(ref_feats_list)
    for i, (f, rf) in enumerate(zip(features_list, ref_feats_list)):
        assert f.shape == rf.shape
        assert _close(f, rf), f"img {i} features max diff {_max_diff(f, rf):.4e}"


# ----- shape / contract tests -------------------------------------------------

@cuda_only
def test_return_prefix_tokens_batched():
    model = _build(num_cls_tokens=1, pool_type='cls_token')
    x = torch.randn(2, 3, 14 * 4, 14 * 4, device='cuda')
    with torch.inference_mode(), _autocast():
        out = model.forward_intermediates(
            x, indices=[5, 11], return_prefix_tokens=True, output_fmt='NCHW',
            intermediates_only=True,
        )
    assert len(out) == 2
    for prefix, feat in out:
        assert prefix.shape == (2, 1, model.embed_dim)
        assert feat.shape == (2, model.embed_dim, 4, 4)


@cuda_only
def test_return_prefix_tokens_packed():
    model = _build(num_cls_tokens=1, pool_type='cls_token')
    images = [
        torch.randn(3, 14 * 3, 14 * 5, device='cuda'),
        torch.randn(3, 14 * 4, 14 * 4, device='cuda'),
    ]
    with torch.inference_mode(), _autocast():
        out = model.forward_intermediates(
            images, indices=[11], return_prefix_tokens=True, output_fmt='NCHW',
            intermediates_only=True,
        )
    assert len(out) == 1
    layer = out[0]
    assert len(layer) == 2
    grids = [(3, 5), (4, 4)]
    for img_idx, (prefix, feat) in enumerate(layer):
        gh, gw = grids[img_idx]
        assert prefix.shape == (1, model.embed_dim)
        assert feat.shape == (model.embed_dim, gh, gw)


@cuda_only
@pytest.mark.parametrize('output_fmt', ['NLC', 'NCHW'])
def test_packed_per_image_shapes_no_batch_dim(output_fmt):
    """Per-image NCHW = (C, H, W); per-image NLC = (N, C). No leading batch dim either way."""
    model = _build(num_cls_tokens=1, pool_type='cls_token')
    images = [
        torch.randn(3, 14 * 3, 14 * 5, device='cuda'),
        torch.randn(3, 14 * 4, 14 * 4, device='cuda'),
    ]
    grids = [(3, 5), (4, 4)]
    with torch.inference_mode(), _autocast():
        out = model.forward_intermediates(
            images, indices=[11], output_fmt=output_fmt, intermediates_only=True,
        )
    for img_idx, (gh, gw) in enumerate(grids):
        t = out[0][img_idx]
        if output_fmt == 'NCHW':
            assert t.shape == (model.embed_dim, gh, gw)
        else:
            assert t.shape == (gh * gw, model.embed_dim)


@cuda_only
def test_indices_as_int_selects_last_n():
    """`indices=n` selects the last n blocks, matching the generic helper convention."""
    model = _build(num_cls_tokens=1, pool_type='cls_token')
    x = torch.randn(2, 3, 14 * 4, 14 * 4, device='cuda')
    with torch.inference_mode(), _autocast():
        out_int = model.forward_intermediates(x, indices=3, output_fmt='NLC', intermediates_only=True)
        out_list = model.forward_intermediates(
            x, indices=list(range(model.depth - 3, model.depth)), output_fmt='NLC',
            intermediates_only=True,
        )
    assert len(out_int) == len(out_list) == 3
    for a, b in zip(out_int, out_list):
        assert torch.equal(a, b)


@cuda_only
def test_stop_early_runs_only_through_max_index():
    """With `stop_early=True`, blocks past max(indices) shouldn't run — verify via a hook."""
    model = _build(num_cls_tokens=1, pool_type='cls_token')
    x = torch.randn(2, 3, 14 * 4, 14 * 4, device='cuda')

    max_idx = 5
    ran = [False] * len(model.blocks)

    def make_hook(i):
        def hook(_mod, _inp, _out):
            ran[i] = True
        return hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(model.blocks)]
    try:
        with torch.inference_mode(), _autocast():
            model.forward_intermediates(
                x, indices=[max_idx], stop_early=True, intermediates_only=True,
            )
    finally:
        for h in handles:
            h.remove()
    assert all(ran[: max_idx + 1])
    assert not any(ran[max_idx + 1 :])


@cuda_only
def test_returns_final_when_not_intermediates_only():
    model = _build(num_cls_tokens=1, pool_type='cls_token')
    x = torch.randn(2, 3, 14 * 4, 14 * 4, device='cuda')
    with torch.inference_mode(), _autocast():
        result = model.forward_intermediates(x, indices=[11])
    assert isinstance(result, tuple) and len(result) == 2
    final, intermediates = result
    assert isinstance(final, tuple) and len(final) == 2
    assert isinstance(intermediates, list)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
