# NaViT `forward_intermediates` support

## Motivation

`RadioModel.forward_intermediates` (radio/radio_model.py:273) delegates to
`self.model.forward_intermediates(...)`. The existing implementations are:

- timm `VisionTransformer` (CPE-enabled): wrapper in `radio/enable_cpe_support.py`
- `radio.vision_transformer_xpos.VisionTransformer`: native method
- `extra_models.DinoWrapper`: wrapper

NaViT (`radio/vision_transformer_navit.NaViT`) has no such method, so loading a
RADIO checkpoint backed by NaViT and calling `forward_intermediates` raises:

```
AttributeError: 'NaViT' object has no attribute 'forward_intermediates'
```

This spec adds that method.

## Scope

Full parity with `NaViT.forward()`: batched and packed inputs, ViTDet windowed
attention, matryoshka truncation.

Simplification: only the packed (varlen) algorithm is implemented. Batched input
is handled by splitting along the batch dim into a list, calling the packed
path, and restacking. `forward_intermediates` is a niche debugging/analysis
interface, so the small overhead of going through the list path is acceptable.

## Public API

Added to `NaViT`:

```python
def forward_intermediates(
    self,
    x: Union[Tensor, List[Tensor]],
    indices: Optional[Union[int, List[int], Tuple[int]]] = None,
    return_prefix_tokens: bool = False,
    norm: bool = False,
    stop_early: bool = False,
    output_fmt: str = 'NCHW',
    intermediates_only: bool = False,
    aggregation: Optional[str] = 'sparse',
    inter_feature_normalizer: Optional[IntermediateFeatureNormalizerBase] = None,
    norm_alpha_scheme: str = 'post-alpha',
) -> Union[List[...], Tuple[Tuple[Tensor, ...], List[...]]]
```

Signature matches the kwargs already forwarded by
`RadioModel.forward_intermediates` (radio/radio_model.py:302-313).

## Output structure

| Input type | `intermediates` shape | Final |
|---|---|---|
| `Tensor` `(B, C, H, W)` | `List[Tensor]`, each `(B, C, H, W)` (NCHW) or `(B, N, C)` (NLC) | `(summary, features)` matching `_forward_batched`'s return |
| `List[Tensor]` (variable size) | `List[List[Tensor]]`, outer=layers, inner=images. Each per-image entry `(1, C, H_i, W_i)` (NCHW) or `(N_i, C)` (NLC) | `(summary, features_list)` matching `_forward_packed`'s return |

`return_prefix_tokens=True` wraps each entry as `(prefix, feat)`. In packed mode
that wrapping applies to each per-image entry.

`intermediates_only=True` returns only `intermediates`.

## Implementation

### Dispatch

```python
def forward_intermediates(self, x, ...):
    is_batched = not isinstance(x, (list, tuple))
    images = list(x) if is_batched else list(x)
    result = self._forward_intermediates_packed(images, ...)
    if is_batched:
        result = _restack_to_batched(result, ...)
    return result
```

For `is_batched=True`, splitting `(B, C, H, W)` along dim 0 yields a list of
`(C, H, W)` tensors — the format `NaViTPatchEmbed.forward_list` expects. All
images share `(H, W)` so the per-layer per-image list can be `torch.stack`-ed
back into `(B, C, H, W)`.

### `_forward_intermediates_packed`

Mirrors `_forward_packed` (radio/vision_transformer_navit.py:1413-1554) setup:

1. `patch_embed(images)` → flat `(total_patches, embed_dim)` + per-image grid sizes.
2. Per-image: prepend `cls_token` if `num_prefix_tokens > 0`; if `vitdet_config`
   is set, reorder spatial tokens by window and build `vitdet_cu_seqlens`,
   `vitdet_full_patch_order`.
3. Build `position_ids` via `_create_position_ids`. If ViTDet, gather by
   `vitdet_full_patch_order`.
4. Build `cu_seqlens` (pinned host → device, async copy, same as line 1492-1494).
5. `attn_info` dict (and `attn_info_vitdet` if ViTDet).

Then the block loop captures intermediates:

```python
take_indices, max_index = _take_indices(len(self.blocks), indices)
take_indices = sorted(take_indices)
blocks = self.blocks[:max_index + 1] if stop_early else self.blocks
take_off = 0
accumulator = 0
alpha_sum = 0
num_accumulated = 0
inter_norm = inter_feature_normalizer or NullIntermediateFeatureNormalizer.get_instance(x.dtype, x.device)
post_alpha = (norm_alpha_scheme == 'post-alpha')
intermediates = []

for i, block in enumerate(blocks):
    if self.vitdet_config is not None and i not in self.vitdet_config.global_attn_indices:
        x = block(x, position_ids=position_ids, attn_info=attn_info_vitdet)
    else:
        x = block(x, position_ids=position_ids, attn_info=attn_info)

    if aggregation == 'dense':
        y, alpha = inter_norm(x, i, rot_index=take_indices[take_off], skip=0)
        if post_alpha:
            accumulator = accumulator + y
            alpha_sum = alpha_sum + alpha
        else:
            accumulator = accumulator + alpha * y
            alpha_sum += 1
        num_accumulated += 1

    if i == take_indices[take_off]:
        if aggregation == 'dense':
            alpha = alpha_sum / num_accumulated
            x_ = alpha * accumulator / num_accumulated
            num_accumulated = 0; accumulator = 0; alpha_sum = 0
        else:
            y, alpha = inter_norm(x, i, skip=0)
            x_ = alpha * y
        intermediates.append(self.norm_post(x_) if norm else x_)
        take_off = min(take_off + 1, len(take_indices) - 1)
```

Note `skip=0` here vs `skip=num_summary_tokens` in the generic helper: NaViT's
packed layout has prefix tokens interleaved per image, so a single skip prefix
on the flat sequence doesn't make sense. `inter_feature_normalizer` is only
expected to be useful with `num_prefix_tokens=0` checkpoints in packed mode; if
a caller passes a normalizer with `num_prefix_tokens>0`, the rotation is applied
uniformly across all tokens including prefix tokens. This matches the
limitation noted in the design review and is acceptable for the use case.

After the loop:

1. Apply `norm_post` to the final `x`. Apply matryoshka truncation
   (`_sample_matryoshka_dim` once, applied to both final and intermediates).
2. If ViTDet, `torch.scatter` un-shuffle `x` and every intermediate back to
   the original packed order (same logic as line 1518-1520).
3. Per image, slice via `cu_seqlens` to split prefix tokens from spatial
   features. For each intermediate, build `prefix_list[layer][img]` and
   `feature_list[layer][img]`.
4. Reshape spatial features per image to `(1, C, H_i, W_i)` for NCHW, or
   `(N_i, C)` for NLC.
5. Compute final summary: `attn_pool(x, ...)` for `pool_type='attn'`,
   `torch.stack(cls_tokens_list)` for `pool_type='cls_token'`.

### `_restack_to_batched`

Used when input was `(B, C, H, W)`:

- For each layer, `torch.stack(per_image_list, dim=0)` → `(B, C, H, W)`.
- For final features, same stack → `(B, C, H, W)`.
- Final summary is already a stacked tensor (or `(B, num_tokens, C)` for cls_token).
- `return_prefix_tokens=True` case: stack prefix tensors too.

### Output format `NLC`

`output_fmt='NLC'` skips the `(1, C, H_i, W_i)` reshape and returns flat
`(N_i, C)` per image (packed) or `(B, N, C)` (batched, stacked).

## Files touched

- `radio/vision_transformer_navit.py`: add `forward_intermediates` method plus
  `_forward_intermediates_packed` and `_restack_to_batched` helpers. The
  `_take_indices` helper is small enough to copy locally (or import from
  `radio/forward_intermediates.py`).

No other files change. `RadioModel.forward_intermediates` (radio_model.py:273)
already routes through `self.model.forward_intermediates`; once the method
exists on NaViT, the chain works.

## Non-goals

- `pool_type='attn'`: intermediates do not include a per-layer attention-pooled
  summary. Only the final summary is computed.
- A separate fast path for batched uniform input — the user explicitly accepted
  the niche-interface overhead.
- Modifying the generic `radio/forward_intermediates.py` helper. Existing
  consumers (xpos, CPE-enabled timm ViT, extra_models) stay untouched.

## Risks

1. **`inter_feature_normalizer` + `num_prefix_tokens > 0` in packed mode**:
   rotation will be applied to prefix tokens too. Documented limitation.
2. **`norm_post` with `elementwise_affine=False`** (line 1363): applies fine to
   intermediates. Worth sanity-checking on a real checkpoint before merging.
3. **Memory**: storing L intermediates of shape `(total_tokens, C)` plus the
   un-shuffle scatters in ViTDet mode roughly doubles activation memory during
   the call. Caller's responsibility; matches behavior of the generic helper.

## Verification plan

- Unit: call `model.forward_intermediates(x)` on a `NaViT` instance with both
  `Tensor` and `List[Tensor]` inputs, with and without `vitdet_config`, with
  `num_prefix_tokens` in `{0, 1, 5}`, with `pool_type` in
  `{'cls_token', 'attn'}`. Assert the final output equals `model(x)` and the
  intermediate count matches the requested indices.
- Integration: load the failing MLoRE/RADIO checkpoint and re-run the train
  step that produced the original `AttributeError`.
