# AM-RADIO: Reduce All Domains Into One

Mike Ranzinger, Greg Heinrich, Jan Kautz, Pavlo Molchanov

[NVIDIA Research](https://www.nvidia.com/en-us/research/)

\[[Paper](TODO)\]\[[BibTex](#citing-radio)\]

## Pretrained Models

Refer to `model_results.csv` for model versions and their metrics.

### HuggingFace Hub

_Coming Soon_

### TorchHub

```Python
import torch

# If you don't supply the `version` parameter, the latest ViT version will be returned.
model = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v1', progress=True)
model.cuda().eval()

x = torch.rand(1, 3, 224, 224, device='cuda')

# NOTE: RADIO models expect the input to have values in the range [0, 1]
# NOTE 2: `radio_v1` is a ViT-H/14 model, and supports inputs in the size range `224 < Height < 1008`
#         and `224 < Width < 1008` where each dimension must be divisible by 14.
#         Non-square inputs are supported.
summary, spatial_features = model(x)

# RADIO also supports running in mixed precision, like so:
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    summary, spatial_features = model(x)
```

### Usage

RADIO will return a tuple with two tensors. The `summary` is similar to the `cls_token` in ViT and is meant to represent the general concept of the entire image. It has shape $(B,C)$ with $B$ being the batch dimension, and $C$ being some number of channels. The `spatial_features` represent more localized content which should be suitable for dense tasks such as semantic segmentation, or for integration into an LLM. It has shape $(B,T,D)$ with $T$ being the flattened spatial tokens, and $D$ being the channels for spatial features. Note that $C \neq D$ in general.

Converting to a spatial tensor format can be done using the downsampling size of the model, combined with the input tensor shape. For 'radio_v1', the patch size is 14.
```Python
from einops import rearrange
spatial_features = rearrange(spatial_features, 'b (h w) d -> b d h w', h=x.shape[-2] // patch_size, w=x.shape[-1] // patch_size)
```

The resulting tensor will have shape $(B,D,H,W)$, as is typically seen with computer vision models.

### RADIOv1 Notes

We have trained this model to be flexible in input dimension. It supports inputs with both width and height in the range $[14, 1008]$ as long as both axes are divisible by 14. We have found that summarization tokens work best at $H=W=378$ (although the range $[192, 448]$ works well). For spatial tasks, we used $H=W=518$ to perform linear probing for semantic segmentation, and may perform better for more high-resolution tasks. Going up to $1008$, the model may need additional fine tuning at that resolution for best results.

It is not required that $H=W$ although we have not specifically trained or testing the model in this setting.


## Training

_Coming Soon_

## License

RADIO code and weights are released under the [NSCLv1 License](LICENSE).

## Citing RADIO

If you find this repository useful, please consider giving a star and citation:
```
@misc{ranzinger2023radio,
  title={AM-RADIO: Agglomerative Model -- Reduce All Domains Into One},
  author={Ranzinger, Mike and Heinrich, Greg and Kautz, Jan and Molchanov, Pavlo},
  journal=TODO,
  year={2023}
}
```
