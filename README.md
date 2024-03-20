# AM-RADIO: Reduce All Domains Into One

Mike Ranzinger, Greg Heinrich, Jan Kautz, Pavlo Molchanov

[NVIDIA Research](https://www.nvidia.com/en-us/research/)

\[[Paper](https://arxiv.org/abs/2312.06709)\]\[[BibTex](#citing-radio)\]

## Pretrained Models

| Name       | Architecture | Precision | Teachers                                 | Throughput | Zero Shot Top-1 | kNN Top-1 | ADE20k    | VOC       | GQA       | TextVQA   | VQAv2     | SAM-COCO  |
|------------|--------------|-----------|------------------------------------------|------------|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| radio_v2.1 | ViT-H/16-CPE | BFloat16  | DFN CLIP; OpenAI CLIP; DINOv2; SAM       | 556        | **82.93**       | **86.06** | **51.34** | 84.71     | **63.01** | 56.32     | **79.28** | **76.58** |
| radio_v2   | ViT-H/16-CPE | Float32   | DFN CLIP; OpenAI CLIP; DINOv2; SAM       | 556        | 82.71           | 85.92     | 51.33     |           | 62.78     | **56.37** | 79.00     | 76.21     |
| radio_v1   | ViT-H/14-CPE | Float32   | DFN CLIP; OpenAI CLIP; DINOv2            | 556        | 82.73           | 85.29     | 50.32     | **85.17** | 61.43     | 54.92     | 77.88     |           |
| eradio_v1  | E-RADIO      | Float32   | Meta CLIP; DINOv2                        | 3697       | 77.87           | 83.73     | 45.50     | 79.95     | 59.55     | 46.31     | 72.05     |           |


### TorchHub

```Python
import torch

# If you don't supply the `version` parameter, the latest ViT version will be returned.
model = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2', progress=True)
model.cuda().eval()

x = torch.rand(1, 3, 224, 224, device='cuda')

# NOTE: RADIO models expect the input to have values in the range [0, 1]
# NOTE 2: `radio_v1` is a ViT-H/14 model, and supports inputs in the size range `224 < dim < 1008`
#           where each dimension must be divisible by 14.
#           Non-square inputs are supported.
# NOTE 3: `radio_v2` is a ViT-H/16 model, and supports inputs in the size range `224 < dim < 2048`
#           where each dimension must be divisible by 16.
summary, spatial_features = model(x)

# RADIO also supports running in mixed precision, like so:
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    summary, spatial_features = model(x)
```

### Usage

RADIO and E-RADIO will return a tuple with two tensors.
The `summary` is similar to the `cls_token` in ViT and is meant to represent the general concept of the entire image.
It has shape $(B,C)$ with $B$ being the batch dimension, and $C$ being some number of channels.
The `spatial_features` represent more localized content which should be suitable for dense tasks such as semantic segmentation, or for integration into an LLM.
RADIO and E-RADIO return spatial features in different shapes:

* RADIO: spatial features have shape $(B,T,D)$ with $T$ being the flattened spatial tokens, and $D$ being the channels for spatial features. Note that $C \neq D$ in general.
* E-RADIO: spatial features have shape $(B,H,W,D)$ with $H$ being the height, and $W$ being the width of the spatial features.

For RADIO, converting to a spatial tensor format can be done using the downsampling size of the model, combined with the input tensor shape. For 'radio_v1', the patch size is 14.
```Python
from einops import rearrange
spatial_features = rearrange(spatial_features, 'b (h w) d -> b d h w', h=x.shape[-2] // patch_size, w=x.shape[-1] // patch_size)
```

The resulting tensor will have shape $(B,D,H,W)$, as is typically seen with computer vision models.

### RADIOv1/v2 Notes

We have trained this model to be flexible in input dimension. It supports arbitrary input sizes. There are useful properties set for the returned model that you may query:
```Python
model.patch_size: int
model.max_resolution: int # (Images can be no larger than this value on either dimension)
model.preferred_resolution: Tuple[height, width] # This is the primary resolution that RADIO was trained at, and will likely
                                                 # produce best results for summary tasks. Dense tasks require experimentation
                                                 # to find the best resolution.
model.window_size: Optional[int] # If `vitdet_window_size` was specified, this is that value
model.min_resolution_step: int # Combines `patch_size` and `window_size` to define what each image dimension must be a multiple of.
                               # e.g. If `patch_size == 16`, then both width and height must be x*16
                               # If `patch_size == 14` and `window_size == 8` then width and height must be x*14*8

# For convenience, you can also call this function to get the nearest valid input size for a given image
nearest_height, nearest_width = model.get_nearest_supported_resolution(height=1024, width=1024)
```

RADIO allows non-square inputs. In fact, both RADIOv1 and RADIOv2 achieve higher zero-shot classification scores when allowing the larger image dimension to vary, and only fixing the smaller dimension.

### Adaptors
_(Currently only supported with TorchHub)_

You may additionally specify model adaptors to achieve extra behaviors. Currently, 'clip' is the only supported adaptor. In this mode, radio will return a dict of tuples:

```Python
model = torch.hub.load(..., adaptor_names='clip', ...)

output = model(x)

bb_summary, bb_features = output['backbone']
clip_summary, clip_features = output['clip']
```

Refer to `examples/zero_shot_imagenet.py` for example usage.

### Preprocessing

By default, RADIO expects the input images to have normalized values in the `[0, 1]` range. If you already have an existing data pipeline, and you'd like conditioning to occur there instead of within the RADIO model, you can call this function:

```Python
preprocessor = model.make_preprocessor_external()

images = preprocessor(images)
...
output = model(images)
```

## Training

_Coming Soon_

## License

RADIO code and weights are released under the [NSCLv1 License](LICENSE).

## ~Deprecated: HuggingFace Hub~

Pull the RADIO model from a Python script:

```Python
from transformers import AutoModel
model = AutoModel.from_pretrained("nvidia/RADIO", trust_remote_code=True)
```

Pull the E-RADIO model from a Python script:

```Python
from transformers import AutoModel
model = AutoModel.from_pretrained("nvidia/E-RADIO", trust_remote_code=True)
```

## Citing RADIO

If you find this repository useful, please consider giving a star and citation:
```bibtex
@misc{ranzinger2023amradio,
      title={AM-RADIO: Agglomerative Model -- Reduce All Domains Into One},
      author={Mike Ranzinger and Greg Heinrich and Jan Kautz and Pavlo Molchanov},
      year={2023},
      eprint={2312.06709},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
