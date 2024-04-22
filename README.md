# \[CVPR 2024\] AM-RADIO: Reduce All Domains Into One

<div align="left">
  <img src="assets/radio.png" width="256"/>
</div>

Official PyTorch implementation of \[CVPR 2024\] [**AM-RADIO: Agglomerative Model – Reduce All Domains Into One**](https://arxiv.org/abs/2312.06709v3).

[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/RADIO.svg?style=social)](https://github.com/NVlabs/RADIO/stargazers)

Mike Ranzinger, Greg Heinrich, [Jan Kautz](https://jankautz.com/), [Pavlo Molchanov](https://www.pmolchanov.com/).

[NVIDIA Research](https://www.nvidia.com/en-us/research/)

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

\[[Paper](https://arxiv.org/abs/2312.06709)\]\[[BibTex](#citing-radio)\]

---

AM-RADIO is a framework to distill Large Vision Foundation models into a single one.
RADIO, a new vision foundation model, excels across visual domains, serving as a superior replacement for vision backbones. Integrating CLIP variants, DINOv2, and SAM through distillation, it preserves unique features like text grounding and segmentation correspondence. Outperforming teachers in ImageNet zero-shot (+6.8%), kNN (+2.39%), and linear probing segmentation (+3.8%) and vision-language models (LLaVa 1.5 up to 1.5%), it scales to any resolution, supports non-square images. We offer an efficient variant, E-RADIO, which achieves is 6-10x faster than CLIP and DINOv2.


## News/Release
- [3.21.2024] 🔥 RADIOv2.1 is released. Trained in bf16, improves metrics!
- [2.26.2024] 🔥 AM-RADIO paper has been accepted to ** CVPR 2024 **
- [2.15.2024]  RADIOv2 is released. Trained with DFN CLIP; OpenAI CLIP; DINOv2; SAM teachers. Note that SAM teacher was not used in previous models.
- [1.5.2024] Initial github repo is released.

## Pretrained Models

| Name       | Architecture | Precision | Teachers                                 | Throughput | Zero Shot Top-1 | kNN Top-1 | ADE20k    | VOC       | GQA       | TextVQA   | VQAv2     | SAM-COCO  |
|------------|--------------|-----------|------------------------------------------|------------|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| radio_v2.1 | ViT-H/16-CPE | BFloat16  | DFN CLIP; OpenAI CLIP; DINOv2; SAM       | 556        | **82.93**       | **86.06** | **51.34** | 84.71     | **63.01** | 56.32     | **79.28** | **76.58** |
| radio_v2   | ViT-H/16-CPE | Float32   | DFN CLIP; OpenAI CLIP; DINOv2; SAM       | 556        | 82.71           | 85.92     | 51.33     |           | 62.78     | **56.37** | 79.00     | 76.21     |
| radio_v1   | ViT-H/14-CPE | Float32   | DFN CLIP; OpenAI CLIP; DINOv2            | 556        | 82.73           | 85.29     | 50.32     | **85.17** | 61.43     | 54.92     | 77.88     |           |
| eradio_v1  | E-RADIO      | Float32   | Meta CLIP; DINOv2                        | 3697       | 77.87           | 83.73     | 45.50     | 79.95     | 59.55     | 46.31     | 72.05     |           |


## Quick Start

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

### HuggingFace

```python
import torch
from transformers import AutoModel

hf_repo = "nvidia/RADIO" # For RADIO.
# hf_repo = "nvidia/E-RADIO" # For E-RADIO.

model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
model.eval().cuda()

# Sample inference with random values.
x = torch.randn(
    1,
    3,
    model.config.preferred_resolution[0],
    model.config.preferred_resolution[1],
).cuda()

# Infer using HuggingFace model.
summary, features = model(x)
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

## Star History

<a href="https://star-history.com/#NVlabs/RADIO&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=NVlabs/RADIO&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=NVlabs/RADIO&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=NVlabs/RADIO&type=Date" />
 </picture>
</a>

## Citing RADIO

If you find this repository useful, please consider giving a star and citation:

#### CVPR 2024 Reference:
```bibtex
@inProceedings{ranzinger2023amradio,
  title={AM-RADIO: Agglomerative Visual Foundation Model -- Reduce All Domains Into One},
  author={Mike Ranzinger and Greg Heinrich and Jan Kautz and Pavlo Molchanov},
  booktitle={CVPR},
  year={2024},
}
```

#### ArXiv Reference:
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

## Licenses

Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.
