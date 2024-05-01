[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/RADIO.svg?style=social)](https://github.com/NVlabs/RADIO/stargazers)
[![License](https://img.shields.io/badge/license-NC-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv.2312.06709-blue.svg)](https://arxiv.org/abs/2312.06709)
[![Paper](https://img.shields.io/badge/paper-CVPR.2024-blue.svg)](https://arxiv.org/abs/2312.06709)

# \[CVPR 2024\] AM-RADIO: Reduce All Domains Into One

<!-- <div align="left"> -->
<img src="assets/radio.png" width="256" align="right">
<!-- </div> -->

Official PyTorch implementation of \[CVPR 2024\] [**AM-RADIO: Agglomerative Model – Reduce All Domains Into One**](https://arxiv.org/abs/2312.06709). 



Mike Ranzinger, Greg Heinrich, [Jan Kautz](https://jankautz.com/), [Pavlo Molchanov](https://www.pmolchanov.com/).

[NVIDIA Research](https://www.nvidia.com/en-us/research/)

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

\[[Paper](https://arxiv.org/abs/2312.06709)\]\[[BibTex](#citing-radio)\]

<br clear="left"/>

---
 

## News/Release
- [4.30.2024] 🔥 README is updated with more metrics, Arxiv is updated with new results.
- [3.21.2024] 🔥 RADIOv2.1 is released. Trained in bf16, improves metrics!
- [2.26.2024]  AM-RADIO paper has been accepted to **CVPR 2024**
- [2.15.2024]  RADIOv2 is released. Trained with DFN CLIP; OpenAI CLIP; DINOv2; SAM teachers. Note that SAM teacher was not used in previous models.
- [1.5.2024] Initial github repo is released.

---

## Abstract


AM-RADIO is a framework to distill Large Vision Foundation models into a single one.
RADIO, a new vision foundation model, excels across visual domains, serving as a superior replacement for vision backbones. Integrating CLIP variants, DINOv2, and SAM through distillation, it preserves unique features like text grounding and segmentation correspondence. Outperforming teachers in ImageNet zero-shot (+6.8%), kNN (+2.39%), and linear probing segmentation (+3.8%) and vision-language models (LLaVa 1.5 up to 1.5%), it scales to any resolution, supports non-square images. We offer an efficient variant, E-RADIO, which achieves is 6-10x faster than CLIP and DINOv2.

<div align="left">
  <img src="assets/radio_overview_github.png" width="768"/>
</div>

## Quick start and model versions:

The latest model version is RADIOv2. To load in the TorchHub, use the following command:

```Python
model = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2', progress=True)
```

For ERADIO, use:
```Python
model = torch.hub.load('NVlabs/RADIO', 'radio_model', version='e-radio_v2', progress=True)  
model.model.set_optimal_window_size(IMAGE_SHAPE) #where IMAGE_SHAPE is a tuple of (height, width) of the input image.
```
For the previous version, use `radio_v1` or `eradio_v1` for the E-RADIO model.

For HF hub:
```Python
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

hf_repo = "nvidia/RADIO" # For RADIO.
# hf_repo = "nvidia/E-RADIO" # For E-RADIO.

image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
model.eval().cuda()

image = Image.open('./assets/radio.png').convert('RGB')
pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

summary, features = model(pixel_values)
```

Please see more details on usage in the [Quick Start](#quick-start---torchhub) section. Information on how to load Adapters (teacher specific heads) is also available in the Quick Start section.

<details>
<summary>Previously trained models</summary>

| Name       | Architecture | Precision | Teachers                                 | Throughput | Zero Shot Top-1 | kNN Top-1 | ADE20k    | VOC       | GQA       | TextVQA   | VQAv2     | SAM-COCO  |
|------------|--------------|-----------|------------------------------------------|------------|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| radio_v2.1 | ViT-H/16-CPE | BFloat16  | DFN CLIP; OpenAI CLIP; DINOv2; SAM       | 556        | **82.93**       | **86.06** | **51.34** | 84.71     | **63.01** | 56.32     | **79.28** | **76.58** |
| radio_v2   | ViT-H/16-CPE | Float32   | DFN CLIP; OpenAI CLIP; DINOv2; SAM       | 556        | 82.71           | 85.92     | 51.33     |           | 62.78     | **56.37** | 79.00     | 76.21     |
| radio_v1   | ViT-H/14-CPE | Float32   | DFN CLIP; OpenAI CLIP; DINOv2            | 556        | 82.73           | 85.29     | 50.32     | **85.17** | 61.43     | 54.92     | 77.88     |           |
| eradio_v1  | E-RADIO      | Float32   | Meta CLIP; DINOv2                        | 3697       | 77.87           | 83.73     | 45.50     | 79.95     | 59.55     | 46.31     | 72.05     |           |
</details>

## Results

### Model stats and summarization metrics:

For summarization results we use the summarization token of the model. For Zero-shot we use the corresponding language embedding for most models. For RADIO models we use language embedding from DFN CLIP 378 model.

| Model                  | Params (M) | Resolution | Throughput | ImageNet1K Zero-shot | ImageNet1K k-NN |
|------------------------|------------|------------|------------|---------------------|-----------------|
| OpenCLIP-H/14          | 632        | 224        | 503        | 77.19               | 81.10           |
| MetaCLIP-H/14          | 632        | 224        | 486        | 80.51               | 82.12           |
| SigLIP-L/14            | 428        | 384        | 241        | 82.61               | 85.16           |
| Intern-ViT-6B          | 5,902      | 224        | 63         | 83.20               | 78.43           |
|                        | 5,537      | 448        | 14         |                     | 68.64           |
| DFN CLIP-H/14          | 633        | 378        | 170        | **83.90**           | 85.27           |
| OpenAI CLIP-L/14       | 305        | 336        | 414        | 75.54               | 79.80           |
| DINOv2-g/14-reg        | 1,137      | 224        | 294        | -                   | 83.41           |
| SAM-H/16               | 637        | 1024       | 12         | -                   | 22.12           |
|------------------------|------------|------------|------------|---------------------|-----------------|
| E-RADIO-L              | 391        | 512        | 468        | 80.73               | 83.89           |
| RADIO-ViT-H/16         | 653        | 432        | 158        | 82.93               | **86.06**       |


### Segmentation metrics:
- Segmentation setup: linear probing, simple head
- For SAM COCO results, we replace the vision backbone of the SAM model with the corresponding RADIO model. The decoder is frozen from the original model.


| Model                  | Segmentation ADE20k | Segmentation VOC | SAM COCO |
|------------------------|---------------------|------------------|----------|
| OpenCLIP-H/14          | 40.04               | 68.03            | -        |
| MetaCLIP-H/14          | 35.39               | 62.62            | -        |
| SigLIP-L/14            | 40.53               | 70.31            | -        |
| Intern-ViT-6B          | 47.20               | 76.85            | -        |
|                        | 42.78               | 74.43            | -        |
| DFN CLIP-H/14          | 39.00               | 70.29            | -        |
| OpenAI CLIP-L/14       | 36.51               | 67.04            | -        |
| DINOv2-g/14-reg        | 48.68               | 82.78            | -        |
| SAM-H/16               | 28.08               | 34.34            | 77.18    |
|------------------------|---------------------|------------------|----------|
| E-RADIO-L              | 48.22               | 81.64            | 76.31    |
| RADIO-ViT-H/16 (ours)  | **51.34**           | **84.71**        | 76.23    |


### Vision-language model performance metrics in LLaVa 1.5:

We replace the vision backbone and keep the same LLM and training recipe as in LLaVa 1.5: 

| Model               | GQA                 | POPE                 | TextVQA                 | VQAv2                 |
|---------------------|---------------------|----------------------|-------------------------|-----------------------|
| OpenCLIP-H/14       | 57.94               | 83.61                | 50.48                   | 72.24                 |
| MetaCLIP-H/14       | 60.57               | 84.76                | 53.65                   | 75.71                 |
| SigLIP-L/14         | 57.70               | 84.85                | 56.65                   | 71.94                 |
| Intern-ViT-6B (224) | 60.18               | 84.02                | 52.45                   | 76.75                 |
|               (448) | 61.19               | **87.23**            | **60.36**               | 78.83                 |      
| DFN CLIP-H/14       | 61.73               | 85.91                | 56.78                   | 78.78                 |
| OpenAI CLIP-L/14    | 62.20               | 86.09                | 57.92                   | 78.49                 |
| DINOv2-g/14-reg     | 61.88               | 85.62                | 47.18                   | 76.23                 |
| SAM-H/16            | 49.92               | 81.76                | 43.91                   | 57.65                 |
|---------------------|---------------------|----------------------|-------------------------|-----------------------|
| E-RADIO-L           | 61.70               | 85.07                | 51.47                   | 76.73                 |
| RADIO-ViT-H/16 (ours)| **63.01**          | 86.20                | 56.32                   | **79.28**             |

### Probing 3D Awareness

Probing 3D Awareness: we use the code from [Probing the 3D Awareness of Visual Foundation Models](https://github.com/mbanani/probe3d) and
evaluate our RADIO model and its teachers on monocular depth,
surface normals and multi-view correspondance tasks, using the
NAVI dataset. For each task we report the accuracy, averaged
over all thresholds. RADIO preserves features of DINOv2 and 
performs much better than CLIP analogs. 

| Backbone              | Depth | Surface Normals | Multi-view corr. |
|-----------------------|-------|-----------------|------------------|
| DFN CLIP-H/14         | 52.5  | 23.0            | 20.3             |
| OpenAI CLIP-L/14      | 53.7  | 25.3            | 20.7             |
| DINOv2-g/14-reg       | **83.2**  | **59.6**            | 59.9     |
| SAM-H/16              | 68.2  | 50.3            | 45.3             |
|-----------------------|-------|-----------------|------------------|
| RADIO-ViT-H/16 (ours) | 81.0  | 58.5            | **62.1**         |


## Detailed usage 

<details>
<summary>Torch hub</summary>

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
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

hf_repo = "nvidia/RADIO" # For RADIO.
# hf_repo = "nvidia/E-RADIO" # For E-RADIO.

image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
model.eval().cuda()

image = Image.open('./examples/image1.png').convert('RGB')
pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

summary, features = model(pixel_values)
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

</details>

<details>
<summary>HuggingFace hub</summary>




</details>

<details>
<summary>E-RADIO limitations</summary>

E-RADIO is a more efficient variant of RADIO, but it has some limitations:
- E-RADIO naively supports only images with size divisible by 32. Other resolutions are supported but might result in a performance drop.
- E-RADIO performance is sensative to the window size of the windowed attention in the 3rd and 4th block. For the best performance automatically adjust the window size for the input resolution: `model.model.set_optimal_window_size(IMAGE_SHAPE)`, where `IMAGE_SHAPE` is a tuple of (height, width) of the input image.


</details>

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
