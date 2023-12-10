# AM-RADIO: Reduce All Domains Into One

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
```

## Training

_Coming Soon_
