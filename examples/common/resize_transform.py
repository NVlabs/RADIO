from typing import Iterable, List, Dict, Any
from PIL import Image

import torch

import torchvision.transforms.v2 as transforms


class ResizeTransform(transforms.Transform):
    def __init__(self, size: Iterable[int], resize_multiple: int = 1):
        super().__init__()

        self.size = size
        self.resize_multiple = resize_multiple

    def _get_nearest(self, value: int):
        return int(round(value / self.resize_multiple) * self.resize_multiple)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = transforms._utils.query_size(flat_inputs)

        if len(self.size) == 1:
            # Shortest-side mode.
            # Resize the short dimension of the image to be the specified size,
            # and the other dimension aspect preserving
            min_sz = min(height, width)
            factor = self.size[0] / min_sz

            rs_height = height * factor
            rs_width = width * factor
            size = (rs_height, rs_width)
        elif len(self.size) == 2:
            # Center-crop mode (the actual crop will be done by subsequent transform)
            in_aspect = height / width
            out_aspect = self.size[0] / self.size[1]

            # Input height varies faster than output
            if in_aspect > out_aspect:
                scale = self.size[1] / width
            else:
                scale = self.size[0] / height

            rs_height = height * scale
            rs_width = width * scale
            size = (rs_height, rs_width)
        else:
            raise ValueError("Unsupported resize mode")

        size = tuple(self._get_nearest(d) for d in size)

        return dict(size=size)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, Image.Image):
            inpt = inpt.convert('RGB')

        size = params['size']

        return transforms.functional.resize(inpt, size=size, interpolation=transforms.InterpolationMode.BICUBIC)


def get_standard_transform(resolution: List[int], resize_multiple: int):
    transform = [
        ResizeTransform(resolution, resize_multiple),
    ]
    if len(resolution) == 2:
        transform.append(transforms.CenterCrop(resolution))
    transform.extend([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    transform = transforms.Compose(transform)
    return transform
