from typing import Iterable, List, Dict, Any
from PIL import Image

import torch

import torchvision.transforms.v2 as transforms


class ResizeTransform(transforms.Transform):
    def __init__(self, size: Iterable[int], resize_multiple: int = 1, max_dim: bool = False):
        super().__init__()

        self.size = size
        self.resize_multiple = resize_multiple
        self.max_dim = max_dim

    def _get_nearest(self, value: int):
        return int(round(value / self.resize_multiple) * self.resize_multiple)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = transforms._utils.query_size(flat_inputs)

        if len(self.size) == 1:
            if not self.max_dim:
                # Shortest-side mode.
                # Resize the short dimension of the image to be the specified size,
                # and the other dimension aspect preserving
                min_sz = min(height, width)
                factor = self.size[0] / min_sz
            else:
                # Longest-side mode
                max_sz = max(height, width)
                factor = self.size[0] / max_sz

            rs_height = height * factor
            rs_width = width * factor
            size = (rs_height, rs_width)
        elif len(self.size) == 2:
            # Center-crop mode (the actual crop will be done by subsequent transform)
            in_aspect = height / width
            out_aspect = self.size[0] / self.size[1]

            # Input height varies faster than output
            if (in_aspect > out_aspect and not self.max_dim) or (in_aspect < out_aspect and self.max_dim):
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


class PadToSquare(transforms.Transform):
    def __init__(self, pad_mean = None):
        super().__init__()
        if torch.is_tensor(pad_mean):
            pad_mean = pad_mean.flatten().tolist()
        elif pad_mean is None:
            pad_mean = 0

        self.pad_mean = pad_mean

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = transforms._utils.query_size(flat_inputs)

        max_sz = max(height, width)

        pad_h = max(0, max_sz - height)
        pad_w = max(0, max_sz - width)

        top = pad_h // 2
        left = pad_w // 2
        bottom = pad_h - top
        right = pad_w - left

        return dict(size=(left, top, right, bottom))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        size = params['size']

        ret = transforms.functional.pad(inpt, size, fill=self.pad_mean)
        return ret



def get_standard_transform(resolution: List[int], resize_multiple: int, preprocessor = None, max_dim: bool = False, pad_mean = None):
    transform = [
        ResizeTransform(resolution, resize_multiple, max_dim=max_dim),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
    if len(resolution) == 2:
        transform.append(PadToSquare(pad_mean))
        transform.append(transforms.CenterCrop(resolution))

    if preprocessor is not None:
        transform.append(preprocessor)

    transform = transforms.Compose(transform)
    return transform
