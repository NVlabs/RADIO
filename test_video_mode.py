import sys
import torch
from hubconf import radio_model


if __name__ == "__main__":
    model = radio_model(version=sys.argv[1] if len(sys.argv) > 1 else '').cuda()

    x = torch.rand(8, 3, 192, 192, device='cuda')

    with model.cpe_video_mode(t=4):
        y = model(x)
    pass
