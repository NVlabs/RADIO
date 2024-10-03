import torch
from hubconf import radio_model


if __name__ == "__main__":
    model = radio_model().cuda()

    x = torch.rand(1, 3, 256, 256, device='cuda')

    y = model(x)

    y_int = model.forward_intermediates(x, indices=[1, 5, 7], output_fmt='NCHW')
    y_int = model.forward_intermediates(x, indices=[2, 4, 6], output_fmt='NLC')
