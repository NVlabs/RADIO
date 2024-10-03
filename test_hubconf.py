import torch
from hubconf import radio_model


if __name__ == "__main__":
    model = radio_model().cuda()

    x = torch.rand(1, 3, 256, 256, device='cuda')

    with torch.no_grad():
        y = model(x)

        y_int1 = model.forward_intermediates(x, indices=[1, 5, 7], output_fmt='NCHW')
        y_int2 = model.forward_intermediates(x, indices=[2, 4, 6], output_fmt='NLC')
        y_int3 = model.forward_intermediates(x, indices=[3, 5, 7], return_prefix_tokens=True, output_fmt='NCHW', aggregation='dense', intermediates_only=True)
        pass
