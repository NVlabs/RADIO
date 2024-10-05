import sys
import torch
from hubconf import radio_model


if __name__ == "__main__":
    model = radio_model(version=sys.argv[1] if len(sys.argv) > 1 else '').cuda()

    x = torch.rand(1, 3, 256, 256, device='cuda')

    with torch.no_grad():
        y = model(x)

        op, (int0,) = model.forward_intermediates(x, indices=[-1], output_fmt='NLC', aggregation='sparse')

        diff = (op.features - int0).norm()
        print(f'Output diff: {diff.item():.8f}')

        y_int1 = model.forward_intermediates(x, indices=[1, 5, 7], output_fmt='NCHW')
        y_int2 = model.forward_intermediates(x, indices=[2, 4, 6], output_fmt='NLC')
        y_int3 = model.forward_intermediates(x, indices=[3, 5, 7], return_prefix_tokens=True, output_fmt='NCHW', aggregation='dense', intermediates_only=True)
        y_int4 = model.forward_intermediates(x, indices=[3, 5, 7], return_prefix_tokens=True, output_fmt='NCHW', aggregation='dense', intermediates_only=True, norm_alpha_scheme='pre-alpha')
        y_int5 = model.forward_intermediates(x, indices=[3, 5, 7], return_prefix_tokens=True, output_fmt='NCHW', aggregation='dense', intermediates_only=True, norm_alpha_scheme='none')
        pass
