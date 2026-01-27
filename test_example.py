import os
from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
repo = 'NVlabs/RADIO'
source = 'local' if os.path.exists(repo) else 'github'

model_version="c-radio_v4-h" # for C-RADIOv4-H model (ViT-H/16)
# model_version="c-radio_v4-so400m" # for C-RADIOv4-H model (ViT-H/16)
model = torch.hub.load(repo, 'radio_model', source=source, version=model_version, progress=True, skip_validation=True, force_reload=True)
model.cuda().eval()

x = Image.open('assets/cradio_v4.png').convert('RGB')
x = pil_to_tensor(x).to(dtype=torch.float32, device='cuda')
x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
x = x.unsqueeze(0) # Add a batch dimension

nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

# RADIO expects the input to have values between [0, 1]. It will automatically normalize them to have mean 0 std 1.
summary, spatial_features = model(x)

# By default, RADIO will return the spatial_features in NLC format, with L being a combined height/width dimension.
# You can alternatively ask for the features in the more computer-vision-convenient format NCHW the following way:
summary, spatial_features = model(x, feature_fmt='NCHW')
assert spatial_features.ndim == 4

# RADIO also supports running in mixed precision:
with torch.autocast('cuda', dtype=torch.bfloat16):
    summary, spatial_features = model(x)

# If you'd rather pre-normalize the inputs, then you can do this:
conditioner = model.make_preprocessor_external()

# Now, the model won't change the inputs, and it's up to the user to call `cond_x = conditioner(x)` before
# calling `model(cond_x)`. You most likely would do this if you want to move the conditioning into your
# existing data processing pipeline.
with torch.autocast('cuda', dtype=torch.bfloat16):
    cond_x = conditioner(x)
    summary, spatial_features = model(cond_x)

# Adaptors
# One or more may be specified via the `adaptor_names` argument
model = torch.hub.load(repo, 'radio_model', source=source, version=model_version, progress=True, skip_validation=True, adaptor_names=['siglip2-g'])
model.cuda().eval()

vis_output = model(x)
# These are the usual RADIO features
backbone_summary, backbone_features = vis_output['backbone']
# There will also be summary and feature pairs for each of the loaded adaptors
sig2_vis_summary, sig2_vis_features = vis_output['siglip2-g']

# The 'siglip2-g' and 'clip' adaptors (when available) are special because they also support text tokenization and encoding
sig2_adaptor = model.adaptors['siglip2-g']
text_input = sig2_adaptor.tokenizer(['An image of an alien wearing headphones, with three orbs floating overhead']).to('cuda')
text_tokens = sig2_adaptor.encode_text(text_input, normalize=True)

sim = F.cosine_similarity(sig2_vis_summary, text_tokens)
print(sim)
