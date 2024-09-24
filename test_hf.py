# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

from PIL import Image
from transformers import AutoConfig, AutoModel, CLIPImageProcessor
import torch

from radio.adaptor_base import RadioOutput


def deterministic_grid_init(shape):
    """Return a diverse but deterministic grid of values."""
    indices = torch.stack(
        torch.meshgrid([torch.arange(s, dtype=torch.float32) for s in shape]), dim=-1
    )
    return torch.sin(indices[..., 0]) + torch.cos(indices[..., 1])


def main():
    """Main Routine.

    Pull a model from HuggingFace and make sure its output features
    match those of the corresponding TorchHub model.

    Usage:

    python3 -m test_hf --hf-repo <repo> --torchhub-version <version|/path/to/checkpoint.pth.tar> [--torchhub-repo <repo>]

    Examples:

    python3 -m test_hf --hf-repo nvidia/RADIO --torchhub-version ./radio_v2.1_bf16.pth.tar
    python3 -m test_hf --hf-repo gheinrich/RADIO --torchhub-version ./radio_v2.1_bf16.pth.tar --torchhub-repo NVlabs/RADIO:dev/hf
    python3 -m test_hf --hf-repo gheinrich/RADIO --torchhub-version ./radio-v2.5-l_half.pth.tar --torchhub-repo NVlabs/RADIO:dev/hf
    python3 -m test_hf --hf-repo gheinrich/RADIO --torchhub-version ./radio-v2.5-l_half.pth.tar  --adaptor-names siglip,sam
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", help="Path to the HuggingFace repo", required=True)
    parser.add_argument(
        "--torchhub-version", help="Torchhub version to compare against", required=True
    )
    parser.add_argument(
        "--torchhub-repo", help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )
    parser.add_argument(
        "--adaptor-names",
        default=None,
        type=lambda x: x.split(","),
        required=False,
        help="Comma-separated list of adaptor names",
    )

    args = parser.parse_args()

    hf_config = AutoConfig.from_pretrained(args.hf_repo, trust_remote_code=True)
    if args.adaptor_names is not None:
        # Configure adaptors if specified on the command line.
        # This needs to happen before we instantiate the model.
        hf_config.adaptor_names = args.adaptor_names
    hf_model = AutoModel.from_pretrained(
        args.hf_repo, trust_remote_code=True, config=hf_config
    )
    hf_model.eval().cuda()

    # Sample inference with deterministic values.
    x = deterministic_grid_init(
        (
            1,
            3,
            hf_model.config.preferred_resolution[0],
            hf_model.config.preferred_resolution[1],
        )
    ).cuda()

    # Infer using HuggingFace model.
    hf_output = hf_model(x)
    if isinstance(hf_output, tuple):
        hf_output = dict(backbone=RadioOutput(hf_output[0], hf_output[1]))

    # Infer using TorchHub model.
    torchhub_model = torch.hub.load(
        args.torchhub_repo,
        "radio_model",
        version=args.torchhub_version,
        adaptor_names=args.adaptor_names,
    )
    torchhub_model.cuda().eval()
    torchhub_output = torchhub_model(x)

    if isinstance(torchhub_output, tuple):
        torchhub_output = dict(
            backbone=RadioOutput(torchhub_output[0], torchhub_output[1])
        )

    for k in torchhub_output.keys():
        hf_summary, hf_features = hf_output[k].summary, hf_output[k].features
        torchhub_summary, torchhub_features = (
            torchhub_output[k].summary,
            torchhub_output[k].features,
        )

        print(
            f"[{k}] HF model Sample inference on tensor shape {x.shape} returned summary ",
            f"with shape={hf_summary.shape} and std={hf_summary.std().item():.3}, ",
            f"features with shape={hf_features.shape} and std={hf_features.std().item():.3}",
        )

        print(
            f"[{k}] TorchHub model Sample inference on tensor shape {x.shape} returned summary ",
            f"with shape={torchhub_summary.shape} and std={torchhub_summary.std().item():.3}, ",
            f"features with shape={torchhub_features.shape} and std={torchhub_features.std().item():.3}",
        )

        # Make sure the results are the same.
        assert torch.allclose(hf_summary, torchhub_summary, atol=1e-6)
        assert torch.allclose(hf_features, torchhub_features, atol=1e-6)

    print("All outputs matched!")

    # Infer a sample image.
    image_processor = CLIPImageProcessor.from_pretrained(args.hf_repo)

    image = Image.open("./examples/image1.png").convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    hf_output = hf_model(pixel_values)
    if isinstance(hf_output, tuple):
        hf_output = dict(backbone=RadioOutput(hf_output[0], hf_output[1]))
    for k, v in hf_output.items():
        hf_summary, hf_features = v.summary, v.features
        print(
            f"[{k}] Sample inference on image shape {pixel_values.shape} with "
            f"min={pixel_values.min().item():.3} and max={pixel_values.max().item():.3} returned summary ",
            f"with shape={hf_summary.shape} and std={hf_summary.std().item():.3}, ",
            f"features with shape={hf_features.shape} and std={hf_features.std().item():.3}",
        )


if __name__ == "__main__":
    """Call the main entrypoint."""
    main()
