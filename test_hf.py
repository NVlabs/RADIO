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
from transformers import AutoModel, CLIPImageProcessor
import torch


def main():
    """Main Routine.

    Pull a model from HuggingFace and make sure its output features
    match those of the corresponding TorchHub model.

    Usage:

    python3 -m test_hf --hf-repo <repo> --torchhub-version <version|/path/to/checkpoint.pth.tar> [--torchhub-repo <repo>]

    Examples:

    python3 -m test_hf --hf-repo nvidia/RADIO --torchhub-version ./radio_v2.1_bf16.pth.tar
    python3 -m test_hf --hf-repo gheinrich/RADIO --torchhub-version ./radio_v2.1_bf16.pth.tar --torchhub-repo NVlabs/RADIO:dev/hf
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", help="Path to the HuggingFace repo", required=True)
    parser.add_argument(
        "--torchhub-version", help="Torchhub version to compare against", required=True
    )
    parser.add_argument(
        "--torchhub-repo", help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )

    args = parser.parse_args()

    hf_model = AutoModel.from_pretrained(args.hf_repo, trust_remote_code=True)
    hf_model.eval().cuda()

    # Sample inference with random values.
    x = torch.randn(
        1,
        3,
        hf_model.config.preferred_resolution[0],
        hf_model.config.preferred_resolution[1],
    ).cuda()

    # Infer using HuggingFace model.
    hf_model_summary, hf_model_features = hf_model(x)
    print(
        f"Sample inference on tensor shape {x.shape} returned summary ",
        f"with shape={hf_model_summary.shape} and std={hf_model_summary.std().item():.3}, ",
        f"features with shape={hf_model_features.shape} and std={hf_model_features.std().item():.3}",
    )

    # Infer using TorchHub model.
    torchhub_model = torch.hub.load(
        args.torchhub_repo, "radio_model", version=args.torchhub_version
    )
    torchhub_model.cuda().eval()
    torchhub_model_summary, torchhub_model_features = torchhub_model(x)

    # Make sure the results are the same.
    assert torch.allclose(hf_model_summary, torchhub_model_summary, atol=1e-6)
    assert torch.allclose(hf_model_features, torchhub_model_features, atol=1e-6)

    print("All outputs matched!")

    # Infer a sample image.
    image_processor = CLIPImageProcessor.from_pretrained(args.hf_repo)

    image = Image.open("./examples/image1.png").convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    hf_model_summary, hf_model_features = hf_model(pixel_values)
    print(
        f"Sample inference on image shape {pixel_values.shape} with "
        f"min={pixel_values.min().item():.3} and max={pixel_values.max().item():.3} returned summary ",
        f"with shape={hf_model_summary.shape} and std={hf_model_summary.std().item():.3}, ",
        f"features with shape={hf_model_features.shape} and std={hf_model_features.std().item():.3}",
    )


if __name__ == "__main__":
    """Call the main entrypoint."""
    main()
