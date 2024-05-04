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
from warnings import warn as warning

from timm.models import clean_state_dict
import torch

from hubconf import get_prefix_state_dict
from radio.hf_model import RADIOConfig, RADIOModel


def main():
    """Main Routine.

    Construct and optionally push a RADIO model to Hugging Face.

    Usage:

    python3 -m hf_hub --model <model-name> --checkpoint-path <checkpoint-path> [--push]

    Examples:

    python3 -m hf_hub --hf-repo nvidia/RADIO --checkpoint-path radio_v2.1_bf16.pth.tar --version radio_v2.1 --push
    python3 -m hf_hub --hf-repo nvidia/E-RADIO --checkpoint-path eradio_v2.pth.tar --version e-radio_v2

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", help="Path to the pretrained weights", required=True
    )
    parser.add_argument("--hf-repo", help="Path to the HuggingFace repo", required=True)
    parser.add_argument(
        "--torchhub-repo", help="Path to the TorchHub repo", default="NVlabs/RADIO"
    )
    parser.add_argument("--version", help="(E-)RADIO model version", required=True)
    parser.add_argument(
        "--push", help="Push the model to HuggingFace", action="store_true"
    )
    parser.add_argument(
        "--commit-message", default=None, type=str, required=False, help="The commit message",
    )
    args = parser.parse_args()

    # Load the checkpoint and create the model.
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_args = checkpoint["args"]

    # Extract the state dict from the checkpoint.
    if "state_dict_ema" in checkpoint:
        state_dict = checkpoint["state_dict_ema"]
        # Disable spectral reparametrization for EMA model.
        model_args.spectral_reparam = False
    else:
        state_dict = checkpoint["state_dict"]
    state_dict = clean_state_dict(state_dict)

    # Tell HuggingFace API we need to push the code for the model config and definition.
    RADIOConfig.register_for_auto_class()
    RADIOModel.register_for_auto_class("AutoModel")

    radio_config = RADIOConfig(vars(model_args), version=args.version)
    radio_model = RADIOModel(radio_config)

    # Restore the model weights.
    key_warn = radio_model.model.load_state_dict(
        get_prefix_state_dict(state_dict, "base_model."), strict=False
    )
    if key_warn.missing_keys:
        warning(f"Missing keys in state dict: {key_warn.missing_keys}")
    if key_warn.unexpected_keys:
        warning(f"Unexpected keys in state dict: {key_warn.unexpected_keys}")

    # Restore input conditioner.
    radio_model.input_conditioner.load_state_dict(
        get_prefix_state_dict(state_dict, "input_conditioner.")
    )

    radio_model.eval().cuda()

    # Sample inference with random values.
    x = torch.randn(
        1,
        3,
        radio_model.config.preferred_resolution[0],
        radio_model.config.preferred_resolution[1],
    ).cuda()

    # Infer using HuggingFace model.
    hf_model_summary, hf_model_features = radio_model(x)
    print(
        f"Sample inference on tensor shape {x.shape} returned summary ",
        f"with shape={hf_model_summary.shape} and std={hf_model_summary.std().item():.3}, ",
        f"features with shape={hf_model_features.shape} and std={hf_model_features.std().item():.3}",
    )

    # Infer using TorchHub model.
    print("Infer using TorchHub model...")
    torchhub_model = torch.hub.load(
        args.torchhub_repo, "radio_model", version=args.checkpoint_path, force_reload=True
    )
    torchhub_model.cuda().eval()
    torchhub_model_summary, torchhub_model_features = torchhub_model(x)

    # Make sure the results are the same.
    assert torch.allclose(hf_model_summary, torchhub_model_summary, atol=1e-6)
    assert torch.allclose(hf_model_features, torchhub_model_features, atol=1e-6)
    print("All outputs matched!")

    if args.push:
        # Push to HuggingFace Hub.
        huggingface_repo = args.hf_repo
        commit = radio_model.push_to_hub(huggingface_repo, create_pr=True, commit_message=args.commit_message)
        print(f"Pushed to {commit}")


if __name__ == "__main__":
    """Call the main entrypoiny."""
    main()
