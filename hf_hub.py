# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from timm.models import clean_state_dict
import torch

from hubconf import get_prefix_state_dict
from radio.hf_model import RADIOConfig, RADIOModel


def main():
    """Main Routine.

    Push a RADIO model to Hugging.

    Usage:

    python3 -m hf_hub --checkpoint-path  /path/to/checkpoint.pth.tar
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", help="Path to the pretrained weights", required=True
    )
    parser.add_argument(
        "--huggingface-repo", help="HuggingFace repository", default="nvidia/RADIO"
    )
    parser.add_argument(
        "--huggingface-repo-branch",
        help="HuggingFace repository branch to push to",
        default="main",
    )
    args = parser.parse_args()

    # Tell HuggingFace API we need to push the code for the model config and definition.
    RADIOConfig.register_for_auto_class()
    RADIOModel.register_for_auto_class("AutoModel")

    # Load the checkpoint and create the model.
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_args = checkpoint["args"]
    radio_config = RADIOConfig(
        vars(model_args), return_summary=True, return_spatial_features=True
    )
    radio_model = RADIOModel(radio_config)

    # Restore the model's state.
    state_dict = checkpoint["state_dict"]
    state_dict = clean_state_dict(state_dict)
    radio_model.model.load_state_dict(
        get_prefix_state_dict(state_dict, "base_model."), strict=False
    )
    radio_model.radio_model.input_conditioner.load_state_dict(
        get_prefix_state_dict(state_dict, "input_conditioner.")
    )

    # Push to HuggingFace Hub.
    commit = radio_model.push_to_hub(
        args.huggingface_repo, args.huggingface_repo_branch
    )
    print(f"Pushed to {commit}")


if __name__ == "__main__":
    """Call the main entrypoiny."""
    main()
