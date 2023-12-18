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
from warnings import warn as warning

from timm.models import clean_state_dict
import torch

from hubconf import get_prefix_state_dict
from radio.hf_model import ERADIOConfig, ERADIOModel, RADIOConfig, RADIOModel


hugginface_repo_mapping = {"RADIO": "nvidia/RADIO", "E-RADIO": "nvidia/E-RADIO"}


def main():
    """Main Routine.

    Push a RADIO model to Hugging.

    Usage:

    python3 -m hf_hub --model RADIO --checkpoint-path /path/to/checkpoint.pth.tar
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", help="Path to the pretrained weights", required=True
    )
    parser.add_argument(
        "--model", help="RADIO model type", required=True, choices=["RADIO", "E-RADIO"]
    )
    parser.add_argument(
        "--huggingface-repo-branch",
        help="HuggingFace repository branch to push to",
        default="main",
    )
    args = parser.parse_args()

    # Load the checkpoint and create the model.
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_args = checkpoint["args"]

    # Extract the state dict from the checkpoint.
    state_dict = checkpoint["state_dict"]
    state_dict = clean_state_dict(state_dict)

    # Remove keys in state dict whose size does not match that of the model.
    checkpoint_state_dict = get_prefix_state_dict(state_dict, "base_model.")

    if args.model == "RADIO":
        # Tell HuggingFace API we need to push the code for the model config and definition.
        RADIOConfig.register_for_auto_class()
        RADIOModel.register_for_auto_class("AutoModel")

        radio_config = RADIOConfig(
            vars(model_args), return_summary=True, return_spatial_features=True
        )
        radio_model = RADIOModel(radio_config)

        model_state_dict = radio_model.model.state_dict()
    elif args.model == "E-RADIO":
        # Tell HuggingFace API we need to push the code for the model config and definition.
        ERADIOConfig.register_for_auto_class()
        ERADIOModel.register_for_auto_class("AutoModel")

        radio_config = ERADIOConfig(
            vars(model_args), return_summary=True, return_spatial_features=True
        )
        radio_model = ERADIOModel(radio_config)

        model_state_dict = radio_model.model.state_dict()
        for k, v in model_state_dict.items():
            if k in checkpoint_state_dict:
                if v.size() != checkpoint_state_dict[k].size():
                    warning(
                        f"Removing key {k} from state dict due to shape mismatch: "
                        f"{v.size()} != {checkpoint_state_dict[k].size()}"
                    )
                else:
                    model_state_dict[k] = checkpoint_state_dict[k]
    else:
        raise ValueError(f"Unknown model {args.model}")

    # Restore the model weights.
    radio_model.model.load_state_dict(model_state_dict, strict=False)

    # Restore input conditioner.
    radio_model.input_conditioner.load_state_dict(
        get_prefix_state_dict(state_dict, "input_conditioner.")
    )

    radio_model.eval().cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    summary, features = radio_model(x)
    print(
        f"Sample inference on tensor shape {x.shape} returned summary ",
        f"with shape {summary.shape}, features with shape {features.shape}",
    )

    # Push to HuggingFace Hub.
    huggingface_repo = hugginface_repo_mapping[args.model]
    commit = radio_model.push_to_hub(huggingface_repo, args.huggingface_repo_branch)
    print(f"Pushed to {commit}")


if __name__ == "__main__":
    """Call the main entrypoiny."""
    main()
