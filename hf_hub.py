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
from radio.adaptor_base import RadioOutput
from radio.adaptor_registry import adaptor_registry
from radio.adaptor_mlp import get_mlp_info_from_state
from radio.hf_model import RADIOConfig, RADIOModel
from test_hf import deterministic_grid_init


def replace_prefix_in_state_dict(state_dict, old_prefix, new_prefix, replace_all=False):
    new_state_dict = {}
    for key, value in state_dict.items():
        if replace_all:
            # Replace all occurrences of the old prefix with the new one
            new_key = key.replace(old_prefix, new_prefix)
        else:
            # Replace only the first occurrence of the old prefix
            new_key = key.replace(old_prefix, new_prefix, 1)
        new_state_dict[new_key] = value
    return new_state_dict


def main():
    """Main Routine.

    Construct and optionally push a RADIO model to Hugging Face.

    Usage:

    python3 -m hf_hub --model <model-name> --checkpoint-path <checkpoint-path> [--push]

    Examples:

    python3 -m hf_hub --hf-repo nvidia/RADIO --checkpoint-path radio_v2.1_bf16.pth.tar --version radio_v2.1 --push
    python3 -m hf_hub --hf-repo nvidia/E-RADIO --checkpoint-path eradio_v2.pth.tar --version e-radio_v2
    python3 -m hf_hub --hf-repo gheinrich/RADIO --checkpoint-path ./radio-v2.5-l_half.pth.tar  --version radio_v2.5-l --adaptor-names clip,siglip,dino_v2,sam

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
        "--commit-message",
        default=None,
        type=str,
        required=False,
        help="The commit message",
    )
    parser.add_argument(
        "--adaptor-names",
        default=None,
        type=lambda x: x.split(","),
        required=False,
        help="Comma-separated list of adaptor names",
    )
    args = parser.parse_args()

    # Load the checkpoint and create the model.
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]

    # Remove invalid identifiers.
    invalid_identifiers = [
        "enable-cudnn-attention",
        "device",
        "damp",
    ]
    for invalid_identifier in invalid_identifiers:
        if hasattr(model_args, invalid_identifier):
            print(f'Removing attribute: {invalid_identifier}!')
            delattr(model_args, invalid_identifier)

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

    # Handle adaptors.
    adaptor_names = args.adaptor_names
    if adaptor_names is None:
        adaptor_names = []
    elif isinstance(adaptor_names, str):
        adaptor_names = [adaptor_names]


    # We need to extract the teacher configurations and adaptor states
    # from the checkpoint.
    teachers = model_args.teachers
    adaptor_configs = dict()
    adaptor_states = dict()
    for adaptor_name in adaptor_names:
        for tidx, tconf in enumerate(teachers):
            if tconf["name"] == adaptor_name:
                break
        else:
            raise ValueError(
                f'Unable to find the specified adaptor name. Known names: {list(t["name"] for t in teachers)}'
            )

        pf_idx_head = f"_heads.{tidx}"
        pf_name_head = f"_heads.{adaptor_name}"
        pf_idx_feat = f"_feature_projections.{tidx}"
        pf_name_feat = f"_feature_projections.{adaptor_name}"

        adaptor_state = dict()
        for k, v in state_dict.items():
            if k.startswith(pf_idx_head):
                adaptor_state["summary" + k[len(pf_idx_head) :]] = v
            elif k.startswith(pf_name_head):
                adaptor_state["summary" + k[len(pf_name_head) :]] = v
            elif k.startswith(pf_idx_feat):
                adaptor_state["feature" + k[len(pf_idx_feat) :]] = v
            elif k.startswith(pf_name_feat):
                adaptor_state["feature" + k[len(pf_name_feat) :]] = v
        adaptor_states[adaptor_name] = adaptor_state

        adaptor_config = dict()
        adaptor_config["head_idx"] = tidx

        input_dim, hidden_dim, output_dim, num_inner = get_mlp_info_from_state(
            model_args.mlp_version, adaptor_state, "summary."
        )
        adaptor_config["summary"] = dict()
        adaptor_config["summary"]["input_dim"] = input_dim
        adaptor_config["summary"]["hidden_dim"] = hidden_dim
        adaptor_config["summary"]["output_dim"] = output_dim
        adaptor_config["summary"]["num_inner"] = num_inner

        input_dim, hidden_dim, output_dim, num_inner = get_mlp_info_from_state(
            model_args.mlp_version, adaptor_state, "feature."
        )
        adaptor_config["feature"] = dict()
        adaptor_config["feature"]["input_dim"] = input_dim
        adaptor_config["feature"]["hidden_dim"] = hidden_dim
        adaptor_config["feature"]["output_dim"] = output_dim
        adaptor_config["feature"]["num_inner"] = num_inner
        adaptor_config["feature"]["upsample_factor"] = tconf.get("fd_upsample_factor", 1)
        adaptor_config["feature"]["upsample_rank"] = tconf.get("fd_upsample_rank", None)

        adaptor_configs[adaptor_name] = adaptor_config


    feat_norm_sd = get_prefix_state_dict(state_dict, '_feature_normalizer.')
    feature_normalizer_config = None
    if feat_norm_sd:
        feature_normalizer_config = {
            "embed_dim": feat_norm_sd['mean'].shape[0]
        }

    inter_feat_norm_sd = get_prefix_state_dict(state_dict, '_intermediate_feature_normalizer.')
    inter_feature_normalizer_config = None
    if inter_feat_norm_sd:
        inter_feature_normalizer_config = {
            "num_intermediates": inter_feat_norm_sd['means'].shape[0],
            "embed_dim": inter_feat_norm_sd['means'].shape[1],
            "rot_per_layer": inter_feat_norm_sd['rotation'].ndim == 3,
        }

    model_vars = vars(model_args)
    model_vars.pop('enable-cudnn-attention', None)

    radio_config = RADIOConfig(
        model_vars,
        version=args.version,
        adaptor_names=adaptor_names,
        adaptor_configs=adaptor_configs,
        feature_normalizer_config=feature_normalizer_config,
        inter_feature_normalizer_config=inter_feature_normalizer_config,
    )
    radio_model = RADIOModel(radio_config)

    # Restore the model weights.
    key_warn = radio_model.model.load_state_dict(
        get_prefix_state_dict(state_dict, "base_model."), strict=False
    )
    if key_warn.missing_keys:
        warning(f"Missing keys in state dict: {key_warn.missing_keys}")
    if key_warn.unexpected_keys:
        warning(f"Unexpected keys in state dict: {key_warn.unexpected_keys}")

    # Restore the adaptor weights from their state. This needs to happen
    # after the model is instantiated from the config.
    for adaptor_name, adaptor_state in adaptor_states.items():
        adaptor_state = replace_prefix_in_state_dict(
            adaptor_state, "summary.", "head_mlp."
        )
        adaptor_state = replace_prefix_in_state_dict(
            adaptor_state, "feature.", "feat_mlp."
        )
        radio_model.adaptors[adaptor_name].load_state_dict(adaptor_state)

    # Restore input conditioner.
    radio_model.input_conditioner.load_state_dict(
        get_prefix_state_dict(state_dict, "input_conditioner.")
    )

    # Restore feature normalizer.
    if feat_norm_sd:
        radio_model.radio_model.feature_normalizer.load_state_dict(feat_norm_sd)
    if inter_feat_norm_sd:
        radio_model.radio_model.inter_feature_normalizer.load_state_dict(inter_feat_norm_sd)

    radio_model.eval().cuda()

    # Sample inference with deterministic values.
    x = deterministic_grid_init(
        (
            1,
            3,
            radio_model.config.preferred_resolution[0],
            radio_model.config.preferred_resolution[1],
        )
    ).cuda()

    # Infer using HuggingFace model.
    with torch.no_grad():
        hf_output = radio_model(x)
    if isinstance(hf_output, tuple):
        # The model returns a single tuple if there are no adaptors.
        hf_output = dict(backbone=RadioOutput(hf_output[0], hf_output[1]))
    for k, v in hf_output.items():
        hf_summary, hf_features = v.summary, v.features

        print(
            f"[{k}] HF inference on tensor shape {x.shape} returned summary ",
            f"with shape={hf_summary.shape} and std={hf_summary.std().item():.3}, ",
            f"features with shape={hf_features.shape} and std={hf_features.std().item():.3}",
        )

    with torch.no_grad():
        intermediates = radio_model.radio_model.forward_intermediates(
                            x,
                            indices=[-1],
                            return_prefix_tokens=True,
                            norm=False,
                            stop_early=False,
                            output_fmt='NLC',
                            intermediates_only=True,
                            aggregation="sparse",
                        )
    print(
        f"Intermediates inference returned ",
        f"features with shape={intermediates[0].features.shape} and std={intermediates[0].features.std().item():.3}",
    )
    assert torch.allclose(intermediates[0].features, hf_output["backbone"].features, atol=1e-4)

    # Infer using TorchHub model.
    print("Infer using TorchHub model...")
    torchhub_model = torch.hub.load(
        args.torchhub_repo,
        "radio_model",
        version=args.checkpoint_path,
        force_reload=True,
        adaptor_names=adaptor_names,
    )
    torchhub_model.cuda().eval()

    with torch.no_grad():
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
            f"[{k}] TorchHub inference on tensor shape {x.shape} returned summary ",
            f"with shape={torchhub_summary.shape} and std={torchhub_summary.std().item():.3}, ",
            f"features with shape={torchhub_features.shape} and std={torchhub_features.std().item():.3}",
        )

        # Make sure the shapes are the same.
        assert (
            hf_summary.shape == torchhub_summary.shape
        ), f"{k} Summary shapes do not match! hf={hf_summary.shape}, torchhub={torchhub_summary.shape}"
        assert (
            hf_features.shape == torchhub_features.shape
        ), f"{k} Features shapes do not match! hf={hf_features.shape}, torchhub={torchhub_features.shape}"

        # Make sure the results are the same.
        assert torch.allclose(
            hf_summary, torchhub_summary, atol=1e-6
        ), f"{k} Summaries do not match ({hf_summary.std().item()} vs {torchhub_summary.std().item()})!"
        assert torch.allclose(
            hf_features, torchhub_features, atol=1e-6
        ), f"{k} Features do not match ({hf_features.std().item()} vs {torchhub_features.std().item()})!"

        print(f"{k} outputs matched!")



    print("All outputs matched!")

    if args.push:
        # Push to HuggingFace Hub.
        huggingface_repo = args.hf_repo
        # Clear the adaptor names before pushing so that we default to
        # just returning the backbone features.
        radio_model.config.adaptor_names = None
        commit = radio_model.push_to_hub(
            huggingface_repo, create_pr=True, commit_message=args.commit_message
        )
        print(f"Pushed to {commit}")

if __name__ == "__main__":
    """Call the main entrypoiny."""
    main()
