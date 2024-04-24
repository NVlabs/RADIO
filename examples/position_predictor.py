# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from collections import defaultdict
import math
import os
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, Iterable, Optional, List

from timm.models.vision_transformer import VisionTransformer
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms.v2 as transforms

from datasets import load_dataset_builder, load_dataset
from datasets.iterable_dataset import DistributedConfig
from datasets.distributed import split_dataset_by_node


def main(rank: int = 0, world_size: int = 1):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    distributed = world_size > 1

    import debugpy

    if False:  # local_rank == 0:
        print("Waiting for client...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("Connected!")

    parser = argparse.ArgumentParser(description="kNN Classification Demo")
    parser.add_argument(
        "--use-hf",
        default=False,
        action="store_true",
        help="Use RADIO from HuggingFace Hub",
    )
    parser.add_argument(
        "-v", "--model-version", default="radio_v2", help="Which radio model to load."
    )

    parser.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        type=int,
        default=None,
        help="The input image resolution."
        " If one value is specified, the shortest dimension is resized to this."
        " If two, the image is center cropped."
        " If not specified, center cropped 378px is used.",
    )
    parser.add_argument(
        "-d", "--dataset", default="food101", help="The name of the dataset to classify"
    )
    parser.add_argument(
        "--eval-dataset",
        default=None,
        type=str,
        help="The name of the evaluation dataset, if different than the training one.",
    )
    parser.add_argument(
        "--resize-multiple",
        type=int,
        default=16,
        help="Resize images with dimensions a multiple of this value."
        " This should be equal to the patch size of a ViT (e.g. RADIOv1)",
    )

    parser.add_argument(
        "--train-split", default="train", help="The dataset training split to use"
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="The evaluation split to use. If labels are present, accuracy will be computed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size. If the input is variable sized, then this argument becomes a maximum.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=8,
        type=int,
        help="The number of data loader workers to use per GPU",
    )
    parser.add_argument(
        "-e", "--num-epochs", default=5, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-2, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--torchhub-repo",
        help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )

    args, _ = parser.parse_known_args()

    def rank_print(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    rank_print("Loading model...")
    if args.use_hf:
        from transformers import AutoModel

        radio_model = AutoModel.from_pretrained(
            f"nvidia/{args.model_version}", trust_remote_code=True
        )
    else:
        radio_model = torch.hub.load(
            args.torchhub_repo, "radio_model", version=args.model_version, progress=True
        )

    radio_model.to(device=device)
    rank_print("Done")

    position_predictor = PositionPredictor(radio_model)
    position_predictor.to(device=device)

    if distributed:
        position_predictor = DDP(position_predictor, device_ids=[local_rank], find_unused_parameters=True)

    if args.resolution is None:
        args.resolution = (radio_model.preferred_resolution.height, radio_model.preferred_resolution.width)

    if args.resize_multiple is None:
        args.resize_multiple = radio_model.min_resolution_step

    transform = [
        ResizeTransform(args.resolution, args.resize_multiple),
    ]
    if len(args.resolution) == 2:
        transform.append(transforms.CenterCrop(args.resolution))
    transform.extend(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),]
    )
    transform = transforms.Compose(transform)

    rank_print("Loading dataset...")
    ds_train_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    num_train_examples = ds_train_builder.info.splits[args.train_split].num_examples

    def _prepare(builder):
        for i in range(min(2, world_size)):
            if i == rank or (i > 0 and rank > 0):
                builder.download_and_prepare()
            if world_size > 1:
                dist.barrier()

    _prepare(ds_train_builder)

    if args.eval_dataset:
        ds_eval_builder = load_dataset_builder(
            args.eval_dataset, trust_remote_code=True
        )
        _prepare(ds_eval_builder)
    else:
        ds_eval_builder = ds_train_builder

    num_eval_examples = ds_eval_builder.info.splits[args.eval_split].num_examples

    num_train_steps = _round_up(num_train_examples, args.batch_size * world_size)
    num_eval_steps = _round_up(num_eval_examples, args.batch_size * world_size)

    def _get_dataset(builder, split: str):
        dataset = builder.as_dataset(split=split)
        dataset = dataset.to_iterable_dataset(
            num_shards=world_size * max(1, args.workers)
        )
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        dataset = dataset.map(lambda ex: dict(image=transform(ex["image"]), label=None))

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=_collate,
            pin_memory=args.workers > 0,
            drop_last=False,
        )

        return loader

    train_dataset = _get_dataset(ds_train_builder, args.train_split)
    eval_dataset = _get_dataset(ds_eval_builder, args.eval_split)

    optimizer = torch.optim.Adam(position_predictor.parameters(), lr=args.learning_rate)
    #loss = torch.nn.MSELoss()
    loss = torch.nn.SmoothL1Loss()

    rank_print("Loaded dataset!")
    rank_print(f"Description: {ds_train_builder.info.description}")

    for epoch in range(args.num_epochs):
        train(position_predictor, train_dataset, optimizer, loss, epoch, device, num_train_steps, rank, distributed)

        evaluate(
            position_predictor,
            eval_dataset,
            loss,
            device,
            num_eval_steps,
            rank,
            distributed,
        )


class ResizeTransform(transforms.Transform):
    def __init__(self, size: Iterable[int], resize_multiple: int = 1):
        super().__init__()

        self.size = size
        self.resize_multiple = resize_multiple

    def _get_nearest(self, value: int):
        return int(round(value / self.resize_multiple) * self.resize_multiple)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = transforms._utils.query_size(flat_inputs)

        if len(self.size) == 1:
            # Shortest-side mode.
            # Resize the short dimension of the image to be the specified size,
            # and the other dimension aspect preserving
            min_sz = min(height, width)
            factor = self.size[0] / min_sz

            rs_height = height * factor
            rs_width = width * factor
            size = (rs_height, rs_width)
        elif len(self.size) == 2:
            # Center-crop mode (the actual crop will be done by subsequent transform)
            in_aspect = height / width
            out_aspect = self.size[0] / self.size[1]

            # Input height varies faster than output
            if in_aspect > out_aspect:
                scale = self.size[1] / width
            else:
                scale = self.size[0] / height

            rs_height = height * scale
            rs_width = width * scale
            size = (rs_height, rs_width)
        else:
            raise ValueError("Unsupported resize mode")

        size = tuple(self._get_nearest(d) for d in size)

        return dict(size=size)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, Image.Image):
            inpt = inpt.convert("RGB")

        size = params["size"]

        return transforms.functional.resize(
            inpt, size=size, interpolation=transforms.InterpolationMode.BICUBIC
        )


def _round_up(value, multiple: int):
    return int(math.ceil(value / multiple))


def _collate(samples: List[torch.Tensor]):
    images = [s["image"] for s in samples]
    size_groups = defaultdict(lambda: [])
    for im in images:
        grp = size_groups[im.shape]
        grp.append(im)

    ret = [torch.stack(g) for g in size_groups.values()]

    return ret


class PositionPredictor(torch.nn.Module):
    """
    A model that learns to predict the position of a patch
    token in the output of a RADIO model.
    """

    def __init__(
        self, radio_model: torch.nn.Module, **kwargs,
    ):
        super().__init__()

        self.radio_model = radio_model

        if isinstance(radio_model.model, VisionTransformer):
            embed_dim = radio_model.model.embed_dim
        else:
            raise ValueError(f"Unhandled model type: {radio_model.model}")

        # A 1x1 conv2d with two output channels to predict the x,y coordinates of patch tokens.
        self.conv2d = nn.Conv2d(
            embed_dim, 2, kernel_size=1, stride=1, padding=0, bias=True
        )

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass."""
        _, _, H, W = x.shape

        _, features = self.radio_model(x)

        if isinstance(self.radio_model.model, VisionTransformer):
            # Reshape
            B, _, C = features.shape

            if hasattr(self.radio_model.model, "patch_generator"):
                # Cropped Positional Embedding (CPE) case.
                patch_height = (
                    patch_width
                ) = self.radio_model.model.patch_generator.patch_size
            else:
                # Standard ViT case.
                (
                    patch_height,
                    patch_width,
                ) = self.radio_model.model.patch_embed.patch_size
            features = (
                features.reshape(
                    B, math.ceil(H / patch_height), math.ceil(W / patch_width), C
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        # IMPORTANT: prevent gradients from flowing back towards the backbone.
        features = features.detach()

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        # normalize features
        features = F.normalize(features, dim=1)
        predictions = self.conv2d(features)
        #predictions = torch.tanh(predictions)
        return predictions

    def train(self, mode=True):
        """Intercept call.

        We want to train the conv2d but keep the radio model in eval mode."""
        self.conv2d.train(mode)
        self.radio_model.eval()

    def eval(self):
        self.train(False)


def generate_position_tensor(predictions):
    B, _, H, W = predictions.shape
    # Generate x positions in the range [-1, 1]
    x_positions = torch.linspace(-1, 1, W)

    # Generate y positions in the range [-1, 1]
    y_positions = torch.linspace(-1, 1, H)

    # Expand dimensions to create 2D tensors
    x_positions = x_positions.view(1, -1).expand(H, -1)
    y_positions = y_positions.view(-1, 1).expand(-1, W)

    # Concatenate along the first dimension to create a 3D tensor
    position_tensor = torch.stack([x_positions, y_positions])

    # Unsqueeze to add a batch dimension
    position_tensor = position_tensor.unsqueeze(0).expand(B, -1, -1, -1)

    return position_tensor.to(device=predictions.device, dtype=predictions.dtype)


def train(
    model: PositionPredictor,
    dataset: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epoch: int,
    device: torch.device,
    num_steps: int,
    rank: int,
    distributed: bool,
    overfit_test: Optional[bool] = False,
):
    """Training routine."""
    model.train()

    num_samples = 0
    loss_sum = 0.0

    one_batch = None

    with tqdm(dataset, total=num_steps, disable=rank > 0) as tbar:

        tbar.set_description(f"Train Epoch {epoch}")

        for batches in tbar:
            batches_loss_sum = torch.tensor(0.0).to(device=device)
            batches_num_samples = torch.tensor(0).to(device=device)

            for batch in batches:
                if overfit_test:
                    if one_batch is None:
                        one_batch = batch
                    batch = one_batch
                samples_in_batch = batch.shape[0]
                images = batch.to(device=device, non_blocking=True)

                predictions = model(images)
                targets = generate_position_tensor(predictions)

                #print(f"predictions shape={predictions.shape} min={predictions.min().item()} mean={predictions.mean().item()} max={predictions.max().item()}")
                #print(f"targets shape={targets.shape} min={targets.min().item()} mean={targets.mean().item()} max={targets.max().item()}")

                loss = loss_fn(predictions, targets)

                batches_loss_sum += loss * samples_in_batch
                batches_num_samples += samples_in_batch

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if distributed:
                dist.all_reduce(batches_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(batches_num_samples, op=dist.ReduceOp.SUM)

            loss_sum += batches_loss_sum.item()
            num_samples += batches_num_samples

            tbar.set_postfix(loss=(batches_loss_sum / batches_num_samples).item(), lr=optimizer.param_groups[0]['lr'])

def evaluate(
    model: PositionPredictor,
    dataset: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_steps: int,
    rank: int,
    distributed: bool,
):
    """Evaluation routine."""
    model.eval()

    num_samples = 0
    loss_sum = 0.0

    with tqdm(dataset, total=num_steps, disable=rank > 0) as tbar:

        tbar.set_description(f"Eval")

        for batches in tbar:
            batches_loss_sum = torch.tensor(0.0).to(device=device)
            batches_num_samples = torch.tensor(0).to(device=device)

            for batch in batches:
                samples_in_batch = batch.shape[0]
                images = batch.to(device=device, non_blocking=True)

                predictions = model(images)
                targets = generate_position_tensor(predictions)

                loss = loss_fn(predictions, targets)

                batches_loss_sum += loss * samples_in_batch
                batches_num_samples += samples_in_batch

            if distributed:
                dist.all_reduce(batches_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(batches_num_samples, op=dist.ReduceOp.SUM)

            loss_sum += batches_loss_sum.item()
            num_samples += batches_num_samples

            tbar.set_postfix(loss=(batches_loss_sum / batches_num_samples).item())

    if rank==0:
        print(f"Average validation loss={(loss_sum / num_samples)}")

if __name__ == "__main__":
    rank = 0
    world_size = 1

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    main(rank, world_size)
