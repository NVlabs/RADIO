#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Progressive Reconstruction Fidelity for RADIO1D Models

This script analyzes how reconstruction quality changes with different numbers of 1D tokens.
For each image, the encoder is run once to produce the full compressed representation.
Then we progressively vary the number of tokens fed to the decoder and measure
reconstruction error compared to the full-token reconstruction baseline.

Key metrics:
- MSE (Mean Squared Error): Per-pixel squared difference
- MAE (Mean Absolute Error): Per-pixel absolute difference
- Cosine Similarity: Feature-level similarity
- PSNR (Peak Signal-to-Noise Ratio): Reconstruction quality in dB
"""

import argparse
import csv
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from datasets import load_dataset_builder
from datasets.distributed import split_dataset_by_node

from common import rank_print, load_model, get_standard_transform, collate
from radio.input_conditioner import InputConditioner


def compute_reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute reconstruction metrics between prediction and target.

    Args:
        pred: Predicted features (B, N, C) or (B, C, H, W)
        target: Target features (B, N, C) or (B, C, H, W)

    Returns:
        Dictionary of metric names to values
    """
    # Flatten spatial dimensions if needed
    if pred.dim() == 4:
        pred = pred.flatten(2).transpose(1, 2)  # (B, N, C)
        target = target.flatten(2).transpose(1, 2)

    std_target = target.std(dim=(1, 2), unbiased=True, keepdim=True)
    pred = pred / std_target
    target = target / std_target

    # MSE
    mse = F.mse_loss(pred, target, reduction='mean').item()

    # MAE
    mae = F.l1_loss(pred, target, reduction='mean').item()

    # Cosine similarity (per token, then averaged)
    pred_norm = F.normalize(pred, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)
    cos_sim = (pred_norm * target_norm).sum(dim=-1).mean().item()

    # PSNR (Peak Signal-to-Noise Ratio)
    # Assuming features are roughly in [-1, 1] range, max value = 2
    max_val = 2.0
    psnr = 10 * np.log10(max_val ** 2 / (mse + 1e-10))

    return {
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cos_sim,
        'psnr': psnr,
    }


@torch.no_grad()
def analyze_progressive_reconstruction(
    model,
    images: torch.Tensor,
    preprocessor,
    token_counts: List[int],
    mode: str,
    use_last_tokens: bool = False,
) -> Dict[int, Dict[str, float]]:
    """Analyze reconstruction with different token counts.

    Args:
        model: RADIO1D model
        images: Input images (B, C, H, W)
        preprocessor: Input preprocessor
        token_counts: List of token counts to test
        use_last_tokens: If True, use last N tokens instead of first N

    Returns:
        Dictionary mapping token_count -> metrics dict
    """
    device = images.device

    max_tokens = images.shape[-2] * images.shape[-1] // ((2 * model.patch_generator.patch_size) ** 2)

    with torch.autocast(device.type, dtype=torch.bfloat16):
        # Preprocess images
        p_images = preprocessor(images)

        # Run encoder once with all tokens to get baseline
        # For baseline, we use the maximum available tokens
        encoder_output = model.forward_encoder(p_images, num_tokens=max_tokens)

        # Get the baseline reconstruction (with all tokens)
        baseline_output = model.forward_decoder(
            encoder_output['global_tokens'],
            encoder_output['global_token_mask'],
            encoder_output['encoder_spatial_size'],
            encoder_output['original_spatial_size'],
        )

        baseline_output = baseline_output[:, model.num_prefix_tokens:]

        # Store results for each token count
        results = {}

        # Test different token counts
        for num_tokens in token_counts:
            if mode == 'prefix':
                global_tokens = encoder_output['global_tokens'][:, :num_tokens]
                global_token_mask = encoder_output['global_token_mask'][:, :num_tokens]
            elif mode == 'select':
                global_tokens = encoder_output['global_tokens'][:, num_tokens-1:num_tokens]
                global_token_mask = encoder_output['global_token_mask'][:, num_tokens-1:num_tokens]

            # Run decoder
            decoder_output = model.forward_decoder(
                global_tokens,
                global_token_mask,
                encoder_output['encoder_spatial_size'],
                encoder_output['original_spatial_size'],
            )

            decoder_output = decoder_output[:, model.num_prefix_tokens:]

            # Compute metrics against baseline
            metrics = compute_reconstruction_metrics(decoder_output, baseline_output)
            results[num_tokens] = metrics

    return results


def parse_experiment_and_checkpoint(model_version: str) -> Tuple[str, str]:
    """Parse experiment name and checkpoint name from model version string.

    Args:
        model_version: Model version string (either a path or a model name)

    Returns:
        Tuple of (experiment_name, checkpoint_name)
    """
    # Check if it's a file
    if os.path.isfile(model_version):
        path = Path(model_version)
        # Checkpoint name is the filename without extension(s)
        checkpoint_name = path.name
        # Remove all extensions (handles .pth.tar, .pth, etc.)
        while checkpoint_name != (checkpoint_name := os.path.splitext(checkpoint_name)[0]):
            pass

        # Experiment name is the parent of the checkpoints directory
        experiment_name = path.parent.parent.name

        return experiment_name, checkpoint_name
    else:
        # Not a file path, use the model version as experiment name
        return model_version, 'release'


def write_results_to_csv(
    csv_path: str,
    experiment_name: str,
    checkpoint_name: str,
    summary_stats: Dict[int, Dict[str, float]],
) -> None:
    """Write or update results in a CSV file.

    Args:
        csv_path: Path to the CSV file
        experiment_name: Name of the experiment
        checkpoint_name: Name of the checkpoint
        summary_stats: Dictionary mapping token_count to statistics
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Define CSV columns
    fieldnames = ['Experiment name', 'Checkpoint name', 'Tokens', 'MSE', 'MAE', 'Cosine Sim']

    # Read existing data if file exists
    existing_data = []
    file_exists = os.path.exists(csv_path)

    my_token_counts = set(summary_stats.keys())

    if file_exists:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows with matching experiment + checkpoint (we'll replace them)
                if not (row['Experiment name'] == experiment_name and
                       row['Checkpoint name'] == checkpoint_name and
                       int(row['Tokens']) in my_token_counts):
                    existing_data.append(row)

    # Add new results
    for token_count, stats in sorted(summary_stats.items()):
        existing_data.append({
            'Experiment name': experiment_name,
            'Checkpoint name': checkpoint_name,
            'Tokens': str(token_count),
            'MSE': f"{stats['mse_mean']:.6f}",
            'MAE': f"{stats['mae_mean']:.6f}",
            'Cosine Sim': f"{stats['cos_sim_mean']:.6f}",
        })

    # Write all data back to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)


def generate_plot(csv_path: str, metric: str = 'MSE', mode: str = 'prefix') -> None:
    """Generate a plot from the CSV results.

    Args:
        csv_path: Path to the CSV file
        metric: Metric to plot on y-axis (MSE, MAE, or Cosine Sim)
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)

    # Convert Tokens to numeric
    df['Tokens'] = pd.to_numeric(df['Tokens'])

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Get unique experiments and checkpoints for styling
    experiments = df['Experiment name'].unique()
    checkpoints = df['Checkpoint name'].unique()

    # Define line styles for checkpoints
    line_styles = ['-', '--', '-.', ':']
    checkpoint_styles = {cp: line_styles[i % len(line_styles)]
                        for i, cp in enumerate(checkpoints)}

    # Plot each combination of experiment and checkpoint
    for experiment in experiments:
        for checkpoint in checkpoints:
            mask = (df['Experiment name'] == experiment) & (df['Checkpoint name'] == checkpoint)
            subset = df[mask]

            if len(checkpoints) == 1:
                label = experiment
            else:
                label = f'{experiment} ({checkpoint})'

            if not subset.empty:
                sns.lineplot(
                    data=subset,
                    x='Tokens',
                    y=metric,
                    label=label,
                    linestyle=checkpoint_styles[checkpoint],
                    marker='o'
                )

    plt.xlabel('Number of Prefix Tokens' if mode == 'prefix' else 'Token Index', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.yscale('log')
    plt.title(f'Self Reconstruction: {metric} vs Tokens', fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plot_path = csv_path.replace('.csv', '_plot.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    rank_print(f'Plot saved to: {plot_path}')


def main(rank: int = 0, world_size: int = 1):
    """Main function for progressive reconstruction analysis."""

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    parser = argparse.ArgumentParser(
        description='Analyze RADIO1D reconstruction fidelity with varying token counts'
    )
    parser.add_argument(
        '-v', '--model-version',
        default='',
        help='Which radio1d model to load (e.g., radio1d_so400m_patch16_224, radio1d_large_patch16_224)'
    )
    parser.add_argument(
        '-d', '--dataset',
        default='detection-datasets/coco',
        help='The name of the dataset to use'
    )
    parser.add_argument(
        '--split',
        default='train',
        help='The dataset split to use'
    )
    parser.add_argument(
        '-n', '--num-samples',
        default=100,
        type=int,
        help='The number of samples to analyze'
    )
    parser.add_argument(
        '-r', '--resolution',
        nargs='+',
        type=int,
        default=(512, 512),
        help='The input image resolution. If one value, shortest dim is resized. '
             'If two, image is center cropped. Default: model preferred resolution.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='The batch size'
    )
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--token-counts',
        type=str,
        default='1,2,3,4,5,6,7,8,16,32,64,128,224',
        help='Comma-separated list of token counts to test (e.g., "16,32,64,128")'
    )
    parser.add_argument(
        '--use-last-tokens',
        action='store_true',
        help='Use the last N tokens instead of first N tokens'
    )
    parser.add_argument(
        '--torchhub-repo',
        default='NVlabs/RADIO',
        help='Path to the Torchhub repo'
    )
    parser.add_argument(
        '--mode',
        default='prefix',
        choices=['prefix', 'select'],
        type=str,
        help='Token selection mode',
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Path to output CSV file for results. If specified, results will be written to this file.'
    )
    parser.add_argument(
        '--plot-metric',
        type=str,
        default='MSE',
        choices=['MSE', 'MAE', 'Cosine Sim'],
        help='Metric to plot on y-axis when generating plot (default: MSE)'
    )
    parser.add_argument(
        '--just-plot',
        action='store_true',
        help='If specified, only generate the plot from the existing CSV file',
    )

    args = parser.parse_args()

    if args.just_plot:
        if args.output_csv is None:
            raise ValueError('Must specify --output-csv when using --just-plot')
        generate_plot(args.output_csv, args.plot_metric, args.mode)
        return

    if not args.model_version:
        raise ValueError('You must specify a model version using -v / --model-version')

    # Parse token counts
    token_counts = sorted([int(x.strip()) for x in args.token_counts.split(',')])
    rank_print(f'Testing token counts: {token_counts}')

    # Parse experiment and checkpoint names
    experiment_name, checkpoint_name = parse_experiment_and_checkpoint(args.model_version)
    rank_print(f'Experiment: {experiment_name}, Checkpoint: {checkpoint_name}')

    # Load model
    rank_print('Loading model...')
    model, preprocessor, info = load_model(
        args.model_version,
        torchhub_repo=args.torchhub_repo,
        adaptor_name=None,
    )
    model = model.to(device=device).eval()
    preprocessor = preprocessor.to(device=device).eval()
    rank_print(f'Loaded model: {args.model_version}')

    model = model.model

    # Verify this is a RADIO1D model
    if not hasattr(model, 'forward_encoder') or not hasattr(model, 'forward_decoder'):
        raise ValueError(
            f'Model {args.model_version} does not appear to be a RADIO1D model. '
            'This script requires forward_encoder and forward_decoder methods.'
        )

    # Setup data transform
    patch_size = getattr(model, 'patch_size', 16)
    if args.resolution is None:
        args.resolution = getattr(model, 'preferred_resolution', (224, 224))

    transform = get_standard_transform(
        args.resolution,
        resize_multiple=getattr(model, 'min_resolution_step', patch_size),
        max_dim=False,
        pad_mean=preprocessor.norm_mean if isinstance(preprocessor, InputConditioner) else None,
    )

    # Load dataset
    rank_print('Loading dataset...')
    if not os.path.isdir(args.dataset):
        try:
            ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
        except:
            ds_builder = load_dataset_builder("imagefolder", data_dir=args.dataset)
        ds_builder.download_and_prepare()
        dataset = ds_builder.as_dataset(split=args.split)
        dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, args.workers))
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        dataset = dataset.map(
            lambda ex: dict(
                image=transform(ex['image']),
                label=torch.zeros(1, dtype=torch.int64)
            )
        )
    else:
        dataset = ImageFolder(args.dataset, transform=transform)
        dataset.samples.sort(key=lambda s: s[0])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate, group=False),
        pin_memory=args.workers > 0,
        drop_last=False,
    )
    rank_print('Dataset loaded')

    # Accumulate results across all images
    all_results = {count: {'mse': [], 'mae': [], 'cosine_similarity': [], 'psnr': []}
                   for count in token_counts}

    sample_count = 0

    # Process batches
    rank_print('Processing images...')
    pbar = tqdm(total=args.num_samples, desc='Analyzing reconstruction', disable=rank != 0)

    for batches in loader:
        if sample_count >= args.num_samples:
            break

        for images, _ in batches:
            if sample_count >= args.num_samples:
                break

            images = images.to(device=device, non_blocking=True)

            # Analyze this batch
            batch_results = analyze_progressive_reconstruction(
                model=model,
                images=images,
                preprocessor=preprocessor,
                token_counts=token_counts,
                use_last_tokens=args.use_last_tokens,
                mode=args.mode,
            )

            # Accumulate results
            for count, metrics in batch_results.items():
                for metric_name, value in metrics.items():
                    all_results[count][metric_name].append(value)

            sample_count += images.shape[0]
            pbar.update(images.shape[0])

    pbar.close()

    # Compute statistics
    rank_print('\n' + '=' * 80)
    rank_print('RECONSTRUCTION FIDELITY RESULTS')
    rank_print('=' * 80)
    rank_print(f'Model: {args.model_version}')
    rank_print(f'Samples analyzed: {sample_count}')
    rank_print(f'Use last tokens: {args.use_last_tokens}')
    rank_print('=' * 80)

    # Print results table
    rank_print(f'\n{"Tokens":<10} {"MSE":<12} {"MAE":<12} {"Cosine Sim":<15} {"PSNR (dB)":<12}')
    rank_print('-' * 65)

    summary_stats = {}
    for count in token_counts:
        metrics = all_results[count]
        stats = {
            'mse_mean': np.mean(metrics['mse']),
            'mse_std': np.std(metrics['mse']),
            'mae_mean': np.mean(metrics['mae']),
            'mae_std': np.std(metrics['mae']),
            'cos_sim_mean': np.mean(metrics['cosine_similarity']),
            'cos_sim_std': np.std(metrics['cosine_similarity']),
            'psnr_mean': np.mean(metrics['psnr']),
            'psnr_std': np.std(metrics['psnr']),
        }
        summary_stats[count] = stats

        rank_print(
            f'{count:<10} '
            f'{stats["mse_mean"]:<12.6f} '
            f'{stats["mae_mean"]:<12.6f} '
            f'{stats["cos_sim_mean"]:<15.6f} '
            f'{stats["psnr_mean"]:<12.2f}'
        )

    rank_print('=' * 80)

    # Write results to CSV if requested
    if args.output_csv and rank == 0:
        write_results_to_csv(
            csv_path=args.output_csv,
            experiment_name=experiment_name,
            checkpoint_name=checkpoint_name,
            summary_stats=summary_stats,
        )
        rank_print(f'\nResults written to: {args.output_csv}')

        # Generate plot
        generate_plot(args.output_csv, args.plot_metric)


if __name__ == '__main__':
    main()
