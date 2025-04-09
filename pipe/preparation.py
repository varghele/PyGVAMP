#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VAMPNet Data Preparation Pipeline

This script processes molecular dynamics trajectories and prepares them
as graph datasets for training VAMPNet models.
"""

import os
import torch
from torch_geometric.loader import DataLoader
from datetime import datetime
import json

from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.utils.pipe_utils import find_trajectory_files
from args import parse_prep_args


def setup_output_directory(args):
    """Setup output directory and return paths"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    run_dir = os.path.join(args.output_dir, f"prep_{timestamp}")
    cache_dir = args.cache_dir or os.path.join(run_dir, "cache")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Create paths
    paths = {
        'run_dir': run_dir,
        'cache_dir': cache_dir,
        'config': os.path.join(run_dir, 'prep_config.json'),
        'stats': os.path.join(run_dir, 'dataset_stats.json'),
    }

    return paths


def save_config(args, paths):
    """Save configuration to a JSON file"""
    # Convert args Namespace to a dictionary for saving
    config = vars(args)

    with open(paths['config'], 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {paths['config']}")


def create_and_analyze_dataset(args, paths):
    """Create and analyze the dataset"""
    print(f"Looking for trajectory files in {args.traj_dir}")

    # Find trajectory files
    trajectory_files = find_trajectory_files(
        args.traj_dir,
        file_pattern=args.file_pattern,
        recursive=args.recursive
    )

    if not trajectory_files:
        print("No trajectory files found. Exiting.")
        return None

    print(f"\nCreating dataset with {len(trajectory_files)} trajectory files")

    # Create dataset
    dataset = VAMPNetDataset(
        trajectory_files=trajectory_files,
        topology_file=args.top,
        lag_time=args.lag_time,
        n_neighbors=args.n_neighbors,
        node_embedding_dim=args.node_embedding_dim,
        gaussian_expansion_dim=args.gaussian_expansion_dim,
        selection=args.selection,
        stride=args.stride,
        cache_dir=paths['cache_dir'],
        use_cache=args.use_cache
    )

    print(f"Dataset created with {len(dataset)} samples")

    # Collect and save statistics
    stats = {
        'num_samples': len(dataset),
        'trajectories': [os.path.basename(f) for f in trajectory_files],
        'parameters': {
            'lag_time': args.lag_time,
            'n_neighbors': args.n_neighbors,
            'node_embedding_dim': args.node_embedding_dim,
            'gaussian_expansion_dim': args.gaussian_expansion_dim,
            'selection': args.selection,
            'stride': args.stride
        }
    }

    # Sample a data point to get structure information
    if len(dataset) > 0:
        sample = dataset[0]
        x_t0, x_t1 = sample

        stats['structure'] = {
            'num_nodes': x_t0.num_nodes,
            'num_edges': x_t0.edge_index.shape[1],
            'node_feature_dim': x_t0.x.shape[1],
            'edge_feature_dim': x_t0.edge_attr.shape[1]
        }

    # Save statistics
    with open(paths['stats'], 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Dataset statistics saved to {paths['stats']}")

    return dataset


def analyze_sample_batch(dataset, args):
    """Analyze a sample batch from the dataset"""
    if not dataset or len(dataset) == 0:
        print("Dataset is empty. Cannot analyze sample batch.")
        return

    print(f"\nAnalyzing sample batch (batch_size={args.batch_size})...")

    # Create data loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get first batch
    for batch in loader:
        batch_t0, batch_t1 = batch

        # Print basic information
        print("\n=== BATCH ANALYSIS ===")
        print(f"Batch size: {batch_t0.num_graphs}")
        print(f"Total nodes in batch: {batch_t0.num_nodes}")
        print(f"Node feature dimensions: {batch_t0.x.shape}")
        print(f"Edge feature dimensions: {batch_t0.edge_attr.shape}")
        print(f"Number of edges: {batch_t0.edge_index.shape[1]}")

        # Analyze node features
        t0_node_features_mean = batch_t0.x.mean(dim=0)
        t0_node_features_std = batch_t0.x.std(dim=0)
        print("\n=== Node Feature Statistics ===")
        print(f"Mean: {t0_node_features_mean[:5].tolist()}... (showing first 5)")
        print(f"Std: {t0_node_features_std[:5].tolist()}... (showing first 5)")

        # Analyze edge features
        t0_edge_features_mean = batch_t0.edge_attr.mean(dim=0)
        t0_edge_features_std = batch_t0.edge_attr.std(dim=0)
        print("\n=== Edge Feature Statistics ===")
        print(f"Mean: {t0_edge_features_mean[:5].tolist()}... (showing first 5)")
        print(f"Std: {t0_edge_features_std[:5].tolist()}... (showing first 5)")

        # Compare T0 and T1
        node_features_equal = torch.allclose(batch_t0.x, batch_t1.x)
        print(f"\nNode features are {'the same' if node_features_equal else 'different'} between T0 and T1")

        if not node_features_equal:
            diff = (batch_t0.x - batch_t1.x).abs().mean().item()
            print(f"Average node feature difference: {diff:.4f}")

        break  # Only analyze the first batch

    print("\nSample batch analysis complete")


def main():
    """Main function for data preparation pipeline"""
    # Parse arguments
    args = parse_prep_args()

    # Setup output directory
    paths = setup_output_directory(args)

    # Save configuration
    save_config(args, paths)

    # Create and analyze the dataset
    dataset = create_and_analyze_dataset(args, paths)

    # Analyze a sample batch if requested
    if args.sample_batch and dataset:
        analyze_sample_batch(dataset, args)

    print(f"\nData preparation completed successfully!")
    print(f"Output directory: {paths['run_dir']}")
    print(f"Cache directory: {paths['cache_dir']}")


if __name__ == "__main__":
    main()
