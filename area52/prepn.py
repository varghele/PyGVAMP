#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script for VAMPNet data preparation
"""

import os
import sys
import argparse


# Add parent directory to sys.path to import from your package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from preparation pipeline
from pipe.preparation import create_and_analyze_dataset, setup_output_directory, save_config, analyze_sample_batch


def create_test_prep_args():
    """Create a simple argument namespace for testing preparation"""
    args = argparse.Namespace()

    # Input data settings
    args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/')
    args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    args.file_pattern = '*.xtc'
    args.recursive = True

    # Data processing settings
    args.selection = 'name CA'
    args.stride = 1
    args.lag_time = 20.0
    args.n_neighbors = 20
    args.node_embedding_dim = 16
    args.gaussian_expansion_dim = 8

    # Output settings
    args.output_dir = './area53'
    args.cache_dir = './area53/cache'
    args.use_cache = False
    args.sample_batch = True
    args.batch_size = 2

    return args


def run_test():
    """Run a VAMPNet data preparation test"""
    # Create test arguments
    args = create_test_prep_args()

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
    run_test()
