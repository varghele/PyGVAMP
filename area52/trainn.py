#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script for VAMPNet training
"""

import os
import sys
import argparse

# Add parent directory to sys.path to import from your package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from training pipeline
from pipe.training import create_dataset_and_loader, create_model, train_model, setup_output_directory, save_config


def create_test_args():
    """Create a simple argument namespace for testing"""
    args = argparse.Namespace()

    # Basic settings
    args.encoder_type = 'schnet'

    # Data settings
    args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/')
    args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    args.selection = 'name CA'
    args.stride = 1
    args.lag_time = 20.0
    args.n_neighbors = 20
    args.node_embedding_dim = 16
    args.gaussian_expansion_dim = 8

    # SchNet encoder settings
    args.node_dim = 16
    args.edge_dim = 8
    args.hidden_dim = 32
    args.output_dim = 16
    args.n_interactions = 2
    args.activation = 'tanh'
    args.use_attention = True

    # Classifier settings
    args.n_states = 4
    args.clf_hidden_dim = 32
    args.clf_num_layers = 2
    args.clf_dropout = 0.0
    args.clf_activation = 'relu'
    args.clf_norm = None

    # Training settings
    args.epochs = 3  # Keep this low for testing
    args.batch_size = 64
    args.lr = 0.001
    args.weight_decay = 1e-5
    args.clip_grad = None
    args.cpu = True  # Use CPU for testing

    # Output settings
    args.output_dir = './area53'
    args.cache_dir = './area53/cache'
    args.use_cache = True
    args.save_every = 0  # Don't save intermediates
    args.run_name = 'test_run'

    return args


def run_test():
    """Run a VAMPNet training test"""
    # Create test arguments
    args = create_test_args()

    # Setup output directory
    paths = setup_output_directory(args)

    # Save configuration
    save_config(args, paths)

    # Create dataset and loader
    dataset, loader = create_dataset_and_loader(args)

    # Create model
    model = create_model(args, dataset)
    print(f"Created VAMPNet model with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    scores = train_model(args, model, loader, paths)

    print(f"Training completed successfully. Results saved to {paths['run_dir']}")


if __name__ == "__main__":
    run_test()
