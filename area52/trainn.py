#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script for VAMPNet training
"""

import os
import sys
import argparse

# Add parent directory to sys.path to import from your package
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from training pipeline
from pipe.training import create_dataset_and_loader, create_model, train_model, setup_output_directory, save_config


def create_test_args():
    """Create a simple argument namespace for testing"""
    args = argparse.Namespace()

    # Basic settings
    args.encoder_type = 'schnet'
    #args.encoder_type = 'meta'

    # Data settings
    args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/')
    args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    args.selection = 'name CA'
    args.stride = 1
    args.lag_time = 10.0
    args.n_neighbors = 10
    args.node_embedding_dim = 16
    args.gaussian_expansion_dim = 16

    # SchNet encoder settings
    args.node_dim = 16
    args.edge_dim = 16
    args.hidden_dim = 32
    args.output_dim = 16
    args.n_interactions = 4
    args.activation = 'tanh'
    args.use_attention = True

    # Meta encoder settings
    """args.meta_node_dim = 16
    args.meta_edge_dim = 16
    args.meta_global_dim = 32
    args.meta_num_node_mlp_layers = 2
    args.meta_num_edge_mlp_layers = 2
    args.meta_num_global_mlp_layers = 2
    args.meta_hidden_dim = 64
    args.meta_output_dim = 32
    args.meta_num_meta_layers = 3
    args.meta_embedding_type = 'global'  # choices: 'node', 'global', 'combined'
    args.meta_activation = 'elu'
    args.meta_norm = 'None'
    args.meta_dropout = 0.0"""

    # Classifier settings
    args.n_states = 4
    args.clf_hidden_dim = 32
    args.clf_num_layers = 2
    args.clf_dropout = 0.0
    args.clf_activation = 'elu'
    args.clf_norm = None# 'LayerNorm'

    # Training settings
    args.epochs = 500
    args.batch_size = 500
    args.lr = 0.001
    args.weight_decay = 1e-5
    args.clip_grad = None
    args.cpu = False  # Use CPU for testing

    # Output settings
    args.output_dir = './area53'
    args.cache_dir = './area53/cache'
    args.use_cache = False
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
