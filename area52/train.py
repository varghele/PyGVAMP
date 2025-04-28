#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script for VAMPNet training
"""

import os
import argparse

# Import functions from training pipeline
from pygv.pipe.training import run_training


def create_test_args():
    """Create a simple argument namespace for testing"""
    args = argparse.Namespace()

    # Basic settings
    args.encoder_type = 'schnet'
    #args.encoder_type = 'meta'

    # Data settings
    args.protein_name = 'TRP'
    #args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/')
    args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/')
    args.file_pattern = '*.dcd'
    #args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein.pdb')
    args.selection = 'name CA'
    args.stride = 100
    args.lag_time = 20.0
    args.n_neighbors = 7
    args.node_embedding_dim = 16
    args.gaussian_expansion_dim = 12 # TODO: This is edge dim!!!

    # SchNet encoder settings
    args.node_dim = 16
    args.edge_dim = 12
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
    args.n_states = 5
    args.clf_hidden_dim = 32
    args.clf_num_layers = 4
    args.clf_dropout = 0.0
    args.clf_activation = 'elu'
    args.clf_norm = None #'BatchNorm' # 'LayerNorm'

    # Embedding settings
    args.use_embedding = True
    args.embedding_in_dim = 16
    args.embedding_hidden_dim = 32
    args.embedding_out_dim = 16
    args.embedding_num_layers = 4
    args.embedding_dropout = 0.0
    args.embedding_act = 'elu'
    args.embedding_norm = None #'LayerNorm'

    # Training settings
    args.epochs = 5
    args.batch_size = 1000
    args.lr = 0.005
    args.weight_decay = 1e-5
    args.clip_grad = None
    args.cpu = False  # Use CPU for testing

    # Testing settngs
    args.max_tau = 200

    # Output settings
    args.output_dir = 'area54'
    args.cache_dir = 'area54/cache'
    args.use_cache = False
    args.save_every = 0  # Don't save intermediates
    args.run_name = 'test_run_new'

    return args


def run_test():
    """Run a VAMPNet training test"""
    # Create test arguments
    args = create_test_args()
    # Run training
    run_training(args)


if __name__ == "__main__":
    run_test()
