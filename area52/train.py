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
    #args.protein_name = 'TRP'
    #args.protein_name = 'NTL'
    args.protein_name= 'AB42'
    #args.protein_name = 'ATR'
    #args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/')
    #args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/')
    #args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/r0/')
    args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/')
    #args.traj_dir = os.path.expanduser('~/PycharmProjects/PyGVAMP/datasets/NTL9/DESRES-Trajectory_NTL9-0-c-alpha')
    args.file_pattern = '*.xtc'
    #args.file_pattern = '*.dcd'
    #args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    #args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein.pdb')
    #args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/prot.pdb')
    #args.top = os.path.expanduser('~/PycharmProjects/PyGVAMP/datasets/NTL9/DESRES-Trajectory_NTL9-0-c-alpha/NTL9-0-c-alpha/NTL9.pdb')
    args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/topol.pdb')
    args.selection = 'name CA'
    #args.selection = '(residue 126 to 146 or residue 221 to 259 or residue 286 to 317) and name CA'

    args.val_split = 0.05
    args.sample_validate_every = 100

    args.stride = 10
    args.lag_time = 10.0
    args.n_neighbors = 10
    args.node_embedding_dim = 32
    args.gaussian_expansion_dim = 16 # TODO: This is edge dim!!!

    # SchNet encoder settings
    args.node_dim = 32
    args.edge_dim = 16
    args.hidden_dim = 32
    args.output_dim = 32
    args.n_interactions = 4
    args.activation = 'tanh'
    args.use_attention = True

    # Meta encoder settings
    """args.meta_node_dim = 16
    args.meta_edge_dim = 12 # TODO: Gaussian expansion dim
    args.meta_global_dim = 32
    args.meta_num_node_mlp_layers = 2
    args.meta_num_edge_mlp_layers = 2
    args.meta_num_global_mlp_layers = 2
    args.meta_hidden_dim = 64
    args.meta_output_dim = 32
    args.meta_num_meta_layers = 3
    args.meta_embedding_type = 'global'  # choices: 'node', 'global', 'combined'
    args.meta_activation = 'leaky_relu'
    args.meta_norm = 'None'
    args.meta_dropout = 0.0"""

    # Classifier settings
    args.n_states = 5
    args.clf_hidden_dim = 32
    args.clf_num_layers = 2
    args.clf_dropout = 0.01
    args.clf_activation = 'leaky_relu'
    args.clf_norm = 'LayerNorm' # 'BatchNorm' #

    # Embedding settings
    args.use_embedding = True
    args.embedding_in_dim = 42 # TODO:This is num of atoms/molecules
    args.embedding_hidden_dim = 64
    args.embedding_out_dim = 32
    args.embedding_num_layers = 2
    args.embedding_dropout = 0.01
    args.embedding_act = 'leaky_relu'
    args.embedding_norm = None # 'BatchNorm' #

    # Training settings
    args.epochs = 25
    args.batch_size = 256
    args.lr = 0.001
    args.weight_decay = 1e-5
    args.clip_grad = None
    args.cpu = False  # Use CPU for testing

    # Testing settngs
    args.max_tau = 200

    # Output settings
    args.output_dir = 'area57'
    args.cache_dir = 'area57/cache'
    args.use_cache = True
    args.save_every = 0  # Don't save intermediates
    args.run_name = 'ab42'

    return args


def run_test():
    """Run a VAMPNet training test"""
    # Create test arguments
    args = create_test_args()
    # Run training
    run_training(args)


if __name__ == "__main__":
    run_test()
