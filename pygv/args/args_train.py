# args/args_train.py

"""
Command line arguments for VAMPNet training script
"""

import argparse


def get_train_parser():
    """Create argument parser for training"""
    parser = argparse.ArgumentParser(
        description='Train a VAMPNet model on molecular dynamics data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Embedding MLP arguments
    embedding_group = parser.add_argument_group('Embedding MLP arguments')
    embedding_group.add_argument('--use_embedding', action='store_true',
                             help='Gradient clipping norm')
    embedding_group.add_argument('--embedding_in_dim', type=int, default=1,
                                 help='Input dimension for embedding MLP (usually 1 for atom types)')
    embedding_group.add_argument('--embedding_hidden_dim', type=int, default=64,
                                 help='Hidden dimension for embedding MLP layers')
    embedding_group.add_argument('--embedding_out_dim', type=int, default=128,
                                 help='Output dimension for embedding MLP')
    embedding_group.add_argument('--embedding_num_layers', type=int, default=3,
                                 help='Number of layers in embedding MLP')
    embedding_group.add_argument('--embedding_dropout', type=float, default=0.1,
                                 help='Dropout rate for embedding MLP')
    embedding_group.add_argument('--embedding_act', type=str, default='relu',
                                 choices=['relu', 'leaky_relu', 'gelu', 'silu', 'tanh'],
                                 help='Activation function for embedding MLP')
    embedding_group.add_argument('--embedding_norm', type=str, default='batch',
                                 choices=['batch', 'layer', 'none'],
                                 help='Normalization type for embedding MLP')

    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--protein_name', required=True,
                            help='Name of the protein that was simulated')
    data_group.add_argument('--traj_dir', required=True,#, nargs='+',
                            help='Path to trajectory directory')
    data_group.add_argument('--file_pattern',
                            help='File extension of trajectory files')
    data_group.add_argument('--top', required=True,
                            help='Path to topology file')
    data_group.add_argument('--selection', default='name CA',
                            help='Atom selection (MDTraj syntax)')
    data_group.add_argument('--stride', type=int, default=1,
                            help='Stride for reading trajectories')
    data_group.add_argument('--lag_time', type=float, default=1.0,
                            help='Lag time in nanoseconds')
    data_group.add_argument('--use_cache', action='store_true',
                            help='Whether to use cached prepared data or run preparation again')
    data_group.add_argument('--cache_dir', default=None,
                            help='Directory to cache processed data')
    data_group.add_argument('--n_neighbors', type=int, default=10,
                            help='Number of neighbors for graph construction')
    data_group.add_argument('--node_embedding_dim', type=int, default=16,
                            help='Dimension for node embeddings')
    data_group.add_argument('--gaussian_expansion_dim', type=int, default=8,
                            help='Dimension for Gaussian expansion of distances')

    # Encoder arguments
    encoder_group = parser.add_argument_group('Encoder')
    encoder_group.add_argument('--encoder_type', type=str, default='schnet',
                               choices=['schnet', 'meta', 'ml3'],
                               help='Type of encoder to use')

    # SchNet specific arguments
    schnet_group = parser.add_argument_group('SchNet Encoder')
    schnet_group.add_argument('--node_dim', type=int, default=32,
                              help='Node dimension (SchNet)')
    schnet_group.add_argument('--edge_dim', type=int, default=16,
                              help='Edge dimension (SchNet)')
    schnet_group.add_argument('--hidden_dim', type=int, default=64,
                              help='Hidden dimension (SchNet)')
    schnet_group.add_argument('--output_dim', type=int, default=32,
                              help='Output dimension (SchNet)')
    schnet_group.add_argument('--n_interactions', type=int, default=3,
                              help='Number of interaction layers (SchNet)')
    schnet_group.add_argument('--activation', default='tanh',
                              help='Activation function (SchNet)')
    schnet_group.add_argument('--use_attention', action='store_true',
                              help='Use attention mechanism (SchNet)')

    # Meta specific arguments
    meta_group = parser.add_argument_group('Meta Encoder')
    meta_group.add_argument('--meta_node_dim', type=int, default=32,
                            help='Node dimension (Meta)')
    meta_group.add_argument('--meta_edge_dim', type=int, default=16,
                            help='Edge dimension (Meta)')
    meta_group.add_argument('--meta_global_dim', type=int, default=32,
                            help='Global dimension (Meta)')
    meta_group.add_argument('--meta_num_node_mlp_layers', type=int, default=2,
                            help='Number of node MLP layers (Meta)')
    meta_group.add_argument('--meta_num_edge_mlp_layers', type=int, default=2,
                            help='Number of edge MLP layers (Meta)')
    meta_group.add_argument('--meta_num_global_mlp_layers', type=int, default=2,
                            help='Number of global MLP layers (Meta)')
    meta_group.add_argument('--meta_hidden_dim', type=int, default=64,
                            help='Hidden dimension (Meta)')
    meta_group.add_argument('--meta_output_dim', type=int, default=32,
                            help='Output dimension (Meta)')
    meta_group.add_argument('--meta_num-meta_layers', type=int, default=3,
                            help='Number of meta layers (Meta)')
    meta_group.add_argument('--meta_embedding_type', type=str, default='combined',
                            choices=['node', 'global', 'combined'],
                            help='Embedding type (Meta)')
    meta_group.add_argument('--meta_activation', default='relu',
                            help='Activation function (Meta)')
    meta_group.add_argument('--meta_norm', default='batch_norm',
                            help='Normalization (Meta)')
    meta_group.add_argument('--meta_dropout', type=float, default=0.0,
                            help='Dropout rate (Meta)')

    # ML3 specific arguments
    ml3_group = parser.add_argument_group('ML3 Encoder')
    ml3_group.add_argument('--ml3_node_dim', type=int, default=32,
                           help='Node dimension (ML3)')
    ml3_group.add_argument('--ml3_edge_dim', type=int, default=16,
                           help='Edge dimension (ML3)')
    ml3_group.add_argument('--ml3_hidden_dim', type=int, default=64,
                           help='Hidden dimension (ML3)')
    ml3_group.add_argument('--ml3_output_dim', type=int, default=32,
                           help='Output dimension (ML3)')
    ml3_group.add_argument('--ml3_num_layers', type=int, default=3,
                           help='Number of layers (ML3)')
    ml3_group.add_argument('--ml3_activation', default='relu',
                           help='Activation function (ML3)')

    # Classifier arguments
    clf_group = parser.add_argument_group('Classifier')
    clf_group.add_argument('--n_states', type=int, default=5,
                           help='Number of states (0 to disable classifier)')
    clf_group.add_argument('--clf_hidden_dim', type=int, default=64,
                           help='Classifier hidden dimension')
    clf_group.add_argument('--clf_num_layers', type=int, default=2,
                           help='Classifier number of layers')
    clf_group.add_argument('--clf_dropout', type=float, default=0.0,
                           help='Classifier dropout rate')
    clf_group.add_argument('--clf_activation', type=str, default='relu',
                           help='Classifier activation function')
    clf_group.add_argument('--clf_norm', type=str, default=None,
                           choices=[None, 'BatchNorm', 'LayerNorm'],
                           help='Classifier normalization layer')

    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=100,
                             help='Number of epochs')
    train_group.add_argument('--batch_size', type=int, default=32,
                             help='Batch size')
    train_group.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_group.add_argument('--weight_decay', type=float, default=1e-5,
                             help='Weight decay')
    train_group.add_argument('--clip_grad', action='store_true',
                             help='Gradient clipping norm')
    train_group.add_argument('--cpu', action='store_true',
                             help='Force CPU usage even if CUDA is available')
    train_group.add_argument('--val_split', type=float, default=0.1,
                             help='Amount of data that training will get validated on')
    train_group.add_argument('--sample_validate_every', type=int, default=100,
                             help='How often (mini)validation should be performed; every n batches')

    # Testing arguments
    test_group = parser.add_argument_group('Testing')
    test_group.add_argument('--max_tau', type=int, default=250,
                            help='Maximum lag time to plot the implied timescales for')

    # Output arguments
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--output_dir', default='./results',
                           help='Directory to save outputs')
    out_group.add_argument('--save_every', type=int, default=10,
                           help='Save model every N epochs (0 to disable)')
    out_group.add_argument('--run_name', default=None,
                           help='Run name for outputs (default: timestamp)')

    return parser


def parse_train_args():
    """Parse training arguments"""
    parser = get_train_parser()
    return parser.parse_args()
