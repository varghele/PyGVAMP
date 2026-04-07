# args/args_encoder.py

"""
Encoder-specific command line arguments for PyGVAMP.

Contains argument groups for SchNet, Meta, ML3, and GIN encoders.
"""

import argparse


def add_encoder_selection_args(parser: argparse.ArgumentParser):
    """Add encoder type selection argument."""
    encoder_group = parser.add_argument_group('Encoder Selection')
    encoder_group.add_argument('--encoder_type', type=str, default='schnet',
                               choices=['schnet', 'meta', 'ml3', 'gin'],
                               help='Type of graph neural network encoder')
    return encoder_group


def add_schnet_args(parser: argparse.ArgumentParser):
    """
    Add SchNet encoder arguments to parser.

    SchNet uses continuous-filter convolutions with optional attention.
    """
    schnet_group = parser.add_argument_group('SchNet Encoder')
    schnet_group.add_argument('--node_dim', type=int, default=16,
                              help='Node feature dimension')
    schnet_group.add_argument('--edge_dim', type=int, default=16,
                              help='Edge feature dimension')
    schnet_group.add_argument('--hidden_dim', type=int, default=128,
                              help='Hidden layer dimension')
    schnet_group.add_argument('--output_dim', type=int, default=64,
                              help='Output dimension')
    schnet_group.add_argument('--n_interactions', type=int, default=3,
                              help='Number of interaction/message-passing layers')
    schnet_group.add_argument('--activation', type=str, default='tanh',
                              choices=['relu', 'tanh', 'leaky_relu', 'elu', 'gelu', 'silu'],
                              help='Activation function')
    schnet_group.add_argument('--use_attention', action='store_true', default=True,
                              help='Use attention mechanism in message passing')
    return schnet_group


def add_meta_args(parser: argparse.ArgumentParser):
    """
    Add Meta encoder arguments to parser.

    Meta encoder uses MetaLayer from PyG with separate edge, node, and global models.
    """
    meta_group = parser.add_argument_group('Meta Encoder')
    meta_group.add_argument('--meta_node_dim', type=int, default=16,
                            help='Node feature dimension')
    meta_group.add_argument('--meta_edge_dim', type=int, default=16,
                            help='Edge feature dimension')
    meta_group.add_argument('--meta_global_dim', type=int, default=16,
                            help='Global feature dimension')
    meta_group.add_argument('--meta_hidden_dim', type=int, default=128,
                            help='Hidden dimension for MLPs')
    meta_group.add_argument('--meta_output_dim', type=int, default=64,
                            help='Output dimension')
    meta_group.add_argument('--meta_num_node_mlp_layers', type=int, default=2,
                            help='Number of layers in node MLP')
    meta_group.add_argument('--meta_num_edge_mlp_layers', type=int, default=2,
                            help='Number of layers in edge MLP')
    meta_group.add_argument('--meta_num_global_mlp_layers', type=int, default=2,
                            help='Number of layers in global MLP')
    meta_group.add_argument('--meta_num_meta_layers', type=int, default=3,
                            help='Number of MetaLayer blocks')
    meta_group.add_argument('--meta_embedding_type', type=str, default='node',
                            choices=['node', 'global', 'combined'],
                            help='Type of embedding to use for output')
    meta_group.add_argument('--meta_use_attention', action='store_true', default=True,
                            help='Use attention mechanism')
    meta_group.add_argument('--meta_activation', type=str, default='relu',
                            choices=['relu', 'tanh', 'leaky_relu', 'elu', 'gelu'],
                            help='Activation function')
    meta_group.add_argument('--meta_norm', type=str, default='batch_norm',
                            choices=['batch_norm', 'layer_norm', 'none'],
                            help='Normalization type')
    meta_group.add_argument('--meta_dropout', type=float, default=0.1,
                            help='Dropout rate')
    return meta_group


def add_ml3_args(parser: argparse.ArgumentParser):
    """
    Add ML3 encoder arguments to parser.

    ML3 uses spectral convolutions with 3-WL expressivity,
    learned edge transformations, skip connections, and optional
    parallel attention.
    """
    ml3_group = parser.add_argument_group('ML3 Encoder')
    # Core dimensions
    ml3_group.add_argument('--ml3_node_dim', type=int, default=16,
                           help='Dimension of input node features')
    ml3_group.add_argument('--ml3_edge_dim', type=int, default=16,
                           help='Dimension of input edge features')
    ml3_group.add_argument('--ml3_hidden_dim', type=int, default=30,
                           help='Hidden dimension for ML3 layers')
    ml3_group.add_argument('--ml3_output_dim', type=int, default=32,
                           help='Output dimension (graph-level)')
    # Layer configuration
    ml3_group.add_argument('--ml3_num_layers', type=int, default=4,
                           help='Number of ML3 interaction layers')
    ml3_group.add_argument('--ml3_nout1', type=int, default=30,
                           help='Convolution output dim in ML3Layer')
    ml3_group.add_argument('--ml3_nout2', type=int, default=2,
                           help='Skip connection output dim (0 to disable)')
    # Attention
    ml3_group.add_argument('--ml3_use_attention', action='store_true', default=True,
                           help='Use parallel attention mechanism')
    # Edge feature mode
    ml3_group.add_argument('--ml3_edge_mode', type=str, default='gaussian',
                           choices=['gaussian', 'spectral'],
                           help='Edge feature mode: gaussian (from dataset) or spectral (eigendecomp)')
    ml3_group.add_argument('--ml3_nfreq', type=int, default=10,
                           help='Number of spectral frequencies (spectral mode)')
    ml3_group.add_argument('--ml3_spectral_dv', type=float, default=1.0,
                           help='Gaussian width for spectral filters (spectral mode)')
    ml3_group.add_argument('--ml3_recfield', type=int, default=1,
                           help='Receptive field for spectral filters (spectral mode)')
    # Activation
    ml3_group.add_argument('--ml3_activation', type=str, default='relu',
                           choices=['relu', 'tanh', 'leaky_relu', 'elu', 'gelu'],
                           help='Activation function')
    ml3_group.add_argument('--ml3_dropout', type=float, default=0.0,
                           help='Dropout rate')
    return ml3_group


def add_encoder_args(parser: argparse.ArgumentParser):
    """
    Add all encoder arguments to a parser.

    This adds the encoder selection argument followed by all encoder-specific groups.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to

    Returns
    -------
    argparse.ArgumentParser
        The parser with added arguments
    """
    add_encoder_selection_args(parser)
    add_schnet_args(parser)
    add_meta_args(parser)
    add_ml3_args(parser)
    return parser
