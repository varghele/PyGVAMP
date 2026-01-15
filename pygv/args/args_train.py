# args/args_train.py

"""
Command line arguments for VAMPNet training script.

Imports common arguments from args_base and encoder arguments from args_encoder,
then adds training-specific arguments.
"""

import argparse
from .args_base import add_common_args
from .args_encoder import add_encoder_args


def add_embedding_args(parser: argparse.ArgumentParser):
    """Add embedding MLP arguments to parser."""
    embedding_group = parser.add_argument_group('Embedding MLP')
    embedding_group.add_argument('--use_embedding', action='store_true',
                                 help='Use embedding MLP for node features')
    embedding_group.add_argument('--embedding_in_dim', type=int, default=None,
                                 help='Input dimension for embedding MLP (auto-inferred if None)')
    embedding_group.add_argument('--embedding_hidden_dim', type=int, default=64,
                                 help='Hidden dimension for embedding MLP layers')
    embedding_group.add_argument('--embedding_out_dim', type=int, default=32,
                                 help='Output dimension for embedding MLP')
    embedding_group.add_argument('--embedding_num_layers', type=int, default=2,
                                 help='Number of layers in embedding MLP')
    embedding_group.add_argument('--embedding_dropout', type=float, default=0.0,
                                 help='Dropout rate for embedding MLP')
    embedding_group.add_argument('--embedding_act', type=str, default='relu',
                                 choices=['relu', 'leaky_relu', 'gelu', 'silu', 'tanh'],
                                 help='Activation function for embedding MLP')
    embedding_group.add_argument('--embedding_norm', type=str, default=None,
                                 choices=['BatchNorm', 'LayerNorm', None],
                                 help='Normalization type for embedding MLP')
    return embedding_group


def add_classifier_args(parser: argparse.ArgumentParser):
    """Add classifier arguments to parser."""
    clf_group = parser.add_argument_group('Classifier')
    clf_group.add_argument('--n_states', type=int, default=5,
                           help='Number of output states (0 to disable classifier)')
    clf_group.add_argument('--clf_hidden_dim', type=int, default=64,
                           help='Hidden dimension for classifier MLP')
    clf_group.add_argument('--clf_num_layers', type=int, default=2,
                           help='Number of layers in classifier MLP')
    clf_group.add_argument('--clf_dropout', type=float, default=0.0,
                           help='Dropout rate for classifier')
    clf_group.add_argument('--clf_activation', type=str, default='relu',
                           choices=['relu', 'leaky_relu', 'gelu', 'tanh'],
                           help='Activation function for classifier')
    clf_group.add_argument('--clf_norm', type=str, default=None,
                           choices=['BatchNorm', 'LayerNorm', None],
                           help='Normalization type for classifier')
    return clf_group


def add_training_args(parser: argparse.ArgumentParser):
    """Add training hyperparameter arguments to parser."""
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_group.add_argument('--batch_size', type=int, default=32,
                             help='Batch size for training')
    train_group.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_group.add_argument('--weight_decay', type=float, default=1e-5,
                             help='Weight decay (L2 regularization)')
    train_group.add_argument('--clip_grad', action='store_true',
                             help='Enable gradient clipping')
    train_group.add_argument('--val_split', type=float, default=0.1,
                             help='Fraction of data to use for validation')
    train_group.add_argument('--sample_validate_every', type=int, default=100,
                             help='Run validation every N batches')
    train_group.add_argument('--save_every', type=int, default=0,
                             help='Save checkpoint every N epochs (0 to disable)')
    return train_group


def add_analysis_args(parser: argparse.ArgumentParser):
    """Add post-training analysis arguments to parser."""
    analysis_group = parser.add_argument_group('Analysis')
    analysis_group.add_argument('--max_tau', type=int, default=250,
                                help='Maximum lag time for implied timescales plot')
    return analysis_group


def get_train_parser():
    """
    Create argument parser for training.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all training arguments
    """
    parser = argparse.ArgumentParser(
        description='Train a VAMPNet model on molecular dynamics data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add common arguments (data, processing, graph, output, hardware)
    add_common_args(parser)

    # Add encoder arguments (encoder selection + SchNet/Meta/ML3)
    add_encoder_args(parser)

    # Add training-specific arguments
    add_embedding_args(parser)
    add_classifier_args(parser)
    add_training_args(parser)
    add_analysis_args(parser)

    return parser


def parse_train_args():
    """Parse training arguments from command line."""
    parser = get_train_parser()
    return parser.parse_args()


if __name__ == "__main__":
    parser = get_train_parser()
    parser.print_help()
