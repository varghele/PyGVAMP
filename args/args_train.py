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

    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--traj', required=True, nargs='+',
                            help='Path(s) to trajectory file(s)')
    data_group.add_argument('--top', required=True,
                            help='Path to topology file')
    data_group.add_argument('--selection', default='name CA',
                            help='Atom selection (MDTraj syntax)')
    data_group.add_argument('--stride', type=int, default=1,
                            help='Stride for reading trajectories')
    data_group.add_argument('--lag-time', type=float, default=1.0,
                            help='Lag time in nanoseconds')
    data_group.add_argument('--cache-dir', default=None,
                            help='Directory to cache processed data')
    data_group.add_argument('--n-neighbors', type=int, default=10,
                            help='Number of neighbors for graph construction')

    # Encoder arguments
    encoder_group = parser.add_argument_group('Encoder')
    encoder_group.add_argument('--encoder-type', type=str, default='schnet',
                               choices=['schnet', 'meta', 'ml3'],
                               help='Type of encoder to use')

    # SchNet specific arguments
    schnet_group = parser.add_argument_group('SchNet Encoder')
    schnet_group.add_argument('--node-dim', type=int, default=32,
                              help='Node dimension (SchNet)')
    schnet_group.add_argument('--edge-dim', type=int, default=16,
                              help='Edge dimension (SchNet)')
    schnet_group.add_argument('--hidden-dim', type=int, default=64,
                              help='Hidden dimension (SchNet)')
    schnet_group.add_argument('--output-dim', type=int, default=32,
                              help='Output dimension (SchNet)')
    schnet_group.add_argument('--n-interactions', type=int, default=3,
                              help='Number of interaction layers (SchNet)')
    schnet_group.add_argument('--activation', default='tanh',
                              help='Activation function (SchNet)')
    schnet_group.add_argument('--use-attention', action='store_true',
                              help='Use attention mechanism (SchNet)')

    # Meta specific arguments
    meta_group = parser.add_argument_group('Meta Encoder')
    meta_group.add_argument('--meta-node-dim', type=int, default=32,
                            help='Node dimension (Meta)')
    meta_group.add_argument('--meta-edge-dim', type=int, default=16,
                            help='Edge dimension (Meta)')
    meta_group.add_argument('--meta-global-dim', type=int, default=32,
                            help='Global dimension (Meta)')
    meta_group.add_argument('--meta-num-node-mlp-layers', type=int, default=2,
                            help='Number of node MLP layers (Meta)')
    meta_group.add_argument('--meta-num-edge-mlp-layers', type=int, default=2,
                            help='Number of edge MLP layers (Meta)')
    meta_group.add_argument('--meta-num-global-mlp-layers', type=int, default=2,
                            help='Number of global MLP layers (Meta)')
    meta_group.add_argument('--meta-hidden-dim', type=int, default=64,
                            help='Hidden dimension (Meta)')
    meta_group.add_argument('--meta-output-dim', type=int, default=32,
                            help='Output dimension (Meta)')
    meta_group.add_argument('--meta-num-meta-layers', type=int, default=3,
                            help='Number of meta layers (Meta)')
    meta_group.add_argument('--meta-embedding-type', type=str, default='combined',
                            choices=['node', 'global', 'combined'],
                            help='Embedding type (Meta)')
    meta_group.add_argument('--meta-activation', default='relu',
                            help='Activation function (Meta)')
    meta_group.add_argument('--meta-norm', default='batch_norm',
                            help='Normalization (Meta)')
    meta_group.add_argument('--meta-dropout', type=float, default=0.0,
                            help='Dropout rate (Meta)')

    # ML3 specific arguments
    ml3_group = parser.add_argument_group('ML3 Encoder')
    ml3_group.add_argument('--ml3-node-dim', type=int, default=32,
                           help='Node dimension (ML3)')
    ml3_group.add_argument('--ml3-edge-dim', type=int, default=16,
                           help='Edge dimension (ML3)')
    ml3_group.add_argument('--ml3-hidden-dim', type=int, default=64,
                           help='Hidden dimension (ML3)')
    ml3_group.add_argument('--ml3-output-dim', type=int, default=32,
                           help='Output dimension (ML3)')
    ml3_group.add_argument('--ml3-num-layers', type=int, default=3,
                           help='Number of layers (ML3)')
    ml3_group.add_argument('--ml3-activation', default='relu',
                           help='Activation function (ML3)')

    # Classifier arguments
    clf_group = parser.add_argument_group('Classifier')
    clf_group.add_argument('--n-states', type=int, default=5,
                           help='Number of states (0 to disable classifier)')
    clf_group.add_argument('--clf-hidden-dim', type=int, default=64,
                           help='Classifier hidden dimension')
    clf_group.add_argument('--clf-num-layers', type=int, default=2,
                           help='Classifier number of layers')
    clf_group.add_argument('--clf-dropout', type=float, default=0.0,
                           help='Classifier dropout rate')
    clf_group.add_argument('--clf-activation', type=str, default='relu',
                           help='Classifier activation function')
    clf_group.add_argument('--clf-norm', type=str, default=None,
                           choices=[None, 'batch_norm', 'layer_norm', 'instance_norm'],
                           help='Classifier normalization layer')

    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=100,
                             help='Number of epochs')
    train_group.add_argument('--batch-size', type=int, default=32,
                             help='Batch size')
    train_group.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=1e-5,
                             help='Weight decay')
    train_group.add_argument('--clip-grad', type=float, default=None,
                             help='Gradient clipping norm')
    train_group.add_argument('--cpu', action='store_true',
                             help='Force CPU usage even if CUDA is available')

    # Output arguments
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--output-dir', default='./results',
                           help='Directory to save outputs')
    out_group.add_argument('--save-every', type=int, default=10,
                           help='Save model every N epochs (0 to disable)')
    out_group.add_argument('--run-name', default=None,
                           help='Run name for outputs (default: timestamp)')

    return parser


def parse_train_args():
    """Parse training arguments"""
    parser = get_train_parser()
    return parser.parse_args()
