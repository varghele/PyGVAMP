# args/args_prep.py

"""
Command line arguments for VAMPNet data preparation script
"""

import argparse


def get_prep_parser():
    """Create argument parser for data preparation"""
    parser = argparse.ArgumentParser(
        description='Prepare molecular dynamics data for VAMPNet training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input data arguments
    data_group = parser.add_argument_group('Input Data')
    data_group.add_argument('--traj_dir', required=True,
                            help='Path to directory containing trajectory files')
    data_group.add_argument('--top', required=True,
                            help='Path to topology file')
    data_group.add_argument('--file-pattern', default='*.xtc',
                            help='Pattern to match trajectory files')
    data_group.add_argument('--recursive', action='store_true',
                            help='Search recursively for trajectory files')

    # Dataset processing arguments
    proc_group = parser.add_argument_group('Data Processing')
    proc_group.add_argument('--selection', default='name CA',
                            help='Atom selection (MDTraj syntax)')
    proc_group.add_argument('--stride', type=int, default=1,
                            help='Stride for reading trajectories')
    proc_group.add_argument('--lag-time', type=float, default=1.0,
                            help='Lag time in nanoseconds')
    proc_group.add_argument('--n-neighbors', type=int, default=10,
                            help='Number of neighbors for graph construction')
    proc_group.add_argument('--node-embedding-dim', type=int, default=16,
                            help='Dimension for node embeddings')
    proc_group.add_argument('--gaussian-expansion-dim', type=int, default=8,
                            help='Dimension for Gaussian expansion of distances')

    # Output arguments
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--output-dir', default='./processed_data',
                           help='Directory to save outputs')
    out_group.add_argument('--cache-dir', default=None,
                           help='Directory to cache processed dataset')
    out_group.add_argument('--use-cache', action='store_true',
                           help='Use cached dataset if available')
    out_group.add_argument('--sample-batch', action='store_true',
                           help='Create and analyze a sample batch')
    out_group.add_argument('--batch-size', type=int, default=1,
                           help='Batch size for sample analysis')

    return parser


def parse_prep_args():
    """Parse data preparation arguments"""
    parser = get_prep_parser()
    return parser.parse_args()
