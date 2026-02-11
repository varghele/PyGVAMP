# args/args_prep.py

"""
Command line arguments for VAMPNet data preparation script.

Imports common arguments from args_base, then adds preparation-specific arguments.
"""

import argparse
from .args_base import add_common_args


def add_prep_args(parser: argparse.ArgumentParser):
    """Add preparation-specific arguments to parser."""
    prep_group = parser.add_argument_group('Preparation Options')
    prep_group.add_argument('--sample_batch', action='store_true',
                            help='Create and analyze a sample batch after preparation')
    prep_group.add_argument('--batch_size', type=int, default=1,
                            help='Batch size for sample analysis')
    prep_group.add_argument('--precompute_graphs', action='store_true',
                            help='Precompute all graphs (uses more memory but faster)')
    prep_group.add_argument('--max_precompute', type=int, default=None,
                            help='Maximum number of graphs to precompute (None for all)')

    # State discovery arguments
    discovery_group = parser.add_argument_group('State Discovery Options')
    discovery_group.add_argument('--discover_states', action='store_true',
                                  help='Run Graph2Vec + clustering for unsupervised state discovery')
    discovery_group.add_argument('--g2v_embedding_dim', type=int, default=512,
                                  help='Graph2Vec embedding dimension')
    discovery_group.add_argument('--g2v_max_degree', type=int, default=3,
                                  help='Graph2Vec WL iteration depth')
    discovery_group.add_argument('--g2v_epochs', type=int, default=50,
                                  help='Graph2Vec training epochs')
    discovery_group.add_argument('--g2v_min_count', type=int, default=10,
                                  help='Minimum subgraph frequency to be included in Graph2Vec vocabulary')
    discovery_group.add_argument('--min_states', type=int, default=2,
                                  help='Minimum number of states to test in clustering')
    discovery_group.add_argument('--max_states', type=int, default=10,
                                  help='Maximum number of states to test in clustering')
    discovery_group.add_argument('--g2v_umap_dim', type=int, nargs='+', default=[2, 3, 5, 6, 7],
                                  help='UMAP dimensionalities to sweep for clustering '
                                       '(e.g., --g2v_umap_dim 2 5 10)')

    return prep_group


def get_prep_parser():
    """
    Create argument parser for data preparation.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all preparation arguments
    """
    parser = argparse.ArgumentParser(
        description='Prepare molecular dynamics data for VAMPNet training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add common arguments (data, processing, graph, output, hardware)
    add_common_args(parser)

    # Add preparation-specific arguments
    add_prep_args(parser)

    return parser


def parse_prep_args():
    """Parse data preparation arguments from command line."""
    parser = get_prep_parser()
    return parser.parse_args()


if __name__ == "__main__":
    parser = get_prep_parser()
    parser.print_help()
