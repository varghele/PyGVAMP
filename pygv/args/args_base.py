# args/args_base.py

"""
Base/common command line arguments shared across all PyGVAMP pipeline phases.

These arguments are used by preparation, training, analysis, and pipeline scripts.
"""

import argparse


def add_data_args(parser: argparse.ArgumentParser):
    """Add data input arguments to parser."""
    data_group = parser.add_argument_group('Data Input')
    data_group.add_argument('--protein_name', type=str, default='protein',
                            help='Name of the protein being analyzed')
    data_group.add_argument('--traj_dir', type=str, required=True,
                            help='Path to directory containing trajectory files')
    data_group.add_argument('--top', type=str, required=True,
                            help='Path to topology file (.pdb)')
    data_group.add_argument('--file_pattern', type=str, default='*.xtc',
                            help='Pattern to match trajectory files (e.g., *.xtc, *.dcd)')
    data_group.add_argument('--recursive', action='store_true',
                            help='Search recursively for trajectory files')
    return data_group


def add_processing_args(parser: argparse.ArgumentParser):
    """Add data processing arguments to parser."""
    proc_group = parser.add_argument_group('Data Processing')
    proc_group.add_argument('--selection', type=str, default='name CA',
                            help='Atom selection string (MDTraj syntax)')
    proc_group.add_argument('--stride', type=int, default=1,
                            help='Stride for reading trajectory frames')
    proc_group.add_argument('--lag_time', type=float, default=1.0,
                            help='Lag time in nanoseconds')
    return proc_group


def add_graph_args(parser: argparse.ArgumentParser):
    """Add graph construction arguments to parser."""
    graph_group = parser.add_argument_group('Graph Construction')
    graph_group.add_argument('--n_neighbors', type=int, default=10,
                             help='Number of nearest neighbors for k-NN graph')
    graph_group.add_argument('--node_embedding_dim', type=int, default=16,
                             help='Dimension for node embeddings')
    graph_group.add_argument('--gaussian_expansion_dim', type=int, default=16,
                             help='Dimension for Gaussian expansion of edge distances')
    return graph_group


def add_output_args(parser: argparse.ArgumentParser):
    """Add output/caching arguments to parser."""
    out_group = parser.add_argument_group('Output & Caching')
    out_group.add_argument('--output_dir', type=str, default='./results',
                           help='Directory to save outputs')
    out_group.add_argument('--cache_dir', type=str, default=None,
                           help='Directory to cache processed datasets')
    out_group.add_argument('--use_cache', action='store_true',
                           help='Use cached dataset if available')
    out_group.add_argument('--run_name', type=str, default=None,
                           help='Name for this run (default: auto-generated timestamp)')
    return out_group


def add_hardware_args(parser: argparse.ArgumentParser):
    """Add hardware/device arguments to parser."""
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument('--cpu', action='store_true',
                          help='Force CPU usage even if CUDA is available')
    return hw_group


def add_common_args(parser: argparse.ArgumentParser):
    """
    Add all common arguments to a parser.

    This is a convenience function that adds all base argument groups:
    - Data input (protein_name, traj_dir, top, file_pattern, recursive)
    - Data processing (selection, stride, lag_time)
    - Graph construction (n_neighbors, node_embedding_dim, gaussian_expansion_dim)
    - Output & caching (output_dir, cache_dir, use_cache, run_name)
    - Hardware (cpu)

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to

    Returns
    -------
    argparse.ArgumentParser
        The parser with added arguments
    """
    add_data_args(parser)
    add_processing_args(parser)
    add_graph_args(parser)
    add_output_args(parser)
    add_hardware_args(parser)
    return parser


def get_base_parser(description: str = 'PyGVAMP base arguments'):
    """
    Create a new parser with all common arguments.

    Parameters
    ----------
    description : str
        Description for the argument parser

    Returns
    -------
    argparse.ArgumentParser
        Parser with all common arguments added
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_common_args(parser)
    return parser