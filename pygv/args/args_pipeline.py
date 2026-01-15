# args/args_pipeline.py

"""
Command line arguments for the complete VAMPNet pipeline.

Imports common and encoder arguments, then adds pipeline control arguments
for running preparation, training, and analysis phases.
"""

import argparse
from .args_base import add_common_args
from .args_encoder import add_encoder_args
from .args_train import add_embedding_args, add_classifier_args, add_training_args


def add_pipeline_control_args(parser: argparse.ArgumentParser):
    """Add pipeline control arguments to parser."""
    pipeline_group = parser.add_argument_group('Pipeline Control')
    pipeline_group.add_argument('--preset', type=str, default=None,
                                help='Configuration preset (e.g., small_schnet, medium_meta, large_ml3)')
    pipeline_group.add_argument('--skip_preparation', action='store_true',
                                help='Skip the preparation phase')
    pipeline_group.add_argument('--skip_training', action='store_true',
                                help='Skip the training phase')
    pipeline_group.add_argument('--skip_analysis', action='store_true',
                                help='Skip the analysis phase')
    pipeline_group.add_argument('--only_preparation', action='store_true',
                                help='Only run preparation phase')
    pipeline_group.add_argument('--only_training', action='store_true',
                                help='Only run training phase')
    pipeline_group.add_argument('--only_analysis', action='store_true',
                                help='Only run analysis phase')
    return pipeline_group


def add_grid_search_args(parser: argparse.ArgumentParser):
    """Add grid search arguments for hyperparameter exploration."""
    grid_group = parser.add_argument_group('Grid Search')
    grid_group.add_argument('--lag_times', type=float, nargs='+', default=None,
                            help='List of lag times to explore (overrides --lag_time)')
    grid_group.add_argument('--n_states_list', type=int, nargs='+', default=None,
                            help='List of n_states values to explore (overrides --n_states)')
    grid_group.add_argument('--select_best_by', type=str, default='vamp2',
                            choices=['vamp1', 'vamp2', 'vampe'],
                            help='Score type to use for selecting best model')
    return grid_group


def add_convenience_args(parser: argparse.ArgumentParser):
    """Add convenience arguments for common use cases."""
    conv_group = parser.add_argument_group('Convenience Options')
    conv_group.add_argument('--hurry', action='store_true',
                            help='Use larger stride for faster processing (testing/debugging)')
    conv_group.add_argument('--quick_test', action='store_true',
                            help='Run a quick test with minimal epochs and data')
    return conv_group


def get_pipeline_parser():
    """
    Create argument parser for the full pipeline.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all pipeline arguments
    """
    parser = argparse.ArgumentParser(
        description='Run the complete VAMPNet pipeline (preparation, training, analysis)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add common arguments (data, processing, graph, output, hardware)
    add_common_args(parser)

    # Add encoder arguments (encoder selection + SchNet/Meta/ML3)
    add_encoder_args(parser)

    # Add training-related arguments
    add_embedding_args(parser)
    add_classifier_args(parser)
    add_training_args(parser)

    # Add pipeline-specific arguments
    add_pipeline_control_args(parser)
    add_grid_search_args(parser)
    add_convenience_args(parser)

    return parser


def parse_pipeline_args():
    """Parse pipeline arguments from command line."""
    parser = get_pipeline_parser()
    return parser.parse_args()


if __name__ == "__main__":
    parser = get_pipeline_parser()
    parser.print_help()
