"""
Pipeline-specific argument parser
"""
import argparse
from .args_base import add_common_args


def parse_pipeline_args():
    """Parse pipeline-specific arguments"""
    parser = argparse.ArgumentParser(
        description='Run complete VAMPNet pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add common arguments (traj_dir, top, etc.)
    add_common_args(parser)

    # Pipeline-specific arguments
    pipeline_group = parser.add_argument_group('Pipeline Control')
    pipeline_group.add_argument('--preset', type=str, default=None,
                                help='Configuration preset (small_schnet, medium_meta, etc.)')
    pipeline_group.add_argument('--model', type=str, default=None,
                                help='Model type (schnet, meta, ml3)')
    pipeline_group.add_argument('--lag_times', type=float, nargs='+', default=[10.0],
                                help='Lag times in nanoseconds')
    pipeline_group.add_argument('--n_states', type=int, nargs='+', default=[5],
                                help='Number of states to explore')
    pipeline_group.add_argument('--cache', action='store_true',
                                help='Cache datasets for reuse')
    pipeline_group.add_argument('--hurry', action='store_true',
                                help='Use larger stride for faster processing')

    # Phase control
    phase_group = parser.add_argument_group('Phase Control')
    phase_group.add_argument('--skip_preparation', action='store_true',
                             help='Skip preparation phase')
    phase_group.add_argument('--skip_training', action='store_true',
                             help='Skip training phase')
    phase_group.add_argument('--only_analysis', action='store_true',
                             help='Only run analysis phase')

    return parser.parse_args()
