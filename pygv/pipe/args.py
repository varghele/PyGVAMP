import argparse


def parse_pipeline_args():
    """Parse command-line arguments for pipeline"""
    parser = argparse.ArgumentParser(
        description='Run complete VAMPNet pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use preset configuration
  python run_pipeline.py --preset medium_schnet --traj_dir ./data --top topology.pdb --cache --hurry

  # Specify lag times and states
  python run_pipeline.py --preset small_meta --traj_dir ./data --top topology.pdb \\
      --lag_times 10 20 50 --n_states 5 8 --cache

  # Custom configuration
  python run_pipeline.py --model schnet --traj_dir ./data --top topology.pdb \\
      --lag_times 10 20 --n_states 5 --epochs 100 --cache
        """
    )

    # Required arguments
    parser.add_argument('--traj_dir', type=str, required=True,
                        help='Directory containing trajectory files')
    parser.add_argument('--top', type=str, required=True,
                        help='Topology file (will be converted if not PDB)')

    # Configuration options
    parser.add_argument('--preset', type=str, default=None,
                        help='Configuration preset (small_schnet, medium_meta, etc.)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model type (schnet, meta, ml3)')

    # Lag times and states
    parser.add_argument('--lag_times', type=float, nargs='+', default=[10.0],
                        help='Lag times in nanoseconds (can specify multiple)')
    parser.add_argument('--n_states', type=int, nargs='+', default=None,
                        help='Number of states (can specify multiple). '
                             'If omitted, determined automatically via state discovery')

    # State discovery
    parser.add_argument('--no_discover_states', action='store_true',
                        help='Disable automatic state discovery (requires --n_states)')
    parser.add_argument('--min_states', type=int, default=None,
                        help='Minimum number of states to test in discovery (default: 2)')
    parser.add_argument('--max_states', type=int, default=None,
                        help='Maximum number of states to test in discovery (default: 10)')

    # Training overrides
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides preset)')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Gradient clipping max norm (default: disabled)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Frame stride for trajectory loading (overrides preset)')
    parser.add_argument('--selection', type=str, default=None,
                        help='MDTraj atom selection string (overrides preset)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--no_continuous', action='store_true',
                        help='Treat each trajectory file as independent '
                             '(time-lagged pairs will not cross file boundaries)')

    # Pipeline control
    parser.add_argument('--cache', action='store_true',
                        help='Cache datasets for faster re-training')
    parser.add_argument('--hurry', action='store_true',
                        help='Use larger stride (up to 10) to speed up processing')

    # Output
    parser.add_argument('--output_dir', type=str, default='./experiments',
                        help='Output directory for experiments')
    parser.add_argument('--protein_name', type=str, default='protein',
                        help='Protein name for labeling')

    # Resume options
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from experiment directory')
    parser.add_argument('--skip_preparation', action='store_true',
                        help='Skip preparation phase')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training phase')
    parser.add_argument('--only_analysis', action='store_true',
                        help='Only run analysis phase')

    return parser.parse_args()