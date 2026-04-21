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
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training and inference (overrides preset)')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Gradient clipping max norm (default: disabled)')
    parser.add_argument('--lr_schedule', type=str, default=None,
                        choices=['none', 'cosine'],
                        help="Learning-rate schedule. 'cosine' anneals LR from --lr to --lr_min "
                             "over --epochs. Default: none (constant LR).")
    parser.add_argument('--lr_min', type=float, default=None,
                        help='Minimum LR for cosine schedule (default: 0.0)')
    parser.add_argument('--auto_stride', action='store_true',
                        help="Per-lag-time runtime subsampling.  For each lag τ, use "
                             "stride = max(1, floor(τ / (10·frame_dt))) on top of the "
                             "preprocessing-level stride.  Requires --timestep when no prepared "
                             "dataset with persisted frame_dt exists.  Stride is fixed within a "
                             "lag time (retrains do not change it).")
    parser.add_argument('--warm_start_retrains', action='store_true',
                        help="On retrain, preserve encoder + embedding + BN running stats and "
                             "replace only the classifier head (and the reversible score module "
                             "for --reversible).  Default: enabled.  The optimizer is always "
                             "reinitialised.  Pass --no_warm_start_retrains to disable.")
    parser.add_argument('--no_warm_start_retrains', action='store_true',
                        help="Disable warm-start: retrains rebuild the model from scratch.")
    parser.add_argument('--max_retrains', type=int, default=None,
                        help="Safety cap on retrain iterations per experiment (default: 5).")
    parser.add_argument('--no_convergence_check', action='store_true',
                        help="Run all --max_retrains rounds even if the diagnostic recommends "
                             "the same k the model was just trained with.  By default the loop "
                             "terminates as soon as the recommendation matches the current k.")
    parser.add_argument('--stride', type=int, default=None,
                        help='Frame stride for trajectory loading (overrides preset)')
    parser.add_argument('--selection', type=str, default=None,
                        help='MDTraj atom selection string (overrides preset)')
    parser.add_argument('--timestep', type=float, default=None,
                        help='Override trajectory timestep in nanoseconds. '
                             'Use when the XTC/DCD time metadata is incorrect '
                             '(e.g. --timestep 0.2 for 0.2 ns between frames)')
    parser.add_argument('--reversible', action='store_true',
                        help='Use RevGraphVAMP (reversible likelihood-based training)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--no_continuous', action='store_true',
                        help='Treat each trajectory file as independent '
                             '(time-lagged pairs will not cross file boundaries)')

    # State diagnostics
    parser.add_argument('--population_threshold', type=float, default=None,
                        help='Min state population fraction (default: 0.02)')
    parser.add_argument('--jsd_threshold', type=float, default=None,
                        help='JSD threshold for redundancy detection (default: 0.05)')

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

    # Preview / validation modes
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview the pipeline configuration and planned experiments without running anything')
    parser.add_argument('--validate_only', action='store_true',
                        help='Validate configuration (topology, trajectories, lag times) then exit')

    return parser.parse_args()