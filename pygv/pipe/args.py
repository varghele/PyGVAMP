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
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help="Enable plateau-based early stopping: stop after this many "
                             "consecutive epochs without a meaningful improvement.  Applies "
                             "to both initial training and retrains.  Off when unset.  "
                             "Suggested non-default: 8.")
    parser.add_argument('--early_stopping_tol', type=float, default=None,
                        help="Relative improvement threshold for the plateau counter.  An "
                             "epoch counts as an improvement only when the gain over the "
                             "plateau reference exceeds this fraction.  Suggested: 5e-4 "
                             "(0.05%% relative, sits above the Val-VAMP plateau noise floor).")
    parser.add_argument('--early_stopping_min_epochs', type=int, default=None,
                        help="Warmup — no early-stopping trigger before this epoch.  Suggested: 10.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Master RNG seed — controls train/val split and dataset-internal "
                             "random state.  Use distinct values (0, 1, 2, ...) across a "
                             "multi-seed sweep.  Default: 42 (historical).")
    # Encoder/graph architecture overrides — override preset defaults for
    # per-experiment reproduction protocols without creating a new preset class.
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Encoder hidden dimension (SchNet/GIN).')
    parser.add_argument('--output_dim', type=int, default=None,
                        help='Encoder output dimension.')
    parser.add_argument('--n_interactions', type=int, default=None,
                        help='Number of message-passing layers.')
    parser.add_argument('--n_neighbors', type=int, default=None,
                        help='k-NN graph neighbors.')
    parser.add_argument('--gaussian_expansion_dim', type=int, default=None,
                        help='Number of RBF/Gaussian edge features.')
    parser.add_argument('--use_attention', dest='use_attention', action='store_const', const=True,
                        default=None, help='Enable attention in the encoder (default off depends on preset).')
    parser.add_argument('--no_use_attention', dest='use_attention', action='store_const', const=False,
                        help='Force-disable attention in the encoder.')
    # Pre-encoder embedding MLP — paired toggle + per-field overrides.
    # When disabled, raw node features (e.g. one-hot atom-type identity) are
    # fed straight into the encoder, which handles the input projection in
    # its first linear layer.
    parser.add_argument('--use_embedding', dest='use_embedding', action='store_const', const=True,
                        default=None, help='Insert a pre-encoder MLP on node features (default depends on preset).')
    parser.add_argument('--no_use_embedding', dest='use_embedding', action='store_const', const=False,
                        help='Skip the pre-encoder MLP; feed raw node features into the encoder.')
    parser.add_argument('--embedding_hidden_dim', type=int, default=None,
                        help='Hidden width of the pre-encoder embedding MLP.')
    parser.add_argument('--embedding_out_dim', type=int, default=None,
                        help='Output width of the pre-encoder embedding MLP (becomes encoder node_dim).')
    parser.add_argument('--embedding_num_layers', type=int, default=None,
                        help='Number of layers in the pre-encoder embedding MLP. 1 = single linear, no activation.')
    parser.add_argument('--embedding_act', type=str, default=None,
                        help='Activation for the pre-encoder embedding MLP (e.g. relu, tanh).')
    parser.add_argument('--embedding_dropout', type=float, default=None,
                        help='Dropout in the pre-encoder embedding MLP.')
    parser.add_argument('--embedding_norm', type=str, default=None,
                        help='Normalization in the pre-encoder embedding MLP (batch_norm, layer_norm, none).')
    # Classifier (softmax) head overrides.
    parser.add_argument('--clf_hidden_dim', type=int, default=None,
                        help='Hidden width of the classifier head.')
    parser.add_argument('--clf_num_layers', type=int, default=None,
                        help='Number of layers in the classifier head. 1 = single linear → softmax.')
    parser.add_argument('--clf_dropout', type=float, default=None,
                        help='Dropout in the classifier head.')
    parser.add_argument('--clf_activation', type=str, default=None,
                        help='Activation in the classifier head (e.g. relu, tanh).')
    parser.add_argument('--clf_norm', type=str, default=None,
                        help='Normalization in the classifier head (batch_norm, layer_norm, none).')
    parser.add_argument('--file_pattern', type=str, default=None,
                        help='Glob for trajectory files within --traj_dir (default: *.xtc).')
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay / L2.')
    parser.add_argument('--val_split', type=float, default=None,
                        help='Validation fraction (e.g. 0.3 for 70/30).')
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