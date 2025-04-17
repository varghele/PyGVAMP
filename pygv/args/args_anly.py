# args/args_anly.py

"""
Command line arguments for VAMPNet data analysis script
"""

import argparse
import os

def get_anly_parser():
    """Create argument parser for data analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze MD trajectory and trained VAMPNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    model_group = parser.add_argument_group('Model parameters')
    model_group.add_argument('--cpu', action='store_true',
                             help='Force CPU usage even if CUDA is available')

    # Simplified approach: just provide the output directory of a trained model
    model_group.add_argument('--output_dir', type=str, required=True,
                             help='Path to the model run output directory (containing models/, plots/, and config.txt)')

    # Optional override of model file
    model_group.add_argument('--model_file', type=str, default=None,
                             help='Specific model file to use (if not specified, best_model.pt will be used)')
    return parser


def parse_config(config_path):
    """Parse the config file and convert to argparse.Namespace"""
    # Read the config file
    with open(config_path, 'r') as f:
        config_text = f.read()

    # Parse the config into a dictionary
    config_dict = {}
    for line in config_text.strip().split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        try:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Convert values to appropriate types
            if value.lower() == 'none':
                value = None
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

            config_dict[key] = value
        except ValueError:
            continue  # Skip lines without '='

    # Convert to Namespace
    args = argparse.Namespace(**config_dict)
    return args


def parse_anly_args():
    """Parse data analysis arguments and merge with config"""
    # Parse command line arguments
    parser = get_anly_parser()
    args = parser.parse_args()

    # Define paths
    run_dir = os.path.abspath(args.output_dir)
    models_dir = os.path.join(run_dir, "models")
    config_file = os.path.join(run_dir, "config.txt")

    # Check if run directory exists
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Model run directory not found at {run_dir}")

    # Check if config file exists
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    # Load config
    config_args = parse_config(config_file)

    # Override config args with command line args
    for key, value in vars(args).items():
        setattr(config_args, key, value)

    # Set model_path based on model_file or best_model.pt
    if args.model_file:
        if os.path.isfile(args.model_file):
            config_args.model_path = args.model_file
        else:
            model_path = os.path.join(models_dir, args.model_file)
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            config_args.model_path = model_path
    else:
        # Try best_model.pt first, then final_model.pt
        best_model_path = os.path.join(models_dir, "best_model.pt")
        final_model_path = os.path.join(models_dir, "final_model.pt")

        if os.path.isfile(best_model_path):
            config_args.model_path = best_model_path
        elif os.path.isfile(final_model_path):
            config_args.model_path = final_model_path
        else:
            raise FileNotFoundError(f"No model files found in {models_dir}")

    # Set analysis directory
    config_args.analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(config_args.analysis_dir, exist_ok=True)

    # Print loaded configuration
    print("\nModel configuration and analysis parameters:")
    print(f"Model path: {config_args.model_path}")
    print(f"Trajectory directory: {config_args.traj_dir}")
    print(f"Topology file: {config_args.top}")
    print(f"Analysis directory: {config_args.analysis_dir}")

    return config_args