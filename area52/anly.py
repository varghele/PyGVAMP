#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for VAMPNet analysis
"""

import os
import sys
from pygv.pipe.analysis import run_analysis
from pygv.args.args_anly import parse_config

def create_test_args():
    """Create test arguments for analysis by loading from config"""
    # Base directory of the trained model
    base_output_dir = os.path.expanduser('area57/atr')
    config_path = os.path.join(base_output_dir, 'config.txt')

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Parse the configuration file
    args = parse_config(config_path)

    # Add required analysis parameters
    args.output_dir = base_output_dir
    args.model_path = os.path.join(base_output_dir, "models/best_model.pt")
    args.analysis_dir = os.path.join(base_output_dir, "analysis")

    # Set CPU flag (optional)
    if not hasattr(args, 'cpu'):
        args.cpu = False

    return args


def test_analysis():
    """Run a test of the analysis pipeline"""
    print("Starting VAMPNet analysis test...")

    # Create test arguments
    args = create_test_args()

    # Verify output directory exists
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        print("Please update the path in create_test_args() function")
        sys.exit(1)

    # Check for models directory
    models_dir = os.path.join(output_dir, "models")
    if not os.path.isdir(models_dir):
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)

    # Check for at least one model file
    best_model = os.path.join(models_dir, "best_model.pt")
    final_model = os.path.join(models_dir, "final_model.pt")
    if not os.path.isfile(best_model) and not os.path.isfile(final_model):
        print(f"Error: No model files found in {models_dir}")
        sys.exit(1)

    print(f"Using model from: {output_dir}")

    # Run the analysis
    try:
        results = run_analysis(args)
        print("Analysis completed successfully!")
        return results
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_analysis()
