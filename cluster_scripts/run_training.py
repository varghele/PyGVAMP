#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line script for VAMPNet training
"""

import os
import sys
from pygv.pipe.training import run_training
from pygv.args.args_train import get_train_parser


def main():
    """Main entry point for VAMPNet training"""

    # Get the argument parser
    parser = get_train_parser()

    # Parse command line arguments
    args = parser.parse_args()

    # Run the training pipeline
    print(f"Starting VAMPNet training for {args.protein_name}")
    run_training(args)
    print(f"Training completed for {args.protein_name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
