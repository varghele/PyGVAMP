#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple multi-experiment runner for PyGVAMP
"""

import os
import argparse
import pandas as pd
from pygv.pipe.training import run_training


def create_args_from_row(row):
    """Create argument namespace from CSV row"""
    args = argparse.Namespace()

    # Set defaults
    args.encoder_type = 'schnet'
    args.val_split = 0.05
    args.sample_validate_every = 100
    args.use_embedding = True
    args.embedding_hidden_dim = 64
    args.embedding_out_dim = 64
    args.embedding_num_layers = 2
    args.embedding_dropout = 0.01
    args.embedding_act = 'leaky_relu'
    args.embedding_norm = None
    args.clip_grad = None
    args.cpu = False
    args.use_cache = True
    args.save_every = 0

    # Override with CSV values
    for column, value in row.items():
        if pd.isna(value) or value == '':
            continue

        # Handle paths
        if column in ['traj_dir', 'top', 'output_dir']:
            setattr(args, column, os.path.expanduser(str(value)))
        # Handle booleans
        elif column in ['use_attention', 'use_embedding', 'cpu', 'use_cache']:
            setattr(args, column, str(value).lower() in ('true', 'yes', '1'))
        # Handle integers
        elif column in ['n_states', 'stride', 'n_neighbors', 'node_embedding_dim',
                        'gaussian_expansion_dim', 'node_dim', 'edge_dim', 'hidden_dim',
                        'output_dim', 'n_interactions', 'clf_hidden_dim', 'clf_num_layers',
                        'epochs', 'batch_size', 'max_tau']:  # Added max_tau here
            setattr(args, column, int(value))
        # Handle floats
        elif column in ['lag_time', 'clf_dropout', 'lr', 'weight_decay']:
            setattr(args, column, float(value))
        # Handle strings (including file_pattern)
        else:
            setattr(args, column, str(value))

    # Set cache_dir if not specified
    if not hasattr(args, 'cache_dir'):
        args.cache_dir = os.path.join(args.output_dir, 'cache') if hasattr(args, 'output_dir') else 'cache'

    return args


def main():
    """Run experiments from CSV file"""
    parser = argparse.ArgumentParser(description="Run PyGVAMP experiments from CSV")
    parser.add_argument("csv_file", help="Path to CSV file with experiment configurations")
    args = parser.parse_args()

    # Load experiments
    df = pd.read_csv(args.csv_file)
    print(f"Found {len(df)} experiments in {args.csv_file}")

    # Run each experiment
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        experiment_name = row.get('run_name', f'experiment_{idx}')
        print(f"\n{'=' * 60}")
        print(f"Running experiment {idx}/{len(df)}: {experiment_name}")
        print(f"{'=' * 60}")

        try:
            # Create arguments and run training
            exp_args = create_args_from_row(row)

            # Print key parameters for verification
            print(f"Protein: {exp_args.protein_name}")
            print(f"States: {exp_args.n_states}")
            print(f"Lag time: {exp_args.lag_time}")
            print(f"Max tau: {exp_args.max_tau}")
            print(f"Epochs: {exp_args.epochs}")
            print(f"Output dir: {exp_args.output_dir}")

            run_training(exp_args)
            print(f"✅ Experiment '{experiment_name}' completed successfully")

        except Exception as e:
            print(f"❌ Experiment '{experiment_name}' failed: {str(e)}")
            continue

    print(f"\n🎉 All experiments completed!")


if __name__ == "__main__":
    main()
