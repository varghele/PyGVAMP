#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VAMPNet Analysis Pipeline

This script provides a command-line interface for analyzing molecular dynamics data
using trained VAMPNet models.
"""
# Import arguments parser
from pygv.args.args_anly import parse_anly_args, parse_config

import os
import sys
import torch
import numpy as np

from pygv.utils.analysis import calculate_state_attention_maps
from pygv.utils.plotting import plot_transition_probabilities, plot_state_attention_maps

from pygv.vampnet.vampnet import VAMPNet
from pygv.utils.analysis import analyze_vampnet_outputs
from torch_geometric.loader import DataLoader
from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.utils.pipe_utils import find_trajectory_files

def get_model_and_traj_directory(args):
    """
    Get paths to the best model and trajectory files, verifying they exist.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing model_path, traj_dir, and analysis_dir

    Returns
    -------
    dict
        Dictionary containing verified paths
    """
    paths = {}

    # Verify model path
    model_path = os.path.expanduser(args.model_path)
    if not os.path.isfile(model_path):
        sys.exit(f"Error: Model file not found at {model_path}")
    paths['model_path'] = model_path

    # Verify trajectory directory
    traj_dir = os.path.expanduser(args.traj_dir)
    if not os.path.isdir(traj_dir):
        sys.exit(f"Error: Trajectory directory not found at {traj_dir}")
    paths['traj_dir'] = traj_dir

    # Verify topology file
    top_file = os.path.expanduser(args.top)
    if not os.path.isfile(top_file):
        sys.exit(f"Error: Topology file not found at {top_file}")
    paths['top_file'] = top_file

    # Set analysis directory
    analysis_dir = os.path.expanduser(args.analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)
    paths['analysis_dir'] = analysis_dir

    return paths


def load_model(model_path, args, device):
    """
    Load a previously saved VAMPNet model.

    Parameters
    ----------
    model_path : str
        Path to the saved model file
    args : argparse.Namespace
        Arguments containing device settings
    device: torch.device
    Returns
    -------
    VAMPNet
        The loaded VAMPNet model on the appropriate device
    """
    # Load the model
    try:
        model = VAMPNet.load_complete_model(model_path, map_location=device)
        model = model.to(device)
        print(f"Model loaded to {device}")
        return model
    except Exception as e:
        sys.exit(f"Error loading model: {str(e)}")

def create_dataset_and_loader(args):
    """Create dataset and data loader"""
    # Getting all trajectories in traj directory
    traj_files = find_trajectory_files(args.traj_dir)

    print("Creating dataset...")
    dataset = VAMPNetDataset(
        trajectory_files=traj_files,
        topology_file=args.top,
        lag_time=args.lag_time,
        n_neighbors=args.n_neighbors,
        node_embedding_dim=args.node_embedding_dim,
        gaussian_expansion_dim=args.gaussian_expansion_dim,
        selection=args.selection,
        stride=args.stride,
        cache_dir=args.cache_dir,
        use_cache=True if args.cache_dir is not None else False
    )

    # Get frames dataset instead of time-lagged pairs dataset
    frames_dataset = dataset.get_frames_dataset(return_pairs=False)

    print(f"Dataset created with {len(dataset)} samples")

    # Create data loader
    loader = DataLoader(
        frames_dataset,
        shuffle=False, # Always false for inference
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
        pin_memory=torch.cuda.is_available() and not args.cpu
    )

    return dataset, loader

def analyze_trajectories(model, paths, args):
    """
    Analyze molecular dynamics trajectories using a trained VAMPNet model.

    Parameters
    ----------
    model : VAMPNet
        The trained VAMPNet model
    paths : dict
        Dictionary containing file paths
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    dict
        Results of the analysis
    """
    print("\nAnalyzing trajectories...")
    print(f"Using trajectory data from: {paths['traj_dir']}")

    # Set model to evaluation mode
    model.eval()

    # Here you would implement your trajectory analysis code
    # For now, we just return a placeholder result
    results = {
        "model": model,
        "paths": paths,
        "args": args
    }

    print("Analysis complete!")
    return results


def run_analysis(args=None):
    """
    Main function to run the analysis pipeline.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Command line arguments. If None, they will be parsed from sys.argv.
    """
    # Parse arguments if not provided
    if args is None:
        args = parse_anly_args()

    # Get paths to model and trajectory files
    paths = get_model_and_traj_directory(args)

    # Load the model and get device
    # Determine device
    if args.cpu is True:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(paths['model_path'], args, device)

    # Print model information
    print(f"\nLoaded model: {type(model).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Print model structure overview
    print("\nModel component overview:")
    if hasattr(model, 'embedding_module'):
        embedding = model.embedding_module if hasattr(model, 'embedding_module') else model.classifier
        print(f"Embedding type: {type(embedding).__name__}")
    if hasattr(model, 'encoder'):
        print(f"Encoder type: {type(model.encoder).__name__}")
    if hasattr(model, 'classifier') or hasattr(model, 'classifier_module'):
        classifier = model.classifier_module if hasattr(model, 'classifier_module') else model.classifier
        print(f"Classifier type: {type(classifier).__name__}")

    # Create dataset and loader for analysis
    dataset, loader = create_dataset_and_loader(args)

    # Get state transition probabilities, graph embeddings and attention scores
    probs, embeddings, attentions, edge_indices = analyze_vampnet_outputs(model=model,
                                                            data_loader=loader,
                                                            save_folder=paths['analysis_dir'],
                                                            batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
                                                            device=device,
                                                            return_tensors=False
                                                            )

    inferred_timestep = dataset._infer_timestep()/1000 # Timestep in nanoseconds

    # Calculate and plot the state transition matrices
    plot_transition_probabilities(probs = probs,
                                  save_dir=paths['analysis_dir'],
                                  protein_name='ab42',#args.protein_name, #TODO: INCLUDE THIS AS AN ARGUMENT
                                  lag_time=args.lag_time,
                                  stride=args.stride,
                                  timestep=inferred_timestep)

    # TODO: THIS IS NOT IMPLEMENTED CORRECTLY SO FAR, YOU NEED THE EDGE INDICES AND PROBABLY REVERSE THE EDGE INDICE GROUPING FROM BEFORE
    # Calculate attention maps with pre-calculated neighbor indices
    state_attention_maps, state_populations = calculate_state_attention_maps(
        attentions=attentions,
        neighbor_indices=neighbor_indices,
        state_assignments=state_assignments,
        num_classes=args.n_states,
        num_atoms=args.num_atoms
    )

    # Plot attention maps
    """plot_state_attention_maps(attention_maps=state_attention_maps,
                              states_order=,
                              n_states=args.n_states,
                              state_populations=state_populations,
                              save_path=,
                              n_atoms=)"""




    # Run the trajectory analysis
    results = analyze_trajectories(model, paths, args)

    print(f"\nResults saved to: {paths['analysis_dir']}")

    return results


if __name__ == "__main__":
    # Get args
    args = parse_anly_args()

    # Run the analysis pipeline
    run_analysis(args=args)
