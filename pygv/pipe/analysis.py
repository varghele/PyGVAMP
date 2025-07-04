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
import mdtraj as md

from pygv.utils.analysis import (calculate_state_edge_attention_maps, generate_state_structures,
                                 save_attention_colored_structures, extract_residue_indices_from_selection)
from pygv.utils.plotting import (plot_transition_probabilities, plot_state_edge_attention_maps,
                                 plot_state_attention_weights, plot_all_residue_attention_directions,
                                 visualize_state_ensemble, visualize_attention_ensemble,
                                 plot_state_network)

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
    traj_files = find_trajectory_files(dataset_path=args.traj_dir,
                                       file_pattern=args.file_pattern)

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
    plot_transition_probabilities(probs=probs,
                                  save_dir=paths['analysis_dir'],
                                  protein_name=args.protein_name,
                                  lag_time=args.lag_time,
                                  stride=args.stride,
                                  timestep=inferred_timestep)

    # Get residue index list for sparse selections
    # Load the topology
    topology = md.load(args.top).topology
    residue_indices, residue_names = extract_residue_indices_from_selection(selection_string=args.selection,
                                                                            topology=topology)

    # Calculate attention maps
    state_attention_maps, state_populations = calculate_state_edge_attention_maps(
        edge_attentions=attentions,
        edge_indices=edge_indices,
        probs=probs,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
    )

    # Plot attention maps
    plot_state_edge_attention_maps(
        state_attention_maps=state_attention_maps,
        state_populations=state_populations,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        threshold=0.001,  # Optional: hide low attention values
        residue_indices=residue_indices
    )

    # Create residue-level attention plot
    plot_state_attention_weights(
        state_attention_maps=state_attention_maps,
        topology_file=args.top,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        plot_sum_direction="target",  # Show attention TO residues
        atom_selection=args.selection,
    )

    # Or to get all perspectives at once
    # TODO: REMOVE THAT PART; IT IS UNNECESSARY TO HAVE BOTH DIRECTIONS
    #plot_all_residue_attention_directions(
    #    state_attention_maps=state_attention_maps,
    #    topology_file=args.top,
    #    save_dir=paths['analysis_dir'],
    #    protein_name=args.protein_name,
    #)
    print("Attention analysis complete")

    print("Generating state structures")
    state_structures = generate_state_structures(
        traj_folder=args.traj_dir,
        topology_file=args.top,
        probs=probs,  # The state probabilities array from your analysis
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        stride=args.stride,
        n_structures=10,  # Generate 10 representative structures per state
        prob_threshold=0.0  # Only consider frames with probability ≥ 0.0
    )

    # Generate visualization
    visualize_state_ensemble(
        state_structures=state_structures,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        image_size=(800, 600)
    )
    print("Representative state structures generated")

    # Generate state ensembles with attention values
    save_attention_colored_structures(
        state_structures=state_structures,
        state_attention_maps=state_attention_maps,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        residue_indices=residue_indices,
        residue_names=residue_names
    )

    # Visualize attention ensembles
    visualize_attention_ensemble(
        state_structures=state_structures,
        state_attention_maps=state_attention_maps,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
    )
    print("Attention-colored visualizations generated")

    lag_frames = int(args.lag_time/inferred_timestep)/args.stride

    plot_state_network(
        probs=probs,
        state_structures=state_structures,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        lag_time=args.lag_time,
        stride=args.stride,
        timestep=inferred_timestep
    )
    print("network of states plotted ")

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
