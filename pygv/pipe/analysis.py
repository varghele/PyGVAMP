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

    Pipeline flow:
    1.  Load model, create dataset
    2.  Inference → probs, embeddings, attentions, edge_indices
    3.  Plot transition matrix (original)
    4.  Run state diagnostics → StateReductionReport
    5.  If merge recommended: merge states, validate, use merged probs downstream
    6.  Save diagnostic report (JSON + plots)
    7.  Calculate attention maps (using final probs)
    8.  Plot attention maps
    9.  Generate state structures
    10. PyMOL visualizations
    11. State network plot
    12. ITS analysis
    13. CK test
    14. Interactive report

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

    # ---- Step 1-2: Create dataset, run inference ----
    dataset, loader = create_dataset_and_loader(args)

    probs, embeddings, attentions, edge_indices = analyze_vampnet_outputs(
        model=model,
        data_loader=loader,
        save_folder=paths['analysis_dir'],
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
        device=device,
        return_tensors=False
    )

    inferred_timestep = dataset._infer_timestep() / 1000  # Timestep in nanoseconds

    # ---- Step 3: Plot original transition matrix ----
    from pygv.utils.analysis import calculate_transition_matrices
    original_transition_matrix, _ = calculate_transition_matrices(
        probs=probs,
        lag_time=args.lag_time,
        stride=args.stride,
        timestep=inferred_timestep,
    )

    plot_transition_probabilities(
        probs=probs,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        lag_time=args.lag_time,
        stride=args.stride,
        timestep=inferred_timestep
    )

    # ---- Step 4-6: State diagnostics and merging ----
    auto_merge = getattr(args, 'auto_merge', True)
    population_threshold = getattr(args, 'population_threshold', 0.02)
    jsd_threshold = getattr(args, 'jsd_threshold', 0.05)
    merge_validation = getattr(args, 'merge_validation', True)
    vamp2_drop_threshold = getattr(args, 'vamp2_drop_threshold', 0.10)

    # Always run diagnostics (even if merging is disabled)
    from pygv.utils.state_diagnostics import recommend_state_reduction
    from pygv.utils.state_merging import merge_and_validate, save_merge_report
    from pygv.utils.plotting import (plot_eigenvalue_spectrum, plot_jsd_heatmap,
                                      plot_diagnostic_summary)

    print("\n" + "=" * 60)
    print("STATE QUALITY DIAGNOSTICS")
    print("=" * 60)

    diagnostic_report = recommend_state_reduction(
        transition_matrix=original_transition_matrix,
        probs=probs,
        population_threshold=population_threshold,
        jsd_threshold=jsd_threshold,
    )
    print(diagnostic_report.summary())

    # Decide whether to merge
    final_probs = probs
    merge_result = None

    if auto_merge and diagnostic_report.recommendation == "merge" and diagnostic_report.merge_groups:
        print(f"\nAttempting to merge {diagnostic_report.original_n_states} → "
              f"{diagnostic_report.effective_n_states} states...")

        merge_result = merge_and_validate(
            probs=probs,
            merge_groups=diagnostic_report.merge_groups,
            lag_time=args.lag_time,
            stride=args.stride,
            timestep=inferred_timestep,
            vamp2_drop_threshold=vamp2_drop_threshold,
            validate=merge_validation,
        )
        print(merge_result.summary())

        if merge_result.validation_passed:
            print("Merge validation PASSED — using merged states for downstream analysis.")
            final_probs = merge_result.merged_probs
        else:
            print("Merge validation FAILED — keeping original states.")
            merge_result = None
    elif diagnostic_report.recommendation == "retrain":
        print(f"\nRecommendation: RETRAIN with ~{diagnostic_report.effective_n_states} states "
              f"(large reduction from {diagnostic_report.original_n_states}).")
    else:
        print("\nNo state merging needed.")

    # Save diagnostic report
    report_path = save_merge_report(
        report=diagnostic_report,
        merge_result=merge_result,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
    )
    print(f"Diagnostic report saved to: {report_path}")

    # Plot diagnostics
    try:
        plot_eigenvalue_spectrum(
            eigenvalues=diagnostic_report.eigenvalues,
            gap_ratios=diagnostic_report.gap_ratios,
            suggested_n_states=diagnostic_report.eigenvalue_gap_suggestion,
            save_dir=paths['analysis_dir'],
            protein_name=args.protein_name,
        )
        plot_jsd_heatmap(
            jsd_matrix=diagnostic_report.jsd_matrix,
            merge_groups=diagnostic_report.merge_groups,
            save_dir=paths['analysis_dir'],
            protein_name=args.protein_name,
        )
        plot_diagnostic_summary(
            diagnostic_report=diagnostic_report,
            original_transition_matrix=original_transition_matrix,
            merge_result=merge_result,
            save_dir=paths['analysis_dir'],
            protein_name=args.protein_name,
        )
        print("Diagnostic plots saved.")
    except Exception as e:
        print(f"Warning: Could not generate diagnostic plots: {e}")

    # ---- Step 7-8: Attention maps (using final_probs) ----
    topology = md.load(args.top).topology
    residue_indices, residue_names = extract_residue_indices_from_selection(
        selection_string=args.selection,
        topology=topology
    )

    state_attention_maps, state_populations = calculate_state_edge_attention_maps(
        edge_attentions=attentions,
        edge_indices=edge_indices,
        probs=final_probs,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
    )

    plot_state_edge_attention_maps(
        state_attention_maps=state_attention_maps,
        state_populations=state_populations,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        threshold=0.001,
        residue_indices=residue_indices
    )

    plot_state_attention_weights(
        state_attention_maps=state_attention_maps,
        topology_file=args.top,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        plot_sum_direction="target",
        atom_selection=args.selection,
    )
    print("Attention analysis complete")

    # ---- Step 9: Generate state structures ----
    print("Generating state structures")
    state_structures = generate_state_structures(
        traj_folder=args.traj_dir,
        topology_file=args.top,
        probs=final_probs,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        stride=args.stride,
        n_structures=10,
        prob_threshold=0.0
    )

    # ---- Step 10: PyMOL visualizations (optional) ----
    try:
        visualize_state_ensemble(
            state_structures=state_structures,
            save_dir=paths['analysis_dir'],
            protein_name=args.protein_name,
            image_size=(800, 600)
        )
        print("Representative state structures generated")
    except Exception as e:
        print(f"Warning: PyMOL visualization skipped: {e}")

    save_attention_colored_structures(
        state_structures=state_structures,
        state_attention_maps=state_attention_maps,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        residue_indices=residue_indices,
        residue_names=residue_names
    )

    try:
        visualize_attention_ensemble(
            state_structures=state_structures,
            state_attention_maps=state_attention_maps,
            save_dir=paths['analysis_dir'],
            protein_name=args.protein_name,
        )
        print("Attention-colored visualizations generated")
    except Exception as e:
        print(f"Warning: PyMOL attention visualization skipped: {e}")

    # ---- Step 11: State network plot ----
    plot_state_network(
        probs=final_probs,
        state_structures=state_structures,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        lag_time=args.lag_time,
        stride=args.stride,
        timestep=inferred_timestep
    )
    print("State network plotted")

    # ---- Step 12: ITS analysis ----
    try:
        from pygv.utils.its import analyze_implied_timescales
        # Generate lag times: 10 points from timestep*stride to 5*lag_time
        effective_timestep = inferred_timestep * args.stride
        max_lag = min(5 * args.lag_time, (len(final_probs) * effective_timestep) / 3)
        its_lag_times = np.linspace(effective_timestep, max_lag, 10).tolist()

        print("\nRunning implied timescales analysis...")
        analyze_implied_timescales(
            probs=final_probs,
            save_dir=paths['analysis_dir'],
            protein_name=args.protein_name,
            lag_times_ns=its_lag_times,
            stride=args.stride,
            timestep=inferred_timestep,
        )
    except Exception as e:
        print(f"Warning: ITS analysis failed: {e}")

    # ---- Step 13: Chapman-Kolmogorov test ----
    try:
        from pygv.utils.ck import run_ck_analysis
        ck_lag_times = [args.lag_time * m for m in [0.5, 1.0, 2.0] if args.lag_time * m > 0]

        print("\nRunning Chapman-Kolmogorov test...")
        run_ck_analysis(
            probs=final_probs,
            save_dir=paths['analysis_dir'],
            protein_name=args.protein_name,
            lag_times_ns=ck_lag_times,
            stride=args.stride,
            timestep=inferred_timestep,
        )
    except Exception as e:
        print(f"Warning: CK test failed: {e}")

    # ---- Step 14: Interactive report ----
    experiment_dir = os.path.dirname(os.path.dirname(paths['analysis_dir']))
    analysis_parent = os.path.join(experiment_dir, 'analysis')
    if os.path.isdir(analysis_parent):
        from pygv.utils.interactive_report import generate_merged_interactive_report
        print("\nGenerating merged interactive report...")
        try:
            report_path = generate_merged_interactive_report(
                experiment_dir=experiment_dir,
                topology_file=args.top,
                protein_name=args.protein_name,
                max_frames=5000,
                stride=args.stride,
                timestep=inferred_timestep,
            )
            if report_path:
                print(f"Merged interactive report: {report_path}")
        except Exception as e:
            print(f"Warning: Could not generate merged interactive report: {e}")

    # Build results dict
    results = {
        "model": model,
        "paths": paths,
        "args": args,
        "probs": final_probs,
        "original_probs": probs,
        "embeddings": embeddings,
        "diagnostic_report": diagnostic_report,
        "merge_result": merge_result,
        "state_attention_maps": state_attention_maps,
        "state_populations": state_populations,
    }

    print(f"\nResults saved to: {paths['analysis_dir']}")

    return results


if __name__ == "__main__":
    # Get args
    args = parse_anly_args()

    # Run the analysis pipeline
    run_analysis(args=args)
