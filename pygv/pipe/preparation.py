#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VAMPNet Data Preparation Pipeline

This script processes molecular dynamics trajectories and prepares them
as graph datasets for training VAMPNet models.
"""

import os
import torch
from torch_geometric.loader import DataLoader
from datetime import datetime
import json
import mdtraj as md

from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.utils.pipe_utils import find_trajectory_files
from pygv.args import parse_prep_args


def setup_output_directory(args):
    """Setup output directory and return paths"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    run_dir = os.path.join(args.output_dir, f"prep_{timestamp}")
    cache_dir = args.cache_dir or os.path.join(run_dir, "cache")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Create paths
    paths = {
        'run_dir': run_dir,
        'cache_dir': cache_dir,
        'config': os.path.join(run_dir, 'prep_config.json'),
        'stats': os.path.join(run_dir, 'dataset_stats.json'),
        'topology_pdb': os.path.join(run_dir, 'topology.pdb'),
        'state_discovery_dir': os.path.join(run_dir, 'state_discovery'),
    }

    return paths


def save_config(args, paths):
    """Save configuration to a JSON file"""
    # Convert args Namespace to a dictionary for saving
    config = vars(args)

    with open(paths['config'], 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {paths['config']}")


def save_topology_as_pdb(topology_file, selection, output_path):
    """
    Load topology and save as PDB file with selected atoms.

    Parameters
    ----------
    topology_file : str
        Path to input topology file (can be .pdb, .mae, .gro, etc.)
    selection : str
        MDTraj selection string (e.g., "name CA")
    output_path : str
        Path where the PDB file will be saved

    Returns
    -------
    dict
        Information about the saved topology
    """
    print(f"Loading topology from {topology_file}")

    # Load topology - MDTraj can handle various formats
    topology = md.load_topology(topology_file)

    # Get selected atom indices
    atom_indices = topology.select(selection)

    if len(atom_indices) == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms!")

    print(f"Selected {len(atom_indices)} atoms with selection: '{selection}'")

    # Create a minimal structure with the topology
    # We need coordinates to save as PDB, so create dummy coordinates at origin
    # These will be replaced by actual coordinates during analysis
    import numpy as np
    n_atoms = topology.n_atoms
    dummy_coords = np.zeros((1, n_atoms, 3))

    # Create trajectory object with full topology
    traj = md.Trajectory(dummy_coords, topology)

    # Slice to selected atoms
    traj_selected = traj.atom_slice(atom_indices)

    # Save as PDB
    traj_selected.save_pdb(output_path)

    print(f"Topology saved as PDB: {output_path}")

    # Collect topology information
    topology_info = {
        'input_file': topology_file,
        'output_pdb': output_path,
        'selection': selection,
        'n_atoms_total': n_atoms,
        'n_atoms_selected': len(atom_indices),
        'atom_indices': atom_indices.tolist(),
        'residues': []
    }

    # Get residue information for selected atoms
    selected_topology = traj_selected.topology
    for residue in selected_topology.residues:
        topology_info['residues'].append({
            'name': residue.name,
            'index': residue.index,
            'n_atoms': residue.n_atoms
        })

    return topology_info


def run_state_discovery(dataset, args, paths):
    """
    Run Graph2Vec + clustering for unsupervised state discovery.

    Parameters
    ----------
    dataset : VAMPNetDataset
        The prepared dataset
    args : Namespace
        Command line arguments
    paths : dict
        Output paths dictionary

    Returns
    -------
    dict
        State discovery results including recommended n_states
    """
    from pygv.clustering.state_discovery import StateDiscovery

    # Get frames dataset (not pairs)
    frames_ds = dataset.get_frames_dataset(return_pairs=False)

    # Create state discovery output directory
    os.makedirs(paths['state_discovery_dir'], exist_ok=True)

    # Run state discovery
    discovery = StateDiscovery(
        embedding_dim=args.g2v_embedding_dim,
        max_degree=args.g2v_max_degree,
        g2v_epochs=args.g2v_epochs,
        max_k=args.max_states,
        min_k=args.min_states,
    )
    discovery.fit(frames_ds)

    # Generate visualizations
    discovery.plot_results(paths['state_discovery_dir'])

    # Get recommendation
    recommended_n_states = discovery.get_recommended_n_states()

    return {
        'recommended_n_states': recommended_n_states,
        'embeddings_shape': list(discovery.get_embeddings().shape),
        'silhouette_scores': {str(k): float(v) for k, v in discovery.silhouette_scores.items()},
        'best_silhouette_k': discovery.best_k,
        'elbow_k': discovery.elbow_k,
    }


def create_and_analyze_dataset(args, paths):
    """Create and analyze the dataset"""

    # First, convert and save topology as PDB
    print("\n=== TOPOLOGY CONVERSION ===")
    topology_info = save_topology_as_pdb(
        topology_file=args.top,
        selection=args.selection,
        output_path=paths['topology_pdb']
    )

    print(f"\nLooking for trajectory files in {args.traj_dir}")

    # Find trajectory files
    trajectory_files = find_trajectory_files(
        args.traj_dir,
        file_pattern=args.file_pattern,
        recursive=args.recursive
    )

    if not trajectory_files:
        print("No trajectory files found. Exiting.")
        return None

    print(f"\nCreating dataset with {len(trajectory_files)} trajectory files")

    # Create dataset
    dataset = VAMPNetDataset(
        trajectory_files=trajectory_files,
        topology_file=args.top,
        lag_time=args.lag_time,
        n_neighbors=args.n_neighbors,
        node_embedding_dim=args.node_embedding_dim,
        gaussian_expansion_dim=args.gaussian_expansion_dim,
        selection=args.selection,
        stride=args.stride,
        cache_dir=paths['cache_dir'],
        use_cache=args.use_cache
    )

    print(f"Dataset created with {len(dataset)} samples")

    # Collect and save statistics
    stats = {
        'num_samples': len(dataset),
        'trajectories': [os.path.basename(f) for f in trajectory_files],
        'topology': {
            'input_file': topology_info['input_file'],
            'output_pdb': topology_info['output_pdb'],
            'n_atoms_selected': topology_info['n_atoms_selected'],
            'n_residues': len(topology_info['residues']),
        },
        'parameters': {
            'lag_time': args.lag_time,
            'n_neighbors': args.n_neighbors,
            'node_embedding_dim': args.node_embedding_dim,
            'gaussian_expansion_dim': args.gaussian_expansion_dim,
            'selection': args.selection,
            'stride': args.stride
        }
    }

    # Sample a data point to get structure information
    if len(dataset) > 0:
        sample = dataset[0]
        x_t0, x_t1 = sample

        stats['structure'] = {
            'num_nodes': x_t0.num_nodes,
            'num_edges': x_t0.edge_index.shape[1],
            'node_feature_dim': x_t0.x.shape[1],
            'edge_feature_dim': x_t0.edge_attr.shape[1]
        }

    # Run state discovery if enabled
    if hasattr(args, 'discover_states') and args.discover_states:
        print("\n=== STATE DISCOVERY ===")
        state_discovery_results = run_state_discovery(dataset, args, paths)
        stats['state_discovery'] = state_discovery_results
        print(f"\nRecommended n_states: {state_discovery_results['recommended_n_states']}")

    # Save statistics
    with open(paths['stats'], 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Dataset statistics saved to {paths['stats']}")

    return dataset


def analyze_sample_batch(dataset, args):
    """Analyze a sample batch from the dataset"""
    if not dataset or len(dataset) == 0:
        print("Dataset is empty. Cannot analyze sample batch.")
        return

    print(f"\nAnalyzing sample batch (batch_size={args.batch_size})...")

    # Create data loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get first batch
    for batch in loader:
        batch_t0, batch_t1 = batch

        # Print basic information
        print("\n=== BATCH ANALYSIS ===")
        print(f"Batch size: {batch_t0.num_graphs}")
        print(f"Total nodes in batch: {batch_t0.num_nodes}")
        print(f"Node feature dimensions: {batch_t0.x.shape}")
        print(f"Edge feature dimensions: {batch_t0.edge_attr.shape}")
        print(f"Number of edges: {batch_t0.edge_index.shape[1]}")

        # Analyze node features
        t0_node_features_mean = batch_t0.x.mean(dim=0)
        t0_node_features_std = batch_t0.x.std(dim=0)
        print("\n=== Node Feature Statistics ===")
        print(f"Mean: {t0_node_features_mean[:5].tolist()}... (showing first 5)")
        print(f"Std: {t0_node_features_std[:5].tolist()}... (showing first 5)")

        # Analyze edge features
        t0_edge_features_mean = batch_t0.edge_attr.mean(dim=0)
        t0_edge_features_std = batch_t0.edge_attr.std(dim=0)
        print("\n=== Edge Feature Statistics ===")
        print(f"Mean: {t0_edge_features_mean[:5].tolist()}... (showing first 5)")
        print(f"Std: {t0_edge_features_std[:5].tolist()}... (showing first 5)")

        # Compare T0 and T1
        node_features_equal = torch.allclose(batch_t0.x, batch_t1.x)
        print(f"\nNode features are {'the same' if node_features_equal else 'different'} between T0 and T1")

        if not node_features_equal:
            diff = (batch_t0.x - batch_t1.x).abs().mean().item()
            print(f"Average node feature difference: {diff:.4f}")

        break  # Only analyze the first batch

    print("\nSample batch analysis complete")


def main():
    """Main function for data preparation pipeline"""
    # Parse arguments
    args = parse_prep_args()

    # Setup output directory
    paths = setup_output_directory(args)

    # Save configuration
    save_config(args, paths)

    # Create and analyze the dataset
    dataset = create_and_analyze_dataset(args, paths)

    # Analyze a sample batch if requested
    if args.sample_batch and dataset:
        analyze_sample_batch(dataset, args)

    print(f"\nData preparation completed successfully!")
    print(f"Output directory: {paths['run_dir']}")
    print(f"Topology PDB: {paths['topology_pdb']}")
    print(f"Cache directory: {paths['cache_dir']}")


if __name__ == "__main__":
    main()
