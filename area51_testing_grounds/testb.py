#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze unique edge connectivity states in MD trajectory using VAMPNet dataset
"""

import os
import sys
import argparse
import torch
import numpy as np
import hashlib
from torch_geometric.loader import DataLoader
from collections import defaultdict
from tqdm import tqdm
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pygv.utils.pipe_utils import find_trajectory_files

# Add parent directory to sys.path to import from your package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the VAMPNet dataset
from pygv.dataset.vampnet_dataset import VAMPNetDataset


def edge_indices_equal_sorted(edge_index1, edge_index2):
    """
    Compare edge indices by sorting them first (more memory efficient).
    """
    # Check if they have the same number of edges
    if edge_index1.shape[1] != edge_index2.shape[1]:
        return False

    # Convert to list of edge tuples and sort
    edges1 = edge_index1.t().tolist()  # Convert to list of [src, dst] pairs
    edges2 = edge_index2.t().tolist()

    # Sort the edge lists (sorts by first element, then second element)
    edges1_sorted = sorted(edges1)
    edges2_sorted = sorted(edges2)

    # Compare sorted edge lists
    return edges1_sorted == edges2_sorted


def create_edge_connectivity_hash(edge_index):
    """
    Create a unique hash for edge connectivity pattern.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index tensor of shape [2, num_edges]

    Returns
    -------
    str
        Unique hash string representing the edge connectivity
    """
    # Convert to list of edge tuples and sort for canonical representation
    edges = edge_index.t().tolist()  # Convert to list of [src, dst] pairs
    edges_sorted = sorted(edges)  # Sort the entire edge list

    # Convert to string representation
    edges_str = str(edges_sorted)
    #print(edges_str)

    # Create hash
    hash_obj = hashlib.md5(edges_str.encode())
    #exit()
    return hash_obj.hexdigest()


def create_test_args():
    """Create a simple argument namespace for testing preparation"""
    args = argparse.Namespace()

    # Input data settings - modify these paths for your data
    #args.traj_dir = os.path.expanduser(
    #    '~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/')
    #args.top = os.path.expanduser(
    #    '~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein.pdb')

    args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/r0/')
    args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/prot.pdb')

    args.file_pattern = '*.xtc'
    args.recursive = True

    # Data processing settings
    args.selection = 'name CA'
    args.stride = 100
    args.lag_time = 20.0
    args.n_neighbors = 4
    args.node_embedding_dim = 16
    args.gaussian_expansion_dim = 16

    # Output settings
    args.output_dir = './connectivity_analysis'
    args.cache_dir = './connectivity_analysis/cache'
    args.use_cache = True
    args.batch_size = 32

    return args


def analyze_edge_connectivity_states(dataset, max_frames=None, save_results=True):
    """
    Analyze unique edge connectivity states in the dataset

    Parameters
    ----------
    dataset : VAMPNetDataset
        The dataset to analyze
    max_frames : int, optional
        Maximum number of frames to analyze (None for all)
    save_results : bool
        Whether to save results to files

    Returns
    -------
    dict
        Analysis results containing unique states, frame mappings, etc.
    """
    print("Analyzing edge connectivity states...")

    # Get frames dataset (individual frames, not time-lagged pairs)
    frames_dataset = dataset.get_frames_dataset(return_pairs=False)

    # Determine number of frames to analyze
    n_frames = len(frames_dataset)
    if max_frames is not None and max_frames < n_frames:
        n_frames = max_frames
        print(f"Analyzing first {n_frames} frames (limited by max_frames)")
    else:
        print(f"Analyzing all {n_frames} frames")

    # Create data loader
    dataloader = DataLoader(
        frames_dataset,
        batch_size=1,  # Process one frame at a time for hash computation
        shuffle=False,
        num_workers=0
    )

    # Storage for analysis
    unique_hashes = {}  # hash -> first_frame_index
    frame_to_hash = {}  # frame_index -> hash
    hash_to_frames = defaultdict(list)  # hash -> list of frame indices
    edge_connectivity_examples = {}  # hash -> edge_index example

    print("Processing frames and computing connectivity hashes...")

    # Process frames
    frame_idx = 0
    for batch in tqdm(dataloader, desc="Processing frames"):
        if frame_idx >= n_frames:
            break

        # Extract edge index from the batch
        edge_index = batch.edge_index

        # Create hash for this edge connectivity
        connectivity_hash = create_edge_connectivity_hash(edge_index)

        # Store mappings
        frame_to_hash[frame_idx] = connectivity_hash
        hash_to_frames[connectivity_hash].append(frame_idx)

        # If this is a new unique connectivity, store it
        if connectivity_hash not in unique_hashes:
            unique_hashes[connectivity_hash] = frame_idx
            edge_connectivity_examples[connectivity_hash] = edge_index.clone()

        frame_idx += 1

    # Analyze results
    n_unique_states = len(unique_hashes)

    print(f"\n{'=' * 60}")
    print("EDGE CONNECTIVITY ANALYSIS RESULTS")
    print(f"{'=' * 60}")
    print(f"Total frames analyzed: {n_frames}")
    print(f"Unique edge connectivity states: {n_unique_states}")
    print(f"Compression ratio: {n_frames / n_unique_states:.2f}:1")

    # Analyze state populations
    state_populations = {hash_val: len(frames) for hash_val, frames in hash_to_frames.items()}
    sorted_states = sorted(state_populations.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 most populated states:")
    for i, (hash_val, population) in enumerate(sorted_states[:10]):
        percentage = (population / n_frames) * 100
        first_frame = unique_hashes[hash_val]
        print(
            f"  {i + 1:2d}. State {hash_val[:8]}... : {population:4d} frames ({percentage:5.1f}%) - First seen: frame {first_frame}")

    # Analyze state distribution
    populations = list(state_populations.values())
    print(f"\nState population statistics:")
    print(f"  Mean population per state: {np.mean(populations):.1f}")
    print(f"  Median population per state: {np.median(populations):.1f}")
    print(f"  Most populated state: {np.max(populations)} frames")
    print(f"  Least populated state: {np.min(populations)} frames")
    print(f"  States with only 1 frame: {sum(1 for p in populations if p == 1)}")

    # Create results dictionary
    results = {
        'n_frames_analyzed': n_frames,
        'n_unique_states': n_unique_states,
        'unique_hashes': unique_hashes,
        'frame_to_hash': frame_to_hash,
        'hash_to_frames': dict(hash_to_frames),
        'state_populations': state_populations,
        'edge_connectivity_examples': edge_connectivity_examples,
        'compression_ratio': n_frames / n_unique_states
    }

    # Save results if requested
    if save_results:
        save_analysis_results(results)

    return results


def save_analysis_results(results, output_dir='./connectivity_analysis'):
    """
    Save analysis results to files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save main results as pickle
    with open(os.path.join(output_dir, 'connectivity_analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Save summary as text
    with open(os.path.join(output_dir, 'connectivity_summary.txt'), 'w') as f:
        f.write("EDGE CONNECTIVITY ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total frames analyzed: {results['n_frames_analyzed']}\n")
        f.write(f"Unique edge connectivity states: {results['n_unique_states']}\n")
        f.write(f"Compression ratio: {results['compression_ratio']:.2f}:1\n\n")

        # Write state populations
        sorted_states = sorted(results['state_populations'].items(), key=lambda x: x[1], reverse=True)
        f.write("State populations (hash -> count):\n")
        for hash_val, population in sorted_states:
            percentage = (population / results['n_frames_analyzed']) * 100
            f.write(f"  {hash_val}: {population} frames ({percentage:.1f}%)\n")

    # Create and save visualizations
    create_visualizations(results, output_dir)

    print(f"\nResults saved to {output_dir}/")


def create_visualizations(results, output_dir):
    """
    Create visualizations of the connectivity analysis
    """
    # State population distribution
    plt.figure(figsize=(15, 10))

    # Plot 1: State population histogram
    plt.subplot(2, 2, 1)
    populations = list(results['state_populations'].values())
    plt.hist(populations, bins=min(50, len(populations)), edgecolor='black', alpha=0.7)
    plt.xlabel('Frames per State')
    plt.ylabel('Number of States')
    plt.title('Distribution of State Populations')
    plt.yscale('log')

    # Plot 2: Cumulative state coverage
    plt.subplot(2, 2, 2)
    sorted_pops = sorted(populations, reverse=True)
    cumulative = np.cumsum(sorted_pops)
    cumulative_pct = cumulative / results['n_frames_analyzed'] * 100
    plt.plot(range(1, len(cumulative) + 1), cumulative_pct)
    plt.xlabel('Number of States (ranked by population)')
    plt.ylabel('Cumulative Coverage (%)')
    plt.title('Cumulative State Coverage')
    plt.grid(True, alpha=0.3)

    # Plot 3: Top 20 states
    plt.subplot(2, 2, 3)
    sorted_states = sorted(results['state_populations'].items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_states[:20]
    if len(top_20) > 0:
        hashes, pops = zip(*top_20)
        short_hashes = [h[:8] + '...' for h in hashes]
        plt.bar(range(len(pops)), pops)
        plt.xlabel('State Rank')
        plt.ylabel('Number of Frames')
        plt.title('Top 20 Most Populated States')
        plt.xticks(range(len(pops)), range(1, len(pops) + 1))

    # Plot 4: State discovery over time
    plt.subplot(2, 2, 4)
    unique_states_over_time = []
    seen_hashes = set()
    for frame_idx in range(results['n_frames_analyzed']):
        hash_val = results['frame_to_hash'][frame_idx]
        if hash_val not in seen_hashes:
            seen_hashes.add(hash_val)
        unique_states_over_time.append(len(seen_hashes))

    plt.plot(unique_states_over_time)
    plt.xlabel('Frame Index')
    plt.ylabel('Cumulative Unique States')
    plt.title('State Discovery Over Time')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'connectivity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}/connectivity_analysis.png")


def compare_connectivity_states(results, state_hash_1, state_hash_2):
    """
    Compare two connectivity states in detail
    """
    if state_hash_1 not in results['edge_connectivity_examples']:
        print(f"State {state_hash_1} not found")
        return

    if state_hash_2 not in results['edge_connectivity_examples']:
        print(f"State {state_hash_2} not found")
        return

    edge_index_1 = results['edge_connectivity_examples'][state_hash_1]
    edge_index_2 = results['edge_connectivity_examples'][state_hash_2]

    print(f"\nComparing states {state_hash_1[:8]}... and {state_hash_2[:8]}...")
    print(f"State 1: {edge_index_1.shape[1]} edges")
    print(f"State 2: {edge_index_2.shape[1]} edges")

    # Check if they're the same
    are_same = edge_indices_equal_sorted(edge_index_1, edge_index_2)
    print(f"States are identical: {are_same}")

    if not are_same:
        # Find differences
        edges_1 = set(map(tuple, edge_index_1.t().tolist()))
        edges_2 = set(map(tuple, edge_index_2.t().tolist()))

        only_in_1 = edges_1 - edges_2
        only_in_2 = edges_2 - edges_1
        common = edges_1 & edges_2

        print(f"Common edges: {len(common)}")
        print(f"Edges only in state 1: {len(only_in_1)}")
        print(f"Edges only in state 2: {len(only_in_2)}")


def compare_connectivity_states_top5(results):
    """
    Compare the top 5 connectivity states in detail
    """
    # Get the top 5 most populated states
    sorted_states = sorted(results['state_populations'].items(), key=lambda x: x[1], reverse=True)
    top_5_states = sorted_states[:5]

    if len(top_5_states) < 2:
        print("Not enough states to compare")
        return

    print(f"\n{'=' * 60}")
    print("TOP 5 CONNECTIVITY STATES COMPARISON")
    print(f"{'=' * 60}")

    # Print overview of top 5 states
    print("\nTop 5 most populated states:")
    for i, (hash_val, population) in enumerate(top_5_states):
        percentage = (population / results['n_frames_analyzed']) * 100
        first_frame = results['unique_hashes'][hash_val]
        print(
            f"  {i + 1}. State {hash_val[:8]}... : {population:4d} frames ({percentage:5.1f}%) - First seen: frame {first_frame}")

    # Compare each pair of top 5 states
    print(f"\n{'=' * 60}")
    print("PAIRWISE COMPARISONS")
    print(f"{'=' * 60}")

    for i in range(len(top_5_states)):
        for j in range(i + 1, len(top_5_states)):
            state_hash_1, pop_1 = top_5_states[i]
            state_hash_2, pop_2 = top_5_states[j]

            print(f"\n--- Comparing State {i + 1} vs State {j + 1} ---")
            print(f"State {i + 1}: {state_hash_1[:8]}... ({pop_1} frames)")
            print(f"State {j + 1}: {state_hash_2[:8]}... ({pop_2} frames)")

            # Check if states exist in examples
            if state_hash_1 not in results['edge_connectivity_examples']:
                print(f"State {state_hash_1[:8]}... not found in examples")
                continue

            if state_hash_2 not in results['edge_connectivity_examples']:
                print(f"State {state_hash_2[:8]}... not found in examples")
                continue

            edge_index_1 = results['edge_connectivity_examples'][state_hash_1]
            edge_index_2 = results['edge_connectivity_examples'][state_hash_2]

            print(f"State {i + 1}: {edge_index_1.shape[1]} edges")
            print(f"State {j + 1}: {edge_index_2.shape[1]} edges")

            # Check if they're the same
            are_same = edge_indices_equal_sorted(edge_index_1, edge_index_2)
            print(f"States are identical: {are_same}")

            if not are_same:
                # Find differences
                edges_1 = set(map(tuple, edge_index_1.t().tolist()))
                edges_2 = set(map(tuple, edge_index_2.t().tolist()))

                only_in_1 = edges_1 - edges_2
                only_in_2 = edges_2 - edges_1
                common = edges_1 & edges_2

                print(f"Common edges: {len(common)}")
                print(f"Edges only in state {i + 1}: {len(only_in_1)}")
                print(f"Edges only in state {j + 1}: {len(only_in_2)}")

                # Calculate similarity percentage
                total_unique_edges = len(edges_1 | edges_2)
                similarity = len(common) / total_unique_edges * 100 if total_unique_edges > 0 else 0
                print(f"Edge similarity: {similarity:.1f}%")
            else:
                print("States have identical edge connectivity!")


def analyze_top5_state_relationships(results):
    """
    Analyze relationships between top 5 states
    """
    sorted_states = sorted(results['state_populations'].items(), key=lambda x: x[1], reverse=True)
    top_5_states = sorted_states[:5]

    if len(top_5_states) < 2:
        print("Not enough states for relationship analysis")
        return

    print(f"\n{'=' * 60}")
    print("TOP 5 STATES RELATIONSHIP ANALYSIS")
    print(f"{'=' * 60}")

    # Create similarity matrix
    n_states = len(top_5_states)
    similarity_matrix = np.zeros((n_states, n_states))
    edge_count_matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                similarity_matrix[i, j] = 100.0  # Self-similarity is 100%
                edge_count_matrix[i, j] = results['edge_connectivity_examples'][top_5_states[i][0]].shape[1]
            else:
                state_hash_1 = top_5_states[i][0]
                state_hash_2 = top_5_states[j][0]

                if (state_hash_1 in results['edge_connectivity_examples'] and
                        state_hash_2 in results['edge_connectivity_examples']):
                    edge_index_1 = results['edge_connectivity_examples'][state_hash_1]
                    edge_index_2 = results['edge_connectivity_examples'][state_hash_2]

                    edges_1 = set(map(tuple, edge_index_1.t().tolist()))
                    edges_2 = set(map(tuple, edge_index_2.t().tolist()))

                    common = edges_1 & edges_2
                    total_unique = edges_1 | edges_2

                    similarity = len(common) / len(total_unique) * 100 if len(total_unique) > 0 else 0
                    similarity_matrix[i, j] = similarity
                    edge_count_matrix[i, j] = len(edges_2)

    # Print similarity matrix
    print("\nEdge Connectivity Similarity Matrix (%):")
    print("State:  ", end="")
    for i in range(n_states):
        print(f"  {i + 1:5d}", end="")
    print()

    for i in range(n_states):
        print(f"State {i + 1}: ", end="")
        for j in range(n_states):
            print(f"{similarity_matrix[i, j]:6.1f}", end="")
        print()

    # Print edge count matrix
    print("\nEdge Count Matrix:")
    print("State:  ", end="")
    for i in range(n_states):
        print(f"  {i + 1:5d}", end="")
    print()

    for i in range(n_states):
        print(f"State {i + 1}: ", end="")
        for j in range(n_states):
            print(f"{int(edge_count_matrix[i, j]):6d}", end="")
        print()

    # Find most and least similar pairs
    max_similarity = 0
    min_similarity = 100
    max_pair = None
    min_pair = None

    for i in range(n_states):
        for j in range(i + 1, n_states):
            sim = similarity_matrix[i, j]
            if sim > max_similarity:
                max_similarity = sim
                max_pair = (i, j)
            if sim < min_similarity:
                min_similarity = sim
                min_pair = (i, j)

    if max_pair:
        print(
            f"\nMost similar states: State {max_pair[0] + 1} and State {max_pair[1] + 1} ({max_similarity:.1f}% similarity)")

    if min_pair:
        print(
            f"Least similar states: State {min_pair[0] + 1} and State {min_pair[1] + 1} ({min_similarity:.1f}% similarity)")

    return similarity_matrix, edge_count_matrix


def main():
    """
    Main function to run the connectivity analysis
    """
    print("Starting edge connectivity analysis...")

    # Create test arguments
    args = create_test_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Find trajectory files
    trajectory_files = find_trajectory_files(
        args.traj_dir,
        file_pattern=args.file_pattern,
        recursive=args.recursive
    )

    try:
        # Create dataset
        print("Creating VAMPNet dataset...")
        dataset = VAMPNetDataset(
            trajectory_files=trajectory_files,
            topology_file=args.top,
            lag_time=args.lag_time,
            n_neighbors=args.n_neighbors,
            node_embedding_dim=args.node_embedding_dim,
            gaussian_expansion_dim=args.gaussian_expansion_dim,
            selection=args.selection,
            stride=args.stride,
            cache_dir=args.cache_dir,
            use_cache=args.use_cache
        )

        print(f"Dataset created successfully!")
        print(f"Total frames in dataset: {dataset.n_frames}")
        print(f"Atoms per frame: {dataset.n_atoms}")

        # Analyze connectivity states
        results = analyze_edge_connectivity_states(
            dataset,
            max_frames=None,  # Analyze all frames, set to smaller number for testing
            save_results=True
        )

        # Example: Compare the two most populated states
        #sorted_states = sorted(results['state_populations'].items(), key=lambda x: x[1], reverse=True)
        #if len(sorted_states) >= 2:
        #    print(f"\nComparing the two most populated states:")
        #    compare_connectivity_states(results, sorted_states[0][0], sorted_states[1][0])

        # NEW: Compare the top 5 most populated states
        print(f"\nAnalyzing top 5 states:")
        compare_connectivity_states_top5(results)

        # NEW: Analyze relationships between top 5 states
        similarity_matrix, edge_count_matrix = analyze_top5_state_relationships(results)

        print(f"\n🎉 Analysis completed successfully!")
        print(f"Found {results['n_unique_states']} unique edge connectivity states")
        print(f"Results saved to {args.output_dir}/")

        return results

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
