import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from tqdm import tqdm
import time
from torch_geometric.data import Data
import os


class OriginalNeighborFinder:
    """Implementation of the original get_nbrs function"""

    def __init__(self):
        pass

    def get_nbrs(self, coords, num_neighbors):
        """
        Original neighbor finding method using BallTree

        Parameters:
        -----------
        coords : np.ndarray
            Coordinates with shape [n_frames, n_atoms, 3]
        num_neighbors : int
            Number of neighbors to find

        Returns:
        --------
        tuple
            (distances, indices) both with shape [n_frames, n_atoms, num_neighbors]
        """
        k_nbr = num_neighbors + 1  # +1 to account for self
        all_inds = []
        all_dists = []

        for i in range(len(coords)):
            tree = BallTree(coords[i], leaf_size=3)
            dist, ind = tree.query(coords[i], k=k_nbr)
            # Exclude self (index 0)
            all_dists.append(dist[:, 1:])
            all_inds.append(ind[:, 1:])

        all_dists = np.array(all_dists)
        all_inds = np.array(all_inds)

        return all_dists, all_inds


class NewNeighborFinder:
    """Implementation of the PyG _create_graph_from_frame method"""

    def __init__(self):
        pass

    def find_neighbors(self, coords, num_neighbors):
        """
        New neighbor finding method using PyTorch

        Parameters:
        -----------
        coords : torch.Tensor
            Coordinates with shape [n_frames, n_atoms, 3]
        num_neighbors : int
            Number of neighbors to find

        Returns:
        --------
        tuple
            (distances, indices, edge_index, edge_attr) where:
            - distances: [n_frames, n_atoms, num_neighbors]
            - indices: [n_frames, n_atoms, num_neighbors]
            - edge_index: list of torch.Tensor with shape [2, n_edges]
            - edge_attr: list of torch.Tensor with shape [n_edges, n_gaussian]
        """
        n_frames = coords.shape[0]
        n_atoms = coords.shape[1]

        # Results containers
        all_distances = []
        all_indices = []
        all_edge_indices = []
        all_edge_attrs = []

        # Process each frame
        for frame_idx in range(n_frames):
            frame_coords = coords[frame_idx]

            # Calculate pairwise distances
            diff = frame_coords.unsqueeze(1) - frame_coords.unsqueeze(0)  # [n_atoms, n_atoms, 3]
            distances = torch.sqrt((diff ** 2).sum(dim=2))  # [n_atoms, n_atoms]

            # Create a mask to identify self-connections (diagonal elements)
            diag_mask = torch.eye(n_atoms, dtype=torch.bool, device=distances.device)

            # Set self-distances to -1 (keep original approach)
            distances[diag_mask] = -1.0

            # Create a mask for valid distances (excluding self-connections)
            valid_mask = ~diag_mask

            # For each node, get indices of the k-nearest neighbors (excluding self)
            frame_nn_indices = []
            frame_nn_distances = []

            for i in range(n_atoms):
                # Get distances from node i to all other nodes
                node_distances = distances[i]
                # Mask out the self-connection
                valid_distances = node_distances[valid_mask[i]]
                # Get indices of valid nodes (excluding self)
                valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]

                # Get top-k nearest neighbors
                values, top_k_indices = torch.topk(
                    valid_distances,
                    min(num_neighbors, len(valid_distances)),
                    largest=False
                )

                # Map back to original indices
                node_nn_indices = valid_indices[top_k_indices]
                # Add to lists
                frame_nn_indices.append(node_nn_indices)
                frame_nn_distances.append(values)

            # Stack indices and distances for all nodes
            frame_nn_indices = torch.stack(frame_nn_indices)
            frame_nn_distances = torch.stack(frame_nn_distances)

            # Create a set to track all edges for ensuring bidirectionality
            edge_set = set()

            # First, collect all the original directional edges
            for i in range(n_atoms):
                for j in frame_nn_indices[i]:
                    edge_set.add((i, j))

            # Create a list of all bidirectional edges
            bidirectional_edges = []
            for source, target in edge_set:
                bidirectional_edges.append((source, target))
                # Add the reverse edge if it doesn't already exist
                if (target, source) not in edge_set:
                    bidirectional_edges.append((target, source))

            # Convert to tensors for source and target indices
            source_indices = torch.tensor([edge[0] for edge in bidirectional_edges], device=distances.device)
            target_indices = torch.tensor([edge[1] for edge in bidirectional_edges], device=distances.device)

            # Create the edge_index tensor
            edge_index = torch.stack([source_indices, target_indices], dim=0)

            # Get edge distances
            edge_distances = distances[source_indices, target_indices]

            # Append to results
            all_distances.append(frame_nn_distances.numpy())
            all_indices.append(frame_nn_indices.numpy())
            all_edge_indices.append(edge_index)
            all_edge_attrs.append(edge_distances)

        return (
            np.array(all_distances),
            np.array(all_indices),
            all_edge_indices,
            all_edge_attrs
        )


def generate_test_data(n_frames=10, n_atoms=100, noise_level=0.1):
    """
    Generate synthetic coordinate data for testing neighbor finding

    Parameters:
    -----------
    n_frames : int
        Number of frames to generate
    n_atoms : int
        Number of atoms per frame
    noise_level : float
        Standard deviation of random noise to add

    Returns:
    --------
    tuple
        (np_coords, torch_coords) where:
        - np_coords: np.ndarray with shape [n_frames, n_atoms, 3]
        - torch_coords: torch.Tensor with shape [n_frames, n_atoms, 3]
    """
    print(f"Generating {n_frames} frames with {n_atoms} atoms each...")

    # Generate base coordinates (grid-like arrangement)
    n_per_dim = int(np.ceil(np.power(n_atoms, 1 / 3)))
    x = np.linspace(0, 10, n_per_dim)
    y = np.linspace(0, 10, n_per_dim)
    z = np.linspace(0, 10, n_per_dim)

    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z)

    # Flatten and take only what we need
    base_coords = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T[:n_atoms]

    # Create frames with noise
    np_coords = np.zeros((n_frames, n_atoms, 3))
    for i in range(n_frames):
        # Add random noise to each frame
        noise = np.random.normal(0, noise_level, (n_atoms, 3))
        np_coords[i] = base_coords + noise

    # Convert to torch
    torch_coords = torch.tensor(np_coords, dtype=torch.float32)

    return np_coords, torch_coords


def compare_neighbors(original_dists, original_inds, new_dists, new_inds, edge_indices):
    """
    Compare the results of both neighbor finding methods

    Parameters:
    -----------
    original_dists : np.ndarray
        Distances from original method
    original_inds : np.ndarray
        Indices from original method
    new_dists : np.ndarray
        Distances from new method
    new_inds : np.ndarray
        Indices from new method
    edge_indices : list
        Edge indices from new method

    Returns:
    --------
    dict
        Dictionary of comparison metrics
    """
    n_frames = original_dists.shape[0]

    # Compare distances
    dist_diffs = []
    for i in range(n_frames):
        # Compute distance differences for each frame
        frame_dist_diff = np.abs(original_dists[i] - new_dists[i])
        dist_diffs.append(frame_dist_diff)

    mean_dist_diff = np.mean([np.mean(d) for d in dist_diffs])
    max_dist_diff = np.max([np.max(d) for d in dist_diffs])

    # Compare indices
    idx_match_rates = []
    for i in range(n_frames):
        # For each frame, calculate percentage of matching indices
        matches = (original_inds[i] == new_inds[i])
        match_rate = np.mean(matches)
        idx_match_rates.append(match_rate)

    avg_idx_match_rate = np.mean(idx_match_rates)

    # Check bidirectionality of edges
    bidirectional_checks = []
    for edge_idx in edge_indices:
        # For each edge (i,j), check if (j,i) exists
        edge_pairs = set(zip(edge_idx[0].tolist(), edge_idx[1].tolist()))
        is_bidirectional = all((target, source) in edge_pairs for source, target in edge_pairs)
        bidirectional_checks.append(is_bidirectional)

    all_bidirectional = all(bidirectional_checks)

    return {
        'mean_distance_diff': mean_dist_diff,
        'max_distance_diff': max_dist_diff,
        'avg_index_match_rate': avg_idx_match_rate,
        'all_bidirectional': all_bidirectional,
        'bidirectional_check_results': bidirectional_checks
    }


def time_comparison(np_coords, torch_coords, num_neighbors, n_runs=5):
    """
    Time both implementations and compare performance

    Parameters:
    -----------
    np_coords : np.ndarray
        Coordinates for original method
    torch_coords : torch.Tensor
        Coordinates for new method
    num_neighbors : int
        Number of neighbors to find
    n_runs : int
        Number of timing runs to average

    Returns:
    --------
    dict
        Timing results for both methods
    """
    original_finder = OriginalNeighborFinder()
    new_finder = NewNeighborFinder()

    # Time original method
    original_times = []
    for _ in range(n_runs):
        start = time.time()
        original_finder.get_nbrs(np_coords, num_neighbors)
        original_times.append(time.time() - start)

    avg_original_time = sum(original_times) / n_runs

    # Time new method
    new_times = []
    for _ in range(n_runs):
        start = time.time()
        new_finder.find_neighbors(torch_coords, num_neighbors)
        new_times.append(time.time() - start)

    avg_new_time = sum(new_times) / n_runs

    # Calculate speedup
    speedup = avg_original_time / avg_new_time if avg_new_time > 0 else float('inf')

    return {
        'original_time': avg_original_time,
        'new_time': avg_new_time,
        'speedup': speedup
    }


def visualize_neighbors(np_coords, original_inds, new_inds, frame_idx=0, atom_idx=0):
    """
    Visualize neighbor differences for a specific atom

    Parameters:
    -----------
    np_coords : np.ndarray
        Coordinates data
    original_inds : np.ndarray
        Indices from original method
    new_inds : np.ndarray
        Indices from new method
    frame_idx : int
        Frame index to visualize
    atom_idx : int
        Atom index to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)

    # Get coordinates for the specified frame
    frame_coords = np_coords[frame_idx]

    # Get neighbors for the specified atom
    original_neighbors = original_inds[frame_idx, atom_idx]
    new_neighbors = new_inds[frame_idx, atom_idx]

    # Identify common and different neighbors
    common_neighbors = np.intersect1d(original_neighbors, new_neighbors)
    only_original = np.setdiff1d(original_neighbors, new_neighbors)
    only_new = np.setdiff1d(new_neighbors, original_neighbors)

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all atoms
    ax.scatter(frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2],
               c='gray', alpha=0.3, label='All atoms')

    # Plot central atom
    ax.scatter(frame_coords[atom_idx, 0], frame_coords[atom_idx, 1], frame_coords[atom_idx, 2],
               c='black', s=100, label='Central atom')

    # Plot common neighbors
    if len(common_neighbors) > 0:
        ax.scatter(frame_coords[common_neighbors, 0], frame_coords[common_neighbors, 1],
                   frame_coords[common_neighbors, 2], c='green', s=60, label='Common neighbors')

    # Plot neighbors only in original
    if len(only_original) > 0:
        ax.scatter(frame_coords[only_original, 0], frame_coords[only_original, 1],
                   frame_coords[only_original, 2], c='blue', s=60, label='Only in original')

    # Plot neighbors only in new
    if len(only_new) > 0:
        ax.scatter(frame_coords[only_new, 0], frame_coords[only_new, 1],
                   frame_coords[only_new, 2], c='red', s=60, label='Only in new')

    # Draw lines from central atom to common neighbors
    for neighbor in common_neighbors:
        ax.plot([frame_coords[atom_idx, 0], frame_coords[neighbor, 0]],
                [frame_coords[atom_idx, 1], frame_coords[neighbor, 1]],
                [frame_coords[atom_idx, 2], frame_coords[neighbor, 2]], 'g-', alpha=0.5)

    # Draw lines from central atom to neighbors only in original
    for neighbor in only_original:
        ax.plot([frame_coords[atom_idx, 0], frame_coords[neighbor, 0]],
                [frame_coords[atom_idx, 1], frame_coords[neighbor, 1]],
                [frame_coords[atom_idx, 2], frame_coords[neighbor, 2]], 'b-', alpha=0.5)

    # Draw lines from central atom to neighbors only in new
    for neighbor in only_new:
        ax.plot([frame_coords[atom_idx, 0], frame_coords[neighbor, 0]],
                [frame_coords[atom_idx, 1], frame_coords[neighbor, 1]],
                [frame_coords[atom_idx, 2], frame_coords[neighbor, 2]], 'r-', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Neighbor Comparison for Atom {atom_idx}, Frame {frame_idx}')
    ax.legend()

    plt.savefig(f"visualizations/neighbor_comparison_frame{frame_idx}_atom{atom_idx}.png", dpi=300)
    plt.close()

    print(f"Neighbor visualization saved to visualizations/neighbor_comparison_frame{frame_idx}_atom{atom_idx}.png")

    # Also generate a summary table
    print("\nNeighbor Comparison Summary:")
    print("--------------------------")
    print(f"Common neighbors: {len(common_neighbors)} of {len(original_neighbors)}")
    print(f"Only in original: {len(only_original)}")
    print(f"Only in new: {len(only_new)}")
    print(f"Match percentage: {len(common_neighbors) / len(original_neighbors) * 100:.1f}%")


def analyze_edge_index_structure(edge_indices, n_atoms):
    """
    Analyze the structure of edge_index tensors from the new method

    Parameters:
    -----------
    edge_indices : list
        List of edge_index tensors
    n_atoms : int
        Number of atoms

    Returns:
    --------
    dict
        Analysis results
    """
    results = {}

    # Analyze first frame as example
    edge_index = edge_indices[0]

    # Count edges
    n_edges = edge_index.shape[1]
    results['n_edges'] = n_edges

    # Average degree (edges per node)
    degrees = np.bincount(edge_index[0].numpy())
    avg_degree = np.mean(degrees)
    results['avg_degree'] = avg_degree

    # Check symmetry - verify that for every edge (i,j) there's an edge (j,i)
    edge_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    symmetric = all((target, source) in edge_pairs for source, target in edge_pairs)
    results['is_symmetric'] = symmetric

    # Calculate typical degree distribution
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=30, alpha=0.7)
    plt.axvline(avg_degree, color='r', linestyle='--', label=f'Avg: {avg_degree:.1f}')
    plt.xlabel('Node Degree')
    plt.ylabel('Count')
    plt.title('Node Degree Distribution')
    plt.legend()
    plt.savefig("visualizations/degree_distribution.png", dpi=300)
    plt.close()

    print(f"Edge structure analysis: {n_edges} edges, {avg_degree:.1f} avg degree, symmetric: {symmetric}")

    return results


def run_comprehensive_comparison():
    """Run a comprehensive comparison of both neighbor finding approaches"""
    print("Starting comprehensive comparison of neighbor finding methods...")

    # Parameters for test
    n_frames = 20
    n_atoms = 1000
    num_neighbors = 50  # Typical value for MD simulations

    # Generate test data
    np_coords, torch_coords = generate_test_data(n_frames, n_atoms)

    # Initialize finders
    original_finder = OriginalNeighborFinder()
    new_finder = NewNeighborFinder()

    # Run original method
    print("\nRunning original neighbor finding method...")
    original_dists, original_inds = original_finder.get_nbrs(np_coords, num_neighbors)

    # Run new method
    print("\nRunning new neighbor finding method...")
    new_dists, new_inds, edge_indices, edge_attrs = new_finder.find_neighbors(torch_coords, num_neighbors)

    # Compare results
    print("\nComparing results...")
    comparison = compare_neighbors(original_dists, original_inds, new_dists, new_inds, edge_indices)

    print(
        f"\nDistance difference - Mean: {comparison['mean_distance_diff']:.6f}, Max: {comparison['max_distance_diff']:.6f}")
    print(f"Index match rate: {comparison['avg_index_match_rate'] * 100:.2f}%")
    print(f"All edges bidirectional: {comparison['all_bidirectional']}")

    # Time comparison
    print("\nRunning timing comparison...")
    timing = time_comparison(np_coords, torch_coords, num_neighbors)

    print(f"Original method: {timing['original_time']:.6f} seconds")
    print(f"New method: {timing['new_time']:.6f} seconds")
    print(f"Speedup: {timing['speedup']:.2f}x")

    # Visualize neighbors for a specific atom
    visualize_neighbors(np_coords, original_inds, new_inds)

    # Analyze edge_index structure
    print("\nAnalyzing edge_index structure...")
    edge_analysis = analyze_edge_index_structure(edge_indices, n_atoms)

    # Create visualization directory
    os.makedirs("visualizations", exist_ok=True)

    # Create timing comparison plot
    plt.figure(figsize=(8, 6))
    plt.bar(['Original', 'New'], [timing['original_time'], timing['new_time']])
    plt.ylabel('Time (seconds)')
    plt.title('Neighbor Finding Performance')
    plt.yscale('log')  # Log scale to better see differences
    plt.text(0, timing['original_time'], f"{timing['original_time']:.6f}s",
             ha='center', va='bottom')
    plt.text(1, timing['new_time'], f"{timing['new_time']:.6f}s\n({timing['speedup']:.2f}x faster)",
             ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("visualizations/timing_comparison.png", dpi=300)

    print("\nSummary report:")
    print("--------------")
    if comparison['avg_index_match_rate'] > 0.95:
        print("✅ Neighbor indices are highly consistent between implementations")
    elif comparison['avg_index_match_rate'] > 0.8:
        print("⚠️ Moderate differences in neighbor indices (>80% match)")
    else:
        print("❌ Significant differences in neighbor indices")

    if comparison['all_bidirectional']:
        print("✅ New implementation correctly maintains bidirectional edges")
    else:
        print("❌ New implementation has issues with bidirectional edges")

    if timing['speedup'] > 1.5:
        print(f"✅ New implementation is {timing['speedup']:.1f}x faster")
    elif timing['speedup'] >= 0.9:
        print("✅ Performance is comparable between implementations")
    else:
        print("⚠️ New implementation is slower than original")

    return {
        'comparison': comparison,
        'timing': timing,
        'edge_analysis': edge_analysis
    }


if __name__ == "__main__":
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        results = run_comprehensive_comparison()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()
