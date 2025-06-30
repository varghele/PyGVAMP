import torch
import torch.nn.functional as F
import numpy as np
import mdtraj as md
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from tqdm import tqdm
import hashlib
import os

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, k_hop_subgraph
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
from torch_scatter import scatter_add, scatter_max


class PyGNSPDK:
    """
    PyTorch Geometric implementation of Neighborhood Subgraph Pairwise Distance Kernel
    Based on the grakel library implementation
    """

    def __init__(self, max_radius=3, max_distance=4, device='cuda'):
        self.max_radius = max_radius
        self.max_distance = max_distance
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def trajectory_to_pyg_graphs(self, trajectory, selection='name CA'):
        """
        Convert MD trajectory to PyTorch Geometric Data objects
        """
        print(f"Converting {trajectory.n_frames} frames to PyG graphs...")

        atom_indices = trajectory.topology.select(selection)
        n_atoms = len(atom_indices)

        # Pre-compute all pairwise distances
        atom_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
        distances = md.compute_distances(trajectory, atom_pairs)

        graphs = []
        for frame_idx in tqdm(range(trajectory.n_frames)):
            # Create node labels (residue types)
            node_labels = torch.zeros(n_atoms, dtype=torch.long, device=self.device)
            for i, atom_idx in enumerate(atom_indices):
                atom = trajectory.topology.atom(atom_idx)
                # Create hash from residue name and sequence number
                label_str = f"{atom.residue.name}_{atom.residue.resSeq}"
                label_hash = hash(label_str) % 1000  # Limit to reasonable range
                node_labels[i] = label_hash

            # Create edges based on distance cutoff (e.g., 12 Angstroms)
            edge_indices = []
            edge_attrs = []

            frame_distances = distances[frame_idx] * 10  # Convert to Angstroms
            for pair_idx, (i, j) in enumerate(atom_pairs):
                if frame_distances[pair_idx] < 12.0:  # Distance cutoff
                    # Add both directions for undirected graph
                    edge_indices.extend([[i, j], [j, i]])
                    edge_attrs.extend([frame_distances[pair_idx], frame_distances[pair_idx]])

            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float, device=self.device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_attr = torch.empty((0,), dtype=torch.float, device=self.device)

            # Create PyG Data object
            graph = Data(
                x=node_labels.unsqueeze(1).float(),  # Node features
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_labels=node_labels,
                num_nodes=n_atoms
            )

            graphs.append(graph)

        return graphs

    def compute_shortest_paths_pyg(self, data):
        """
        Compute shortest paths using PyG operations
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Initialize distance matrix
        dist_matrix = torch.full((num_nodes, num_nodes), float('inf'), device=self.device)
        torch.diagonal(dist_matrix).fill_(0.0)

        # Set direct connections to distance 1
        if edge_index.size(1) > 0:
            dist_matrix[edge_index[0], edge_index[1]] = 1.0

        # Floyd-Warshall algorithm for small graphs
        for k in range(num_nodes):
            dist_ik = dist_matrix[:, k].unsqueeze(1)
            dist_kj = dist_matrix[k, :].unsqueeze(0)
            dist_matrix = torch.min(dist_matrix, dist_ik + dist_kj)

        return dist_matrix

    def extract_neighborhood_subgraph_pyg(self, data, center, radius):
        """
        Extract neighborhood subgraph using PyG's k_hop_subgraph
        """
        if radius == 0:
            subset = torch.tensor([center], device=self.device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_mask = torch.empty((0,), dtype=torch.bool, device=self.device)
        else:
            # Use PyG's k_hop_subgraph for efficient neighborhood extraction
            subset, edge_index, _, edge_mask = k_hop_subgraph(
                center, radius, data.edge_index,
                relabel_nodes=True, num_nodes=data.num_nodes
            )

        # Get node labels for the subgraph
        subgraph_node_labels = data.node_labels[subset]

        # Find center index in the subgraph
        center_idx = torch.where(subset == center)[0]
        if len(center_idx) > 0:
            center_idx = center_idx[0]
        else:
            center_idx = torch.tensor(0, device=self.device)

        return {
            'nodes': subset,
            'edge_index': edge_index,
            'node_labels': subgraph_node_labels,
            'center_idx': center_idx,
            'num_nodes': len(subset)
        }

    def hash_subgraph_pyg(self, subgraph, root_idx, dist_matrix):
        """
        Create hash for subgraph using the grakel-inspired method
        """
        nodes = subgraph['nodes']
        edge_index = subgraph['edge_index']
        node_labels = subgraph['node_labels']
        num_nodes = subgraph['num_nodes']

        if num_nodes == 0:
            return hash("EMPTY") % (2 ** 32)

        # Get distances from root to all nodes in subgraph
        root_global_idx = nodes[root_idx]
        distances_from_root = []

        for i, node_global_idx in enumerate(nodes):
            if root_global_idx < dist_matrix.shape[0] and node_global_idx < dist_matrix.shape[1]:
                dist = dist_matrix[root_global_idx, node_global_idx].item()
                if dist == float('inf'):
                    dist = -1  # Disconnected nodes
                distances_from_root.append(int(dist))
            else:
                distances_from_root.append(-1)

        # Create node encoding: (distance_from_root, label)
        node_encodings = []
        for i in range(num_nodes):
            dist = distances_from_root[i]
            label = node_labels[i].item()
            node_encodings.append((dist, label))

        # Sort for canonical representation
        node_encodings.sort()

        # Create edge encoding
        edge_encodings = []
        if edge_index.size(1) > 0:
            edges = edge_index.t().cpu().numpy()
            for edge in edges:
                i, j = edge
                if i < j:  # Avoid duplicates in undirected graph
                    enc_i = node_encodings[i] if i < len(node_encodings) else (-1, -1)
                    enc_j = node_encodings[j] if j < len(node_encodings) else (-1, -1)
                    edge_encodings.append((min(enc_i, enc_j), max(enc_i, enc_j)))

        edge_encodings.sort()

        # Create canonical string
        canonical_str = f"NODES:{node_encodings}|EDGES:{edge_encodings}"

        # Use hashlib for consistent hashing (similar to grakel's APHash)
        return int(hashlib.md5(canonical_str.encode()).hexdigest()[:8], 16)

    def compute_nspdk_features_pyg(self, data):
        """
        Compute NSPDK features for a single PyG graph
        """
        features = defaultdict(int)
        num_nodes = data.num_nodes

        # Compute shortest path matrix
        dist_matrix = self.compute_shortest_paths_pyg(data)

        # For each pair of nodes
        for i in range(num_nodes):
            for j in range(i, num_nodes):  # Only consider i <= j to avoid duplicates
                distance = dist_matrix[i, j].item()

                if distance > self.max_distance or distance == float('inf'):
                    continue

                # For each radius
                for radius in range(self.max_radius + 1):
                    # Extract neighborhood subgraphs
                    subgraph_i = self.extract_neighborhood_subgraph_pyg(data, i, radius)
                    subgraph_j = self.extract_neighborhood_subgraph_pyg(data, j, radius)

                    # Compute hashes
                    hash_i = self.hash_subgraph_pyg(subgraph_i, subgraph_i['center_idx'], dist_matrix)
                    hash_j = self.hash_subgraph_pyg(subgraph_j, subgraph_j['center_idx'], dist_matrix)

                    # Create feature key (order-independent)
                    if hash_i <= hash_j:
                        feature_key = f"r{radius}_d{int(distance)}_{hash_i}_{hash_j}"
                    else:
                        feature_key = f"r{radius}_d{int(distance)}_{hash_j}_{hash_i}"

                    features[feature_key] += 1

        return features

    def compute_kernel_matrix_pyg(self, graphs):
        """
        Compute NSPDK kernel matrix using PyG operations
        """
        print("Computing NSPDK features for all PyG graphs...")

        # Compute features for all graphs
        all_features = []
        for graph in tqdm(graphs):
            features = self.compute_nspdk_features_pyg(graph)
            all_features.append(features)

        # Get all unique feature keys
        all_keys = set()
        for features in all_features:
            all_keys.update(features.keys())
        all_keys = sorted(list(all_keys))

        print(f"Found {len(all_keys)} unique NSPDK features")

        # Convert to feature vectors
        feature_vectors = torch.zeros(len(graphs), len(all_keys), device=self.device)

        for i, features in enumerate(all_features):
            for j, key in enumerate(all_keys):
                feature_vectors[i, j] = features.get(key, 0)

        # Compute kernel matrix
        print("Computing kernel matrix...")
        kernel_matrix = torch.mm(feature_vectors, feature_vectors.t())

        # Normalize
        norms = torch.sqrt(torch.diagonal(kernel_matrix))
        norms = torch.where(norms > 1e-10, norms, torch.ones_like(norms))
        kernel_matrix = kernel_matrix / torch.outer(norms, norms)

        return kernel_matrix.cpu().numpy(), feature_vectors.cpu().numpy(), all_keys


def cluster_md_trajectory_pyg_nspdk(trajectory_file, topology_file, n_clusters=5,
                                    selection='name CA', max_radius=3, max_distance=4,
                                    device='cuda'):
    """
    Cluster MD trajectory frames using PyTorch Geometric NSPDK
    """
    # Load trajectory
    print(f"Loading trajectory from {trajectory_file}...")
    traj = md.load(trajectory_file, top=topology_file)
    print(f"Loaded {traj.n_frames} frames")

    # Initialize PyG NSPDK
    nspdk = PyGNSPDK(
        max_radius=max_radius,
        max_distance=max_distance,
        device=device
    )

    # Convert to PyG graphs
    graphs = nspdk.trajectory_to_pyg_graphs(traj, selection=selection)

    # Compute kernel matrix
    kernel_matrix, feature_vectors, feature_keys = nspdk.compute_kernel_matrix_pyg(graphs)

    # Perform spectral clustering
    print(f"Performing spectral clustering with {n_clusters} clusters...")
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )

    cluster_labels = clustering.fit_predict(kernel_matrix)

    # Print cluster statistics
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster populations:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} frames ({count / len(cluster_labels) * 100:.1f}%)")

    return cluster_labels, kernel_matrix, feature_vectors, feature_keys


def analyze_pyg_nspdk_results(cluster_labels, kernel_matrix, trajectory_file, topology_file,
                              output_dir="pyg_nspdk_analysis"):
    """
    Analyze the results of PyG NSPDK clustering
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory for analysis
    traj = md.load(trajectory_file, top=topology_file)

    # Compute representative structures for each cluster
    unique_labels = np.unique(cluster_labels)

    for cluster_id in unique_labels:
        cluster_frames = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_frames) == 0:
            continue

        # Get centroid frame (frame with highest average similarity to others in cluster)
        cluster_similarities = kernel_matrix[np.ix_(cluster_frames, cluster_frames)]
        avg_similarities = np.mean(cluster_similarities, axis=1)
        centroid_idx = cluster_frames[np.argmax(avg_similarities)]

        # Save representative structure
        centroid_structure = traj[centroid_idx]
        centroid_structure.save_pdb(f"{output_dir}/cluster_{cluster_id}_representative.pdb")

        print(f"Cluster {cluster_id}: {len(cluster_frames)} frames, "
              f"representative frame {centroid_idx}")

    # Save clustering results
    np.save(f"{output_dir}/cluster_labels.npy", cluster_labels)
    np.save(f"{output_dir}/kernel_matrix.npy", kernel_matrix)

    print(f"Analysis results saved to {output_dir}/")


# Example usage function
def main():
    """
    Example of using PyTorch Geometric NSPDK for MD trajectory clustering
    """
    # Parameters
    trajectory_file = "trajectory.xtc"  # Your trajectory file
    topology_file = "topology.pdb"  # Your topology file

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Perform NSPDK clustering
    cluster_labels, kernel_matrix, feature_vectors, feature_keys = cluster_md_trajectory_pyg_nspdk(
        trajectory_file=trajectory_file,
        topology_file=topology_file,
        n_clusters=5,
        selection='name CA',
        max_radius=3,
        max_distance=4,
        device=device
    )

    # Analyze results
    analyze_pyg_nspdk_results(cluster_labels, kernel_matrix, trajectory_file, topology_file)

    print("PyTorch Geometric NSPDK clustering completed!")

    return cluster_labels, kernel_matrix


if __name__ == "__main__":
    main()
