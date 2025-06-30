import torch
import torch.nn.functional as F
import numpy as np
import mdtraj as md
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from tqdm import tqdm
import hashlib
import os
from scipy.sparse import csr_matrix


import torch_geometric

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse, k_hop_subgraph
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
from torch_scatter import scatter_add, scatter_max


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, trajectory, atom_indices, atom_pairs, device):
        self.trajectory = trajectory
        self.atom_indices = atom_indices
        self.atom_pairs = atom_pairs
        self.device = device
        self.n_atoms = len(atom_indices)

        # Pre-compute node labels once
        self.node_labels = torch.zeros(self.n_atoms, dtype=torch.long)
        for i, atom_idx in enumerate(atom_indices):
            atom = trajectory.topology.atom(atom_idx)
            label_str = f"{atom.residue.name}_{atom.residue.resSeq}"
            label_hash = hash(label_str) % 1000
            self.node_labels[i] = label_hash

    def __len__(self):
        return self.trajectory.n_frames

    def __getitem__(self, idx):
        # Extract single frame
        frame = self.trajectory[idx]

        # Compute distances for this frame only
        distances = md.compute_distances(frame, self.atom_pairs)[0] * 10  # Convert to Angstroms

        # Create edges based on distance cutoff
        edge_indices = []
        edge_attrs = []

        for pair_idx, (i, j) in enumerate(self.atom_pairs):
            if distances[pair_idx] < 12.0:  # Distance cutoff
                edge_indices.extend([[i, j], [j, i]])
                edge_attrs.extend([distances[pair_idx], distances[pair_idx]])

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float)

        # Create PyG Data object
        return Data(
            x=self.node_labels.clone().unsqueeze(1).float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_labels=self.node_labels.clone(),
            num_nodes=self.n_atoms,
            frame_idx=idx
        )


class PyGNSPDK:
    """
    PyTorch Geometric implementation of Neighborhood Subgraph Pairwise Distance Kernel
    Based on the grakel library implementation
    """

    def __init__(self, r=3, d=4, device='cuda', batch_size=32):
        """
        Initialize NSPDK with parameters matching grakel implementation

        Parameters
        ----------
        r : int, default=3
            Maximum radius for neighborhood subgraphs
        d : int, default=4
            Maximum distance between subgraph pairs
        device : str
            PyTorch device
        batch_size : int
            Batch size for processing large trajectories
        """
        self.r = r  # Maximum radius
        self.d = d  # Maximum distance
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # Storage for fitted data (matching grakel's approach)
        self._fit_keys = {}
        self._ngx = 0  # Number of graphs in fit
        self._ngy = 0  # Number of graphs in transform
        self.X = {}  # Fitted feature matrices
        self._X_level_norm_factor = {}

    def trajectory_to_pyg_dataset(self, trajectory, selection='name CA'):
        """
        Convert MD trajectory to PyTorch Geometric dataset with DataLoader
        This addresses the memory issue by using batched processing
        """
        print(f"Converting {trajectory.n_frames} frames to PyG dataset...")

        atom_indices = trajectory.topology.select(selection)
        n_atoms = len(atom_indices)

        # Pre-compute all pairwise distances in batches to save memory
        atom_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]

        # Process trajectory in chunks to avoid memory issues
        chunk_size = min(1000, trajectory.n_frames)  # Process 1000 frames at a time

        dataset = TrajectoryDataset(trajectory, atom_indices, atom_pairs, self.device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=0)  # num_workers=0 to avoid multiprocessing issues

        return dataloader

    def compute_shortest_paths_batch(self, batch):
        """
        Compute shortest paths for a batch of graphs
        """
        shortest_paths = []

        for i in range(batch.num_graphs):
            # Extract single graph from batch
            mask = batch.batch == i
            num_nodes = mask.sum().item()

            if num_nodes == 0:
                continue

            # Get edges for this graph
            edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
            graph_edges = batch.edge_index[:, edge_mask]

            # Remap node indices to 0-based for this graph
            node_mapping = torch.zeros(batch.num_nodes, dtype=torch.long, device=self.device)
            node_mapping[mask] = torch.arange(num_nodes, device=self.device)
            graph_edges = node_mapping[graph_edges]

            # Compute shortest paths using Floyd-Warshall
            dist_matrix = torch.full((num_nodes, num_nodes), float('inf'), device=self.device)
            torch.diagonal(dist_matrix).fill_(0.0)

            # Set direct connections
            if graph_edges.size(1) > 0:
                dist_matrix[graph_edges[0], graph_edges[1]] = 1.0

            # Floyd-Warshall algorithm
            for k in range(num_nodes):
                dist_ik = dist_matrix[:, k].unsqueeze(1)
                dist_kj = dist_matrix[k, :].unsqueeze(0)
                dist_matrix = torch.min(dist_matrix, dist_ik + dist_kj)

            shortest_paths.append(dist_matrix)

        return shortest_paths

    def extract_neighborhoods_batch(self, batch, shortest_paths):
        """
        Extract neighborhood subgraphs for all radii and compute features
        """
        all_features = defaultdict(dict)

        for graph_idx in range(batch.num_graphs):
            if graph_idx >= len(shortest_paths):
                continue

            mask = batch.batch == graph_idx
            num_nodes = mask.sum().item()
            node_labels = batch.node_labels[mask]
            dist_matrix = shortest_paths[graph_idx]

            # Extract neighborhoods for each radius
            neighborhoods = {}
            for r in range(self.r + 1):
                neighborhoods[r] = {}
                for center in range(num_nodes):
                    if r == 0:
                        neighborhoods[r][center] = [center]
                    else:
                        # Find all nodes within radius r
                        neighbors = torch.where(dist_matrix[center] <= r)[0].tolist()
                        neighborhoods[r][center] = sorted(neighbors)

            # Compute distance pairs and hash neighborhoods
            H = self._hash_neighborhoods_torch(num_nodes, node_labels, neighborhoods, dist_matrix)

            # Extract features for each distance level
            for d_level in range(self.d + 1):
                for i in range(num_nodes):
                    for j in range(i, num_nodes):  # Only consider i <= j to avoid duplicates
                        distance = dist_matrix[i, j].item()

                        if distance == d_level and distance != float('inf'):
                            for r in range(self.r + 1):
                                hash_i = H.get((r, i), 0)
                                hash_j = H.get((r, j), 0)

                                # Create feature key (order-independent)
                                if hash_i <= hash_j:
                                    feature_key = (hash_i, hash_j)
                                else:
                                    feature_key = (hash_j, hash_i)

                                level_key = (r, d_level)
                                if level_key not in all_features:
                                    all_features[level_key] = defaultdict(int)

                                all_features[level_key][(graph_idx, feature_key)] += 1

        return all_features

    def _hash_neighborhoods_torch(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """
        Hash neighborhoods using the same algorithm as grakel
        """
        H = {}

        for center in range(num_nodes):
            for radius in range(self.r + 1):
                sub_vertices = neighborhoods[radius][center]

                # Create encoding string similar to grakel's hash_graph function
                encoding = ""

                # Make labels for vertices (similar to grakel's Lv)
                vertex_labels = {}
                for i in sub_vertices:
                    label_parts = []
                    for j in sub_vertices:
                        if i < len(dist_matrix) and j < len(dist_matrix):
                            dist = dist_matrix[i, j].item()
                            if dist != float('inf'):
                                node_label = node_labels[j].item() if j < len(node_labels) else 0
                                label_parts.append(f"{int(dist)},{node_label}")

                    vertex_label = "|".join(sorted(label_parts))
                    vertex_labels[i] = vertex_label
                    encoding += vertex_label + "."

                if encoding.endswith("."):
                    encoding = encoding[:-1] + ":"

                # Add edge information (simplified for efficiency)
                for i in sub_vertices:
                    for j in sub_vertices:
                        if i < j and i < len(dist_matrix) and j < len(dist_matrix):
                            dist = dist_matrix[i, j].item()
                            if dist == 1.0:  # Direct edge
                                encoding += f"{vertex_labels.get(i, '')},{vertex_labels.get(j, '')},1_"

                # Use hashlib for consistent hashing
                hash_value = int(hashlib.md5(encoding.encode()).hexdigest()[:8], 16)
                H[(radius, center)] = hash_value

        return H

    def fit(self, dataloader):
        """
        Fit the NSPDK kernel on trajectory data
        """
        print("Fitting NSPDK kernel...")

        all_features = defaultdict(dict)
        all_keys = defaultdict(dict)
        graph_count = 0

        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = batch.to(self.device)

            # Compute shortest paths for this batch
            shortest_paths = self.compute_shortest_paths_batch(batch)

            # Extract features for this batch
            batch_features = self.extract_neighborhoods_batch(batch, shortest_paths)

            # Merge features with global feature dictionary
            for level_key, features in batch_features.items():
                for (local_graph_idx, feature_key), count in features.items():
                    global_graph_idx = graph_count + local_graph_idx

                    # Index feature keys
                    keys = all_keys[level_key]
                    if feature_key not in keys:
                        keys[feature_key] = len(keys)

                    feature_idx = keys[feature_key]
                    all_features[level_key][(global_graph_idx, feature_idx)] = count

            graph_count += batch.num_graphs

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Convert to sparse matrices (matching grakel's format)
        self.X = {}
        for level_key, features in all_features.items():
            if len(features) > 0:
                indices, data = zip(*features.items())
                rows, cols = zip(*indices)

                self.X[level_key] = csr_matrix(
                    (data, (rows, cols)),
                    shape=(graph_count, len(all_keys[level_key])),
                    dtype=np.int64
                )

        self._fit_keys = all_keys
        self._ngx = graph_count

        print(f"Fitted NSPDK on {graph_count} graphs with {len(self.X)} feature levels")
        return self

    def compute_kernel_matrix(self):
        """
        Compute the kernel matrix from fitted features
        """
        print("Computing kernel matrix...")

        # Compute normalization factors
        self._X_level_norm_factor = {
            key: np.array(M.power(2).sum(-1)).flatten()
            for key, M in self.X.items()
        }

        # Compute kernel matrix
        S = np.zeros((self._ngx, self._ngx))

        for level_key, M in self.X.items():
            # Compute dot product kernel for this level
            K = M.dot(M.T).toarray()
            K_diag = K.diagonal()

            # Normalize
            with np.errstate(divide='ignore', invalid='ignore'):
                Q = K / np.sqrt(np.outer(K_diag, K_diag))
                Q = np.nan_to_num(Q, nan=1.0)

            S += Q

        # Average over all levels
        if len(self.X) > 0:
            S /= len(self.X)

        return S

    def fit_transform(self, dataloader):
        """
        Fit and transform in one step
        """
        self.fit(dataloader)
        return self.compute_kernel_matrix()


def cluster_md_trajectory_pyg_nspdk(trajectory_file, topology_file, n_clusters=5,
                                    selection='name CA', r=3, d=4, device='cuda',
                                    batch_size=32):
    """
    Cluster MD trajectory using PyTorch Geometric NSPDK with memory-efficient processing
    """
    # Load trajectory
    print(f"Loading trajectory from {trajectory_file}...")
    traj = md.load(trajectory_file, top=topology_file)
    print(f"Loaded {traj.n_frames} frames")

    # Initialize NSPDK
    nspdk = PyGNSPDK(r=r, d=d, device=device, batch_size=batch_size)

    # Convert to PyG dataset with DataLoader
    dataloader = nspdk.trajectory_to_pyg_dataset(traj, selection=selection)

    # Fit and compute kernel matrix
    kernel_matrix = nspdk.fit_transform(dataloader)

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

    return cluster_labels, kernel_matrix, nspdk


def analyze_nspdk_results(cluster_labels, kernel_matrix, trajectory_file, topology_file,
                          output_dir="nspdk_analysis"):
    """
    Analyze NSPDK clustering results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory for analysis
    traj = md.load(trajectory_file, top=topology_file)

    # Save results
    np.save(f"{output_dir}/cluster_labels.npy", cluster_labels)
    np.save(f"{output_dir}/kernel_matrix.npy", kernel_matrix)

    # Compute representative structures
    unique_labels = np.unique(cluster_labels)

    for cluster_id in unique_labels:
        cluster_frames = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_frames) == 0:
            continue

        # Find centroid frame
        cluster_similarities = kernel_matrix[np.ix_(cluster_frames, cluster_frames)]
        avg_similarities = np.mean(cluster_similarities, axis=1)
        centroid_idx = cluster_frames[np.argmax(avg_similarities)]

        # Save representative structure
        centroid_structure = traj[centroid_idx]
        centroid_structure.save_pdb(f"{output_dir}/cluster_{cluster_id}_representative.pdb")

        print(f"Cluster {cluster_id}: {len(cluster_frames)} frames, "
              f"representative frame {centroid_idx}")

    print(f"Analysis results saved to {output_dir}/")


# Example usage
def main():
    """
    Example usage of PyTorch Geometric NSPDK
    """
    # Parameters
    trajectory_file = "trajectory.xtc"
    topology_file = "topology.pdb"

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Perform NSPDK clustering with memory-efficient processing
    cluster_labels, kernel_matrix, nspdk = cluster_md_trajectory_pyg_nspdk(
        trajectory_file=trajectory_file,
        topology_file=topology_file,
        n_clusters=5,
        selection='name CA',
        r=3,  # Maximum radius
        d=4,  # Maximum distance
        device=device,
        batch_size=16  # Adjust based on available memory
    )

    # Analyze results
    analyze_nspdk_results(cluster_labels, kernel_matrix, trajectory_file, topology_file)

    print("PyTorch Geometric NSPDK clustering completed!")

    return cluster_labels, kernel_matrix, nspdk


if __name__ == "__main__":
    main()
