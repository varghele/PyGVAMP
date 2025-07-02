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

    def _hash_neighborhoods_torch_old(self, num_nodes, node_labels, neighborhoods, dist_matrix):
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

    def _hash_neighborhoods_torch_optim(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """
        Optimized hash neighborhoods using vectorized operations and efficient string handling
        """
        H = {}
        device = dist_matrix.device

        # Pre-convert tensors to numpy for faster CPU operations
        dist_matrix_np = dist_matrix.cpu().numpy()
        node_labels_np = node_labels.cpu().numpy()

        # Pre-allocate string builders for better memory efficiency
        for center in range(num_nodes):
            for radius in range(self.r + 1):
                sub_vertices = neighborhoods[radius][center]

                if len(sub_vertices) == 0:
                    H[(radius, center)] = hash("EMPTY") % (2 ** 32)
                    continue

                # Convert to numpy arrays for vectorized operations
                sub_vertices_arr = np.array(sub_vertices)

                # Vectorized distance and label extraction
                sub_dist_matrix = dist_matrix_np[np.ix_(sub_vertices_arr, sub_vertices_arr)]
                sub_node_labels = node_labels_np[sub_vertices_arr]

                # Create vertex labels using vectorized operations
                vertex_labels = []
                for i, vertex_idx in enumerate(sub_vertices_arr):
                    # Vectorized distance-label pair creation
                    valid_mask = sub_dist_matrix[i] != float('inf')
                    valid_distances = sub_dist_matrix[i][valid_mask].astype(int)
                    valid_labels = sub_node_labels[valid_mask]

                    # Create label parts using list comprehension (faster than loop)
                    label_parts = [f"{dist},{label}" for dist, label in zip(valid_distances, valid_labels)]
                    label_parts.sort()  # Sort once instead of during join

                    vertex_label = "|".join(label_parts)
                    vertex_labels.append(vertex_label)

                # Build encoding using list for efficient concatenation
                encoding_parts = []
                encoding_parts.extend(vertex_labels)
                encoding_parts.append(":")

                # Add edge information efficiently
                edge_parts = []
                for i in range(len(sub_vertices)):
                    for j in range(i + 1, len(sub_vertices)):  # Only upper triangle
                        vi, vj = sub_vertices[i], sub_vertices[j]
                        if vi < len(dist_matrix_np) and vj < len(dist_matrix_np):
                            if dist_matrix_np[vi, vj] == 1.0:  # Direct edge
                                edge_parts.append(f"{vertex_labels[i]},{vertex_labels[j]},1")

                # Join everything at once (much faster than incremental concatenation)
                if edge_parts:
                    encoding_parts.extend(edge_parts)

                encoding = ".".join(encoding_parts[:-1]) + ":" + "_".join(
                    encoding_parts[-len(edge_parts):]) if edge_parts else ".".join(encoding_parts)

                # Use hashlib for consistent hashing
                hash_value = int(hashlib.md5(encoding.encode()).hexdigest()[:8], 16)
                H[(radius, center)] = hash_value

        return H

    def _hash_neighborhoods_torch_ultra_optim(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """
        Ultra-optimized version using batch processing and minimal string operations
        """
        H = {}

        # Convert to numpy once
        dist_matrix_np = dist_matrix.cpu().numpy()
        node_labels_np = node_labels.cpu().numpy()

        # Group neighborhoods by size for batch processing
        neighborhoods_by_size = defaultdict(list)
        for center in range(num_nodes):
            for radius in range(self.r + 1):
                sub_vertices = neighborhoods[radius][center]
                size = len(sub_vertices)
                neighborhoods_by_size[size].append((radius, center, sub_vertices))

        # Process each size group in batch
        for size, neighborhood_list in neighborhoods_by_size.items():
            if size == 0:
                for radius, center, _ in neighborhood_list:
                    H[(radius, center)] = hash("EMPTY") % (2 ** 32)
                continue

            # Batch process neighborhoods of the same size
            for radius, center, sub_vertices in neighborhood_list:
                sub_vertices_arr = np.array(sub_vertices)

                # Use numpy's advanced indexing for speed
                sub_distances = dist_matrix_np[sub_vertices_arr][:, sub_vertices_arr]
                sub_labels = node_labels_np[sub_vertices_arr]

                # Create a more efficient encoding using numerical hashing
                # Instead of string operations, use numerical combinations
                vertex_hashes = []
                for i in range(size):
                    # Create a numerical hash for each vertex's neighborhood
                    valid_mask = sub_distances[i] != float('inf')
                    if np.any(valid_mask):
                        # Combine distances and labels numerically
                        dist_label_pairs = sub_distances[i][valid_mask].astype(int) * 1000 + sub_labels[valid_mask]
                        dist_label_pairs.sort()
                        vertex_hash = hash(tuple(dist_label_pairs)) % (2 ** 16)
                    else:
                        vertex_hash = 0
                    vertex_hashes.append(vertex_hash)

                # Create edge hash
                edge_hash = 0
                for i in range(size):
                    for j in range(i + 1, size):
                        if sub_distances[i, j] == 1.0:
                            edge_hash ^= hash((min(vertex_hashes[i], vertex_hashes[j]),
                                               max(vertex_hashes[i], vertex_hashes[j]))) % (2 ** 16)

                # Combine vertex and edge hashes
                vertex_hashes.sort()  # Canonical ordering
                final_hash = hash((tuple(vertex_hashes), edge_hash)) % (2 ** 32)
                H[(radius, center)] = final_hash

        return H

    def _hash_neighborhoods_torch(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """
        Ultra-optimized version using batch processing and minimal string operations
        WITH DETAILED TIMING MARKERS
        """
        import time

        # Overall timing
        start_total = time.time()

        H = {}

        # TIMING MARKER 1: Data conversion
        start_conversion = time.time()
        dist_matrix_np = dist_matrix.cpu().numpy()
        node_labels_np = node_labels.cpu().numpy()
        end_conversion = time.time()
        print(f"Data conversion time: {end_conversion - start_conversion:.6f}s")

        # TIMING MARKER 2: Neighborhood grouping
        start_grouping = time.time()
        neighborhoods_by_size = defaultdict(list)
        for center in range(num_nodes):
            for radius in range(self.r + 1):
                sub_vertices = neighborhoods[radius][center]
                size = len(sub_vertices)
                neighborhoods_by_size[size].append((radius, center, sub_vertices))
        end_grouping = time.time()
        print(f"Neighborhood grouping time: {end_grouping - start_grouping:.6f}s")

        # TIMING MARKER 3: Processing by size groups
        start_processing = time.time()

        # Track individual operations within processing
        time_empty_handling = 0
        time_array_conversion = 0
        time_indexing = 0
        time_vertex_hashing = 0
        time_edge_hashing = 0
        time_final_hashing = 0

        for size, neighborhood_list in neighborhoods_by_size.items():
            # TIMING MARKER 3a: Empty neighborhoods
            start_empty = time.time()
            if size == 0:
                for radius, center, _ in neighborhood_list:
                    H[(radius, center)] = hash("EMPTY") % (2 ** 32)
                time_empty_handling += time.time() - start_empty
                continue
            time_empty_handling += time.time() - start_empty

            # Process each neighborhood in this size group
            for radius, center, sub_vertices in neighborhood_list:
                # TIMING MARKER 3b: Array conversion
                start_array = time.time()
                sub_vertices_arr = np.array(sub_vertices)
                time_array_conversion += time.time() - start_array

                # TIMING MARKER 3c: Numpy indexing operations
                start_indexing = time.time()
                sub_distances = dist_matrix_np[sub_vertices_arr][:, sub_vertices_arr]
                sub_labels = node_labels_np[sub_vertices_arr]
                time_indexing += time.time() - start_indexing

                # TIMING MARKER 3d: Vertex hash computation
                start_vertex_hash = time.time()
                vertex_hashes = []
                for i in range(size):
                    # Create a numerical hash for each vertex's neighborhood
                    valid_mask = sub_distances[i] != float('inf')
                    if np.any(valid_mask):
                        # Combine distances and labels numerically
                        dist_label_pairs = sub_distances[i][valid_mask].astype(int) * 1000 + sub_labels[valid_mask]
                        dist_label_pairs.sort()
                        vertex_hash = hash(tuple(dist_label_pairs)) % (2 ** 16)
                    else:
                        vertex_hash = 0
                    vertex_hashes.append(vertex_hash)
                time_vertex_hashing += time.time() - start_vertex_hash

                # TIMING MARKER 3e: Edge hash computation
                #start_edge_hash = time.time()
                #edge_hash = 0
                #for i in range(size):
                #    for j in range(i + 1, size):
                #        if sub_distances[i, j] == 1.0:
                #            edge_hash ^= hash((min(vertex_hashes[i], vertex_hashes[j]),
                #                               max(vertex_hashes[i], vertex_hashes[j]))) % (2 ** 16)
                #time_edge_hashing += time.time() - start_edge_hash

                # HIGHLY OPTIMIZED EDGE HASHING
                start_edge_hash = time.time()

                # Create edge hash using vectorized operations
                edge_hash = 0
                if size > 1:
                    # Find edges (distance == 1) using vectorized operations
                    edge_mask = (sub_distances == 1.0)
                    edge_indices = np.where(np.triu(edge_mask, k=1))

                    if len(edge_indices[0]) > 0:
                        # Vectorized edge hash computation
                        vertex_hashes_arr = np.array(vertex_hashes)
                        i_vals = edge_indices[0]
                        j_vals = edge_indices[1]

                        min_hashes = np.minimum(vertex_hashes_arr[i_vals], vertex_hashes_arr[j_vals])
                        max_hashes = np.maximum(vertex_hashes_arr[i_vals], vertex_hashes_arr[j_vals])

                        # Combine hashes efficiently
                        for min_h, max_h in zip(min_hashes, max_hashes):
                            edge_hash ^= hash((int(min_h), int(max_h))) % (2 ** 16)

                time_edge_hashing += time.time() - start_edge_hash

                # TIMING MARKER 3f: Final hash combination
                start_final_hash = time.time()
                vertex_hashes.sort()  # Canonical ordering
                final_hash = hash((tuple(vertex_hashes), edge_hash)) % (2 ** 32)
                H[(radius, center)] = final_hash
                time_final_hashing += time.time() - start_final_hash

        end_processing = time.time()
        print(f"Total processing time: {end_processing - start_processing:.6f}s")

        # Print detailed breakdown of processing time
        print(f"  - Empty handling: {time_empty_handling:.6f}s")
        print(f"  - Array conversion: {time_array_conversion:.6f}s")
        print(f"  - Numpy indexing: {time_indexing:.6f}s")
        print(f"  - Vertex hashing: {time_vertex_hashing:.6f}s")
        print(f"  - Edge hashing: {time_edge_hashing:.6f}s")
        print(f"  - Final hashing: {time_final_hashing:.6f}s")

        end_total = time.time()
        print(f"TOTAL FUNCTION TIME: {end_total - start_total:.6f}s")
        print("-" * 50)

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
        batch_size=128  # Adjust based on available memory
    )

    # Analyze results
    analyze_nspdk_results(cluster_labels, kernel_matrix, trajectory_file, topology_file)

    print("PyTorch Geometric NSPDK clustering completed!")

    return cluster_labels, kernel_matrix, nspdk


if __name__ == "__main__":
    main()
