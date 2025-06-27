import torch
import numpy as np
import mdtraj as md
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from tqdm import tqdm

from torch_geometric.data import Data
from torch_sparse import SparseTensor



class PyGProteinNSPDK:
    """
    PyTorch Geometric-based Neighborhood Subgraph Pairwise Distance Kernel for protein structures
    """

    def __init__(self, max_radius=3, max_distance=10, contact_cutoff=8.0, device='cuda'):
        self.max_radius = max_radius
        self.max_distance = max_distance
        self.contact_cutoff = contact_cutoff
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def trajectory_to_pyg_graphs(self, trajectory, selection='name CA'):
        """
        Convert MD trajectory to PyTorch Geometric Data objects

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            MD trajectory
        selection : str
            Atom selection string

        Returns
        -------
        graphs : list
            List of PyTorch Geometric Data objects
        """
        print(f"Converting {trajectory.n_frames} frames to PyG graphs...")

        # Select atoms
        atom_indices = trajectory.topology.select(selection)
        n_atoms = len(atom_indices)

        # Pre-compute all pairwise distances for efficiency
        print("Computing pairwise distances...")
        atom_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
        distances = md.compute_distances(trajectory, atom_pairs)

        graphs = []
        for frame_idx in tqdm(range(trajectory.n_frames)):
            # Create node features (simple encoding for now)
            node_features = torch.zeros(n_atoms, 1, device=self.device)

            # Create node labels
            node_labels = torch.zeros(n_atoms, dtype=torch.long, device=self.device)
            for i, atom_idx in enumerate(atom_indices):
                atom = trajectory.topology.atom(atom_idx)
                # Simple label encoding
                label_hash = hash(f"{atom.residue.name}_{atom.name}") % 1000
                node_labels[i] = label_hash
                node_features[i, 0] = label_hash  # Use as feature too

            # Create edge index based on distance cutoff
            edge_indices = []
            edge_attrs = []

            frame_distances = distances[frame_idx]
            for pair_idx, (i, j) in enumerate(atom_pairs):
                if frame_distances[pair_idx] < self.contact_cutoff / 10.0:  # Convert to nm
                    # Add both directions for undirected graph
                    edge_indices.extend([[i, j], [j, i]])
                    edge_attrs.extend([frame_distances[pair_idx], frame_distances[pair_idx]])

            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float, device=self.device).unsqueeze(1)
            else:
                # Handle isolated nodes
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_attr = torch.empty((0, 1), dtype=torch.float, device=self.device)

            # Create PyG Data object
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_labels=node_labels,
                num_nodes=n_atoms
            )

            graphs.append(graph)

        return graphs

    def compute_shortest_paths_pyg(self, data):
        """
        Compute all-pairs shortest paths using PyG sparse operations
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Create adjacency matrix
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=torch.ones(edge_index.size(1), device=self.device),
            sparse_sizes=(num_nodes, num_nodes)
        )

        # Initialize distance matrix
        dist_matrix = torch.full((num_nodes, num_nodes), float('inf'), device=self.device)

        # Set distances for direct connections
        adj_dense = adj.to_dense()
        dist_matrix[adj_dense > 0] = 1.0

        # Set diagonal to 0
        torch.diagonal(dist_matrix).fill_(0.0)

        # Floyd-Warshall algorithm (optimized for small graphs)
        for k in range(num_nodes):
            dist_ik = dist_matrix[:, k].unsqueeze(1)  # [num_nodes, 1]
            dist_kj = dist_matrix[k, :].unsqueeze(0)  # [1, num_nodes]
            dist_matrix = torch.min(dist_matrix, dist_ik + dist_kj)

        return dist_matrix

    def extract_neighborhood_subgraph_pyg(self, data, center, radius):
        """
        Extract neighborhood subgraph using PyG operations
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        if radius == 0:
            # Only the center node
            subgraph_nodes = torch.tensor([center], device=self.device)
        else:
            # Use BFS to find nodes within radius
            visited = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            current_nodes = torch.tensor([center], device=self.device)
            visited[center] = True

            for r in range(radius):
                if len(current_nodes) == 0:
                    break

                # Find neighbors of current nodes
                mask = torch.isin(edge_index[0], current_nodes)
                neighbors = edge_index[1][mask]

                # Filter unvisited neighbors
                new_neighbors = neighbors[~visited[neighbors]]
                visited[new_neighbors] = True
                current_nodes = new_neighbors

            subgraph_nodes = torch.where(visited)[0]

        # Extract subgraph
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        node_mask[subgraph_nodes] = True

        # Filter edges
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        subgraph_edge_index = edge_index[:, edge_mask]

        # Remap node indices
        node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=self.device)
        node_mapping[subgraph_nodes] = torch.arange(len(subgraph_nodes), device=self.device)

        remapped_edge_index = node_mapping[subgraph_edge_index]

        # Get center index in subgraph
        center_idx = node_mapping[center]

        return {
            'nodes': subgraph_nodes,
            'edge_index': remapped_edge_index,
            'node_labels': data.node_labels[subgraph_nodes],
            'center_idx': center_idx,
            'num_nodes': len(subgraph_nodes)
        }

    def subgraph_to_hash_pyg(self, subgraph, root_idx):
        """
        Convert subgraph to hash using PyG operations
        """
        edge_index = subgraph['edge_index']
        node_labels = subgraph['node_labels']
        num_nodes = subgraph['num_nodes']

        if num_nodes == 0:
            return hash("EMPTY")

        # Compute distances from root using BFS
        dist_from_root = torch.full((num_nodes,), float('inf'), device=self.device)
        dist_from_root[root_idx] = 0.0

        # BFS implementation
        queue = [root_idx.item()]
        while queue:
            current = queue.pop(0)
            current_dist = dist_from_root[current]

            # Find neighbors
            neighbors = edge_index[1][edge_index[0] == current]
            for neighbor in neighbors:
                neighbor = neighbor.item()
                if dist_from_root[neighbor] > current_dist + 1:
                    dist_from_root[neighbor] = current_dist + 1
                    queue.append(neighbor)

        # Create canonical representation
        node_features = []
        for i in range(num_nodes):
            dist = dist_from_root[i].item()
            if dist == float('inf'):
                dist = -1  # Disconnected nodes
            label = node_labels[i].item()
            node_features.append((dist, label))

        # Sort for canonical form
        node_features.sort()

        # Create edge list
        edge_features = []
        edge_list = edge_index.t().cpu().numpy()
        for edge in edge_list:
            i, j = edge
            if i < j:  # Avoid duplicates in undirected graph
                feat_i = node_features[i]
                feat_j = node_features[j]
                edge_features.append((min(feat_i, feat_j), max(feat_i, feat_j)))

        edge_features.sort()

        # Create hash
        canonical_str = f"NODES:{node_features}|EDGES:{edge_features}"
        return hash(canonical_str) % (2 ** 32)

    def compute_nspdk_features_pyg(self, data):
        """
        Compute NSPDK features for a single PyG graph
        """
        num_nodes = data.num_nodes
        features = defaultdict(int)

        # Compute shortest path distances
        dist_matrix = self.compute_shortest_paths_pyg(data)

        # For each pair of nodes
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                distance = dist_matrix[i, j].item()

                if distance > self.max_distance or distance == float('inf'):
                    continue

                # For each radius
                for radius in range(self.max_radius + 1):
                    # Extract neighborhood subgraphs
                    subgraph_i = self.extract_neighborhood_subgraph_pyg(data, i, radius)
                    subgraph_j = self.extract_neighborhood_subgraph_pyg(data, j, radius)

                    # Convert to hashes
                    hash_i = self.subgraph_to_hash_pyg(subgraph_i, subgraph_i['center_idx'])
                    hash_j = self.subgraph_to_hash_pyg(subgraph_j, subgraph_j['center_idx'])

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

        # Convert to feature vectors (use sparse representation)
        feature_vectors = torch.zeros(len(graphs), len(all_keys), device=self.device)

        for i, features in enumerate(all_features):
            for j, key in enumerate(all_keys):
                feature_vectors[i, j] = features.get(key, 0)

        # Compute kernel matrix using PyTorch
        print("Computing kernel matrix...")
        kernel_matrix = torch.mm(feature_vectors, feature_vectors.t())

        # Normalize
        norms = torch.sqrt(torch.diagonal(kernel_matrix))
        # Avoid division by zero
        norms = torch.where(norms > 1e-10, norms, torch.ones_like(norms))
        kernel_matrix = kernel_matrix / torch.outer(norms, norms)

        return kernel_matrix.cpu().numpy(), feature_vectors.cpu().numpy(), all_keys


# Batched version for very large trajectories
class BatchedPyGNSPDK(PyGProteinNSPDK):
    """
    Batched version using PyG's batching capabilities
    """

    def __init__(self, max_radius=3, max_distance=10, contact_cutoff=8.0,
                 device='cuda', batch_size=32):
        super().__init__(max_radius, max_distance, contact_cutoff, device)
        self.batch_size = batch_size

    def compute_kernel_matrix_batched(self, graphs):
        """
        Compute kernel matrix in batches using PyG's Batch
        """
        n_graphs = len(graphs)

        # Compute features in batches
        print("Computing NSPDK features in batches...")
        all_features = []

        for i in tqdm(range(0, n_graphs, self.batch_size)):
            batch_end = min(i + self.batch_size, n_graphs)
            batch_graphs = graphs[i:batch_end]

            # Process each graph in the batch
            batch_features = []
            for graph in batch_graphs:
                features = self.compute_nspdk_features_pyg(graph)
                batch_features.append(features)

            all_features.extend(batch_features)

            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Get all unique feature keys
        all_keys = set()
        for features in all_features:
            all_keys.update(features.keys())
        all_keys = sorted(list(all_keys))

        print(f"Found {len(all_keys)} unique NSPDK features")

        # Compute kernel matrix in blocks to save memory
        kernel_matrix = np.zeros((n_graphs, n_graphs))

        for i in tqdm(range(0, n_graphs, self.batch_size), desc="Computing kernel matrix"):
            i_end = min(i + self.batch_size, n_graphs)

            # Convert batch to feature vectors
            batch_features_i = torch.zeros(i_end - i, len(all_keys), device=self.device)
            for idx, features in enumerate(all_features[i:i_end]):
                for j, key in enumerate(all_keys):
                    batch_features_i[idx, j] = features.get(key, 0)

            for j in range(0, n_graphs, self.batch_size):
                j_end = min(j + self.batch_size, n_graphs)

                # Convert batch to feature vectors
                batch_features_j = torch.zeros(j_end - j, len(all_keys), device=self.device)
                for idx, features in enumerate(all_features[j:j_end]):
                    for k, key in enumerate(all_keys):
                        batch_features_j[idx, k] = features.get(key, 0)

                # Compute kernel block
                kernel_block = torch.mm(batch_features_i, batch_features_j.t())
                kernel_matrix[i:i_end, j:j_end] = kernel_block.cpu().numpy()

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Normalize
        norms = np.sqrt(np.diagonal(kernel_matrix))
        norms = np.where(norms > 1e-10, norms, 1.0)
        kernel_matrix = kernel_matrix / np.outer(norms, norms)

        return kernel_matrix, None, all_keys


def cluster_md_trajectory_pyg_nspdk(trajectory_file, topology_file, n_clusters=5,
                                    selection='name CA', max_radius=2, max_distance=8,
                                    device='cuda', use_batched=False, batch_size=32):
    """
    Cluster MD trajectory frames using PyTorch Geometric-based NSPDK

    Parameters
    ----------
    trajectory_file : str
        Path to trajectory file
    topology_file : str
        Path to topology file
    n_clusters : int
        Number of clusters
    selection : str
        Atom selection for graph construction
    max_radius : int
        Maximum radius for neighborhood subgraphs
    max_distance : int
        Maximum distance between subgraph pairs
    device : str
        PyTorch device ('cuda' or 'cpu')
    use_batched : bool
        Whether to use batched processing
    batch_size : int
        Batch size for batched processing

    Returns
    -------
    cluster_labels : np.ndarray
        Cluster assignment for each frame
    kernel_matrix : np.ndarray
        NSPDK kernel matrix
    """

    # Load trajectory
    print(f"Loading trajectory from {trajectory_file}...")
    traj = md.load(trajectory_file, top=topology_file)
    print(f"Loaded {traj.n_frames} frames with {traj.n_atoms} atoms")

    # Initialize NSPDK
    if use_batched:
        nspdk = BatchedPyGNSPDK(
            max_radius=max_radius,
            max_distance=max_distance,
            device=device,
            batch_size=batch_size
        )
    else:
        nspdk = PyGProteinNSPDK(
            max_radius=max_radius,
            max_distance=max_distance,
            device=device
        )

    # Convert trajectory to PyG graphs
    graphs = nspdk.trajectory_to_pyg_graphs(traj, selection=selection)

    # Compute kernel matrix
    if use_batched:
        kernel_matrix, feature_vectors, feature_keys = nspdk.compute_kernel_matrix_batched(graphs)
    else:
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


# Example usage
def main():
    """
    Example of using PyTorch Geometric-based NSPDK for MD trajectory clustering
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
        selection='name CA',  # Use CA atoms
        max_radius=2,  # Neighborhood radius
        max_distance=8,  # Maximum distance between subgraph pairs
        device=device,
        use_batched=True,  # Use batched processing for large trajectories
        batch_size=32
    )

    # Save results
    np.save("pyg_nspdk_cluster_labels.npy", cluster_labels)
    np.save("pyg_nspdk_kernel_matrix.npy", kernel_matrix)

    print("PyTorch Geometric NSPDK clustering completed!")

    return cluster_labels, kernel_matrix


if __name__ == "__main__":
    main()
