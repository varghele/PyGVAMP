import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator
import random
from tqdm import tqdm
import pickle
import os
import glob
from pygv.dataset.vampnet_dataset import VAMPNetDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch_geometric.datasets import TUDataset

def find_xtc_files(base_path):
    """
    Find all .xtc files within the given directory and its subdirectories

    Args:
        base_path: Base directory to search

    Returns:
        List of absolute paths to .xtc files
    """
    # Ensure base_path exists
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    # Use recursive glob to find all .xtc files
    xtc_files = glob.glob(os.path.join(base_path, '**', '*.xtc'), recursive=True)

    # Sort the files for consistency
    xtc_files.sort()

    print(f"Found {len(xtc_files)} .xtc files in {base_path} and subdirectories")

    # Print the first few files for verification
    if xtc_files:
        print("Sample files:")
        for file in xtc_files[:5]:  # Show the first 5 files
            print(f"  - {file}")
        if len(xtc_files) > 5:
            print(f"  ... and {len(xtc_files) - 5} more")

    return xtc_files

class Graph2Vec(nn.Module):
    """
    Batch-wise PyTorch Geometric implementation of graph2vec algorithm.

    Optimized for large datasets with many large graphs (>300 nodes, >20k graphs).
    Based on the paper: "graph2vec: Learning Distributed Representations of Graphs"
    by Narayanan et al. (2017) [[9]]
    """

    def __init__(self,
                 embedding_dim: int = 128,
                 max_degree: int = 2,
                 negative_samples: int = 5,
                 learning_rate: float = 0.025,
                 epochs: int = 10,
                 min_count: int = 5,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 vocab_cache_path: Optional[str] = None):
        """
        Initialize Graph2Vec model with batch processing capabilities.

        Args:
            embedding_dim: Dimension of graph embeddings
            max_degree: Maximum degree of rooted subgraphs to consider
            negative_samples: Number of negative samples for training
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            min_count: Minimum frequency for subgraph to be included in vocabulary
            batch_size: Batch size for processing graphs
            num_workers: Number of workers for DataLoader
            vocab_cache_path: Path to cache vocabulary (saves time on repeated runs)
        """
        super(Graph2Vec, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_degree = max_degree
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_count = min_count
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_cache_path = vocab_cache_path

        # Will be initialized after vocabulary creation
        self.subgraph_vocab = {}
        self.vocab_size = 0
        self.graph_embeddings = None
        self.subgraph_embeddings = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_node_neighbors(self, edge_index: torch.Tensor, node: int) -> List[int]:
        """
        Get neighbors of a node from edge_index efficiently.

        Args:
            edge_index: Edge index tensor of shape (2, num_edges)
            node: Target node

        Returns:
            List of neighbor node indices
        """
        # Find edges where the node appears
        mask = (edge_index[0] == node) | (edge_index[1] == node)
        if not mask.any():
            return []

        relevant_edges = edge_index[:, mask]

        # Get unique neighbors
        neighbors = torch.cat([relevant_edges[0], relevant_edges[1]])
        neighbors = neighbors[neighbors != node]  # Remove self
        neighbors = torch.unique(neighbors)

        return neighbors.tolist()

    def _get_wl_subgraph(self,
                         edge_index: torch.Tensor,
                         node_features: torch.Tensor,
                         root: int,
                         degree: int) -> str:
        """
        Extract rooted subgraph using Weisfeiler-Lehman relabeling.

        Args:
            edge_index: Edge index tensor
            node_features: Node feature tensor (can be None)
            root: Root node for subgraph extraction
            degree: Degree of subgraph to extract

        Returns:
            String representation of the rooted subgraph
        """
        if degree == 0:
            # Return node label (use first feature or node degree)
            if node_features is not None and node_features.size(1) > 0:
                label = str(int(node_features[root, 0].item()))
            else:
                # Use node degree as label for unlabeled graphs
                node_degree = len(self._get_node_neighbors(edge_index, root))
                label = str(node_degree)
            return label

        # Get neighbors
        neighbors = self._get_node_neighbors(edge_index, root)

        # Get subgraphs of degree d-1 for all neighbors
        neighbor_subgraphs = []
        for neighbor in neighbors:
            neighbor_sg = self._get_wl_subgraph(edge_index, node_features, neighbor, degree - 1)
            neighbor_subgraphs.append(neighbor_sg)

        # Sort neighbor subgraphs for canonical representation
        neighbor_subgraphs.sort()

        # Get subgraph of degree d-1 for root
        root_subgraph = self._get_wl_subgraph(edge_index, node_features, root, degree - 1)

        # Combine root subgraph with sorted neighbor subgraphs
        combined = root_subgraph + ''.join(neighbor_subgraphs)

        return combined

    def _extract_subgraphs_batch(self, batch: Batch) -> List[List[str]]:
        """
        Extract subgraphs from a batch of graphs efficiently using iterative WL.

        Args:
            batch: Batched PyG Data object

        Returns:
            List of subgraphs for each graph in the batch
        """
        batch_subgraphs = []

        # Process each graph in the batch
        graph_slices = batch.ptr

        for i in range(len(graph_slices) - 1):
            start_idx = graph_slices[i].item()
            end_idx = graph_slices[i + 1].item()

            num_nodes = end_idx - start_idx

            # Get edges for this graph (adjust indices to be relative to this graph)
            edge_mask = (batch.edge_index[0] >= start_idx) & (batch.edge_index[0] < end_idx)
            graph_edge_index = batch.edge_index[:, edge_mask] - start_idx

            # Build adjacency list once for this graph
            adj_list = [[] for _ in range(num_nodes)]
            for j in range(graph_edge_index.size(1)):
                u, v = graph_edge_index[0, j].item(), graph_edge_index[1, j].item()
                adj_list[u].append(v)
                adj_list[v].append(u)  # undirected graph

            # Get initial node labels
            if hasattr(batch, 'x') and batch.x is not None:
                initial_labels = [str(int(batch.x[start_idx + node, 0].item()))
                                  for node in range(num_nodes)]
            else:
                # Use node degree as initial label for unlabeled graphs
                initial_labels = [str(len(adj_list[node])) for node in range(num_nodes)]

            # Extract subgraphs using iterative WL
            graph_subgraphs = self._get_wl_subgraphs_iterative(
                adj_list, initial_labels, num_nodes
            )

            batch_subgraphs.append(graph_subgraphs)

        return batch_subgraphs

    def _get_wl_subgraphs_iterative(self, adj_list: List[List[int]],
                                    initial_labels: List[str],
                                    num_nodes: int) -> List[str]:
        """
        Extract all rooted subgraphs using iterative WL relabeling.

        Args:
            adj_list: Adjacency list representation of the graph
            initial_labels: Initial node labels
            num_nodes: Number of nodes in the graph

        Returns:
            List of all subgraph identifiers for this graph
        """
        all_subgraphs = []

        # Initialize current labels
        current_labels = initial_labels.copy()

        # Store labels at each iteration for subgraph extraction
        labels_at_iteration = [current_labels.copy()]

        # Iterative WL relabeling
        for iteration in range(1, self.max_degree + 1):
            new_labels = []

            for node in range(num_nodes):
                # Get neighbor labels and sort them
                neighbor_labels = [current_labels[neighbor] for neighbor in adj_list[node]]
                neighbor_labels.sort()

                # Create new label by combining current label with sorted neighbor labels
                combined_label = current_labels[node] + ''.join(neighbor_labels)

                # Use hash to create compact identifier
                new_label = str(hash(combined_label) % (2 ** 31))  # Keep it positive and bounded
                new_labels.append(new_label)

            current_labels = new_labels
            labels_at_iteration.append(current_labels.copy())

        # Extract rooted subgraphs for each node at each degree
        for node in range(num_nodes):
            for degree in range(self.max_degree + 1):
                # The subgraph identifier is simply the node's label at that iteration
                subgraph_id = labels_at_iteration[degree][node]
                all_subgraphs.append(subgraph_id)

        return all_subgraphs

    def _negative_sampling(self, positive_subgraphs: List[int]) -> List[int]:
        """
        Generate negative samples for training.

        Args:
            positive_subgraphs: List of positive subgraph indices

        Returns:
            List of negative subgraph indices
        """
        positive_set = set(positive_subgraphs)
        negative_samples = []

        while len(negative_samples) < self.negative_samples:
            neg_sample = random.randint(0, self.vocab_size - 1)
            if neg_sample not in positive_set:
                negative_samples.append(neg_sample)

        return negative_samples

    def fit(self, dataset, num_graphs: int) -> 'Graph2Vec':
        """
        Fit the Graph2Vec model on a dataset using batch processing.
        Optimized version with cached subgraph extraction and vectorized training.

        Args:
            dataset: PyTorch Geometric dataset or list of Data objects
            num_graphs: Total number of graphs in the dataset

        Returns:
            Self (fitted model)
        """
        # Create DataLoader for vocabulary building
        vocab_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Build vocabulary
        self.subgraph_vocab = self._build_vocabulary(vocab_dataloader)
        self.vocab_size = len(self.subgraph_vocab)

        if self.vocab_size == 0:
            raise ValueError("No subgraphs found in vocabulary")

        # Initialize embeddings
        self.graph_embeddings = nn.Embedding(num_graphs, self.embedding_dim).to(self.device)
        self.subgraph_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.device)

        # Initialize with small random values
        nn.init.uniform_(self.graph_embeddings.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        nn.init.uniform_(self.subgraph_embeddings.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)

        # PRE-EXTRACT AND CACHE ALL SUBGRAPHS
        print("Pre-extracting and caching subgraphs for all graphs...")
        cached_subgraphs = self._cache_all_subgraphs(dataset, num_graphs)

        # Setup optimizer
        optimizer = torch.optim.SGD([
            {'params': self.graph_embeddings.parameters()},
            {'params': self.subgraph_embeddings.parameters()}
        ], lr=self.learning_rate)

        # Training loop
        print("Training Graph2Vec model...")
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_training_steps = 0

            # Shuffle graph indices for this epoch
            graph_indices = torch.randperm(num_graphs).tolist()

            # Process graphs in batches for efficient training
            for batch_start in tqdm(range(0, num_graphs, self.batch_size),
                                    desc=f"Epoch {epoch + 1}/{self.epochs}"):
                batch_end = min(batch_start + self.batch_size, num_graphs)
                batch_graph_indices = graph_indices[batch_start:batch_end]

                # Collect all subgraphs for this batch
                batch_positive_pairs = []  # (graph_idx, subgraph_idx) pairs

                for graph_idx in batch_graph_indices:
                    if graph_idx in cached_subgraphs:
                        subgraph_indices = cached_subgraphs[graph_idx]
                        for subgraph_idx in subgraph_indices:
                            batch_positive_pairs.append((graph_idx, subgraph_idx))

                if not batch_positive_pairs:
                    continue

                # Convert to tensors for vectorized operations
                batch_graph_ids = torch.tensor([pair[0] for pair in batch_positive_pairs], device=self.device)
                batch_subgraph_ids = torch.tensor([pair[1] for pair in batch_positive_pairs], device=self.device)

                # VECTORIZED TRAINING STEP
                optimizer.zero_grad()

                # Get embeddings for all positive pairs at once
                graph_embs = self.graph_embeddings(batch_graph_ids)  # Shape: (num_pairs, embedding_dim)
                subgraph_embs = self.subgraph_embeddings(batch_subgraph_ids)  # Shape: (num_pairs, embedding_dim)

                # Compute positive scores (vectorized dot product)
                pos_scores = torch.sigmoid(torch.sum(graph_embs * subgraph_embs, dim=1))  # Shape: (num_pairs,)
                pos_loss = -torch.mean(torch.log(pos_scores + 1e-8))

                # VECTORIZED NEGATIVE SAMPLING
                neg_loss = self._compute_negative_loss_vectorized(batch_graph_ids, batch_subgraph_ids)

                # Total loss
                total_loss_batch = pos_loss + neg_loss
                total_loss_batch.backward()
                optimizer.step()

                total_loss += total_loss_batch.item()
                num_training_steps += 1

            avg_loss = total_loss / max(num_training_steps, 1)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        return self

    def _cache_all_subgraphs(self, dataset, num_graphs: int) -> Dict[int, List[int]]:
        """
        Pre-extract and cache subgraphs for all graphs to avoid repeated computation.

        Args:
            dataset: PyTorch Geometric dataset
            num_graphs: Total number of graphs

        Returns:
            Dictionary mapping graph_idx to list of subgraph indices
        """
        cached_subgraphs = {}

        # Create DataLoader for caching
        cache_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        graph_idx = 0
        for batch in tqdm(cache_dataloader, desc="Caching subgraphs"):
            batch = batch.to(self.device)
            batch_subgraphs = self._extract_subgraphs_batch(batch)

            # Store subgraphs for each graph in the batch
            for i, graph_subgraphs in enumerate(batch_subgraphs):
                if graph_idx < num_graphs:
                    # Convert subgraph strings to indices
                    subgraph_indices = [self.subgraph_vocab[sg] for sg in graph_subgraphs
                                        if sg in self.subgraph_vocab]
                    cached_subgraphs[graph_idx] = subgraph_indices
                    graph_idx += 1

        return cached_subgraphs

    def _compute_negative_loss_vectorized(self, batch_graph_ids: torch.Tensor,
                                          batch_subgraph_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute negative sampling loss in a vectorized manner.

        Args:
            batch_graph_ids: Tensor of graph indices
            batch_subgraph_ids: Tensor of positive subgraph indices

        Returns:
            Negative sampling loss
        """
        num_pairs = batch_graph_ids.size(0)

        # Generate negative samples (vectorized)
        neg_samples = torch.randint(0, self.vocab_size,
                                    (num_pairs, self.negative_samples),
                                    device=self.device)

        # Ensure negative samples don't overlap with positive samples
        # (This is a simplified version - in practice, you might want more sophisticated filtering)
        positive_set = set(batch_subgraph_ids.cpu().tolist())
        mask = torch.ones_like(neg_samples, dtype=torch.bool)
        for i, pos_idx in enumerate(batch_subgraph_ids):
            mask[i] = neg_samples[i] != pos_idx.item()

        # Get graph embeddings for negative sampling
        graph_embs_expanded = batch_graph_ids.unsqueeze(1).expand(-1, self.negative_samples)  # (num_pairs, neg_samples)
        graph_embs_neg = self.graph_embeddings(graph_embs_expanded.reshape(-1))  # (num_pairs * neg_samples, emb_dim)
        graph_embs_neg = graph_embs_neg.view(num_pairs, self.negative_samples, -1)  # (num_pairs, neg_samples, emb_dim)

        # Get negative subgraph embeddings
        neg_subgraph_embs = self.subgraph_embeddings(neg_samples.reshape(-1))  # (num_pairs * neg_samples, emb_dim)
        neg_subgraph_embs = neg_subgraph_embs.view(num_pairs, self.negative_samples,
                                                   -1)  # (num_pairs, neg_samples, emb_dim)

        # Compute negative scores (vectorized)
        neg_scores = torch.sigmoid(torch.sum(graph_embs_neg * neg_subgraph_embs, dim=2))  # (num_pairs, neg_samples)
        neg_loss = -torch.mean(torch.log(1 - neg_scores + 1e-8))

        return neg_loss

    def _build_vocabulary(self, dataloader: DataLoader) -> Dict[str, int]:
        """
        Build vocabulary with proper contiguous indexing (fixed version).
        """
        # Try to load cached vocabulary
        if self.vocab_cache_path and os.path.exists(self.vocab_cache_path):
            print(f"Loading cached vocabulary from {self.vocab_cache_path}")
            with open(self.vocab_cache_path, 'rb') as f:
                return pickle.load(f)

        print("Building vocabulary from dataset...")
        subgraph_counter = Counter()

        for batch in tqdm(dataloader, desc="Processing batches for vocabulary"):
            batch = batch.to(self.device)
            batch_subgraphs = self._extract_subgraphs_batch(batch)

            # Count subgraphs
            for graph_subgraphs in batch_subgraphs:
                for subgraph in graph_subgraphs:
                    subgraph_counter[subgraph] += 1

        # FIXED: Create vocabulary with proper contiguous indexing
        filtered_subgraphs = [(sg, count) for sg, count in subgraph_counter.items()
                              if count >= self.min_count]
        vocab = {sg: idx for idx, (sg, count) in enumerate(filtered_subgraphs)}

        # Cache vocabulary if path provided
        if self.vocab_cache_path:
            print(f"Caching vocabulary to {self.vocab_cache_path}")
            os.makedirs(os.path.dirname(self.vocab_cache_path), exist_ok=True)
            with open(self.vocab_cache_path, 'wb') as f:
                pickle.dump(vocab, f)

        print(f"Created vocabulary with {len(vocab)} subgraphs")
        return vocab

    def get_embeddings(self) -> torch.Tensor:
        """
        Get the learned graph embeddings.

        Returns:
            Tensor of shape (num_graphs, embedding_dim) containing graph embeddings
        """
        if self.graph_embeddings is None:
            raise ValueError("Model has not been fitted yet")

        return self.graph_embeddings.weight.detach().cpu()

    def transform(self, dataset) -> torch.Tensor:
        """
        Transform new graphs to embeddings using the fitted model with batch processing.

        Args:
            dataset: PyTorch Geometric dataset or list of Data objects

        Returns:
            Tensor of embeddings for the input graphs
        """
        if self.graph_embeddings is None:
            raise ValueError("Model has not been fitted yet")

        # Create DataLoader for inference
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Transforming graphs"):
                batch = batch.to(self.device)
                batch_subgraphs = self._extract_subgraphs_batch(batch)

                # Process each graph in the batch
                for graph_subgraphs in batch_subgraphs:
                    # Extract subgraphs and convert to indices
                    subgraph_indices = [self.subgraph_vocab[sg] for sg in graph_subgraphs
                                        if sg in self.subgraph_vocab]

                    if subgraph_indices:
                        # Average subgraph embeddings to get graph embedding
                        subgraph_embs = self.subgraph_embeddings(torch.tensor(subgraph_indices, device=self.device))
                        graph_emb = torch.mean(subgraph_embs, dim=0)
                    else:
                        # If no known subgraphs, return zero embedding
                        graph_emb = torch.zeros(self.embedding_dim, device=self.device)

                    all_embeddings.append(graph_emb.cpu())

        return torch.stack(all_embeddings)

    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'graph_embeddings': self.graph_embeddings.state_dict() if self.graph_embeddings else None,
            'subgraph_embeddings': self.subgraph_embeddings.state_dict() if self.subgraph_embeddings else None,
            'subgraph_vocab': self.subgraph_vocab,
            'vocab_size': self.vocab_size,
            'config': {
                'embedding_dim': self.embedding_dim,
                'max_degree': self.max_degree,
                'negative_samples': self.negative_samples,
                'min_count': self.min_count
            }
        }, path)

    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.subgraph_vocab = checkpoint['subgraph_vocab']
        self.vocab_size = checkpoint['vocab_size']

        if checkpoint['graph_embeddings'] is not None:
            num_graphs = list(checkpoint['graph_embeddings'].values())[0].shape[0]
            self.graph_embeddings = nn.Embedding(num_graphs, self.embedding_dim).to(self.device)
            self.graph_embeddings.load_state_dict(checkpoint['graph_embeddings'])

        if checkpoint['subgraph_embeddings'] is not None:
            self.subgraph_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.device)
            self.subgraph_embeddings.load_state_dict(checkpoint['subgraph_embeddings'])


# Example usage for large datasets
def create_large_dataset_example():
    """Create example for large dataset usage."""
    # Example with a real dataset
    try:
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        print(f"Dataset size: {len(dataset)}")

        # Initialize model with batch processing
        model = Graph2Vec(
            embedding_dim=128,
            max_degree=2,
            negative_samples=10,
            learning_rate=0.001,
            epochs=3,
            batch_size=1024,  # Adjust based on your GPU memory
            num_workers=8,  # Adjust based on your CPU cores
            vocab_cache_path='./cache/proteins_vocab_ab42.pkl'
        )


        """base_path = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/"
        #base_path = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ATR/"
        # First, let's find all the .xtc files
        xtc_files = find_xtc_files(base_path)
        print(xtc_files)

        # Assuming you have a topology file in the same directory or nearby
        # You might need to adjust this path
        topology_file = os.path.join(base_path, "topol.pdb")  # Adjust as needed
        #topology_file = os.path.join(base_path, "prot.pdb")  # Adjust as needed

        # Initialize the dataset
        dataset = VAMPNetDataset(
            trajectory_files=xtc_files,
            topology_file=topology_file,
            lag_time=20,  # Lag time in nanoseconds
            n_neighbors=20,  # Number of neighbors for graph construction
            node_embedding_dim=16,
            gaussian_expansion_dim=8,
            selection="name CA",  # Select only C-alpha atoms
            stride=40,  # Take every 2nd frame to reduce dataset size
            cache_dir="testdata",
            use_cache=False
        )

        ds_for_vocab = dataset.get_frames_dataset(return_pairs=False)"""

        # Fit model
        print("Fitting model...")
        model.fit(dataset, len(dataset))

        # Get embeddings
        embeddings = model.get_embeddings()
        print(f"Graph embeddings shape: {embeddings.shape}")

        ## Determine optimal number of clusters
        silhouette_scores = []
        K_range = range(2, 20)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        optimal_k = K_range[np.argmax(silhouette_scores)]
        #optimal_k = 4
        print(f"Optimal number of clusters: {optimal_k}")

        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Reduce to 2D for visualization
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot with cluster colors
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'MD Simulation States (k={optimal_k} clusters)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

        # Save model
        model.save_model('graph2vec_proteins.pt')

        # Transform new graphs (example with subset)
        test_subset = dataset[:10]
        new_embeddings = model.transform(test_subset)
        print(f"New graph embeddings shape: {new_embeddings.shape}")

    except Exception as e:
        print(f"Could not run example with real dataset: {e}")
        print("Make sure to install torch_geometric datasets or provide your own dataset")





def test_classification(dataset_name='MUTAG'):
    # Load dataset
    dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)

    # Train Graph2Vec
    model = Graph2Vec(embedding_dim=128, max_degree=2, epochs=10)
    model.fit(dataset, len(dataset))

    # Get embeddings and labels
    embeddings = model.get_embeddings().numpy()
    labels = [data.y.item() for data in dataset]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.1, random_state=42
    )

    # Train classifier
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{dataset_name} accuracy: {accuracy:.3f}")
    return accuracy


def create_test_graphs():
    """Create simple test graphs with known structural patterns."""
    graphs = []

    # Create 3 groups of structurally similar graphs

    # Group 1: Star graphs (1 central node connected to others)
    for i in range(10):
        num_outer = 5
        edges = [[0, j] for j in range(1, num_outer + 1)]
        edges += [[j, 0] for j in range(1, num_outer + 1)]  # undirected
        edge_index = torch.tensor(edges).t().contiguous()
        x = torch.ones(num_outer + 1, 1)  # simple features
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([0])))

    # Group 2: Path graphs (linear chain)
    for i in range(10):
        num_nodes = 6
        edges = [[j, j + 1] for j in range(num_nodes - 1)]
        edges += [[j + 1, j] for j in range(num_nodes - 1)]  # undirected
        edge_index = torch.tensor(edges).t().contiguous()
        x = torch.ones(num_nodes, 1)
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([1])))

    # Group 3: Cycle graphs (ring structure)
    for i in range(10):
        num_nodes = 6
        edges = [[j, (j + 1) % num_nodes] for j in range(num_nodes)]
        edges += [[(j + 1) % num_nodes, j] for j in range(num_nodes)]  # undirected
        edge_index = torch.tensor(edges).t().contiguous()
        x = torch.ones(num_nodes, 1)
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([2])))

    return graphs


def test_clustering():
    # Use synthetic data with known clusters
    test_graphs = create_test_graphs()
    true_labels = [data.y.item() for data in test_graphs]

    # Train Graph2Vec
    model = Graph2Vec(embedding_dim=64, max_degree=2, epochs=5)
    model.fit(test_graphs, len(test_graphs))

    # Get embeddings
    embeddings = model.get_embeddings().numpy()

    # Cluster
    kmeans = KMeans(n_clusters=3, random_state=42)
    pred_labels = kmeans.fit_predict(embeddings)

    # Evaluate
    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"Clustering ARI: {ari:.3f}")

    # Should get high ARI (>0.8) for synthetic data
    return ari


def debug_subgraph_extraction():
    """Debug the subgraph extraction process."""
    # Create simple test graphs
    test_graphs = create_test_graphs()

    model = Graph2Vec(embedding_dim=64, max_degree=2, epochs=1)

    # Test vocabulary building
    dataloader = DataLoader(test_graphs, batch_size=5, shuffle=False)
    vocab = model._build_vocabulary(dataloader)

    print(f"Vocabulary size: {len(vocab)}")
    print("Sample vocabulary entries:")
    for i, (sg, idx) in enumerate(list(vocab.items())[:10]):
        print(f"  {idx}: {sg}")

    # Check if different graph types produce different subgraphs
    batch = Batch.from_data_list(test_graphs[:3])  # One from each type
    batch_subgraphs = model._extract_subgraphs_batch(batch)

    print("\nSubgraphs per graph:")
    for i, subgraphs in enumerate(batch_subgraphs):
        print(f"Graph {i} ({test_graphs[i].y.item()}): {len(subgraphs)} subgraphs")
        print(f"  Sample: {subgraphs}")


def debug_training_step():
    """Debug a single training step."""
    test_graphs = create_test_graphs()[:6]  # Small subset

    model = Graph2Vec(embedding_dim=32, max_degree=1, epochs=1, batch_size=3)

    # Build vocabulary
    dataloader = DataLoader(test_graphs, batch_size=3, shuffle=False)
    model.subgraph_vocab = model._build_vocabulary(dataloader)
    model.vocab_size = len(model.subgraph_vocab)

    # Initialize embeddings
    model.graph_embeddings = nn.Embedding(len(test_graphs), 32)
    model.subgraph_embeddings = nn.Embedding(model.vocab_size, 32)

    # Cache subgraphs
    cached_subgraphs = model._cache_all_subgraphs(test_graphs, len(test_graphs))

    print(f"Cached subgraphs for {len(cached_subgraphs)} graphs")
    for graph_idx, subgraph_indices in cached_subgraphs.items():
        print(f"Graph {graph_idx}: {len(subgraph_indices)} subgraphs")
        if len(subgraph_indices) == 0:
            print(f"  WARNING: Graph {graph_idx} has no valid subgraphs!")

    # Test one training step
    batch_graph_indices = [0, 1, 2]
    batch_positive_pairs = []

    for graph_idx in batch_graph_indices:
        if graph_idx in cached_subgraphs:
            subgraph_indices = cached_subgraphs[graph_idx]
            for subgraph_idx in subgraph_indices:
                batch_positive_pairs.append((graph_idx, subgraph_idx))

    print(f"\nTraining pairs: {len(batch_positive_pairs)}")
    if len(batch_positive_pairs) == 0:
        print("ERROR: No training pairs found!")
        return

    # Check embedding access
    batch_graph_ids = torch.tensor([pair[0] for pair in batch_positive_pairs])
    batch_subgraph_ids = torch.tensor([pair[1] for pair in batch_positive_pairs])

    print(f"Graph IDs range: {batch_graph_ids.min()}-{batch_graph_ids.max()}")
    print(f"Subgraph IDs range: {batch_subgraph_ids.min()}-{batch_subgraph_ids.max()}")
    print(f"Max graph embedding index: {model.graph_embeddings.num_embeddings - 1}")
    print(f"Max subgraph embedding index: {model.subgraph_embeddings.num_embeddings - 1}")

    # Check for index out of bounds
    if batch_graph_ids.max() >= model.graph_embeddings.num_embeddings:
        print("ERROR: Graph index out of bounds!")
    if batch_subgraph_ids.max() >= model.subgraph_embeddings.num_embeddings:
        print("ERROR: Subgraph index out of bounds!")


def debug_loss_computation():
    """Debug the loss computation."""
    # Simple test case
    embedding_dim = 32
    vocab_size = 100
    num_graphs = 10

    graph_embeddings = nn.Embedding(num_graphs, embedding_dim)
    subgraph_embeddings = nn.Embedding(vocab_size, embedding_dim)

    # Test positive loss
    graph_ids = torch.tensor([0, 1, 2])
    subgraph_ids = torch.tensor([10, 20, 30])

    graph_embs = graph_embeddings(graph_ids)
    subgraph_embs = subgraph_embeddings(subgraph_ids)

    # Compute positive scores
    pos_scores = torch.sigmoid(torch.sum(graph_embs * subgraph_embs, dim=1))
    pos_loss = -torch.mean(torch.log(pos_scores + 1e-8))

    print(f"Positive scores: {pos_scores}")
    print(f"Positive loss: {pos_loss.item()}")

    # Test negative loss
    neg_samples = torch.randint(0, vocab_size, (len(graph_ids), 5))
    print(f"Negative samples shape: {neg_samples.shape}")

    # Check if loss is reasonable
    if pos_loss.item() > 10 or pos_loss.item() < 0:
        print("WARNING: Positive loss seems unreasonable")

    if torch.isnan(pos_loss) or torch.isinf(pos_loss):
        print("ERROR: Loss is NaN or Inf")





def debug_wl_differences():
    """Check if WL produces different results for different structures."""

    # Create clearly different graphs
    # Triangle
    triangle_edges = [[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]]
    triangle = Data(edge_index=torch.tensor(triangle_edges).t().contiguous(),
                    x=torch.ones(3, 1))

    # Path
    path_edges = [[0, 1], [1, 2], [1, 0], [2, 1]]
    path = Data(edge_index=torch.tensor(path_edges).t().contiguous(),
                x=torch.ones(3, 1))

    graphs = [triangle, path]

    model = Graph2Vec(max_degree=2)

    # Test subgraph extraction
    batch = Batch.from_data_list(graphs)
    batch_subgraphs = model._extract_subgraphs_batch(batch)

    print("Triangle subgraphs:", set(batch_subgraphs[0]))
    print("Path subgraphs:", set(batch_subgraphs[1]))
    print("Overlap:", set(batch_subgraphs[0]) & set(batch_subgraphs[1]))

    if set(batch_subgraphs[0]) == set(batch_subgraphs[1]):
        print("ERROR: Identical subgraphs for different structures!")
    else:
        print("SUCCESS: Different subgraphs for different structures")


def minimal_test():
    """Minimal test to verify basic functionality."""
    # Create very simple graphs
    graphs = []

    # Graph 1: Triangle (should cluster together)
    edges1 = [[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]]
    edge_index1 = torch.tensor(edges1).t().contiguous()
    graphs.append(Data(edge_index=edge_index1, x=torch.ones(3, 1)))
    graphs.append(Data(edge_index=edge_index1, x=torch.ones(3, 1)))  # Duplicate

    # Graph 2: Path (should cluster together)
    edges2 = [[0, 1], [1, 2], [1, 0], [2, 1]]
    edge_index2 = torch.tensor(edges2).t().contiguous()
    graphs.append(Data(edge_index=edge_index2, x=torch.ones(3, 1)))
    graphs.append(Data(edge_index=edge_index2, x=torch.ones(3, 1)))  # Duplicate

    print(f"Created {len(graphs)} simple graphs")

    # Train with minimal settings
    model = Graph2Vec(
        embedding_dim=16,
        max_degree=1,  # Very small degree
        epochs=20,
        batch_size=2,
        min_count=1,  # Accept all subgraphs
        negative_samples=2
    )

    try:
        model.fit(graphs, len(graphs))
        embeddings = model.get_embeddings()
        print(f"Embeddings shape: {embeddings.shape}")

        # Check if similar graphs have similar embeddings
        sim_01 = F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
        sim_23 = F.cosine_similarity(embeddings[2:3], embeddings[3:4]).item()
        sim_02 = F.cosine_similarity(embeddings[0:1], embeddings[2:3]).item()

        print(f"Similarity between identical triangles: {sim_01:.3f}")
        print(f"Similarity between identical paths: {sim_23:.3f}")
        print(f"Similarity between triangle and path: {sim_02:.3f}")

        if sim_01 > 0.8 and sim_23 > 0.8 and sim_02 < 0.5:
            print("SUCCESS: Basic functionality works!")
        else:
            print("ISSUE: Embeddings don't show expected similarity patterns")

    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":

    #create_large_dataset_example()
    # Test on benchmark datasets
    #test_classification('MUTAG')  # Should get ~80-85%
    #test_classification('PROTEINS')  # Should get ~70-75%
    #test_clustering()
    #debug_subgraph_extraction()
    #debug_wl_differences()
    #debug_training_step()
    #debug_loss_computation()
    minimal_test()