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
    from torch_geometric.datasets import TUDataset

    # Example with a real dataset
    try:
        """dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
        print(f"Dataset size: {len(dataset)}")"""

        # Initialize model with batch processing
        model = Graph2Vec(
            embedding_dim=128,
            max_degree=2,
            negative_samples=5,
            learning_rate=0.025,
            epochs=3,
            batch_size=1024,  # Adjust based on your GPU memory
            num_workers=8,  # Adjust based on your CPU cores
            vocab_cache_path='./cache/proteins_vocab.pkl'
        )


        #base_path = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ab42/trajectories/trajectories/red/"
        base_path = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ATR/"
        # First, let's find all the .xtc files
        xtc_files = find_xtc_files(base_path)
        print(xtc_files)

        # Assuming you have a topology file in the same directory or nearby
        # You might need to adjust this path
        # topology_file = os.path.join(base_path, "topol.pdb")  # Adjust as needed
        topology_file = os.path.join(base_path, "prot.pdb")  # Adjust as needed

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

        ds_for_vocab = dataset.get_frames_dataset(return_pairs=False)

        # Fit model
        print("Fitting model...")
        model.fit(ds_for_vocab, len(ds_for_vocab))

        # Get embeddings
        embeddings = model.get_embeddings()
        print(f"Graph embeddings shape: {embeddings.shape}")

        # Save model
        model.save_model('graph2vec_proteins.pt')

        # Transform new graphs (example with subset)
        test_subset = dataset[:10]
        new_embeddings = model.transform(test_subset)
        print(f"New graph embeddings shape: {new_embeddings.shape}")

    except Exception as e:
        print(f"Could not run example with real dataset: {e}")
        print("Make sure to install torch_geometric datasets or provide your own dataset")


if __name__ == "__main__":
    create_large_dataset_example()
