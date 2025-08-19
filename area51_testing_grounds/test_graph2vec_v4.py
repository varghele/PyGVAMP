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
        Extract subgraphs from a batch of graphs efficiently.

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

            # Extract subgraph for this specific graph
            graph_nodes = list(range(start_idx, end_idx))
            num_nodes = len(graph_nodes)

            # Get edges for this graph (adjust indices to be relative to this graph)
            edge_mask = (batch.edge_index[0] >= start_idx) & (batch.edge_index[0] < end_idx)
            graph_edge_index = batch.edge_index[:, edge_mask] - start_idx

            # Get node features for this graph
            if hasattr(batch, 'x') and batch.x is not None:
                graph_node_features = batch.x[start_idx:end_idx]
            else:
                graph_node_features = None

            graph_subgraphs = []

            # Extract rooted subgraphs for each node and each degree
            for node in range(num_nodes):
                for deg in range(self.max_degree + 1):
                    subgraph_str = self._get_wl_subgraph(graph_edge_index, graph_node_features, node, deg)
                    graph_subgraphs.append(subgraph_str)

            batch_subgraphs.append(graph_subgraphs)

        return batch_subgraphs

    def _build_vocabulary(self, dataloader: DataLoader) -> Dict[str, int]:
        """
        Build vocabulary from all graphs in the dataset using batched processing.

        Args:
            dataloader: DataLoader containing all graphs

        Returns:
            Dictionary mapping subgraph strings to indices
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

        # First filter, then enumerate
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

        Args:
            dataset: PyTorch Geometric dataset or list of Data objects
            num_graphs: Total number of graphs in the dataset

        Returns:
            Self (fitted model)
        """
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle for vocabulary building
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Build vocabulary
        self.subgraph_vocab = self._build_vocabulary(dataloader)
        self.vocab_size = len(self.subgraph_vocab)

        if self.vocab_size == 0:
            raise ValueError("No subgraphs found in vocabulary")

        # Initialize embeddings
        self.graph_embeddings = nn.Embedding(num_graphs, self.embedding_dim).to(self.device)
        self.subgraph_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.device)

        # Initialize with small random values as described in the paper [[9]]
        nn.init.uniform_(self.graph_embeddings.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        nn.init.uniform_(self.subgraph_embeddings.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)

        # Setup optimizer
        optimizer = torch.optim.SGD([
            {'params': self.graph_embeddings.parameters()},
            {'params': self.subgraph_embeddings.parameters()}
        ], lr=self.learning_rate)

        # Training loop following Algorithm 1 from the paper [[9]]
        print("Training Graph2Vec model...")
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0

            # Create shuffled dataloader for training
            train_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                #num_workers=self.num_workers,
                #pin_memory=True if torch.cuda.is_available() else False
            )

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")):
                batch = batch.to(self.device)
                batch_subgraphs = self._extract_subgraphs_batch(batch)

                # Process each graph in the batch
                graph_slices = batch.ptr
                for i in range(len(graph_slices) - 1):
                    # Calculate actual graph index in the dataset
                    graph_idx = batch_idx * self.batch_size + i
                    if graph_idx >= num_graphs:
                        continue

                    graph_subgraphs = batch_subgraphs[i]

                    # Convert to indices (filter out unknown subgraphs)
                    subgraph_indices = [self.subgraph_vocab[sg] for sg in graph_subgraphs
                                        if sg in self.subgraph_vocab]

                    if not subgraph_indices:
                        continue

                    # Training step for each subgraph using skipgram with negative sampling
                    for subgraph_idx in subgraph_indices:
                        optimizer.zero_grad()

                        # Positive sample
                        graph_emb = self.graph_embeddings(torch.tensor([graph_idx], device=self.device))
                        subgraph_emb = self.subgraph_embeddings(torch.tensor([subgraph_idx], device=self.device))

                        pos_score = torch.sigmoid(torch.dot(graph_emb.squeeze(), subgraph_emb.squeeze()))
                        pos_loss = -torch.log(pos_score + 1e-8)

                        # Negative samples
                        neg_indices = self._negative_sampling([subgraph_idx])
                        neg_loss = 0.0

                        for neg_idx in neg_indices:
                            neg_subgraph_emb = self.subgraph_embeddings(torch.tensor([neg_idx], device=self.device))
                            neg_score = torch.sigmoid(torch.dot(graph_emb.squeeze(), neg_subgraph_emb.squeeze()))
                            neg_loss += -torch.log(1 - neg_score + 1e-8)

                        # Total loss
                        loss = pos_loss + neg_loss / len(neg_indices)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        return self

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
            batch_size=64,  # Adjust based on your GPU memory
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
