import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree, to_dense_adj
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import random
from tqdm import tqdm
import hashlib


class Graph2Vec(nn.Module):
    """
    PyTorch Geometric implementation of graph2vec algorithm.

    Based on the paper: "graph2vec: Learning Distributed Representations of Graphs"
    by Narayanan et al. (2017) [[11]]
    """

    def __init__(self,
                 embedding_dim: int = 128,
                 max_degree: int = 2,
                 negative_samples: int = 5,
                 learning_rate: float = 0.025,
                 epochs: int = 10,
                 min_count: int = 1):
        """
        Initialize Graph2Vec model.

        Args:
            embedding_dim: Dimension of graph embeddings
            max_degree: Maximum degree of rooted subgraphs to consider
            negative_samples: Number of negative samples for training
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            min_count: Minimum frequency for subgraph to be included in vocabulary
        """
        super(Graph2Vec, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_degree = max_degree
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_count = min_count

        # Will be initialized after vocabulary creation
        self.subgraph_vocab = {}
        self.vocab_size = 0
        self.graph_embeddings = None
        self.subgraph_embeddings = None

    def _get_node_neighbors(self, edge_index: torch.Tensor, node: int, num_nodes: int) -> List[int]:
        """
        Get neighbors of a node from edge_index.

        Args:
            edge_index: Edge index tensor of shape (2, num_edges)
            node: Target node
            num_nodes: Total number of nodes

        Returns:
            List of neighbor node indices
        """
        # Find edges where the node appears
        mask = (edge_index[0] == node) | (edge_index[1] == node)
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
                         degree: int,
                         num_nodes: int) -> str:
        """
        Extract rooted subgraph using Weisfeiler-Lehman relabeling.

        Args:
            edge_index: Edge index tensor
            node_features: Node feature tensor (can be None)
            root: Root node for subgraph extraction
            degree: Degree of subgraph to extract
            num_nodes: Total number of nodes

        Returns:
            String representation of the rooted subgraph
        """
        if degree == 0:
            # Return node label (use first feature or node degree)
            if node_features is not None and node_features.size(1) > 0:
                label = str(int(node_features[root, 0].item()))
            else:
                # Use node degree as label for unlabeled graphs
                node_degree = len(self._get_node_neighbors(edge_index, root, num_nodes))
                label = str(node_degree)
            return label

        # Get neighbors
        neighbors = self._get_node_neighbors(edge_index, root, num_nodes)

        # Get subgraphs of degree d-1 for all neighbors
        neighbor_subgraphs = []
        for neighbor in neighbors:
            neighbor_sg = self._get_wl_subgraph(edge_index, node_features, neighbor, degree - 1, num_nodes)
            neighbor_subgraphs.append(neighbor_sg)

        # Sort neighbor subgraphs for canonical representation
        neighbor_subgraphs.sort()

        # Get subgraph of degree d-1 for root
        root_subgraph = self._get_wl_subgraph(edge_index, node_features, root, degree - 1, num_nodes)

        # Combine root subgraph with sorted neighbor subgraphs
        combined = root_subgraph + ''.join(neighbor_subgraphs)

        return combined

    def _extract_subgraphs(self, graph_list: List[Data]) -> Tuple[List[List[str]], Dict[str, int]]:
        """
        Extract all rooted subgraphs from a list of graphs.

        Args:
            graph_list: List of PyTorch Geometric Data objects

        Returns:
            Tuple of (list of subgraphs per graph, subgraph vocabulary)
        """
        all_subgraphs = []
        subgraph_counter = Counter()

        print("Extracting rooted subgraphs...")
        for graph_data in tqdm(graph_list):
            edge_index = graph_data.edge_index
            node_features = getattr(graph_data, 'x', None)
            num_nodes = graph_data.num_nodes

            graph_subgraphs = []

            # Extract rooted subgraphs for each node and each degree
            for node in range(num_nodes):
                for deg in range(self.max_degree + 1):
                    subgraph_str = self._get_wl_subgraph(edge_index, node_features, node, deg, num_nodes)
                    graph_subgraphs.append(subgraph_str)
                    subgraph_counter[subgraph_str] += 1

            all_subgraphs.append(graph_subgraphs)

        # Create vocabulary (filter by min_count)
        vocab = {sg: idx for idx, (sg, count) in enumerate(subgraph_counter.items())
                 if count >= self.min_count}

        print(f"Created vocabulary with {len(vocab)} subgraphs")
        return all_subgraphs, vocab

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

    def fit(self, graph_list: List[Data]) -> 'Graph2Vec':
        """
        Fit the Graph2Vec model on a list of graphs.

        Args:
            graph_list: List of PyTorch Geometric Data objects

        Returns:
            Self (fitted model)
        """
        # Extract subgraphs and create vocabulary
        all_subgraphs, self.subgraph_vocab = self._extract_subgraphs(graph_list)
        self.vocab_size = len(self.subgraph_vocab)

        if self.vocab_size == 0:
            raise ValueError("No subgraphs found in vocabulary")

        # Initialize embeddings
        num_graphs = len(graph_list)
        self.graph_embeddings = nn.Embedding(num_graphs, self.embedding_dim)
        self.subgraph_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Initialize with small random values as described in the paper [[11]]
        nn.init.uniform_(self.graph_embeddings.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        nn.init.uniform_(self.subgraph_embeddings.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)

        # Setup optimizer
        optimizer = torch.optim.SGD([
            {'params': self.graph_embeddings.parameters()},
            {'params': self.subgraph_embeddings.parameters()}
        ], lr=self.learning_rate)

        # Training loop following Algorithm 1 from the paper [[11]]
        print("Training Graph2Vec model...")
        for epoch in range(self.epochs):
            total_loss = 0.0

            # Shuffle graphs for each epoch
            graph_indices = list(range(num_graphs))
            random.shuffle(graph_indices)

            for graph_idx in tqdm(graph_indices, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                # Get subgraphs for this graph
                graph_subgraphs = all_subgraphs[graph_idx]

                # Convert to indices (filter out unknown subgraphs)
                subgraph_indices = [self.subgraph_vocab[sg] for sg in graph_subgraphs
                                    if sg in self.subgraph_vocab]

                if not subgraph_indices:
                    continue

                # Training step for each subgraph using skipgram with negative sampling
                for subgraph_idx in subgraph_indices:
                    optimizer.zero_grad()

                    # Positive sample
                    graph_emb = self.graph_embeddings(torch.tensor([graph_idx]))
                    subgraph_emb = self.subgraph_embeddings(torch.tensor([subgraph_idx]))

                    pos_score = torch.sigmoid(torch.dot(graph_emb.squeeze(), subgraph_emb.squeeze()))
                    pos_loss = -torch.log(pos_score + 1e-8)

                    # Negative samples
                    neg_indices = self._negative_sampling([subgraph_idx])
                    neg_loss = 0.0

                    for neg_idx in neg_indices:
                        neg_subgraph_emb = self.subgraph_embeddings(torch.tensor([neg_idx]))
                        neg_score = torch.sigmoid(torch.dot(graph_emb.squeeze(), neg_subgraph_emb.squeeze()))
                        neg_loss += -torch.log(1 - neg_score + 1e-8)

                    # Total loss
                    loss = pos_loss + neg_loss / len(neg_indices)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            print(f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(graph_indices):.4f}")

        return self

    def get_embeddings(self) -> torch.Tensor:
        """
        Get the learned graph embeddings.

        Returns:
            Tensor of shape (num_graphs, embedding_dim) containing graph embeddings
        """
        if self.graph_embeddings is None:
            raise ValueError("Model has not been fitted yet")

        return self.graph_embeddings.weight.detach()

    def transform(self, graph_list: List[Data]) -> torch.Tensor:
        """
        Transform new graphs to embeddings using the fitted model.

        Args:
            graph_list: List of PyTorch Geometric Data objects

        Returns:
            Tensor of embeddings for the input graphs
        """
        if self.graph_embeddings is None:
            raise ValueError("Model has not been fitted yet")

        embeddings = []

        for graph_data in graph_list:
            edge_index = graph_data.edge_index
            node_features = getattr(graph_data, 'x', None)
            num_nodes = graph_data.num_nodes

            # Extract subgraphs
            graph_subgraphs = []
            for node in range(num_nodes):
                for deg in range(self.max_degree + 1):
                    subgraph_str = self._get_wl_subgraph(edge_index, node_features, node, deg, num_nodes)
                    if subgraph_str in self.subgraph_vocab:
                        graph_subgraphs.append(self.subgraph_vocab[subgraph_str])

            if graph_subgraphs:
                # Average subgraph embeddings to get graph embedding
                subgraph_embs = self.subgraph_embeddings(torch.tensor(graph_subgraphs))
                graph_emb = torch.mean(subgraph_embs, dim=0)
            else:
                # If no known subgraphs, return zero embedding
                graph_emb = torch.zeros(self.embedding_dim)

            embeddings.append(graph_emb)

        return torch.stack(embeddings)


# Utility functions for creating sample graphs
def create_sample_graphs() -> List[Data]:
    """Create sample graphs for testing."""
    graphs = []

    # Create a few simple graphs
    for i in range(5):
        num_nodes = random.randint(5, 15)
        edge_prob = 0.3

        # Create random edges
        edges = []
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if random.random() < edge_prob:
                    edges.extend([[u, v], [v, u]])  # Undirected

        if edges:
            edge_index = torch.tensor(edges).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Random node features
        x = torch.randn(num_nodes, 3)

        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs


def load_graph_dataset(dataset_name: str) -> List[Data]:
    """
    Load a graph dataset from PyTorch Geometric.

    Args:
        dataset_name: Name of the dataset ('MUTAG', 'PROTEINS', etc.)

    Returns:
        List of Data objects
    """
    try:
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)
        return [data for data in dataset]
    except ImportError:
        print("TUDataset not available, using sample graphs")
        return create_sample_graphs()


if __name__ == "__main__":
    # Example usage
    print("Creating sample graphs...")
    graphs = create_sample_graphs()

    print("Initializing Graph2Vec model...")
    model = Graph2Vec(
        embedding_dim=64,
        max_degree=2,
        negative_samples=5,
        learning_rate=0.025,
        epochs=5
    )

    print("Fitting model...")
    model.fit(graphs)

    print("Getting embeddings...")
    embeddings = model.get_embeddings()
    print(f"Graph embeddings shape: {embeddings.shape}")

    print("Testing transform on new graphs...")
    new_graphs = create_sample_graphs()[:2]
    new_embeddings = model.transform(new_graphs)
    print(f"New graph embeddings shape: {new_embeddings.shape}")

    # Example with real dataset (if available)
    try:
        print("\nTesting with MUTAG dataset...")
        mutag_graphs = load_graph_dataset('MUTAG')[:50]  # Use subset for quick testing

        model_mutag = Graph2Vec(embedding_dim=128, max_degree=2, epochs=3)
        model_mutag.fit(mutag_graphs)

        mutag_embeddings = model_mutag.get_embeddings()
        print(f"MUTAG embeddings shape: {mutag_embeddings.shape}")

    except Exception as e:
        print(f"Could not test with MUTAG dataset: {e}")
