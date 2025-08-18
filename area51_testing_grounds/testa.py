import time

import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def edge_indices_equal_pyg(edge_index1, edge_index2, num_nodes=None):
    """
    Compare edge indices using PyG's adjacency matrix conversion.
    """
    # Determine number of nodes if not provided
    if num_nodes is None:
        num_nodes = max(edge_index1.max().item(), edge_index2.max().item()) + 1

    # Convert to dense adjacency matrices
    adj1 = to_dense_adj(edge_index1, max_num_nodes=num_nodes)[0]
    adj2 = to_dense_adj(edge_index2, max_num_nodes=num_nodes)[0]

    # Compare adjacency matrices
    return torch.equal(adj1, adj2)


def edge_indices_equal_undirected(edge_index1, edge_index2):
    """
    Compare edge indices for undirected graphs by creating canonical representation.
    Each edge (i,j) is represented as (min(i,j), max(i,j)).
    """

    def canonicalize_edges(edge_index):
        """Convert edges to canonical form for undirected graphs."""
        # Make sure each edge is represented as (smaller_node, larger_node)
        edge_index_canonical = torch.stack([
            torch.min(edge_index[0], edge_index[1]),
            torch.max(edge_index[0], edge_index[1])
        ])

        # Sort edges lexicographically and remove duplicates
        edge_index_canonical = edge_index_canonical.t()
        edge_index_canonical = torch.unique(edge_index_canonical, dim=0)
        edge_index_canonical = edge_index_canonical.sort(dim=0)[0]

        return edge_index_canonical

    # Check if they have the same number of unique edges
    canonical1 = canonicalize_edges(edge_index1)
    canonical2 = canonicalize_edges(edge_index2)

    if canonical1.shape[0] != canonical2.shape[0]:
        return False

    return torch.equal(canonical1, canonical2)


def edge_indices_equal_sorted(edge_index1, edge_index2):
    """
    Compare edge indices by sorting them first (more memory efficient).
    """
    # Check if they have the same number of edges
    if edge_index1.shape[1] != edge_index2.shape[1]:
        return False

    # Sort edges lexicographically
    edges1_sorted = edge_index1.t().sort(dim=0)[0]
    edges2_sorted = edge_index2.t().sort(dim=0)[0]

    # Compare sorted tensors
    return torch.equal(edges1_sorted, edges2_sorted)


def edge_indices_equal(edge_index1, edge_index2):
    """
    Compare two edge index tensors to see if they represent the same graph.

    Parameters
    ----------
    edge_index1, edge_index2 : torch.Tensor
        Edge index tensors of shape [2, num_edges]

    Returns
    -------
    bool
        True if both tensors represent the same set of edges
    """
    # Check if they have the same number of edges
    if edge_index1.shape[1] != edge_index2.shape[1]:
        return False

    # Convert to sets of tuples for comparison
    edges1 = set(map(tuple, edge_index1.t().tolist()))
    edges2 = set(map(tuple, edge_index2.t().tolist()))

    return edges1 == edges2



def benchmark_comparison_methods():
    """Benchmark different edge comparison methods."""

    # Create test data
    num_edges = 1000000
    num_nodes = 10000

    # Random edge indices
    edge_index1 = torch.randint(0, num_nodes, (2, num_edges))

    # Shuffle the same edges
    perm = torch.randperm(num_edges)
    edge_index2 = edge_index1[:, perm]

    methods = {
        'set_based': edge_indices_equal,
        'sorted': edge_indices_equal_sorted,
        'pyg_adjacency': lambda x, y: edge_indices_equal_pyg(x, y, num_nodes),
    }

    for name, method in methods.items():
        start_time = time.time()
        result = method(edge_index1, edge_index2)
        end_time = time.time()

        print(f"{name}: {result} ({end_time - start_time:.4f}s)")


# Run benchmark
benchmark_comparison_methods()
