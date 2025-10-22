import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.nn.models import MLP


class ChebEncoder(nn.Module):
    """ Chebyshev Spectral Graph Convolution Encoder.
    This encoder uses multiple Chebyshev convolution layers to learn node
    representations and aggregates them to produce graph-level embeddings.
    Args:
        dim_in (int): Input feature dimension for nodes
        num_edge_type (int): Number of edge feature dimensions (not used in ChebConv)
        dim_hidden (int): Hidden dimension for internal representations
        dim_out (int): Output dimension for final graph embeddings
        t (int): Number of Chebyshev convolution iterations (default: 4)
        K (int): Chebyshev filter size (default: 3)
    """

    def __init__(self, dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3):
        super(ChebEncoder, self).__init__()

        self.num_edge_type = num_edge_type
        self.t = t  # Number of message passing iterations
        self.K = K  # Chebyshev filter size

        # Transform input node features to hidden dimension
        self.node_trans = nn.Linear(dim_in, dim_hidden)

        # Transform edge features to hidden dimension (kept for compatibility)
        # Note: ChebConv doesn't use edge features, but we keep this for structure consistency
        self.edge_trans = nn.Linear(num_edge_type, dim_hidden)

        # Chebyshev convolution layer
        # ChebConv uses spectral convolution with Chebyshev polynomials
        self.conv = ChebConv(dim_hidden, dim_hidden, K=K)

        # Final linear layer to map concatenated features to output dimension
        # Input size is dim_hidden * t because we concatenate all t iterations
        self.linear = nn.Linear(dim_hidden * self.t, dim_out)

    def embed_node(self, x, edge_index, edge_attr):
        """
        Compute node embeddings using multiple Chebyshev convolution layers.

        Args:
            x (torch.Tensor): Node features [num_nodes, dim_in]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, num_edge_type]
                Note: ChebConv doesn't use edge attributes, kept for compatibility

        Returns:
            tuple: (final_node_embeddings, concatenated_all_iterations)
                - final_node_embeddings: [num_nodes, dim_hidden]
                - concatenated_all_iterations: [num_nodes, dim_hidden * t]
        """
        # Transform node features to hidden dimension
        x = self.node_trans(x.float())  # [num_nodes, dim_hidden]

        # Transform and squeeze edge features to hidden dimension
        # Note: This is computed but not used by ChebConv, kept for structure consistency
        edge_attr = self.edge_trans(edge_attr.float()).squeeze(1)  # [num_edges, dim_hidden]

        # Store embeddings from each iteration for concatenation
        all_x = []

        # Apply Chebyshev convolution t times
        # Note: ChebConv only uses x and edge_index, not edge_attr
        for _ in range(self.t):
            x = self.conv(x=x, edge_index=edge_index)
            all_x.append(x)

        # Concatenate all iterations along feature dimension
        all_x = torch.cat(all_x, dim=-1)  # [num_nodes, dim_hidden * t]

        return x, all_x

    def embed_graph(self, all_x, graph_ids, node_mask=None):
        """
        Aggregate node embeddings to create graph-level embeddings.

        Args:
            all_x (torch.Tensor): Concatenated node embeddings [num_nodes, dim_hidden * t]
            graph_ids (torch.Tensor): Graph assignment for each node [num_nodes]
            node_mask (torch.Tensor, optional): Mask to exclude certain nodes

        Returns:
            torch.Tensor: Graph-level embeddings [num_graphs, dim_out]
        """
        # Initialize result tensor for all graphs
        num_graphs = graph_ids[-1] + 1
        res = torch.zeros((num_graphs, all_x.shape[-1]), device=all_x.device)

        # Apply node mask if provided (exclude masked nodes from aggregation)
        if node_mask is not None:
            graph_ids = graph_ids[~node_mask]
            all_x = all_x[~node_mask]

        # Sum node embeddings for each graph
        res.index_add_(0, graph_ids, all_x)

        # Transform to final output dimension
        res = self.linear(res)  # [num_graphs, dim_out]

        return res

    def forward(self, batch, return_x=False):
        """
        Forward pass of the Chebyshev encoder.

        Args:
            batch: PyTorch Geometric batch object containing:
                - x: Node features
                - edge_index: Edge connectivity
                - edge_attr: Edge features (not used by ChebConv)
                - batch: Graph assignment for each node
                - num_graphs: Number of graphs in batch
            return_x (bool): Whether to return final node embeddings

        Returns:
            torch.Tensor or tuple: Graph embeddings [num_graphs, dim_out]
                If return_x=True, also returns final node embeddings
        """
        # Extract batch components
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        # Compute node embeddings
        final_x, all_x = self.embed_node(x, edge_index, edge_attr)

        # Initialize result tensor for graph embeddings
        res = torch.zeros((batch.num_graphs, all_x.shape[-1]), device=all_x.device)

        # Aggregate node embeddings by graph using batch assignment
        res.index_add_(0, batch.batch, all_x)

        # Transform to final output dimension
        res = self.linear(res)  # [num_graphs, dim_out]

        # Return graph embeddings and optionally final node embeddings
        if return_x:
            return res, final_x
        return res
