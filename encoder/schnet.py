import torch
import torch.nn as nn
from torch_geometric.nn import MLP, global_mean_pool
from src.components.blocks.gcn_interaction import GCNInteraction


class SchNetEncoder(nn.Module):
    """
    SchNet encoder that works with PyTorch Geometric graph data format.
    Uses GCNInteraction blocks with CFConv layers for message passing.

    Parameters
    ----------
    node_dim : int
        Dimension of node features
    edge_dim : int
        Dimension of edge features (Gaussian RBF expansion)
    hidden_dim : int
        Dimension of hidden layers
    output_dim : int
        Dimension of output embeddings
    n_interactions : int
        Number of interaction blocks
    activation : str
        Activation function to use ('relu', 'tanh', etc.)
    use_attention : bool
        Whether to use attention in CFConv layers
    """

    def __init__(
            self,
            node_dim,
            edge_dim,
            hidden_dim=64,
            output_dim=32,
            n_interactions=3,
            activation='tanh',
            use_attention=True
    ):
        super(SchNetEncoder, self).__init__()

        # Initial embedding MLP
        self.embedding = MLP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=1,
            act=activation,
            bias=True
        )

        # Convert string activation to module for GCNInteraction
        act_module = self._get_activation_module(activation)

        # Interaction blocks
        self.interactions = nn.ModuleList()
        for _ in range(n_interactions):
            interaction = GCNInteraction(
                n_inputs=hidden_dim,
                n_gaussians=edge_dim,
                n_filters=hidden_dim,
                activation=act_module,
                use_attention=use_attention
            )
            self.interactions.append(interaction)

        # Output MLP for graph-level embeddings
        self.output_layers = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=2,
            act=activation,
            bias=True
        )

    def _get_activation_module(self, activation_name):
        """Convert activation name to module for GCNInteraction"""
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name == 'elu':
            return nn.ELU()
        elif activation_name == 'gelu':
            return nn.GELU()
        else:
            # Default to Tanh as in original SchNet
            return nn.Tanh()

    def forward(self, data):
        """
        Forward pass for SchNet encoder.

        Parameters
        ----------
        data : torch_geometric.data.Data
            PyG data object with x (node features), edge_index, and edge_attr

        Returns
        -------
        torch.Tensor
            Graph-level embeddings
        """
        # Extract node and edge features from PyG data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None

        # If no batch information, assume single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Convert to proper format for GCNInteraction
        batch_size = batch.max().item() + 1
        node_counts = torch.bincount(batch)
        max_nodes = node_counts.max().item()

        # Prepare tensors for batched processing
        node_features = torch.zeros(
            batch_size, max_nodes, x.size(1),
            device=x.device, dtype=x.dtype
        )

        # Fill in node features tensor
        for i in range(x.size(0)):
            b = batch[i].item()
            node_idx = torch.sum(batch[:i] == b).item()  # Local index within its graph
            node_features[b, node_idx] = x[i]

        # Initial embedding
        h = self.embedding(node_features)

        # Process edge features and create neighbor lists for GCNInteraction
        rbf_expansion, neighbor_lists = self._prepare_graph_inputs(
            edge_index, edge_attr, batch, node_counts, max_nodes
        )

        # Apply interaction blocks
        for interaction in self.interactions:
            output_features, _ = interaction(h, rbf_expansion, neighbor_lists)
            h = h + output_features  # Residual connection

        # Global pooling for each graph in batch
        # First, reshape to match PyG's expected format
        h_flat = self._reshape_for_pyg(h, batch, node_counts)

        # Use PyG's global pooling
        pooled = global_mean_pool(h_flat, batch)

        # Final output transformation
        output = self.output_layers(pooled)

        return output

    def _prepare_graph_inputs(self, edge_index, edge_attr, batch, node_counts, max_nodes):
        """
        Convert PyG graph data to format expected by GCNInteraction.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices [2, num_edges]
        edge_attr : torch.Tensor
            Edge features [num_edges, edge_dim]
        batch : torch.Tensor
            Batch assignment for nodes
        node_counts : torch.Tensor
            Number of nodes per graph in batch
        max_nodes : int
            Maximum number of nodes in any graph

        Returns
        -------
        tuple
            rbf_expansion and neighbor_lists tensors
        """
        device = edge_index.device
        batch_size = batch.max().item() + 1
        edge_dim = edge_attr.size(1)

        # Find maximum number of neighbors for any node
        max_neighbors = 0
        for node_idx in range(edge_index.size(1)):
            src = edge_index[0, node_idx]
            src_neighbors = (edge_index[0] == src).sum().item()
            max_neighbors = max(max_neighbors, src_neighbors)

        # Initialize tensors
        rbf_expansion = torch.zeros(
            batch_size, max_nodes, max_neighbors, edge_dim,
            device=device, dtype=edge_attr.dtype
        )
        neighbor_lists = torch.zeros(
            batch_size, max_nodes, max_neighbors,
            device=device, dtype=torch.long
        )

        # Track neighbor count for each node
        neighbor_counts = torch.zeros(
            batch_size, max_nodes,
            device=device, dtype=torch.long
        )

        # Create offset maps to convert global indices to local indices
        offset_map = {}
        curr_offset = 0
        for b in range(batch_size):
            n_nodes = node_counts[b].item()
            offset_map[b] = curr_offset
            curr_offset += n_nodes

        # Fill tensors
        for e in range(edge_index.size(1)):
            src, dst = edge_index[0, e].item(), edge_index[1, e].item()
            src_batch = batch[src].item()
            dst_batch = batch[dst].item()

            # Only process edges within the same graph
            if src_batch == dst_batch:
                # Get local indices
                src_local = src - offset_map[src_batch]
                dst_local = dst - offset_map[dst_batch]

                # Add neighbor info
                n_idx = neighbor_counts[src_batch, src_local].item()
                if n_idx < max_neighbors:
                    neighbor_lists[src_batch, src_local, n_idx] = dst_local
                    rbf_expansion[src_batch, src_local, n_idx] = edge_attr[e]
                    neighbor_counts[src_batch, src_local] += 1

        return rbf_expansion, neighbor_lists

    def _reshape_for_pyg(self, h, batch, node_counts):
        """
        Reshape node features from [batch_size, max_nodes, hidden_dim]
        to PyG format [total_nodes, hidden_dim]

        Parameters
        ----------
        h : torch.Tensor
            Node features in batched format [batch_size, max_nodes, hidden_dim]
        batch : torch.Tensor
            Batch assignment for nodes
        node_counts : torch.Tensor
            Number of nodes per graph

        Returns
        -------
        torch.Tensor
            Reshaped node features [total_nodes, hidden_dim]
        """
        batch_size, max_nodes, hidden_dim = h.shape
        total_nodes = int(node_counts.sum().item())

        # Initialize output tensor
        h_flat = torch.zeros(total_nodes, hidden_dim, device=h.device, dtype=h.dtype)

        # Fill output tensor
        node_idx = 0
        for b in range(batch_size):
            n_nodes = node_counts[b].item()
            h_flat[node_idx:node_idx + n_nodes] = h[b, :n_nodes]
            node_idx += n_nodes

        return h_flat

    def initialize_weights(self, method='xavier', gain=1.0):
        """
        Apply custom weight initialization to all linear layers in the model.

        Parameters
        ----------
        method : str
            Initialization method ('xavier', 'kaiming', 'identity')
        gain : float
            Gain factor for xavier initialization
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == 'identity':
                    if module.weight.size(0) == module.weight.size(1):  # Square matrix
                        nn.init.eye_(module.weight)
                    else:
                        nn.init.xavier_uniform_(module.weight, gain=gain)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
