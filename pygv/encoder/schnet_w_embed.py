import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MLP
import torch_scatter


class CFConv(MessagePassing):
    """
    Continuous Filter Convolution for PyTorch Geometric graph data.

    Based on SchNet (Schütt et al., 2018) with attention mechanism from
    GraphVAMPNet (Ghorbani et al., 2022) and RevGraphVAMP (Huang et al. 2024).
    """

    def __init__(self,
                 n_gaussians,
                 n_filters,
                 cutoff=5.0,
                 activation='tanh',
                 normalization=None,
                 use_attention=True,
                 aggr='add'):

        # Initialize the MessagePassing module with specified aggregation
        super(CFConv, self).__init__(aggr=aggr)

        # Convert string activation to module if needed
        if isinstance(activation, str):
            activation = self._get_activation(activation)

        # Create filter generator using MLP
        self.filter_generator = MLP(
            in_channels=n_gaussians,
            hidden_channels=n_filters,
            out_channels=n_filters,
            num_layers=2,
            act=activation
        )

        # Initialize attention components
        self.use_attention = use_attention
        if use_attention:
            self.nbr_filter = nn.Parameter(torch.Tensor(n_filters, 1))
            nn.init.xavier_uniform_(self.nbr_filter, gain=1.414)

        # Store parameters
        self.cutoff = cutoff  # Not implemented yet
        self.normalization = normalization
        self.n_filters = n_filters
        self._attention_weights = None

    def _get_activation(self, activation_name):
        """Convert activation name to module"""
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
            return nn.Tanh()

    def forward(self, x, edge_index, edge_attr):
        """Forward pass of CFConv layer."""
        # Normalize edge attributes (RBF expansion)
        edge_attr_norm = edge_attr / (edge_attr.norm(dim=-1, keepdim=True) + 1e-8)

        # Generate filters from RBF expansion
        filters = self.filter_generator(edge_attr_norm)

        # Clear previous attention weights
        self._attention_weights = None

        # Propagate messages
        out = self.propagate(edge_index, x=x, filters=filters)

        # Apply normalization if provided
        if self.normalization is not None:
            out = self.normalization(out)

        return out, self._attention_weights

    def message(self, x_j, filters, edge_index_i):
        """Message function for message passing."""
        # Apply filters to source node features
        messages = x_j * filters

        # If using attention, compute attention weights
        if self.use_attention:
            # Compute attention weights
            attention = torch.matmul(messages, self.nbr_filter).squeeze(-1)

            # Store attention weights for later access
            self._attention_weights = attention

            # Get indices for each source node's edges
            source_nodes = edge_index_i.unique()
            attention_list = []

            # Apply softmax for each source node separately
            for src in source_nodes:
                # Find edges coming from this source
                mask = (edge_index_i == src)
                if mask.sum() > 0:
                    # Apply softmax to attention scores for this source node
                    src_attention = attention[mask]
                    src_attention = F.softmax(src_attention, dim=0)
                    attention_list.append(src_attention)

            # Concatenate all normalized attention scores
            if attention_list:
                normalized_attention = torch.cat(attention_list)
                # Apply normalized attention weights
                messages = messages * normalized_attention.view(-1, 1)

        return messages

    def update(self, aggr_out):
        """Update function for message passing."""
        return aggr_out


class GCNInteraction(nn.Module):
    """SchNet-style interaction block using the PyG-compatible CFConv."""

    def __init__(self, n_inputs, n_gaussians, n_filters,
                 activation='tanh', use_attention=True):
        super(GCNInteraction, self).__init__()

        # Convert string activation to module if needed
        if isinstance(activation, str):
            activation = self._get_activation(activation)

        # Initial atom-wise layer
        self.initial_projection = MLP(
            in_channels=n_inputs,
            hidden_channels=n_filters,
            out_channels=n_filters,
            num_layers=1,
            bias=True,
            act=None  # No activation
        )

        # Continuous filter convolution
        self.cfconv = CFConv(
            n_gaussians=n_gaussians,
            n_filters=n_filters,
            activation=activation,
            use_attention=use_attention
        )

        # Output MLPs
        self.output_layers = MLP(
            in_channels=n_filters,
            hidden_channels=n_filters,
            out_channels=n_inputs,  # Return to input size for residual
            num_layers=2,
            bias=True,
            act=activation
        )

    def _get_activation(self, activation_name):
        """Convert activation name to module"""
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
            return nn.Tanh()

    def forward(self, x, edge_index, edge_attr):
        """Forward pass for the interaction block."""
        # Initial projection
        h = self.initial_projection(x)

        # Apply continuous filter convolution
        h, attention = self.cfconv(h, edge_index, edge_attr)

        # Output transformation
        output = self.output_layers(h)

        return output, attention


class SchNetEncoder(nn.Module):
    """SchNet encoder using PyTorch Geometric native operations."""

    def __init__(
            self,
            node_dim,
            edge_dim,
            hidden_dim=16,
            output_dim=32,
            n_interactions=3,
            activation='tanh',
            use_attention=True
    ):
        super(SchNetEncoder, self).__init__()

        # Initial embedding
        self.embedding = MLP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=1,
            act=activation
        )

        # Interaction blocks
        self.interactions = nn.ModuleList([
            GCNInteraction(
                n_inputs=hidden_dim,
                n_gaussians=edge_dim,
                n_filters=hidden_dim,
                activation=activation,
                use_attention=use_attention
            ) for _ in range(n_interactions)
        ])

        # Output network
        self.output_network = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=2,
            act=activation
        )

        # Store dimensions for debugging
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass for SchNet encoder with separate inputs rather than PyG data object.

        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, node_dim]
        edge_index : torch.Tensor
            Edge indices [2, num_edges]
        edge_attr : torch.Tensor
            Edge features [num_edges, edge_dim]
        batch : torch.Tensor, optional
            Batch assignment for nodes

        Returns:
        --------
        Tuple[torch.Tensor, Tuple]
            - graph_embeddings: Graph-level embeddings [batch_size, output_dim]
            - (node_embeddings, attentions): Additional outputs
        """
        # If no batch information provided, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial embedding
        h = self.embedding(x)

        # Apply interaction blocks with residual connections
        attentions = []
        for interaction in self.interactions:
            delta, attention = interaction(h, edge_index, edge_attr)
            h = h + delta  # Residual connection
            if attention is not None:
                attentions.append(attention)

        # Global pooling (mean of node features per graph)
        from torch_geometric.nn import global_mean_pool
        pooled = global_mean_pool(h, batch)

        # Final output transformation
        output = self.output_network(pooled)

        return output, (h, attentions)
