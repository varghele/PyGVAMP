import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MLP
from torch_geometric.utils import softmax
from torch.jit import script
from torch_geometric.nn import global_mean_pool


def init_weights(m):
    if type(m) == torch.nn.Linear:
        #torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.fill_(0.0)


class CFConv(MessagePassing):
    """
    Continuous Filter Convolution for PyTorch Geometric graph data.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_channels,
                 hidden_channels=None,
                 activation='tanh',
                 use_attention=True,
                 aggr='add'):

        super(CFConv, self).__init__(aggr=aggr)

        # Set default hidden channels if not provided
        if hidden_channels is None:
            hidden_channels = out_channels

        # Set channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        # Convert string activation to module if needed
        self.activation = self._get_activation(activation) if isinstance(activation, str) else activation

        # Filter network transforms edge features to weights
        self.filter_network = MLP(
            in_channels=edge_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            act=self.activation,
            norm=None,
            plain_last=True
        ).apply(init_weights)

        # Add node projection layer if input and output dimensions differ
        self.has_node_projection = (in_channels != out_channels)
        if self.has_node_projection:
            self.node_projection = nn.Linear(in_channels, out_channels)

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention_vector = nn.Parameter(torch.Tensor(out_channels, 1))
            nn.init.xavier_uniform_(self.attention_vector, gain=1.414)

        # For storing attention weights
        self._attention_weights = None

    def _get_activation(self, activation_name):
        """Convert activation name to module"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations.get(activation_name.lower(), nn.Tanh())

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of CFConv layer.
        """
        # Normalize edge attributes for numerical stability
        # Todo: check if this normalization is responsible for differences
        edge_attr_norm = edge_attr / (edge_attr.norm(dim=-1, keepdim=True) + 1e-8)
        #edge_attr_norm = edge_attr

        # Generate weights from edge attributes
        edge_weights = self.filter_network(edge_attr_norm)

        # Clear previous attention weights
        self._attention_weights = None

        # Project input features to output dimension if needed
        if self.has_node_projection:
            x = self.node_projection(x)

        # Message passing
        out = self.propagate(edge_index, x=x, edge_weights=edge_weights)

        return out, self._attention_weights

    # Define JIT-compiled helper functions
    @script
    def compute_attention(messages, attention_vector):
        """Compute attention scores using matrix multiplication"""
        return torch.matmul(messages, attention_vector).squeeze(-1)

    @script
    def apply_attention(messages, attention):
        """Apply attention weights to messages"""
        return messages * attention.view(-1, 1)

    def message(self, x_j, edge_weights, edge_index_i):
        """
        Optimized message function that uses PyG's built-in softmax for better performance.
        """
        # Apply edge weights to source node features (keep dimension check for debugging)
        if x_j.size(1) != edge_weights.size(1):
            print(f"Dimension mismatch - x_j: {x_j.shape}, edge_weights: {edge_weights.shape}")

        # Compute messages by element-wise multiplication
        messages = x_j * edge_weights

        # If using attention, compute attention weights with PyG's optimized functions
        if self.use_attention:
            # Compute attention score for each edge
            attention = torch.matmul(messages, self.attention_vector).squeeze(-1)

            # Use PyG's optimized softmax instead of manual per-node implementation
            normalized_attention = softmax(attention, edge_index_i)

            # Apply attention weights to messages
            #messages = messages * normalized_attention.view(-1, 1)

            # Store raw attention weights for later access
            self._attention_weights = attention

            # Use JIT-compiled function for applying attention
            messages = self.apply_attention(messages, normalized_attention)

        return messages

    def update(self, aggr_out):
        """No additional update step needed"""
        return aggr_out


class GCNInteraction(nn.Module):
    """SchNet-style interaction block using the PyG-compatible CFConv."""

    def __init__(self, in_channels, edge_channels, hidden_channels,
                 activation='tanh', use_attention=True):
        super(GCNInteraction, self).__init__()

        # Initial dense layer (matching classic implementation)
        self.initial_dense = nn.Linear(in_channels, in_channels, bias=False)

        # CFConv layer - use in_channels for both input and output to maintain dimensions
        self.cfconv = CFConv(
            in_channels=in_channels,
            out_channels=in_channels,  # Same as input to preserve dimensions
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            use_attention=use_attention
        )

        # Output transformation
        self.output_layer = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,  # Back to input dimension for residual
            num_layers=2,
            act=activation
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for interaction block.
        """
        # Initial dense layer (matching classic implementation)
        h = self.initial_dense(x)

        # Apply continuous filter convolution
        #h, attention = self.cfconv(x, edge_index, edge_attr)
        conv_output, attention = self.cfconv(h, edge_index, edge_attr)

        # Transform back to input dimension
        #output = self.output_layer(h)
        output = self.output_layer(conv_output)

        return output, attention


class SchNetEncoderNoEmbed(nn.Module):
    """SchNet encoder using PyTorch Geometric native operations."""

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
        super(SchNetEncoderNoEmbed, self).__init__()

        self.node_dim = node_dim
        self.output_dim = output_dim

        # Interaction blocks
        self.interactions = nn.ModuleList([
            GCNInteraction(
                in_channels=node_dim,
                edge_channels=edge_dim,
                hidden_channels=hidden_dim,
                activation=activation,
                use_attention=use_attention
            ) for _ in range(n_interactions)
        ])

        # Output network for graph-level readout
        self.output_network = MLP(
            in_channels=node_dim,  # Use node_dim as each interaction preserves dimensions
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=2,
            act=activation,
            #norm="LayerNorm"
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass for SchNet encoder.
        """
        # If no batch information provided, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Node features start as x (no initial embedding)
        h = x

        # Store attention weights from each layer
        attentions = []

        # Apply interaction blocks with residual connections
        for interaction in self.interactions:
            delta, attention = interaction(h, edge_index, edge_attr)
            h = h + delta  # Residual connection
            if attention is not None:
                attentions.append(attention)

        # Global pooling (mean of node features per graph)
        pooled = global_mean_pool(h, batch)

        # Final output transformation
        output = self.output_network(pooled)

        return output, (h, attentions)
