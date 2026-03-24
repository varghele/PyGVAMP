import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MLP
from torch_geometric.utils import softmax
from torch_geometric.nn import global_mean_pool


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.fill_(0.0)


class GINEConvWithAttention(MessagePassing):
    """
    GIN-E convolution with edge features and bolt-on attention.

    Implements the GIN message-passing rule with edge feature injection
    (like GINEConv) plus a learnable attention mechanism identical to
    SchNet's CFConv attention.

    Message:  m_ij = activation(x_j + edge_proj(edge_attr))
    Attention: a_ij = softmax_j(m_ij @ attention_vector)
    Aggregate: agg_i = sum(a_ij * m_ij)
    Update:   h_i = update_mlp((1 + eps) * x_i + agg_i)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_channels,
                 hidden_channels=None,
                 activation='tanh',
                 use_attention=True,
                 aggr='add'):

        super(GINEConvWithAttention, self).__init__(aggr=aggr)

        if hidden_channels is None:
            hidden_channels = out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.activation = self._get_activation(activation) if isinstance(activation, str) else activation

        # Learnable epsilon for GIN self-loop weighting
        self.eps = nn.Parameter(torch.zeros(1))

        # Project edge features into node feature space
        self.edge_projection = nn.Linear(edge_channels, in_channels)

        # Node projection if dimensions differ
        self.has_node_projection = (in_channels != out_channels)
        if self.has_node_projection:
            self.node_projection = nn.Linear(in_channels, out_channels)

        # MLP applied after aggregation (core GIN component)
        self.update_mlp = MLP(
            in_channels=out_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            act=self.activation,
            norm=None,
            plain_last=True
        ).apply(init_weights)

        # Attention mechanism (identical to SchNet's CFConv)
        self.use_attention = use_attention
        if use_attention:
            self.attention_vector = nn.Parameter(torch.Tensor(in_channels, 1))
            nn.init.xavier_uniform_(self.attention_vector, gain=1.414)

        # For storing attention weights
        self._attention_weights = None

    def _get_activation(self, activation_name):
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
        # Normalize edge attributes for numerical stability
        edge_attr_norm = edge_attr / (edge_attr.norm(dim=-1, keepdim=True) + 1e-8)

        self._attention_weights = None

        # Project input if dimensions differ
        if self.has_node_projection:
            x_proj = self.node_projection(x)
        else:
            x_proj = x

        # Message passing aggregation
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr_norm)

        # Project aggregated messages if needed
        if self.has_node_projection:
            agg = self.node_projection(agg) if agg.size(-1) != x_proj.size(-1) else agg

        # GIN update: MLP((1 + eps) * x_i + aggregate)
        out = self.update_mlp((1 + self.eps) * x_proj + agg)

        return out, self._attention_weights

    def message(self, x_j, edge_attr, edge_index_i):
        # Project edge features and add to source node features
        edge_proj = self.edge_projection(edge_attr)
        messages = self.activation(x_j + edge_proj)

        # Bolt-on attention (identical to SchNet's CFConv)
        if self.use_attention:
            attention = torch.matmul(messages, self.attention_vector).squeeze(-1)
            normalized_attention = softmax(attention, edge_index_i)
            self._attention_weights = normalized_attention
            messages = messages * normalized_attention.view(-1, 1)

        return messages

    def update(self, aggr_out):
        return aggr_out


class GINInteraction(nn.Module):
    """GIN interaction block, mirroring SchNet's GCNInteraction."""

    def __init__(self, in_channels, edge_channels, hidden_channels,
                 activation='tanh', use_attention=True):
        super(GINInteraction, self).__init__()

        self.initial_dense = nn.Linear(in_channels, in_channels, bias=False)

        self.gin_conv = GINEConvWithAttention(
            in_channels=in_channels,
            out_channels=in_channels,
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            use_attention=use_attention
        )

        self.output_layer = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_layers=2,
            act=activation
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.initial_dense(x)
        conv_output, attention = self.gin_conv(h, edge_index, edge_attr)
        output = self.output_layer(conv_output)
        return output, attention


class GINEncoder(nn.Module):
    """GIN encoder using GINEConv with edge features and bolt-on attention."""

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
        super(GINEncoder, self).__init__()

        self.node_dim = node_dim
        self.output_dim = output_dim

        self.interactions = nn.ModuleList([
            GINInteraction(
                in_channels=node_dim,
                edge_channels=edge_dim,
                hidden_channels=hidden_dim,
                activation=activation,
                use_attention=use_attention
            ) for _ in range(n_interactions)
        ])

        self.output_network = MLP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=2,
            act=activation,
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        attentions = []

        for interaction in self.interactions:
            delta, attention = interaction(h, edge_index, edge_attr)
            h = h + delta
            if attention is not None:
                attentions.append(attention)

        pooled = global_mean_pool(h, batch)
        output = self.output_network(pooled)

        return output, (h, attentions)
