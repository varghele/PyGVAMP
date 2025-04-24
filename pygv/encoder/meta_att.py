import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer, MLP, global_mean_pool
try:
    from torch_scatter import scatter_mean, scatter_add
except ImportError:
    from pygv.utils.alternative_torch_scatter import scatter_mean
from typing import Optional, Union, Callable, Literal


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class EdgeModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_edge_mlp_layers, act, norm, dropout):
        super().__init__()
        self.edge_mlp = MLP(
            in_channels=2 * node_dim + edge_dim, # no global_dim
            hidden_channels=hidden_dim,
            out_channels=edge_dim,
            num_layers=num_edge_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        ).apply(init_weights)

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeAttention(torch.nn.Module):
    """
    Attention layer for nodes in a graph.
    This computes attention weights between nodes based on their features.
    """

    def __init__(self, node_dim, attn_dim=32, dropout=0.0):
        super().__init__()
        self.node_dim = node_dim
        self.attn_dim = attn_dim

        # Layers for computing attention scores
        self.query = nn.Linear(node_dim, attn_dim)
        self.key = nn.Linear(node_dim, attn_dim)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, batch=None):
        """
        Compute attention weights for nodes.

        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, node_dim]
        edge_index : torch.Tensor
            Edge indices [2, num_edges]
        batch : torch.Tensor, optional
            Batch assignment for nodes

        Returns:
        --------
        tuple
            (x, attention_weights)
            - x: Original node features
            - attention_weights: Attention weights for edges
        """
        # Project node features to query and key spaces
        q = self.query(x)  # [num_nodes, attn_dim]
        k = self.key(x)  # [num_nodes, attn_dim]

        # Get source and target nodes for each edge
        src, dst = edge_index

        # Compute attention scores between connected nodes
        # a_{ij} = q_i^T · k_j
        src_q = q[src]  # [num_edges, attn_dim]
        dst_k = k[dst]  # [num_edges, attn_dim]

        # Compute attention scores
        attn_scores = torch.sum(src_q * dst_k, dim=1)  # [num_edges]

        # Normalize attention scores per source node using scatter_softmax
        from torch_scatter import scatter_softmax
        attn_weights = scatter_softmax(attn_scores, src, dim=0)  # [num_edges]

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        return x, attn_weights


class NodeModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_node_mlp_layers, act, norm, dropout, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Attention layer for computing edge importance
        if use_attention:
            self.attention = NodeAttention(node_dim=node_dim, attn_dim=hidden_dim, dropout=dropout)

        self.node_mlp_1 = MLP(
            in_channels=node_dim + edge_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_node_mlp_layers,
            act=act,
            norm="BatchNorm",  # norm,
            dropout=dropout
        ).apply(init_weights)

        self.node_mlp_2 = MLP(
            in_channels=node_dim + hidden_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=node_dim,
            num_layers=num_node_mlp_layers,
            act=act,
            norm="BatchNorm",  # norm,
            dropout=dropout
        ).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index

        # Apply attention if enabled
        attentions = None
        if self.use_attention:
            _, attentions = self.attention(x, edge_index, batch)

        # Create message = concat(source_node_features, edge_features)
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)

        # Apply attention weights to messages if available
        if attentions is not None:
            out = out * attentions.unsqueeze(-1)  # Weight messages by attention

        # Aggregate messages at target nodes
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))

        # Update node features
        out = torch.cat([x, out], dim=1)
        updated_nodes = self.node_mlp_2(out)

        return updated_nodes, attentions

class GlobalModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, num_global_mlp_layers, act, norm, dropout):
        super().__init__()
        self.global_mlp = MLP(
            in_channels=node_dim + edge_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=global_dim,
            num_layers=num_global_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        ).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([
            scatter_mean(x, batch, dim=0),
            scatter_mean(edge_attr, batch[edge_index[0]], dim=0)
        ], dim=1)
        return self.global_mlp(out)


class Meta(torch.nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            global_dim: int,
            num_node_mlp_layers: int,
            num_edge_mlp_layers: int,
            num_global_mlp_layers: int,
            hidden_dim: int,
            output_dim: int,
            num_meta_layers: int,
            embedding_type: Literal["node", "global", "combined"],
            use_attention: bool = True,
            act: Union[str, Callable] = "relu",
            norm: Optional[str] = "batch_norm",
            dropout: float = 0.0,
    ):
        super().__init__()
        self.num_meta_layers = num_meta_layers
        self.embedding_type = embedding_type
        self.use_attention = use_attention

        # Store all necessary attributes
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.num_edge_mlp_layers = num_edge_mlp_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = act
        self.norm = norm
        self.dropout = dropout

        # Message passing layers
        self.layers = torch.nn.ModuleList([
            MetaLayer(
                edge_model=EdgeModel(node_dim, edge_dim, hidden_dim, num_edge_mlp_layers, act, norm, dropout),
                node_model=NodeModel(node_dim, edge_dim, hidden_dim, num_node_mlp_layers, act, norm, dropout,
                                     use_attention=use_attention),
                global_model=GlobalModel(node_dim, edge_dim, global_dim, hidden_dim, num_global_mlp_layers, act, norm,
                                         dropout)
            ) for _ in range(num_meta_layers)
        ])

        # Determine input dimension for embedding projection
        if embedding_type == "node":
            embedding_in_dim = node_dim
        elif embedding_type == "global":
            embedding_in_dim = global_dim
        else:  # combined
            embedding_in_dim = node_dim + global_dim

        # Embedding projection layer (optional, for reducing dimensions or additional non-linearity)
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_in_dim, output_dim),
            self._get_activation(act)
        )

    def _get_activation(self, act):
        if act == "relu":
            return nn.ReLU()
        elif act == "leaky_relu":
            return nn.LeakyReLU()
        elif act == "gelu":
            return nn.GELU()
        elif act == "elu":
            return nn.ELU()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.Identity()

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initialize random global attribute u with the same size as node features
        u = torch.randn(batch.max() + 1, self.global_dim, device=x.device)

        # Store attention weights from all layers
        attentions = []

        # Message passing with attention
        for layer in self.layers:
            if self.use_attention:
                # Modified MetaLayer forward call that handles attention
                # Assumes node_model returns (x_new, attention)
                edge_out = layer.edge_model(x[edge_index[0]], x[edge_index[1]], edge_attr, u[batch][edge_index[0]],
                                            batch[edge_index[0]])
                x_out, layer_attention = layer.node_model(x, edge_index, edge_out, u, batch)
                u_out = layer.global_model(x_out, edge_index, edge_out, u, batch)

                # Store attention from this layer
                attentions.append(layer_attention)

                # Update variables for next layer
                x, edge_attr, u = x_out, edge_out, u_out
            else:
                # Standard MetaLayer processing without attention
                x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)

        # Prepare embeddings based on type
        if self.embedding_type == "node":
            # For node-level tasks, keep node embeddings
            embeddings = x
            # Can also provide graph-level embeddings by pooling nodes
            graph_embeddings = global_mean_pool(x, batch)
        elif self.embedding_type == "global":
            # For graph-level tasks, use global features
            embeddings = u
            graph_embeddings = u
        else:  # combined
            # Combine node and global features
            node_embeddings = torch.cat([x, u[batch]], dim=1)
            embeddings = node_embeddings
            # Pool to get graph-level embeddings
            graph_embeddings = global_mean_pool(node_embeddings, batch)

        # Project embeddings to desired output dimension
        projected_embeddings = self.embedding_projection(embeddings)
        projected_graph_embeddings = self.embedding_projection(graph_embeddings)

        # Return with attention values
        return projected_graph_embeddings, (projected_embeddings, x, edge_attr, u, attentions)



