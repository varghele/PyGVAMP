import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer, MLP
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


class NodeModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_node_mlp_layers, act, norm, dropout):
        super().__init__()
        self.node_mlp_1 = MLP(
            in_channels=node_dim + edge_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_node_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        ).apply(init_weights)
        self.node_mlp_2 = MLP(
            in_channels=node_dim + hidden_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=node_dim,
            num_layers=num_node_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        ).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)  # Removed u[batch]
        return self.node_mlp_2(out)


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
            act: Union[str, Callable] = "relu",
            norm: Optional[str] = "batch_norm",
            dropout: float = 0.0,
    ):
        super().__init__()
        self.num_meta_layers = num_meta_layers
        self.embedding_type = embedding_type

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
                node_model=NodeModel(node_dim, edge_dim, hidden_dim, num_node_mlp_layers, act, norm, dropout),
                global_model=GlobalModel(node_dim, edge_dim, global_dim, hidden_dim, num_global_mlp_layers, act, norm,
                                         dropout)
            ) for _ in range(num_meta_layers)
        ])

        # Determine input dimension for embedding projection
        if embedding_type == "node":
            embedding_in_dim = hidden_dim
        elif embedding_type == "global":
            embedding_in_dim = hidden_dim
        else:  # combined
            embedding_in_dim = 2 * hidden_dim

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
        #u = torch.randn(batch.max() + 1, x.size(1), device=x.device)
        u = torch.randn(batch.max() + 1, self.global_dim, device=x.device)

        # Message passing
        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)

        # Prepare embeddings based on type
        if self.embedding_type == "node":
            # For node-level tasks, keep node embeddings
            embeddings = x
            # Can also provide graph-level embeddings by pooling nodes
            graph_embeddings = nn.global_mean_pool(x, batch)
        elif self.embedding_type == "global":
            # For graph-level tasks, use global features
            embeddings = u
            graph_embeddings = u
        else:  # combined
            # Combine node and global features
            node_embeddings = torch.cat([x, u[batch]], dim=1)
            embeddings = node_embeddings
            # Pool to get graph-level embeddings
            graph_embeddings = nn.global_mean_pool(node_embeddings, batch)

        # Project embeddings to desired output dimension
        projected_embeddings = self.embedding_projection(embeddings)
        projected_graph_embeddings = self.embedding_projection(graph_embeddings)

        return projected_graph_embeddings, (projected_embeddings, x, edge_attr, u)
