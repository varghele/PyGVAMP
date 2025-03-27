import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer, MLP
try:
    from torch_scatter import scatter_mean, scatter_add
except ImportError:
    from utils.alternative_torch_scatter import scatter_mean
from typing import Optional, Union, Callable, Literal, List


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
        )

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
        )
        self.node_mlp_2 = MLP(
            in_channels=node_dim + hidden_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=node_dim,
            num_layers=num_node_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        )

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
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([
            scatter_mean(x, batch, dim=0),
            scatter_mean(edge_attr, batch[edge_index[0]], dim=0)
        ], dim=1)
        return self.global_mlp(out)


class Meta(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,  # New parameter for embedding output dimension
            num_layers: int,
            embedding_type: Literal["node", "global", "combined"],
            act: Union[str, Callable] = "relu",
            norm: Optional[str] = "batch_norm",
            dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_type = embedding_type

        # Store all necessary attributes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = act
        self.norm = norm
        self.dropout = dropout

        # Message passing layers
        self.layers = torch.nn.ModuleList([
            MetaLayer(
                edge_model=EdgeModel(hidden_dim, act, norm, dropout),
                node_model=NodeModel(hidden_dim, act, norm, dropout),
                global_model=GlobalModel(hidden_dim, act, norm, dropout)
            ) for _ in range(num_layers)
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
        u = torch.randn(batch.max() + 1, x.size(1), device=x.device)

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



# Example usage:
if __name__ == "__main__":
    # Model parameters
    model = MPGNN(
        node_dim=32,
        edge_dim=16,
        global_dim=8,
        hidden_dim=64,
        num_layers=3,
        num_encoder_layers=2,
        num_edge_mlp_layers=2,
        num_node_mlp_layers=2,
        num_global_mlp_layers=2,
        shift_predictor_hidden_dim=[128, 64, 32],  # Custom architecture for shift predictor
        shift_predictor_layers=4,
        embedding_type="combined",  # Use both node and global features
        act="relu",
        norm="batch_norm",
        dropout=0.1
    )

    # Example data
    num_nodes = 10
    num_edges = 15
    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 16)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[5:] = 1  # Second half of nodes belong to second graph

    # Forward pass
    shifts, (node_embeddings, edge_embeddings, global_embeddings) = model(x, edge_index, edge_attr, batch)

    print(f"Predicted shifts shape: {shifts.shape}")  # [num_nodes, 1]
