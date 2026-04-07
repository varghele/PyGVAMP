import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn import MLP, global_add_pool
from torch_geometric.utils import softmax
from pygv.utils.alternative_torch_scatter import scatter_add


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SpectralDesign:
    """
    Computes spectral edge features via Laplacian eigendecomposition.

    For each graph in the batch, computes Gaussian spectral filters from
    the normalized Laplacian eigenvalues/eigenvectors, then extracts
    per-edge features.
    """

    def __init__(self, nfreq=10, dv=1.0, recfield=1):
        self.nfreq = nfreq
        self.dv = dv
        self.recfield = recfield

    def compute(self, edge_index, batch, num_nodes, device):
        """
        Compute spectral edge features for all graphs in the batch.

        Returns: [total_edges, nfreq + 1] tensor of edge features.
        """
        src, dst = edge_index
        # Determine graph membership for each node
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

        graph_ids = torch.unique(batch)
        all_edge_features = torch.zeros(edge_index.size(1), self.nfreq + 1, device=device)

        for gid in graph_ids:
            node_mask = batch == gid
            node_indices = torch.where(node_mask)[0]
            n = node_indices.size(0)

            # Find edges belonging to this graph
            edge_mask = node_mask[src] & node_mask[dst]
            graph_edge_indices = torch.where(edge_mask)[0]

            if n == 0 or graph_edge_indices.size(0) == 0:
                continue

            # Map global node indices to local
            local_map = torch.zeros(num_nodes, dtype=torch.long, device=device)
            local_map[node_indices] = torch.arange(n, device=device)
            local_src = local_map[src[edge_mask]]
            local_dst = local_map[dst[edge_mask]]

            # Build adjacency matrix
            A = torch.zeros(n, n, device=device)
            A[local_src, local_dst] = 1.0

            # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
            deg = A.sum(dim=1)
            deg_inv_sqrt = torch.zeros_like(deg)
            mask = deg > 0
            deg_inv_sqrt[mask] = deg[mask].pow(-0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            L = torch.eye(n, device=device) - D_inv_sqrt @ A @ D_inv_sqrt

            # Eigendecomposition
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(L)
            except Exception:
                # Fallback: use identity features
                all_edge_features[graph_edge_indices, -1] = 1.0
                continue

            # Create nfreq Gaussian filters
            # Centers evenly spaced in [0, 2] (eigenvalues of normalized Laplacian are in [0, 2])
            centers = torch.linspace(0, 2, self.nfreq, device=device)

            # Receptive field mask: M = (A + I)^recfield, binarized
            M = A + torch.eye(n, device=device)
            for _ in range(self.recfield - 1):
                M = M @ (A + torch.eye(n, device=device))
            M = (M > 0).float()

            for k in range(self.nfreq):
                # Gaussian filter in spectral domain
                g_k = torch.exp(-self.dv * (eigenvalues - centers[k]) ** 2)
                # Apply filter: F_k = U @ diag(g_k) @ U^T
                F_k = eigenvectors @ torch.diag(g_k) @ eigenvectors.T
                # Apply receptive field mask
                F_k = F_k * M
                # Extract edge features
                all_edge_features[graph_edge_indices, k] = F_k[local_src, local_dst]

            # Identity channel (self-connection indicator)
            identity = torch.eye(n, device=device)
            all_edge_features[graph_edge_indices, -1] = identity[local_src, local_dst]

        return all_edge_features


class SpectConvWithAttention(nn.Module):
    """
    Spectral convolution with parallel attention aggregation.

    Drops MessagePassing in favor of explicit scatter_add with dual
    aggregation per spectral coefficient:
    - Sum branch: preserves 3-WL expressiveness (identical to original SpectConv)
    - Attention branch: adds learned per-neighbor re-weighting

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        K: Number of spectral coefficients
        selfconn: Whether to use self-connection (last weight)
        use_attention: Whether to add parallel attention branch
        bias: Whether to use bias
    """

    def __init__(self, in_channels, out_channels, K=1, selfconn=True,
                 use_attention=True, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.selfconn = selfconn
        self.use_attention = use_attention
        self.K_spectral = K  # number of spectral coefficients (before self-conn)

        total_K = K + 1 if selfconn else K
        self.weight = Parameter(torch.Tensor(total_K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if use_attention:
            self.attention_vector = Parameter(torch.Tensor(in_channels, 1))
            nn.init.xavier_uniform_(self.attention_vector, gain=1.414)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: [n_nodes, in_channels]
            edge_index: [2, n_edges]
            edge_attr: [n_edges, K_spectral]

        Returns:
            out: [n_nodes, out_channels]
            att_scores: [n_edges] or None
        """
        src, dst = edge_index
        n_nodes = x.size(0)
        K = self.K_spectral

        sum_out = torch.zeros(n_nodes, self.out_channels, device=x.device, dtype=x.dtype)

        # Self-connection (last weight)
        if self.selfconn:
            sum_out = sum_out + torch.matmul(x, self.weight[-1])

        # Collect messages across all K spectral coefficients
        all_msg = [] if self.use_attention else None

        for k in range(K):
            # spectral-weighted neighbor features
            msg_k = edge_attr[:, k].view(-1, 1) * x[src]  # [n_edges, in_channels]
            sum_agg_k = scatter_add(msg_k, dst, dim=0, dim_size=n_nodes)
            sum_out = sum_out + torch.matmul(sum_agg_k, self.weight[k])
            if self.use_attention:
                all_msg.append(msg_k)

        att_scores = None
        if self.use_attention and len(all_msg) > 0:
            # Single attention score per edge (averaged across K coefficients)
            pooled_msg = torch.stack(all_msg).mean(dim=0)  # [n_edges, in_channels]
            raw_att = torch.matmul(pooled_msg, self.attention_vector).squeeze(-1)  # [n_edges]
            att_scores = softmax(raw_att, dst, num_nodes=n_nodes)

            att_out = torch.zeros(n_nodes, self.out_channels, device=x.device, dtype=x.dtype)
            for k in range(K):
                att_msg_k = att_scores.view(-1, 1) * all_msg[k]
                att_agg_k = scatter_add(att_msg_k, dst, dim=0, dim_size=n_nodes)
                att_out = att_out + torch.matmul(att_agg_k, self.weight[k])

            out = sum_out + att_out
        else:
            out = sum_out

        if self.bias is not None:
            out = out + self.bias

        return out, att_scores


class ML3Layer(nn.Module):
    """
    ML3 message-passing layer with edge transformation and skip connection.

    Uses SpectConvWithAttention instead of the original SpectConv/MessagePassing.
    """

    def __init__(self, learnedge=True, nedgeinput=1, nedgeoutput=1,
                 ninp=1, nout1=30, nout2=2, use_attention=True):
        super().__init__()

        self.learnedge = learnedge
        self.nout2 = nout2

        if self.learnedge:
            self.edge_transform = nn.ModuleDict({
                'linear1': nn.Linear(nedgeinput, 2 * nedgeinput, bias=False),
                'linear2': nn.Linear(nedgeinput, 2 * nedgeinput, bias=False),
                'linear3': nn.Linear(nedgeinput, 2 * nedgeinput, bias=False),
                'linear4': nn.Linear(4 * nedgeinput, nedgeoutput, bias=False)
            })
        else:
            nedgeoutput = nedgeinput

        self.conv = SpectConvWithAttention(
            ninp, nout1, nedgeoutput, selfconn=False, use_attention=use_attention
        )

        if nout2 > 0:
            self.skip_connection = nn.ModuleDict({
                'linear1': nn.Linear(ninp, nout2),
                'linear2': nn.Linear(ninp, nout2)
            })

    def forward(self, x, edge_index, edge_attr):
        """
        Returns:
            out: [n_nodes, nout1 + nout2]
            att_scores: [n_edges] or None
        """
        if self.learnedge:
            linear_part = F.relu(self.edge_transform['linear1'](edge_attr))
            gated_part = (torch.tanh(self.edge_transform['linear2'](edge_attr)) *
                          torch.tanh(self.edge_transform['linear3'](edge_attr)))
            tmp = torch.cat([linear_part, gated_part], dim=1)
            edge_attr = F.relu(self.edge_transform['linear4'](tmp))

        conv_output, att_scores = self.conv(x, edge_index, edge_attr)
        conv_output = F.relu(conv_output)

        if self.nout2 > 0:
            skip_output = (torch.tanh(self.skip_connection['linear1'](x)) *
                           torch.tanh(self.skip_connection['linear2'](x)))
            return torch.cat([conv_output, skip_output], dim=1), att_scores
        else:
            return conv_output, att_scores


class ML3Interaction(nn.Module):
    """
    ML3 interaction block with residual connection.

    Mirrors GINInteraction: initial projection → ML3Layer → output projection → residual.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=30, nout1=30, nout2=2,
                 use_attention=True, learnedge=True):
        super().__init__()

        self.initial_dense = nn.Linear(node_dim, node_dim, bias=False)

        self.ml3_layer = ML3Layer(
            learnedge=learnedge,
            nedgeinput=edge_dim,
            nedgeoutput=edge_dim,
            ninp=node_dim,
            nout1=nout1,
            nout2=nout2,
            use_attention=use_attention
        )

        # Project back to node_dim for residual connection
        self.output_layer = nn.Linear(nout1 + nout2, node_dim)

    def forward(self, h, edge_index, edge_feat):
        """
        Returns:
            delta: [n_nodes, node_dim] — residual update
            att_scores: [n_edges] or None
        """
        h_proj = self.initial_dense(h)
        layer_out, att_scores = self.ml3_layer(h_proj, edge_index, edge_feat)
        delta = self.output_layer(layer_out)
        return delta, att_scores


class ML3Encoder(nn.Module):
    """
    ML3 encoder using spectral convolutions with 3-WL expressivity and
    optional parallel attention.

    Follows the same interface as GINEncoder:
    - forward(x, edge_index, edge_attr, batch) → (output, (h, attentions))
    - output shape: [batch_size, output_dim]

    Supports two edge feature modes:
    - 'gaussian': Uses dataset-provided Gaussian-expanded distance features,
      projected through an edge encoder MLP (default)
    - 'spectral': Computes Laplacian eigendecomposition per graph for
      spectral edge features

    Args:
        node_dim: Input node feature dimension
        edge_dim: Input edge feature dimension (for gaussian mode)
        hidden_dim: Hidden dimension for ML3 layers (nout1)
        output_dim: Final graph-level output dimension
        num_layers: Number of ML3 interaction layers
        activation: Activation function name
        use_attention: Whether to use parallel attention
        edge_mode: 'gaussian' or 'spectral'
        nfreq: Number of spectral frequencies (spectral mode)
        spectral_dv: Gaussian width for spectral filters (spectral mode)
        recfield: Receptive field for spectral filters (spectral mode)
        nout1: Convolution output dim in ML3Layer
        nout2: Skip connection output dim in ML3Layer (0 to disable)
    """

    def __init__(
            self,
            node_dim,
            edge_dim,
            hidden_dim=30,
            output_dim=32,
            num_layers=4,
            activation='relu',
            use_attention=True,
            edge_mode='gaussian',
            nfreq=10,
            spectral_dv=1.0,
            recfield=1,
            nout1=30,
            nout2=2,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.edge_mode = edge_mode
        self.use_attention = use_attention

        # Internal working dimension for nodes
        internal_node_dim = hidden_dim

        # Node encoder: project input features to internal dim
        self.node_encoder = MLP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=internal_node_dim,
            num_layers=2,
            act=activation,
            norm=None,
        ).apply(init_weights)

        # Edge handling depends on mode
        if edge_mode == 'gaussian':
            internal_edge_dim = edge_dim  # keep original edge dim
            self.edge_encoder = MLP(
                in_channels=edge_dim,
                hidden_channels=hidden_dim,
                out_channels=internal_edge_dim,
                num_layers=2,
                act=activation,
                norm=None,
            ).apply(init_weights)
            self.spectral_design = None
        elif edge_mode == 'spectral':
            internal_edge_dim = nfreq + 1
            self.edge_encoder = None
            self.spectral_design = SpectralDesign(
                nfreq=nfreq, dv=spectral_dv, recfield=recfield
            )
        else:
            raise ValueError(f"Unknown edge_mode: {edge_mode}. Use 'gaussian' or 'spectral'.")

        # Interaction layers
        self.interactions = nn.ModuleList([
            ML3Interaction(
                node_dim=internal_node_dim,
                edge_dim=internal_edge_dim,
                hidden_dim=hidden_dim,
                nout1=nout1,
                nout2=nout2,
                use_attention=use_attention,
                learnedge=True,
            ) for _ in range(num_layers)
        ])

        # Output network: graph-level projection
        self.output_network = MLP(
            in_channels=internal_node_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=2,
            act=activation,
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass.

        Args:
            x: [n_nodes, node_dim] node features
            edge_index: [2, n_edges] edge connectivity
            edge_attr: [n_edges, edge_dim] edge features
            batch: [n_nodes] batch assignment

        Returns:
            output: [batch_size, output_dim] graph-level embeddings
            aux: (h, attentions) — node embeddings and attention weights per layer
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Encode node features
        h = self.node_encoder(x)

        # Encode/compute edge features
        if self.edge_mode == 'gaussian':
            edge_feat = self.edge_encoder(edge_attr)
        else:
            edge_feat = self.spectral_design.compute(
                edge_index, batch, x.size(0), x.device
            )

        # Message passing with residual connections
        attentions = []
        for interaction in self.interactions:
            delta, att = interaction(h, edge_index, edge_feat)
            h = h + delta  # residual
            if att is not None:
                attentions.append(att)

        # Graph-level pooling (add-pool, faithful to ML3 paper)
        pooled = global_add_pool(h, batch)

        # Output projection
        output = self.output_network(pooled)  # [batch_size, output_dim]

        return output, (h, attentions)
