from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, to_dense_adj
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU


def glorot(tensor):
    """Glorot uniform initialization for better gradient flow."""
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    """Zero initialization."""
    if tensor is not None:
        tensor.data.fill_(0)


class SpectralConvolution(MessagePassing):
    """ Spectral Convolution Layer for GNNML3
    Mathematical Background:
    - Implements spectral convolution using precomputed spectral supports
    - Each support C'(s) = U * diag(Φs(λ)) * U^T represents a different frequency filter
    - Φs(λ) = exp(-b(λ - fs)²) creates Gaussian-like frequency responses
    - Power series representation: C'(s) = α₀L⁰ + α₁L¹ + α₂L² + ...
    - This gives access to all powers of Laplacian without explicit computation

    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        K (int): Number of spectral supports (convolution kernels)
        selfconn (bool): Whether to include self-connection (identity transform)
        depthwise (bool): Whether to use depthwise separable convolution
        bias (bool): Whether to use bias parameters
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 K=1,
                 selfconn=True,
                 depthwise=False,
                 bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')  # Sum aggregation for message passing
        super(SpectralConvolution, self).__init__(**kwargs)

        assert K > 0, "Number of supports must be positive"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise = depthwise
        self.selfconn = selfconn

        # Add extra support for self-connection if enabled
        if self.selfconn:
            K = K + 1

        if self.depthwise:
            # Depthwise separable convolution: separate spatial and channel mixing
            self.DSweight = Parameter(torch.Tensor(K, in_channels))
            self.nsup = K
            K = 1  # Only one output weight matrix needed

        # Main learnable weights: [num_supports, in_channels, out_channels]
        # Each support gets its own transformation matrix
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot initialization."""
        glorot(self.weight)
        zeros(self.bias)
        if self.depthwise:
            zeros(self.DSweight)

    def forward(self,
                x,
                edge_index,
                edge_attr,
                edge_weight: OptTensor = None,
                batch: OptTensor = None,
                lambda_max: OptTensor = None):
        """
        Forward pass of spectral convolution.

        Mathematical Background:
        For each spectral support s:
        1. h_s = C(s) @ x  (message passing with s-th spectral support)
        2. out += h_s @ W_s  (apply learnable transformation)

        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (Tensor): Edge connectivity [2, num_edges]
            edge_attr (Tensor): Precomputed spectral supports [num_edges, num_supports]
                               Each column contains coefficients for one spectral filter

        Returns:
            Tensor: Transformed node features [num_nodes, out_channels]
        """
        Tx_0 = x  # Initial node features
        out = 0

        if not self.depthwise:
            # Standard convolution: each support has its own weight matrix
            enditr = self.weight.size(0)

            # Handle self-connection (identity transformation)
            if self.selfconn:
                # Self-connection: direct transformation without message passing
                # Implements: out += x @ W_self
                out = torch.matmul(Tx_0, self.weight[-1])
                enditr -= 1

            # Apply each spectral support
            for i in range(0, enditr):
                # Message passing with i-th spectral support
                # edge_attr[:, i] contains precomputed spectral coefficients
                h = self.propagate(edge_index, x=Tx_0, norm=edge_attr[:, i], size=None)

                # Apply learnable transformation for this support
                # Implements: out += (C(i) @ x) @ W_i
                out = out + torch.matmul(h, self.weight[i])

        else:
            # Depthwise separable convolution
            enditr = self.nsup

            if self.selfconn:
                # Self-connection with depthwise weights
                out = Tx_0 * self.DSweight[-1]
                enditr -= 1

            # First support with special handling
            out = out + (1 + self.DSweight[0:1, :]) * self.propagate(
                edge_index, x=Tx_0, norm=edge_attr[:, 0], size=None
            )

            # Remaining supports
            for i in range(1, enditr):
                out = out + self.DSweight[i:i + 1, :] * self.propagate(
                    edge_index, x=Tx_0, norm=edge_attr[:, i], size=None
                )

            # Final pointwise convolution
            out = torch.matmul(out, self.weight[0])

        # Add bias if present
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        """
        Message function for spectral convolution.

        Mathematical Background:
        - norm contains precomputed spectral coefficients from C'(s)
        - Each edge (i,j) gets weighted by the spectral support value
        - Implements: message_ij = C'(s)_ij * x_j

        Args:
            x_j (Tensor): Source node features
            norm (Tensor): Spectral coefficients for current support

        Returns:
            Tensor: Weighted messages
        """
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.weight.size(0))


class GNNML3Layer(torch.nn.Module):
    """ Complete GNNML3 Layer implementing 3-WL equivalent message passing.
    Mathematical Background:
    This layer achieves 3-WL expressive power through two key components:

    1. Spectral Convolution with Learned Edge Features:
       - C = M ⊙ mlp4(mlp1(C') | mlp2(C') ⊙ mlp3(C'))
       - Where C' contains precomputed spectral supports
       - The ⊙ operation (element-wise multiplication) is crucial for 3-WL power

    2. Element-wise Multiplication in Node Updates:
       - Implements the ⊙ operation from MATLANG L₃ = {·, ⊤, 1, diag, tr, ⊙}
       - According to theory: L₃ operations → 3-WL equivalent
       - Enables counting complex substructures (triangles, 4-cycles, tailed triangles)

    MATLANG Theory Connection:
    - L₁ = {·, ⊤, 1, diag} → 1-WL equivalent (traditional MPNNs)
    - L₂ = L₁ ∪ {tr} → Between 1-WL and 3-WL
    - L₃ = L₂ ∪ {⊙} → 3-WL equivalent (this implementation)

    Args:
        learnedge (bool): Whether to learn edge transformations
        nedgeinput (int): Input dimension of edge features (spectral supports)
        nedgeoutput (int): Output dimension of learned edge features
        ninp (int): Input node feature dimension
        nout1 (int): Output dimension of spectral convolution
        nout2 (int): Output dimension of element-wise multiplication branch
    """

    def __init__(self, learnedge, nedgeinput, nedgeoutput, ninp, nout1, nout2):
        super(GNNML3Layer, self).__init__()

        self.learnedge = learnedge
        self.nout2 = nout2

        if self.learnedge:
            # Edge feature learning MLPs (Equation 5 from paper)
            # Implements: mlp4(mlp1(C') | mlp2(C') ⊙ mlp3(C'))

            # mlp1: Linear transformation of spectral supports
            self.fc1_1 = torch.nn.Linear(nedgeinput, 2 * nedgeinput, bias=False)

            # mlp2, mlp3: For element-wise multiplication branch
            # The ⊙ operation here contributes to 3-WL expressive power
            self.fc1_2 = torch.nn.Linear(nedgeinput, 2 * nedgeinput, bias=False)
            self.fc1_3 = torch.nn.Linear(nedgeinput, 2 * nedgeinput, bias=False)

            # mlp4: Final combination of linear and multiplicative terms
            self.fc1_4 = torch.nn.Linear(4 * nedgeinput, nedgeoutput, bias=False)
        else:
            # If not learning edge features, use input dimension as output
            nedgeoutput = nedgeinput

        # Spectral convolution layer
        # Uses learned/precomputed spectral supports for message passing
        self.conv1 = SpectralConvolution(ninp, nout1, nedgeoutput, selfconn=False)

        if nout2 > 0:
            # Element-wise multiplication branch (crucial for 3-WL power!)
            # Implements: mlp5(x) ⊙ mlp6(x)
            # This is the key operation that breaks 1-WL limitations
            self.fc11 = torch.nn.Linear(ninp, nout2)  # mlp5
            self.fc12 = torch.nn.Linear(ninp, nout2)  # mlp6

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass implementing 3-WL equivalent computation.

        Mathematical Operations:
        1. Learn edge transformations (if enabled):
           C = mlp4(mlp1(C') | mlp2(C') ⊙ mlp3(C'))

        2. Spectral message passing:
           h_spectral = σ(∑_s C(s) @ x @ W_s)

        3. Element-wise multiplication (key for 3-WL):
           h_element = tanh(mlp5(x)) ⊙ tanh(mlp6(x))

        4. Concatenate results:
           output = [h_spectral | h_element]

        Args:
            x (Tensor): Node features [num_nodes, ninp]
            edge_index (Tensor): Edge connectivity [2, num_edges]
            edge_attr (Tensor): Precomputed spectral supports [num_edges, nedgeinput]

        Returns:
            Tensor: Updated node features [num_nodes, nout1 + nout2]
        """

        # Step 1: Learn edge transformations (if enabled)
        if self.learnedge:
            # Implement: mlp4(mlp1(C') | mlp2(C') ⊙ mlp3(C'))
            # This learns optimal combinations of spectral supports

            # Linear branch: mlp1(C')
            linear_branch = F.relu(self.fc1_1(edge_attr))

            # Multiplicative branch: mlp2(C') ⊙ mlp3(C')
            # Element-wise multiplication contributes to 3-WL power
            mult_branch = (torch.tanh(self.fc1_2(edge_attr)) *
                          torch.tanh(self.fc1_3(edge_attr)))

            # Concatenate and combine: mlp4([linear | multiplicative])
            combined = torch.cat([linear_branch, mult_branch], dim=1)
            edge_attr = F.relu(self.fc1_4(combined))

        # Step 2: Combine spectral convolution with element-wise operations
        if self.nout2 > 0:
            # Spectral convolution branch
            spectral_output = F.relu(self.conv1(x, edge_index, edge_attr))

            # Element-wise multiplication branch (crucial for 3-WL!)
            # Implements: tanh(mlp5(x)) ⊙ tanh(mlp6(x))
            # This operation enables counting complex substructures
            element_wise_output = (torch.tanh(self.fc11(x)) *
                                 torch.tanh(self.fc12(x)))

            # Concatenate both branches
            # Final representation combines spectral and multiplicative information
            x = torch.cat([spectral_output, element_wise_output], dim=1)
        else:
            # Only spectral convolution (still benefits from learned edge features)
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        return x

class GNNML3Model(nn.Module):
    """
    Complete GNNML3 Model.
    Full GNNML3 architecture with preprocessing and multiple layers.
    """

    def __init__(self, dim_in, dim_out, dim_hidden=64, num_layers=3, num_supports=5,
                 bandwidth=5.0, use_adjacency=False):
        super(GNNML3Model, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.num_supports = num_supports
        self.bandwidth = bandwidth
        self.use_adjacency = use_adjacency

        # Input transformation
        self.input_transform = nn.Linear(dim_in, dim_hidden)

        # GNNML3 layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            # Each layer learns edge features and has both spectral + element-wise branches
            layer = GNNML3Layer(
                learnedge=True,
                nedgeinput=num_supports,
                nedgeoutput=num_supports,
                ninp=dim_hidden,
                nout1=dim_hidden // 2,  # Spectral branch
                nout2=dim_hidden // 2  # Element-wise branch
            )
            self.gnn_layers.append(layer)

        # Output transformation
        self.output_transform = nn.Linear(dim_hidden, dim_out)

    def preprocess_graph(self, data):
        """
        Preprocessing step: compute spectral supports (Algorithm 1 from paper).

        Mathematical Background:
        1. Compute eigendecomposition: L = U * diag(λ) * U^T
        2. Create spectral supports: C'(s) = U * diag(Φs(λ)) * U^T
        3. Φs(λ) = exp(-b(λ - fs)²) creates Gaussian frequency responses
        4. Extract sparse edge features from supports
        """
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)

        # Create adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

        # Compute Laplacian or use adjacency
        if self.use_adjacency:
            # Use adjacency matrix directly
            basis_matrix = adj
        else:
            # Use normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
            degree = adj.sum(dim=1)
            degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
            degree_inv_sqrt = torch.diag(degree_inv_sqrt)
            basis_matrix = torch.eye(num_nodes, device=adj.device) - \
                           degree_inv_sqrt @ adj @ degree_inv_sqrt

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(basis_matrix)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
        except:
            # Fallback for numerical issues
            eigenvalues = torch.ones(num_nodes, device=adj.device)
            eigenvectors = torch.eye(num_nodes, device=adj.device)

        # Create receptive field mask (1-hop neighborhood + self-connections)
        mask = adj + torch.eye(num_nodes, device=adj.device)

        # Generate spectral supports
        lambda_min, lambda_max = eigenvalues.min(), eigenvalues.max()
        edge_features = []

        for s in range(self.num_supports):
            # Uniform frequency sampling
            if self.num_supports > 1:
                fs = lambda_min + (s / (self.num_supports - 1)) * (lambda_max - lambda_min)
            else:
                fs = (lambda_min + lambda_max) / 2

            # Gaussian frequency response: Φs(λ) = exp(-b(λ - fs)²)
            freq_response = torch.exp(-self.bandwidth * (eigenvalues - fs) ** 2)

            # Create spectral support: C'(s) = U * diag(Φs(λ)) * U^T
            support = eigenvectors @ torch.diag(freq_response) @ eigenvectors.T

            # Apply receptive field mask and extract edge features
            masked_support = mask * support

            # Extract edge features for current support
            edge_attr_s = masked_support[edge_index[0], edge_index[1]]
            edge_features.append(edge_attr_s)

        # Stack edge features: [num_edges, num_supports]
        edge_attr = torch.stack(edge_features, dim=1)

        return edge_attr

    def forward(self, data):
        """Forward pass of complete GNNML3 model."""
        x, edge_index = data.x, data.edge_index

        # Preprocess: compute spectral supports
        edge_attr = self.preprocess_graph(data)

        # Input transformation
        x = self.input_transform(x)

        # Apply GNNML3 layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        # Global pooling (sum aggregation)
        if hasattr(data, 'batch') and data.batch is not None:
            # Batch processing
            batch_size = data.batch.max().item() + 1
            out = torch.zeros(batch_size, x.size(1), device=x.device)
            out.index_add_(0, data.batch, x)
        else:
            # Single graph - sum all nodes
            out = x.sum(dim=0, keepdim=True)

        # Output transformation
        out = self.output_transform(out)

        return out


