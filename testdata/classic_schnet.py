import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import List, Union, Optional

class LinearLayer:
    """
    Factory class for creating linear layers with customizable initialization and components.
    """

    @staticmethod
    def create(
            d_in: int,
            d_out: int,
            bias: bool = True,
            activation: Optional[nn.Module] = None,
            dropout: float = 0,
            weight_init: Union[str, float, int, None] = 'xavier',
            gain: float = 1.0
    ) -> List[nn.Module]:
        """
        Creates a linear layer with optional activation and dropout.

        Parameters
        ----------
        d_in : int
            Input dimension
        d_out : int
            Output dimension
        bias : bool, optional
            Whether to include bias, by default True
        activation : nn.Module, optional
            Activation function to use, by default None
        dropout : float, optional
            Dropout probability, by default 0
        weight_init : Union[str, float, int, None], optional
            Weight initialization method ('xavier', 'identity', 'kaiming')
            or constant value, by default 'xavier'
        gain : float, optional
            Gain factor for xavier initialization, by default 1.0

        Returns
        -------
        List[nn.Module]
            List of layer components

        Raises
        ------
        TypeError
            If activation is not a valid nn.Module
        ValueError
            If weight_init method is not recognized
        """
        # Create linear layer
        linear = nn.Linear(d_in, d_out, bias=bias)

        # Initialize weights
        with torch.no_grad():
            if weight_init == 'xavier':
                nn.init.xavier_uniform_(linear.weight, gain=gain)
                if bias:
                    nn.init.zeros_(linear.bias)
            elif weight_init == 'kaiming':
                nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
                if bias:
                    nn.init.zeros_(linear.bias)
            elif weight_init == 'identity':
                nn.init.eye_(linear.weight)
                if bias:
                    nn.init.zeros_(linear.bias)
            elif isinstance(weight_init, (int, float)):
                nn.init.constant_(linear.weight, weight_init)
                if bias:
                    nn.init.zeros_(linear.bias)
            elif weight_init is not None:
                raise ValueError(f"Unsupported weight initialization: {weight_init}")

        # Build layer sequence
        layers = [linear]

        # Add activation if specified
        if activation is not None:
            if not isinstance(activation, nn.Module):
                raise TypeError(f'Activation {activation} is not a valid torch.nn.Module')
            layers.append(activation)

        # Add dropout if specified
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        return layers

    @staticmethod
    def get_activation(activation_type: str) -> nn.Module:
        """
        Get activation function by name.

        Parameters
        ----------
        activation_type : str
            Name of activation function ('relu', 'tanh', 'sigmoid', etc.)

        Returns
        -------
        nn.Module
            Activation function
        """
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
        }
        if activation_type not in activations:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        return activations[activation_type]


class CFConv(nn.Module):
    """
    Continuous Filter Convolution layer as described in SchNet (Schütt et al., 2018) with optional attention mechanism.
    Attention mechanism and hyperbolic tangent come from GraphVAMPNet (Ghorbani et al., 2022).
    See also RevGraphVAMP: (Huang et al. 2024)

    This layer performs the following operations:
    1. Generates continuous filters from radial basis function expansions
    2. Applies these filters to neighboring atom features
    3. Aggregates the filtered features either through attention or summation

    Parameters
    ----------
    n_gaussians : int
        Number of Gaussian functions used in the radial basis expansion
    n_filters : int
        Number of filters to be generated and output features
    cutoff : float, optional
        Interaction cutoff distance, by default 5.0
    activation : nn.Module, optional
        Activation function for the filter generator network, by default nn.Tanh()
    use_attention : bool, optional
        Whether to use attention mechanism for feature aggregation, by default True

    Attributes
    ----------
    filter_generator : nn.Sequential
        Neural network that generates filters from RBF expansions
    nbr_filter : nn.Parameter
        Learnable parameter for attention mechanism (only if use_attention=True)
    use_attention : bool
        Flag indicating whether attention mechanism is used
    cutoff : float
        Cutoff distance for interactions
    """

    def __init__(self,
                 n_gaussians,
                 n_filters,
                 cutoff=5.0,
                 activation=nn.Tanh(),
                 normalization_layer=None,
                 use_attention=True):
        super().__init__()

        # Create filter generator using LinearLayer
        filter_layers = (
            LinearLayer.create(
                d_in=n_gaussians,
                d_out=n_filters,
                bias=True,
                activation=activation,
                weight_init='xavier'
            ) +
            LinearLayer.create(
                d_in=n_filters,
                d_out=n_filters,
                bias=True,
                weight_init='xavier'
            )
        )
        self.filter_generator = nn.Sequential(*filter_layers)

        # Initialize attention components
        self.use_attention = use_attention
        if use_attention:
            self.nbr_filter = nn.Parameter(torch.Tensor(n_filters, 1))
            nn.init.xavier_uniform_(self.nbr_filter, gain=1.414)

        self.cutoff = cutoff #NOT IMPLEMENTED YET! DOESNT DO ANYTHING

        # Add normalization layer to mirror original
        self.normalization_layer = normalization_layer

    def forward(self, features, rbf_expansion, neighbor_list):
        """
        Forward pass of the continuous filter convolution layer.

        Parameters
        ----------
        features : torch.Tensor
            Input atom features of shape [batch_size, n_atoms, n_features]
        rbf_expansion : torch.Tensor
            Radial basis function expansion of interatomic distances
            of shape [batch_size, n_atoms, n_neighbors, n_gaussians]
        neighbor_list : torch.Tensor
            Indices of neighboring atoms of shape [batch_size, n_atoms, n_neighbors]

        Returns
        -------
        tuple
            - aggregated_features : torch.Tensor
                Convolved and aggregated features of shape [batch_size, n_atoms, n_filters]
            - nbr_filter : torch.Tensor or None
                Attention weights if use_attention=True, None otherwise
        """
        # Ensure all inputs are on the same device
        device = features.device
        rbf_expansion = rbf_expansion.to(device)
        neighbor_list = neighbor_list.to(device)

        # Feature tensor needs to also be transformed from [n_frames, n_atoms, n_features]
		# to [n_frames, n_atoms, n_neighbors, n_features]
        batch_size, n_atoms, n_neighbors = neighbor_list.size()

        # Generate continuous filters from RBF expansion
        # Filter has size [n_frames, n_atoms, n_neighbors, n_features]
        #conv_filter = self.filter_generator(rbf_expansion.to(torch.float32))
        #conv_filter = conv_filter.to(device) #TODO: this is the old version! Mybe re-enable it

        rbf_norm = rbf_expansion / (rbf_expansion.norm(dim=-1, keepdim=True) + 1e-8)
        conv_filter = self.filter_generator(rbf_norm.to(torch.float32))
        conv_filter=conv_filter.to(device)

        # Ensure neighbor_list is int64 and properly shaped for gathering
        neighbor_list = neighbor_list.to(torch.int64)
        # size [n_frames, n_atoms*n_neighbors, 1]
        neighbor_list = neighbor_list.reshape(-1, n_atoms * n_neighbors, 1)
        # size [n_frames, n_atoms*n_neighbors, n_features]
        neighbor_list = neighbor_list.expand(-1, -1, features.size(2))

        # Gather features of neighboring atoms
        neighbor_features = torch.gather(features, 1, neighbor_list)
        # Reshape back to [n_frames, n_atoms, n_neighbors, n_features] for element-wise multiplication
        neighbor_features = neighbor_features.reshape(batch_size, n_atoms, n_neighbors, -1)

        # Apply continuous filters to neighbor features
        # element-wise multiplication of the features with the convolutional filter
        conv_features = neighbor_features * conv_filter

        # Aggregate features using either attention or summation
        if self.use_attention:
            # Move nbr_filter to correct device
            nbr_filter = self.nbr_filter.to(device)
            # Compute attention weights and apply them
            # [B, N, M, n_features]
            attention_weights = torch.matmul(conv_features, nbr_filter).squeeze(-1)
            # [B, N, 1, M]
            attention_weights = F.softmax(attention_weights, dim=-1)
            # [B, N, M]
            aggregated_features = torch.einsum('bij,bijc->bic', attention_weights, conv_features)

            if self.normalization_layer is not None:
                #if isinstance(self.normalization_layer, NeighborNormLayer):
                #    return self.normalization_layer(aggregated_features, n_neighbors)
                #else:
                return self.normalization_layer(aggregated_features)

            return aggregated_features, attention_weights
        else:
            # Simple summation aggregation
            aggregated_features = conv_features.sum(dim=2)
            return aggregated_features, None



class GCNInteraction(nn.Module):
    """
    Graph Convolutional Network (GCN) interaction block.

    An interaction block consists of:
    1. Atom-wise linear layer without activation
    2. Continuous filter convolution with optional attention
    3. Atom-wise linear layer with activation
    4. Atom-wise linear layer without activation

    The output forms an additive residual connection with the original input features.

    Parameters
    ----------
    n_inputs : int
        Number of input features
    n_gaussians : int
        Number of gaussians used in the radial basis function
    n_filters : int
        Number of filters for the continuous filter convolution
    activation : nn.Module, optional
        Activation function for the atom-wise layers, default=nn.Tanh()
    use_attention : bool, optional
        Whether to use attention in the CFConv layer, default=True
    """

    def __init__(self, n_inputs: int, n_gaussians: int, n_filters: int,
                 activation: nn.Module = nn.Tanh(), use_attention: bool = True):
        super().__init__()

        # Initial atom-wise layer without activation
        self.initial_dense = nn.Sequential(
            *LinearLayer.create(
                d_in=n_inputs,
                d_out=n_filters,
                bias=False,
                activation=None
            )
        )

        # Continuous filter convolution
        self.cfconv = CFConv(
            n_gaussians=n_gaussians,
            n_filters=n_filters,
            activation=activation,
            use_attention=use_attention
        )

        # Output layers
        output_layers = (
            LinearLayer.create(
                d_in=n_filters,
                d_out=n_filters,
                bias=True,
                activation=activation
            ) +
            LinearLayer.create(
                d_in=n_filters,
                d_out=n_filters,
                bias=True
            )
        )
        self.output_dense = nn.Sequential(*output_layers)

    def forward(self, features: torch.Tensor, rbf_expansion: torch.Tensor,
                neighbor_list: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute interaction block forward pass.

        Parameters
        ----------
        features : torch.Tensor
            Input features from embedding or interaction layer [batch_size, n_atoms, n_features]
        rbf_expansion : torch.Tensor
            Radial basis function expansion of distances [batch_size, n_atoms, n_neighbors, n_gaussians]
        neighbor_list : torch.Tensor
            Indices of neighboring atoms [batch_size, n_atoms, n_neighbors]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - output_features: Output features [batch_size, n_atoms, n_filters]
            - attention: Attention weights from CFConv layer (None if use_attention=False)
        """
        # Initial dense layer
        init_features = self.initial_dense(features)

        # Continuous filter convolution
        conv_output, attention = self.cfconv(init_features, rbf_expansion, neighbor_list)

        # Output dense layers
        output_features = self.output_dense(conv_output)

        return output_features, attention
