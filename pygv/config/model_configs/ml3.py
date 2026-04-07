"""
ML3 encoder configuration
"""
from dataclasses import dataclass
from typing import Optional
from ..base_config import BaseConfig


@dataclass
class ML3Config(BaseConfig):
    """
    Configuration specific to ML3 encoder.

    ML3 uses spectral convolutions with higher-order (3-WL) expressivity,
    featuring learned edge transformations, skip connections, and optional
    parallel attention.
    """
    encoder_type: str = "ml3"

    # Core dimensions
    ml3_node_dim: int = 16          # Dimension of input node features
    ml3_edge_dim: int = 16          # Dimension of input edge features
    ml3_hidden_dim: int = 30        # Hidden dimension for ML3 layers
    ml3_output_dim: int = 32        # Output dimension (graph-level)

    # Layer configuration
    ml3_num_layers: int = 4         # Number of ML3 interaction layers
    ml3_nout1: int = 30             # Convolution output dim in ML3Layer
    ml3_nout2: int = 2              # Skip connection output dim (0 to disable)

    # Attention
    ml3_use_attention: bool = True  # Whether to use parallel attention

    # Edge feature mode
    ml3_edge_mode: str = "gaussian"  # 'gaussian' or 'spectral'
    ml3_nfreq: int = 10             # Number of spectral frequencies (spectral mode)
    ml3_spectral_dv: float = 1.0    # Gaussian width for spectral filters
    ml3_recfield: int = 1           # Receptive field for spectral filters

    # Regularization and activation
    ml3_activation: str = "relu"         # Activation function
    ml3_dropout: float = 0.0             # Dropout rate
