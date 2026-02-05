"""
ML3 (GNNML3) encoder configuration
"""
from dataclasses import dataclass
from typing import Optional
from ..base_config import BaseConfig


@dataclass
class ML3Config(BaseConfig):
    """
    Configuration specific to ML3 (GNNML3) encoder.

    ML3 uses spectral convolutions with higher-order expressivity,
    featuring learned edge transformations and skip connections.
    """
    encoder_type: str = "ml3"

    # Core dimensions
    ml3_node_dim: int = 16          # Dimension of encoded node features
    ml3_edge_dim: int = 16          # Dimension of encoded edge features
    ml3_global_dim: int = 0         # Dimension of global features (0 = not used)
    ml3_hidden_dim: int = 30        # Hidden dimension for ML3 layers
    ml3_output_dim: int = 32        # Output dimension (also skip connection dim)

    # Layer configuration
    ml3_num_layers: int = 4              # Number of ML3 layers
    ml3_num_encoder_layers: int = 2      # Layers in node/edge encoder MLPs

    # Shift predictor configuration
    ml3_shift_predictor_hidden_dim: int = 32   # Hidden dim for shift predictor
    ml3_shift_predictor_layers: int = 1        # Layers in shift predictor

    # Embedding type: "node", "global", or "combined"
    ml3_embedding_type: str = "node"

    # Regularization and activation
    ml3_activation: str = "relu"         # Activation function
    ml3_norm: Optional[str] = "batch_norm"  # Normalization type
    ml3_dropout: float = 0.0             # Dropout rate
