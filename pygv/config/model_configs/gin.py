"""
GIN encoder configuration
"""
from dataclasses import dataclass
from ..base_config import BaseConfig


@dataclass
class GINConfig(BaseConfig):
    """Configuration specific to GIN encoder"""
    encoder_type: str = "gin"

    # GIN specific parameters (same semantics as SchNet)
    node_dim: int = 16
    edge_dim: int = 16
    hidden_dim: int = 128
    output_dim: int = 64
    n_interactions: int = 3
    activation: str = "tanh"
    use_attention: bool = True
