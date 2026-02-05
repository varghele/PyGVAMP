"""
SchNet encoder configuration
"""
from dataclasses import dataclass
from ..base_config import BaseConfig


@dataclass
class SchNetConfig(BaseConfig):
    """Configuration specific to SchNet encoder"""
    encoder_type: str = "schnet"

    # SchNet specific parameters
    node_dim: int = 16
    edge_dim: int = 16
    hidden_dim: int = 128
    output_dim: int = 64
    n_interactions: int = 3
    activation: str = "tanh"
    use_attention: bool = True
