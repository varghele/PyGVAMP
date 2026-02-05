"""
Meta encoder configuration
"""
from dataclasses import dataclass
from typing import Optional
from ..base_config import BaseConfig


@dataclass
class MetaConfig(BaseConfig):
    """
    Configuration specific to Meta encoder with attention.

    Meta uses MetaLayer from PyG with separate edge, node, and global models,
    plus optional attention mechanism for edge importance weighting.
    """
    encoder_type: str = "meta"

    # Core dimensions
    meta_node_dim: int = 16          # Node feature dimension
    meta_edge_dim: int = 16          # Edge feature dimension
    meta_global_dim: int = 16        # Global feature dimension
    meta_hidden_dim: int = 128       # Hidden dimension for MLPs
    meta_output_dim: int = 64        # Output dimension

    # MLP layer configuration
    meta_num_node_mlp_layers: int = 2     # Layers in node MLP
    meta_num_edge_mlp_layers: int = 2     # Layers in edge MLP
    meta_num_global_mlp_layers: int = 2   # Layers in global MLP
    meta_num_meta_layers: int = 3         # Number of MetaLayer blocks

    # Embedding type: "node", "global", or "combined"
    meta_embedding_type: str = "node"

    # Attention
    meta_use_attention: bool = True       # Whether to use attention mechanism

    # Regularization and activation
    meta_activation: str = "relu"         # Activation function
    meta_norm: Optional[str] = "batch_norm"  # Normalization type
    meta_dropout: float = 0.1             # Dropout rate
