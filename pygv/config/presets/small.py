"""Small model configuration for small molecular systems (e.g., small molecules, ligands)"""
from ..base_config import SchNetConfig, MetaConfig, ML3Config


class SmallSchNetConfig(SchNetConfig):
    """Small SchNet configuration for small molecular graphs"""
    # Dataset
    stride: int = 20
    batch_size: int = 16

    # Model
    hidden_dim: int = 64
    output_dim: int = 32
    n_interactions: int = 2
    n_states: int = 3

    # Training
    epochs: int = 50
    lr: float = 0.001

    # Embedding
    embedding_hidden_dim: int = 32
    embedding_out_dim: int = 16


class SmallMetaConfig(MetaConfig):
    """Small Meta configuration for small molecular graphs"""
    # Dataset
    stride: int = 20
    batch_size: int = 16

    # Model
    meta_hidden_dim: int = 64
    meta_output_dim: int = 32
    meta_num_meta_layers: int = 2
    n_states: int = 3

    # Training
    epochs: int = 50
    lr: float = 0.001

    # Embedding
    embedding_hidden_dim: int = 32
    embedding_out_dim: int = 16


class SmallML3Config(ML3Config):
    """Small ML3 configuration for small molecular graphs"""
    # Dataset
    stride: int = 20
    batch_size: int = 16

    # Model
    ml3_hidden_dim: int = 20
    ml3_output_dim: int = 16
    ml3_num_layers: int = 2
    n_states: int = 3

    # Training
    epochs: int = 50
    lr: float = 0.001

    # Embedding
    embedding_hidden_dim: int = 32
    embedding_out_dim: int = 16
