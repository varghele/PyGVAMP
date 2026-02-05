"""Medium model configuration for medium-sized molecular systems (e.g., peptides, small protein domains)"""
from ..model_configs import SchNetConfig, MetaConfig, ML3Config


class MediumSchNetConfig(SchNetConfig):
    """Medium SchNet configuration for medium-sized molecular graphs"""
    # Dataset
    stride: int = 10
    batch_size: int = 32

    # Model
    hidden_dim: int = 128
    output_dim: int = 64
    n_interactions: int = 3
    n_states: int = 5

    # Training
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0001

    # Embedding
    embedding_hidden_dim: int = 64
    embedding_out_dim: int = 32

    # Classifier
    clf_hidden_dim: int = 64
    clf_num_layers: int = 2


class MediumMetaConfig(MetaConfig):
    """Medium Meta configuration for medium-sized molecular graphs"""
    # Dataset
    stride: int = 10
    batch_size: int = 32

    # Model
    meta_hidden_dim: int = 128
    meta_output_dim: int = 64
    meta_num_meta_layers: int = 3
    n_states: int = 5

    # Training
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0001

    # Embedding
    embedding_hidden_dim: int = 64
    embedding_out_dim: int = 32

    # Classifier
    clf_hidden_dim: int = 64
    clf_num_layers: int = 2


class MediumML3Config(ML3Config):
    """Medium ML3 configuration for medium-sized molecular graphs"""
    # Dataset
    stride: int = 10
    batch_size: int = 32

    # Model
    ml3_hidden_dim: int = 30
    ml3_output_dim: int = 32
    ml3_num_layers: int = 4
    n_states: int = 5

    # Training
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0001

    # Embedding
    embedding_hidden_dim: int = 64
    embedding_out_dim: int = 32

    # Classifier
    clf_hidden_dim: int = 64
    clf_num_layers: int = 2