"""Large model configuration for large molecular systems (e.g., proteins, protein complexes)"""
from ..model_configs import SchNetConfig, MetaConfig, ML3Config


class LargeSchNetConfig(SchNetConfig):
    """Large SchNet configuration for large molecular graphs (proteins)"""
    # Dataset
    stride: int = 5
    batch_size: int = 64

    # Model
    node_dim: int = 32
    edge_dim: int = 32
    hidden_dim: int = 256
    output_dim: int = 128
    n_interactions: int = 4
    n_states: int = 7

    # Training
    epochs: int = 200
    lr: float = 0.0005
    weight_decay: float = 0.00005

    # Embedding
    embedding_hidden_dim: int = 128
    embedding_out_dim: int = 64
    embedding_num_layers: int = 3

    # Classifier
    clf_hidden_dim: int = 128
    clf_num_layers: int = 3


class LargeMetaConfig(MetaConfig):
    """Large Meta configuration for large molecular graphs (proteins)"""
    # Dataset
    stride: int = 5
    batch_size: int = 64

    # Model
    meta_node_dim: int = 32
    meta_edge_dim: int = 32
    meta_global_dim: int = 32
    meta_hidden_dim: int = 256
    meta_output_dim: int = 128
    meta_num_meta_layers: int = 4
    n_states: int = 7

    # Training
    epochs: int = 200
    lr: float = 0.0005
    weight_decay: float = 0.00005

    # Embedding
    embedding_hidden_dim: int = 128
    embedding_out_dim: int = 64
    embedding_num_layers: int = 3

    # Classifier
    clf_hidden_dim: int = 128
    clf_num_layers: int = 3


class LargeML3Config(ML3Config):
    """Large ML3 configuration for large molecular graphs (proteins)"""
    # Dataset
    stride: int = 5
    batch_size: int = 64

    # Model
    ml3_node_dim: int = 32
    ml3_edge_dim: int = 32
    ml3_hidden_dim: int = 64
    ml3_output_dim: int = 64
    ml3_num_layers: int = 6
    ml3_num_encoder_layers: int = 3
    n_states: int = 7

    # Training
    epochs: int = 200
    lr: float = 0.0005
    weight_decay: float = 0.00005

    # Embedding
    embedding_hidden_dim: int = 128
    embedding_out_dim: int = 64
    embedding_num_layers: int = 3

    # Classifier
    clf_hidden_dim: int = 128
    clf_num_layers: int = 3