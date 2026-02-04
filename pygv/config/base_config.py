"""
Base configuration class for VAMPNet training
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import yaml
import json


@dataclass
class BaseConfig:
    """Base configuration class that all configs inherit from"""

    # Dataset parameters
    traj_dir: str = None
    top: str = None
    file_pattern: str = "*.xtc"
    recursive: bool = True
    selection: str = "name CA"
    stride: int = 10
    lag_time: float = 20.0
    continuous: bool = True  # If False, time-lagged pairs won't cross trajectory boundaries

    # Graph construction
    n_neighbors: int = 4
    node_embedding_dim: int = 16
    gaussian_expansion_dim: int = 16

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.0001
    val_split: float = 0.2

    # Output and caching
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    use_cache: bool = True
    run_name: Optional[str] = None

    # Model parameters
    encoder_type: str = "schnet"
    n_states: int = 5

    # Optimization
    save_every: int = 10
    sample_validate_every: int = 5

    # Analysis
    protein_name: str = "protein"
    max_tau: Optional[int] = None

    # Hardware
    cpu: bool = False

    # Embedding parameters
    use_embedding: bool = True
    embedding_in_dim: int = None  # Will be inferred from dataset
    embedding_hidden_dim: int = 64
    embedding_out_dim: int = 32
    embedding_num_layers: int = 2
    embedding_dropout: float = 0.0
    embedding_act: str = "relu"
    embedding_norm: Optional[str] = None

    # Classifier parameters
    clf_hidden_dim: int = 64
    clf_num_layers: int = 2
    clf_dropout: float = 0.1
    clf_activation: str = "relu"
    clf_norm: Optional[str] = "batch_norm"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def to_yaml(self, path: str):
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_json(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def update(self, **kwargs):
        """Update config with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    def merge(self, other_config):
        """Merge with another config (other_config takes precedence)"""
        for key, value in other_config.to_dict().items():
            if value is not None:
                setattr(self, key, value)


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
