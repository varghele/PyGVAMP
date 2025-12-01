"""
Configuration management for PyGVAMP
"""
from .base_config import BaseConfig, SchNetConfig, MetaConfig, ML3Config
from .presets.small import SmallSchNetConfig, SmallMetaConfig
from .presets.medium import MediumSchNetConfig, MediumMetaConfig
from .presets.large import LargeSchNetConfig, LargeMetaConfig

# Registry of available configurations
CONFIG_REGISTRY = {
    # Base configs
    'base': BaseConfig,
    'schnet': SchNetConfig,
    'meta': MetaConfig,
    'ml3': ML3Config,

    # Small presets
    'small_schnet': SmallSchNetConfig,
    'small_meta': SmallMetaConfig,

    # Medium presets
    'medium_schnet': MediumSchNetConfig,
    'medium_meta': MediumMetaConfig,

    # Large presets
    'large_schnet': LargeSchNetConfig,
    'large_meta': LargeMetaConfig,
}


def get_config(preset_name: str, **overrides):
    """
    Get a configuration by preset name with optional overrides

    Parameters
    ----------
    preset_name : str
        Name of the preset configuration
    **overrides : dict
        Additional parameters to override in the config

    Returns
    -------
    config : BaseConfig
        Configuration object

    Examples
    --------
    >>> config = get_config('medium_schnet', epochs=150, lr=0.0005)
    >>> config = get_config('small_meta', n_states=4)
    """
    if preset_name not in CONFIG_REGISTRY:
        available = ', '.join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    config_class = CONFIG_REGISTRY[preset_name]
    config = config_class()

    # Apply overrides
    if overrides:
        config.update(**overrides)

    return config


def list_presets():
    """List all available configuration presets"""
    print("Available configuration presets:")
    print("\nBase configurations:")
    for name in ['base', 'schnet', 'meta', 'ml3']:
        if name in CONFIG_REGISTRY:
            print(f"  - {name}")

    print("\nSmall presets (quick experiments):")
    for name in CONFIG_REGISTRY:
        if name.startswith('small_'):
            print(f"  - {name}")

    print("\nMedium presets (standard training):")
    for name in CONFIG_REGISTRY:
        if name.startswith('medium_'):
            print(f"  - {name}")

    print("\nLarge presets (production training):")
    for name in CONFIG_REGISTRY:
        if name.startswith('large_'):
            print(f"  - {name}")


__all__ = [
    'BaseConfig',
    'SchNetConfig',
    'MetaConfig',
    'ML3Config',
    'get_config',
    'list_presets',
    'CONFIG_REGISTRY',
]
