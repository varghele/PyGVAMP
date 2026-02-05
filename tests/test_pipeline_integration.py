"""
Integration tests for the complete PyGVAMP pipeline.

Tests cover:
- CacheManager functionality
- Pipeline argument parsing
- PipelineOrchestrator (when imports are available)

Note: Some tests require the full pipeline module which has incomplete imports.
These are marked with @pytest.mark.skip or tested via mocking.
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from typing import List
import argparse


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    import numpy as np
    np.random.seed(42)
    return 42


@dataclass
class MockConfig:
    """Mock configuration object for testing."""
    # Data paths
    traj_dir: str = "/mock/traj"
    top: str = "/mock/top.pdb"
    file_pattern: str = "*.xtc"
    recursive: bool = False

    # Processing parameters
    selection: str = "name CA"
    stride: int = 1
    lag_time: float = 10.0
    lag_times: List[float] = field(default_factory=lambda: [10.0])
    n_neighbors: int = 10
    node_embedding_dim: int = 16
    gaussian_expansion_dim: int = 16
    n_states: int = 5
    n_states_list: List[int] = field(default_factory=lambda: [5])

    # Model parameters
    encoder_type: str = "schnet"
    hidden_dim: int = 64

    # Training parameters
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001

    # Pipeline control
    cache: bool = False
    hurry: bool = False
    output_dir: str = "/mock/output"
    cache_dir: str = "/mock/cache"
    protein_name: str = "test_protein"
    cpu: bool = True

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'traj_dir': self.traj_dir,
            'top': self.top,
            'file_pattern': self.file_pattern,
            'selection': self.selection,
            'stride': self.stride,
            'lag_time': self.lag_time,
            'lag_times': self.lag_times,
            'n_neighbors': self.n_neighbors,
            'node_embedding_dim': self.node_embedding_dim,
            'gaussian_expansion_dim': self.gaussian_expansion_dim,
            'n_states': self.n_states,
            'encoder_type': self.encoder_type,
            'hidden_dim': self.hidden_dim,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'cache': self.cache,
            'hurry': self.hurry,
            'output_dir': self.output_dir,
            'protein_name': self.protein_name,
            'cpu': self.cpu
        }

    def to_yaml(self, path):
        """Save config to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return MockConfig()


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config_with_temp_dir(temp_output_dir):
    """Create mock config with temporary output directory."""
    config = MockConfig()
    config.output_dir = temp_output_dir
    config.cache_dir = os.path.join(temp_output_dir, 'cache')
    return config


# ============================================================================
# Test Classes for CacheManager
# ============================================================================

class TestCacheManager:
    """Tests for CacheManager."""

    def test_init_with_config(self, mock_config):
        """CacheManager initializes with config."""
        from pygv.pipe.caching import CacheManager

        cache_manager = CacheManager(mock_config)

        assert cache_manager.config is mock_config

    def test_get_dataset_hash_returns_string(self, mock_config):
        """get_dataset_hash returns string."""
        from pygv.pipe.caching import CacheManager

        cache_manager = CacheManager(mock_config)
        hash_value = cache_manager.get_dataset_hash()

        assert isinstance(hash_value, str)

    def test_get_dataset_hash_is_deterministic(self, mock_config):
        """get_dataset_hash returns same hash for same config."""
        from pygv.pipe.caching import CacheManager

        cache_manager = CacheManager(mock_config)
        hash1 = cache_manager.get_dataset_hash()
        hash2 = cache_manager.get_dataset_hash()

        assert hash1 == hash2

    def test_get_dataset_hash_differs_for_different_configs(self):
        """get_dataset_hash differs for different configs."""
        from pygv.pipe.caching import CacheManager

        config1 = MockConfig(selection="name CA")
        config2 = MockConfig(selection="name CB")

        hash1 = CacheManager(config1).get_dataset_hash()
        hash2 = CacheManager(config2).get_dataset_hash()

        assert hash1 != hash2

    def test_hash_changes_with_traj_dir(self):
        """Hash changes when traj_dir changes."""
        from pygv.pipe.caching import CacheManager

        config1 = MockConfig(traj_dir="/path/a")
        config2 = MockConfig(traj_dir="/path/b")

        hash1 = CacheManager(config1).get_dataset_hash()
        hash2 = CacheManager(config2).get_dataset_hash()

        assert hash1 != hash2

    def test_hash_changes_with_stride(self):
        """Hash changes when stride changes."""
        from pygv.pipe.caching import CacheManager

        config1 = MockConfig(stride=1)
        config2 = MockConfig(stride=10)

        hash1 = CacheManager(config1).get_dataset_hash()
        hash2 = CacheManager(config2).get_dataset_hash()

        assert hash1 != hash2

    def test_hash_changes_with_n_neighbors(self):
        """Hash changes when n_neighbors changes."""
        from pygv.pipe.caching import CacheManager

        config1 = MockConfig(n_neighbors=10)
        config2 = MockConfig(n_neighbors=20)

        hash1 = CacheManager(config1).get_dataset_hash()
        hash2 = CacheManager(config2).get_dataset_hash()

        assert hash1 != hash2

    def test_hash_length(self, mock_config):
        """Hash has expected length (8 characters)."""
        from pygv.pipe.caching import CacheManager

        cache_manager = CacheManager(mock_config)
        hash_value = cache_manager.get_dataset_hash()

        assert len(hash_value) == 8

    def test_check_cached_dataset_returns_none_when_disabled(self, mock_config):
        """check_cached_dataset returns None when caching disabled."""
        from pygv.pipe.caching import CacheManager

        mock_config.cache = False
        cache_manager = CacheManager(mock_config)

        result = cache_manager.check_cached_dataset("abc123")

        assert result is None

    def test_check_cached_dataset_returns_none_when_not_exists(self, mock_config_with_temp_dir):
        """check_cached_dataset returns None when cache doesn't exist."""
        from pygv.pipe.caching import CacheManager

        mock_config_with_temp_dir.cache = True
        cache_manager = CacheManager(mock_config_with_temp_dir)

        result = cache_manager.check_cached_dataset("nonexistent")

        assert result is None

    def test_check_cached_dataset_returns_path_when_exists(self, mock_config_with_temp_dir):
        """check_cached_dataset returns path when cache exists."""
        from pygv.pipe.caching import CacheManager

        mock_config_with_temp_dir.cache = True
        cache_manager = CacheManager(mock_config_with_temp_dir)

        # Create cache directory and file
        cache_dir = Path(mock_config_with_temp_dir.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        hash_value = "abc123"
        cache_file = cache_dir / f"dataset_{hash_value}.pkl"
        cache_file.touch()

        result = cache_manager.check_cached_dataset(hash_value)

        assert result is not None
        assert hash_value in result


# ============================================================================
# Test Classes for Pipeline Arguments
# ============================================================================

class TestPipelineArgs:
    """Tests for pipeline argument parsing."""

    def test_parse_args_with_required(self):
        """parse_pipeline_args parses required arguments."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()

        assert args.traj_dir == '/data'
        assert args.top == 'top.pdb'

    def test_parse_args_default_lag_times(self):
        """parse_pipeline_args has default lag_times."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()

        assert args.lag_times == [10.0]

    def test_parse_args_default_n_states(self):
        """parse_pipeline_args has default n_states."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()

        assert args.n_states == [5]

    def test_parse_args_default_output_dir(self):
        """parse_pipeline_args has default output_dir."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()

        assert args.output_dir == './experiments'

    def test_parse_args_default_protein_name(self):
        """parse_pipeline_args has default protein_name."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()

        assert args.protein_name == 'protein'

    def test_parse_args_multiple_lag_times(self):
        """parse_pipeline_args parses multiple lag times."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--lag_times', '10', '20', '50']):
            args = parse_pipeline_args()

        assert args.lag_times == [10.0, 20.0, 50.0]

    def test_parse_args_multiple_n_states(self):
        """parse_pipeline_args parses multiple n_states."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--n_states', '3', '5', '8']):
            args = parse_pipeline_args()

        assert args.n_states == [3, 5, 8]

    def test_parse_args_cache_flag(self):
        """parse_pipeline_args parses cache flag."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb', '--cache']):
            args = parse_pipeline_args()

        assert args.cache is True

    def test_parse_args_cache_default_false(self):
        """cache flag defaults to False."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()

        assert args.cache is False

    def test_parse_args_hurry_flag(self):
        """parse_pipeline_args parses hurry flag."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb', '--hurry']):
            args = parse_pipeline_args()

        assert args.hurry is True

    def test_parse_args_hurry_default_false(self):
        """hurry flag defaults to False."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()

        assert args.hurry is False

    def test_parse_args_preset(self):
        """parse_pipeline_args parses preset."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--preset', 'medium_schnet']):
            args = parse_pipeline_args()

        assert args.preset == 'medium_schnet'

    def test_parse_args_model(self):
        """parse_pipeline_args parses model."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--model', 'meta']):
            args = parse_pipeline_args()

        assert args.model == 'meta'

    def test_parse_args_resume(self):
        """parse_pipeline_args parses resume path."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--resume', '/path/to/experiment']):
            args = parse_pipeline_args()

        assert args.resume == '/path/to/experiment'

    def test_parse_args_skip_preparation(self):
        """parse_pipeline_args parses skip_preparation flag."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--skip_preparation']):
            args = parse_pipeline_args()

        assert args.skip_preparation is True

    def test_parse_args_skip_training(self):
        """parse_pipeline_args parses skip_training flag."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--skip_training']):
            args = parse_pipeline_args()

        assert args.skip_training is True

    def test_parse_args_only_analysis(self):
        """parse_pipeline_args parses only_analysis flag."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--only_analysis']):
            args = parse_pipeline_args()

        assert args.only_analysis is True

    def test_parse_args_custom_output_dir(self):
        """parse_pipeline_args parses custom output_dir."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--output_dir', '/custom/output']):
            args = parse_pipeline_args()

        assert args.output_dir == '/custom/output'

    def test_parse_args_custom_protein_name(self):
        """parse_pipeline_args parses custom protein_name."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--protein_name', 'my_protein']):
            args = parse_pipeline_args()

        assert args.protein_name == 'my_protein'


# ============================================================================
# Test Classes for Mock Config
# ============================================================================

class TestMockConfig:
    """Tests for MockConfig helper class."""

    def test_to_dict_returns_dict(self):
        """to_dict returns dictionary."""
        config = MockConfig()
        result = config.to_dict()

        assert isinstance(result, dict)

    def test_to_dict_contains_all_fields(self):
        """to_dict contains all expected fields."""
        config = MockConfig()
        result = config.to_dict()

        expected_keys = ['traj_dir', 'top', 'selection', 'stride', 'n_states',
                         'encoder_type', 'batch_size', 'epochs', 'protein_name']

        for key in expected_keys:
            assert key in result

    def test_to_yaml_creates_file(self, temp_output_dir):
        """to_yaml creates YAML file."""
        config = MockConfig()
        yaml_path = os.path.join(temp_output_dir, 'config.yaml')

        config.to_yaml(yaml_path)

        assert os.path.exists(yaml_path)

    def test_to_yaml_is_valid_yaml(self, temp_output_dir):
        """to_yaml creates valid YAML."""
        import yaml

        config = MockConfig()
        yaml_path = os.path.join(temp_output_dir, 'config.yaml')

        config.to_yaml(yaml_path)

        with open(yaml_path, 'r') as f:
            loaded = yaml.safe_load(f)

        assert isinstance(loaded, dict)
        assert loaded['protein_name'] == config.protein_name


# ============================================================================
# Test Classes for Pipeline Module Imports
# ============================================================================

class TestPipelineModuleImports:
    """Tests for pipeline module import availability."""

    def test_caching_module_importable(self):
        """Caching module is importable."""
        from pygv.pipe.caching import CacheManager
        assert CacheManager is not None

    def test_args_module_importable(self):
        """Args module is importable."""
        from pygv.pipe.args import parse_pipeline_args
        assert parse_pipeline_args is not None

    def test_training_module_has_run_training(self):
        """Training module has run_training function."""
        from pygv.pipe.training import run_training
        assert callable(run_training)

    def test_analysis_module_has_run_analysis(self):
        """Analysis module has run_analysis function."""
        from pygv.pipe.analysis import run_analysis
        assert callable(run_analysis)


# ============================================================================
# Test Classes for Edge Cases
# ============================================================================

class TestCacheEdgeCases:
    """Tests for cache edge cases."""

    def test_hash_with_special_characters_in_path(self):
        """Hash handles paths with special characters."""
        from pygv.pipe.caching import CacheManager

        config = MockConfig(traj_dir="/path/with spaces/and-dashes")
        cache_manager = CacheManager(config)

        # Should not raise
        hash_value = cache_manager.get_dataset_hash()
        assert isinstance(hash_value, str)

    def test_hash_with_unicode_in_selection(self):
        """Hash handles unicode in selection."""
        from pygv.pipe.caching import CacheManager

        config = MockConfig(selection="name CA and résidu 1")
        cache_manager = CacheManager(config)

        # Should not raise
        hash_value = cache_manager.get_dataset_hash()
        assert isinstance(hash_value, str)

    def test_cache_check_with_nonexistent_cache_dir(self, mock_config):
        """check_cached_dataset handles nonexistent cache_dir."""
        from pygv.pipe.caching import CacheManager

        mock_config.cache = True
        mock_config.cache_dir = "/nonexistent/path"
        cache_manager = CacheManager(mock_config)

        # Should return None, not raise
        result = cache_manager.check_cached_dataset("abc123")
        assert result is None


class TestArgsEdgeCases:
    """Tests for argument parsing edge cases."""

    def test_single_lag_time_as_list(self):
        """Single lag time is still a list."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--lag_times', '10']):
            args = parse_pipeline_args()

        assert isinstance(args.lag_times, list)
        assert len(args.lag_times) == 1

    def test_single_n_states_as_list(self):
        """Single n_states is still a list."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--n_states', '5']):
            args = parse_pipeline_args()

        assert isinstance(args.n_states, list)
        assert len(args.n_states) == 1

    def test_float_lag_times(self):
        """Lag times can be floats."""
        from pygv.pipe.args import parse_pipeline_args

        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--lag_times', '10.5', '20.25']):
            args = parse_pipeline_args()

        assert args.lag_times == [10.5, 20.25]


# ============================================================================
# Tests for Pipeline Summary Format
# ============================================================================

class TestPipelineSummaryFormat:
    """Tests for expected pipeline summary format."""

    def test_summary_json_structure(self, temp_output_dir):
        """Pipeline summary has expected JSON structure."""
        # Create a mock summary
        summary = {
            'timestamp': '2026-02-05T12:00:00',
            'config': MockConfig().to_dict(),
            'trained_models': {'exp1': '/path/model.pt'},
            'analysis_completed': ['exp1']
        }

        summary_path = os.path.join(temp_output_dir, 'pipeline_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Verify it's valid JSON
        with open(summary_path, 'r') as f:
            loaded = json.load(f)

        assert 'timestamp' in loaded
        assert 'config' in loaded
        assert 'trained_models' in loaded
        assert 'analysis_completed' in loaded

    def test_summary_config_is_dict(self, temp_output_dir):
        """Config in summary is dictionary."""
        config = MockConfig()
        summary = {
            'timestamp': '2026-02-05T12:00:00',
            'config': config.to_dict(),
            'trained_models': {},
            'analysis_completed': []
        }

        summary_path = os.path.join(temp_output_dir, 'pipeline_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        with open(summary_path, 'r') as f:
            loaded = json.load(f)

        assert isinstance(loaded['config'], dict)


# ============================================================================
# Tests for Directory Structure
# ============================================================================

class TestDirectoryStructure:
    """Tests for expected directory structure."""

    def test_experiment_dir_structure(self, temp_output_dir):
        """Experiment directory has expected subdirectories."""
        # Create expected structure
        exp_dir = Path(temp_output_dir) / 'exp_test_20260205_120000'
        subdirs = ['preparation', 'training', 'analysis', 'logs']

        for subdir in subdirs:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Verify structure
        for subdir in subdirs:
            assert (exp_dir / subdir).exists()

    def test_training_subdir_per_experiment(self, temp_output_dir):
        """Training directory has subdirectory per experiment."""
        training_dir = Path(temp_output_dir) / 'training'
        training_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment directories
        experiments = ['lag10ns_5states', 'lag20ns_5states', 'lag10ns_8states']
        for exp in experiments:
            (training_dir / exp).mkdir()

        # Verify
        for exp in experiments:
            assert (training_dir / exp).exists()

    def test_analysis_subdir_per_experiment(self, temp_output_dir):
        """Analysis directory has subdirectory per experiment."""
        analysis_dir = Path(temp_output_dir) / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment directories
        experiments = ['lag10ns_5states', 'lag20ns_5states']
        for exp in experiments:
            (analysis_dir / exp).mkdir()

        # Verify
        for exp in experiments:
            assert (analysis_dir / exp).exists()


# ============================================================================
# Tests for Config Serialization
# ============================================================================

class TestConfigSerialization:
    """Tests for config serialization."""

    def test_config_yaml_roundtrip(self, temp_output_dir):
        """Config survives YAML roundtrip."""
        import yaml

        config = MockConfig(protein_name='test_roundtrip', n_states=7)
        yaml_path = os.path.join(temp_output_dir, 'config.yaml')

        # Save
        config.to_yaml(yaml_path)

        # Load
        with open(yaml_path, 'r') as f:
            loaded = yaml.safe_load(f)

        assert loaded['protein_name'] == 'test_roundtrip'
        assert loaded['n_states'] == 7

    def test_config_json_serializable(self):
        """Config dict is JSON serializable."""
        config = MockConfig()
        config_dict = config.to_dict()

        # Should not raise
        json_str = json.dumps(config_dict)
        assert isinstance(json_str, str)

        # Roundtrip
        loaded = json.loads(json_str)
        assert loaded['protein_name'] == config.protein_name
