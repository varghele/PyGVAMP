"""
Integration tests for the PyGVAMP pipeline.

Part 1: Unit tests for args, config, directory structure (fast, no data needed)
Part 2: End-to-end tests with real MD trajectory data (marked @pytest.mark.integration)

End-to-end tests require trajectory data at /home/iwe81/vi/PYGVAMP/datasets/.
To skip them:  pytest tests/ -m "not integration"
To run them:   pytest tests/test_pipeline_integration.py -v --tb=short

Run with: pytest tests/test_pipeline_integration.py -v
"""

import json
import os
import pytest
import sys
from pathlib import Path
from unittest.mock import patch
import argparse


# ============================================================================
# Test data paths for integration tests
# ============================================================================

AB42_TRAJ_DIR = "/home/iwe81/vi/PYGVAMP/datasets/ab42/trajectories/red/r1"
AB42_TOPOLOGY = "/home/iwe81/vi/PYGVAMP/datasets/ab42/trajectories/red/topol.pdb"

NTL9_TRAJ_DIR = "/home/iwe81/vi/PYGVAMP/datasets/NTL9/DESRES-Trajectory_NTL9-0-c-alpha/NTL9-0-c-alpha"
NTL9_TOPOLOGY = "/home/iwe81/vi/PYGVAMP/datasets/NTL9/DESRES-Trajectory_NTL9-0-c-alpha/NTL9-0-c-alpha/NTL9.pdb"

DATA_AVAILABLE = os.path.isfile(AB42_TOPOLOGY)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    return str(tmp_path)


# ============================================================================
# Part 1: Unit tests (fast, no data needed)
# ============================================================================

class TestPipelineArgs:
    """Tests for pipeline argument parsing."""

    def test_parse_args_with_required(self):
        from pygv.pipe.args import parse_pipeline_args
        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()
        assert args.traj_dir == '/data'
        assert args.top == 'top.pdb'

    def test_parse_args_default_lag_times(self):
        from pygv.pipe.args import parse_pipeline_args
        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()
        assert args.lag_times == [10.0]

    def test_parse_args_default_n_states(self):
        from pygv.pipe.args import parse_pipeline_args
        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb']):
            args = parse_pipeline_args()
        assert args.n_states is None

    def test_parse_args_multiple_lag_times(self):
        from pygv.pipe.args import parse_pipeline_args
        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--lag_times', '10', '20', '50']):
            args = parse_pipeline_args()
        assert args.lag_times == [10.0, 20.0, 50.0]

    def test_parse_args_multiple_n_states(self):
        from pygv.pipe.args import parse_pipeline_args
        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--n_states', '3', '5', '8']):
            args = parse_pipeline_args()
        assert args.n_states == [3, 5, 8]

    def test_parse_args_flags(self):
        from pygv.pipe.args import parse_pipeline_args
        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--cache', '--hurry', '--cpu',
                                '--skip_preparation', '--skip_training']):
            args = parse_pipeline_args()
        assert args.cache is True
        assert args.hurry is True
        assert args.cpu is True
        assert args.skip_preparation is True
        assert args.skip_training is True

    def test_parse_args_preset(self):
        from pygv.pipe.args import parse_pipeline_args
        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--preset', 'medium_schnet']):
            args = parse_pipeline_args()
        assert args.preset == 'medium_schnet'

    def test_parse_args_resume(self):
        from pygv.pipe.args import parse_pipeline_args
        with patch('sys.argv', ['prog', '--traj_dir', '/data', '--top', 'top.pdb',
                                '--resume', '/path/to/experiment']):
            args = parse_pipeline_args()
        assert args.resume == '/path/to/experiment'


class TestPipelineModuleImports:
    """Verify all pipeline modules are importable."""

    def test_args_module(self):
        from pygv.pipe.args import parse_pipeline_args
        assert callable(parse_pipeline_args)

    def test_training_module(self):
        from pygv.pipe.training import run_training
        assert callable(run_training)

    def test_analysis_module(self):
        from pygv.pipe.analysis import run_analysis
        assert callable(run_analysis)

    def test_preparation_module(self):
        from pygv.pipe.preparation import run_preparation
        assert callable(run_preparation)

    def test_master_pipeline_module(self):
        from pygv.pipe.master_pipeline import PipelineOrchestrator
        assert PipelineOrchestrator is not None


class TestTopologyValidation:
    """Test topology file validation."""

    def test_valid_pdb_extension(self):
        from pygv.pipe.master_pipeline import validate_topology_file
        # Should not raise for .pdb
        validate_topology_file("/fake/path/topology.pdb")

    def test_valid_gro_extension(self):
        from pygv.pipe.master_pipeline import validate_topology_file
        validate_topology_file("/fake/path/topology.gro")

    def test_invalid_psf_extension(self):
        from pygv.pipe.master_pipeline import validate_topology_file
        with pytest.raises(SystemExit):
            validate_topology_file("/fake/path/topology.psf")

    def test_invalid_prmtop_extension(self):
        from pygv.pipe.master_pipeline import validate_topology_file
        with pytest.raises(SystemExit):
            validate_topology_file("/fake/path/topology.prmtop")


class TestDirectoryStructure:
    """Test PipelineOrchestrator creates expected directory structure."""

    def test_experiment_dir_structure(self, tmp_path):
        from pygv.config import BaseConfig
        from pygv.pipe.master_pipeline import PipelineOrchestrator

        config = BaseConfig()
        config.output_dir = str(tmp_path)
        config.protein_name = "test"
        config.cache = False

        orchestrator = PipelineOrchestrator(config)
        dirs = orchestrator.setup_experiment_directory()

        assert dirs['preparation'].exists()
        assert dirs['training'].exists()
        assert dirs['analysis'].exists()
        assert dirs['logs'].exists()
        assert (orchestrator.experiment_dir / 'config.yaml').exists()


class TestDryRun:
    """Test dry run mode."""

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="AB42 test data not found")
    def test_dry_run_prints_summary(self, tmp_path, capsys):
        from pygv.pipe.master_pipeline import _print_dry_run_summary
        from pygv.config import BaseConfig

        config = BaseConfig()
        config.traj_dir = AB42_TRAJ_DIR
        config.top = AB42_TOPOLOGY
        config.file_pattern = "traj0000.xtc"
        config.output_dir = str(tmp_path)
        config.protein_name = "ab42_test"
        config.encoder_type = "schnet"
        config.lag_times = [2.5]
        config.n_states = 2
        config.cache = False
        config.hurry = False
        config.cpu = True

        args = argparse.Namespace(
            only_analysis=False, skip_preparation=False,
            skip_training=False, resume=None,
        )
        _print_dry_run_summary(config, args)

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "schnet" in captured.out
        assert "No actions taken" in captured.out


# ============================================================================
# Part 2: End-to-end integration tests (require real data)
# ============================================================================

def _make_config(tmp_path, traj_dir, top, file_pattern="*.xtc", selection="name CA",
                 stride=10, lag_time=2.5, encoder_type="schnet", n_states=2,
                 epochs=3, timestep=None, continuous=True):
    """Create a minimal config for fast integration testing.

    Uses the proper encoder-specific config classes (SchNetConfig, GINConfig,
    ML3Config) so that to_dict() includes all encoder fields.
    """
    from pygv.config.base_config import SchNetConfig, GINConfig, ML3Config

    # Pick the right config class for the encoder
    config_cls = {
        "schnet": SchNetConfig,
        "gin": GINConfig,
        "ml3": ML3Config,
    }.get(encoder_type, SchNetConfig)

    config = config_cls()

    # Data
    config.traj_dir = traj_dir
    config.top = top
    config.file_pattern = file_pattern
    config.recursive = False
    config.selection = selection
    config.stride = stride
    config.lag_time = lag_time
    config.lag_times = [lag_time]
    config.continuous = continuous
    if timestep is not None:
        config.timestep = timestep

    # Graph — small dims
    config.n_neighbors = 4
    config.node_embedding_dim = 16
    config.gaussian_expansion_dim = 16

    # Training — minimal
    config.epochs = epochs
    config.batch_size = 4
    config.lr = 0.001
    config.weight_decay = 0.0
    config.val_split = 0.2
    config.clip_grad = 1.0
    config.save_every = 100
    config.sample_validate_every = 100
    config.cpu = True

    # Model
    config.n_states = n_states

    # Tiny encoder dims (SchNet/GIN share the same field names)
    if encoder_type in ("schnet", "gin"):
        config.node_dim = 16
        config.edge_dim = 16
        config.hidden_dim = 8
        config.output_dim = 4
        config.n_interactions = 2
        config.activation = "relu"
        config.use_attention = False
    elif encoder_type == "ml3":
        # node_dim/edge_dim needed by run_training() for dataset dim inference
        config.node_dim = 16
        config.edge_dim = 16
        config.ml3_node_dim = 16
        config.ml3_edge_dim = 16
        config.ml3_hidden_dim = 8
        config.ml3_output_dim = 4
        config.ml3_num_layers = 2
        config.ml3_activation = "relu"
        config.ml3_dropout = 0.0
        config.ml3_use_attention = False
        config.ml3_edge_mode = "gaussian"
        config.ml3_nfreq = 5
        config.ml3_spectral_dv = 1.0
        config.ml3_recfield = 1
        config.ml3_nout1 = 8
        config.ml3_nout2 = 2

    # Embedding
    config.use_embedding = True
    config.embedding_in_dim = None
    config.embedding_hidden_dim = 16
    config.embedding_out_dim = 16
    config.embedding_num_layers = 2
    config.embedding_dropout = 0.0
    config.embedding_act = "relu"
    config.embedding_norm = None

    # Classifier (no BatchNorm — crashes on single-sample last batches with tiny data)
    config.clf_hidden_dim = 16
    config.clf_num_layers = 2
    config.clf_dropout = 0.0
    config.clf_activation = "relu"
    config.clf_norm = None

    # Output
    config.output_dir = str(tmp_path)
    config.cache = False
    config.hurry = False
    config.protein_name = "test_protein"
    config.run_name = None

    # State discovery — disabled for speed
    config.discover_states = False
    config.g2v_embedding_dim = 32
    config.g2v_max_degree = 2
    config.g2v_epochs = 5
    config.g2v_min_count = 1
    config.g2v_min_count_decay = None
    config.g2v_umap_dim = [2]
    config.min_states = 2
    config.max_states = 5

    # State diagnostics
    config.population_threshold = 0.02
    config.jsd_threshold = 0.05

    # Analysis
    config.max_tau = None

    return config


def _assert_preparation_outputs(prep_dir):
    """Verify preparation phase produced expected outputs."""
    subdirs = [d for d in os.listdir(prep_dir) if os.path.isdir(os.path.join(prep_dir, d))]
    assert len(subdirs) >= 1, f"Expected run dir in {prep_dir}, got {subdirs}"
    run_dir = os.path.join(prep_dir, sorted(subdirs)[-1])

    assert os.path.isfile(os.path.join(run_dir, "dataset_stats.json")), \
        "dataset_stats.json not found"
    assert os.path.isfile(os.path.join(run_dir, "topology.pdb")), \
        "topology.pdb not found"

    with open(os.path.join(run_dir, "dataset_stats.json")) as f:
        stats = json.load(f)
    assert isinstance(stats, dict), "dataset_stats.json should be a dict"
    return run_dir


def _assert_training_outputs(training_dir):
    """Verify training phase produced best_model.pt."""
    subdirs = [d for d in os.listdir(training_dir)
               if os.path.isdir(os.path.join(training_dir, d))]
    assert len(subdirs) >= 1, f"No experiment dirs in {training_dir}"

    for exp_name in subdirs:
        exp_dir = os.path.join(training_dir, exp_name)
        model_found = False
        for root, dirs, files in os.walk(exp_dir):
            if "best_model.pt" in files:
                model_found = True
                break
        assert model_found, f"best_model.pt not found under {exp_dir}"


def _assert_analysis_outputs(analysis_dir):
    """Verify analysis phase produced diagnostic outputs."""
    subdirs = [d for d in os.listdir(analysis_dir)
               if os.path.isdir(os.path.join(analysis_dir, d))]
    assert len(subdirs) >= 1, f"No analysis dirs in {analysis_dir}"

    for exp_name in subdirs:
        exp_dir = os.path.join(analysis_dir, exp_name)
        files = os.listdir(exp_dir)
        has_output = any("diagnostic" in f or f.endswith(".png") for f in files)
        assert has_output, f"No outputs in {exp_dir}, files: {files}"


def _assert_pipeline_summary(experiment_dir):
    """Verify pipeline_summary.json exists and is valid."""
    summary_path = os.path.join(experiment_dir, "pipeline_summary.json")
    assert os.path.isfile(summary_path), "pipeline_summary.json not found"

    with open(summary_path) as f:
        summary = json.load(f)
    assert "timestamp" in summary
    assert "config" in summary
    assert "trained_models" in summary
    assert len(summary["trained_models"]) >= 1, "No trained models in summary"


@pytest.mark.integration
@pytest.mark.skipif(not DATA_AVAILABLE, reason="AB42 test data not found")
class TestFullPipeline:
    """End-to-end pipeline tests with real trajectory data."""

    def test_full_pipeline_schnet_xtc(self, tmp_path):
        """
        Full pipeline: AB42 .xtc data + SchNet encoder.
        3 files × 252 frames, stride=10 → ~75 frames, lag=2.5ns, 3 epochs.
        """
        from pygv.pipe.master_pipeline import PipelineOrchestrator

        config = _make_config(
            tmp_path, traj_dir=AB42_TRAJ_DIR, top=AB42_TOPOLOGY,
            file_pattern="traj000[0-9].xtc", selection="name CA",
            stride=10, lag_time=2.5, encoder_type="schnet", epochs=3,
        )

        orchestrator = PipelineOrchestrator(config)
        results = orchestrator.run_complete_pipeline()

        assert results is not None, "Pipeline returned None"
        assert len(results["trained_models"]) >= 1

        exp_dir = str(results["experiment_dir"])
        _assert_preparation_outputs(os.path.join(exp_dir, "preparation"))
        _assert_training_outputs(os.path.join(exp_dir, "training"))
        _assert_analysis_outputs(os.path.join(exp_dir, "analysis"))
        _assert_pipeline_summary(exp_dir)

    @pytest.mark.skipif(
        not os.path.isfile(NTL9_TOPOLOGY),
        reason="NTL9 test data not found"
    )
    def test_full_pipeline_gin_dcd(self, tmp_path):
        """
        Full pipeline: NTL9 .dcd data + GIN encoder.
        1 file × 100k frames, stride=1000 → ~100 frames, lag=1ns, 3 epochs.
        """
        from pygv.pipe.master_pipeline import PipelineOrchestrator

        config = _make_config(
            tmp_path, traj_dir=NTL9_TRAJ_DIR, top=NTL9_TOPOLOGY,
            file_pattern="*-000.dcd", selection="all",
            stride=1000, lag_time=1.0, encoder_type="gin", epochs=3,
            timestep=0.001,
        )

        orchestrator = PipelineOrchestrator(config)
        results = orchestrator.run_complete_pipeline()

        assert results is not None, "Pipeline returned None"
        assert len(results["trained_models"]) >= 1

        exp_dir = str(results["experiment_dir"])
        _assert_preparation_outputs(os.path.join(exp_dir, "preparation"))
        _assert_training_outputs(os.path.join(exp_dir, "training"))
        _assert_analysis_outputs(os.path.join(exp_dir, "analysis"))
        _assert_pipeline_summary(exp_dir)

    def test_full_pipeline_ml3_xtc(self, tmp_path):
        """
        Full pipeline: AB42 .xtc data + ML3 encoder.
        Validates the rewritten ML3Encoder works end-to-end.
        """
        from pygv.pipe.master_pipeline import PipelineOrchestrator

        config = _make_config(
            tmp_path, traj_dir=AB42_TRAJ_DIR, top=AB42_TOPOLOGY,
            file_pattern="traj000[0-9].xtc", selection="name CA",
            stride=10, lag_time=2.5, encoder_type="ml3", epochs=2,
        )

        orchestrator = PipelineOrchestrator(config)
        results = orchestrator.run_complete_pipeline()

        assert results is not None, "Pipeline returned None"
        assert len(results["trained_models"]) >= 1
        _assert_pipeline_summary(str(results["experiment_dir"]))


@pytest.mark.integration
@pytest.mark.skipif(not DATA_AVAILABLE, reason="AB42 test data not found")
class TestPreparationPhase:
    """Test preparation phase in isolation."""

    def test_preparation_creates_dataset(self, tmp_path):
        """Verify preparation produces a valid dataset with stats."""
        from pygv.pipe.master_pipeline import PipelineOrchestrator

        config = _make_config(
            tmp_path, traj_dir=AB42_TRAJ_DIR, top=AB42_TOPOLOGY,
            file_pattern="traj0000.xtc", stride=50, lag_time=12.5,
        )

        orchestrator = PipelineOrchestrator(config)
        dirs = orchestrator.setup_experiment_directory()
        dataset_path, _ = orchestrator.run_preparation_phase(dirs)

        assert dataset_path is not None
        assert os.path.isdir(dataset_path)
        assert os.path.isfile(os.path.join(dataset_path, "dataset_stats.json"))


@pytest.mark.integration
@pytest.mark.skipif(not DATA_AVAILABLE, reason="AB42 test data not found")
class TestLagTimeValidation:
    """Test lag time validation against real trajectory."""

    def test_valid_lag_time_passes(self, tmp_path):
        """Compatible lag time should pass validation."""
        from pygv.pipe.master_pipeline import validate_lag_times

        config = _make_config(
            tmp_path, traj_dir=AB42_TRAJ_DIR, top=AB42_TOPOLOGY,
            file_pattern="traj0000.xtc", stride=10, lag_time=2.5,
        )
        # Should not raise
        validate_lag_times(config)

    def test_invalid_lag_time_fails(self, tmp_path):
        """Incompatible lag time should fail validation."""
        from pygv.pipe.master_pipeline import validate_lag_times

        config = _make_config(
            tmp_path, traj_dir=AB42_TRAJ_DIR, top=AB42_TOPOLOGY,
            file_pattern="traj0000.xtc", stride=10, lag_time=3.0,
            # effective_dt = 0.25 * 10 = 2.5 ns; 3.0 is not a multiple
        )
        with pytest.raises(SystemExit):
            validate_lag_times(config)


@pytest.mark.integration
@pytest.mark.skipif(not DATA_AVAILABLE, reason="AB42 test data not found")
class TestResume:
    """Test pipeline resume from existing experiment."""

    def test_resume_skips_existing_models(self, tmp_path):
        """Resume should skip already-trained models."""
        from pygv.pipe.master_pipeline import PipelineOrchestrator

        config = _make_config(
            tmp_path, traj_dir=AB42_TRAJ_DIR, top=AB42_TOPOLOGY,
            file_pattern="traj000[0-9].xtc", stride=50, lag_time=12.5, epochs=2,
        )

        # First run
        orchestrator = PipelineOrchestrator(config)
        results = orchestrator.run_complete_pipeline()
        assert results is not None
        exp_dir = str(results["experiment_dir"])

        # Resume — should skip training
        orchestrator2 = PipelineOrchestrator(config)
        results2 = orchestrator2.run_complete_pipeline(
            skip_preparation=True, resume=exp_dir,
        )
        assert results2 is not None
        assert len(results2["trained_models"]) >= 1


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
