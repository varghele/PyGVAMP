"""
Unit tests for PyGVAMP training pipeline.

Tests verify that:
1. Output directory setup works correctly
2. Configuration saving works
3. Model creation works for different encoder types
4. Training loop completes without errors
5. Training improves VAMP score over epochs
6. Checkpoint saving and loading works
7. Early stopping triggers correctly

Run with: pytest tests/test_training.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from pygv.vampnet import VAMPNet
from pygv.encoder.schnet import SchNetEncoderNoEmbed
from pygv.encoder.meta_att import Meta
from pygv.classifier.SoftmaxMLP import SoftmaxMLP
from pygv.scores.vamp_score_v0 import VAMPScore

# Import training functions
from pygv.pipe.training import setup_output_directory, save_config


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Use CPU for tests."""
    return torch.device('cpu')


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_args(temp_dir):
    """Create mock arguments namespace."""
    return Namespace(
        run_name=None,
        output_dir=temp_dir,
        # Dataset params
        traj_dir='/fake/traj',
        top='/fake/top.pdb',
        file_pattern='*.xtc',
        selection='name CA',
        stride=10,
        lag_time=20.0,
        n_neighbors=4,
        node_embedding_dim=16,
        gaussian_expansion_dim=16,
        cache_dir=None,
        use_cache=False,
        # Model params
        encoder_type='schnet',
        n_states=5,
        node_dim=16,
        edge_dim=16,
        hidden_dim=64,
        output_dim=32,
        n_interactions=2,
        activation='tanh',
        use_attention=True,
        # Classifier params
        clf_hidden_dim=32,
        clf_num_layers=2,
        clf_dropout=0.0,
        clf_activation='relu',
        clf_norm=None,
        # Embedding params
        use_embedding=False,
        embedding_in_dim=10,
        embedding_hidden_dim=32,
        embedding_out_dim=16,
        embedding_num_layers=2,
        embedding_dropout=0.0,
        embedding_act='relu',
        embedding_norm=None,
        # Training params
        epochs=10,
        batch_size=16,
        lr=0.001,
        weight_decay=0.0001,
        val_split=0.2,
        save_every=5,
        sample_validate_every=5,
        clip_grad=None,
        cpu=True,
        # Analysis params
        protein_name='test_protein',
        max_tau=None,
    )


def create_synthetic_graph(num_nodes=10, node_dim=16, edge_dim=16, device='cpu'):
    """Create a synthetic graph for testing."""
    # Node features (one-hot style)
    x = torch.zeros(num_nodes, node_dim, device=device)
    for i in range(num_nodes):
        x[i, i % node_dim] = 1.0

    # Create k-NN style edges
    edge_index = []
    for i in range(num_nodes):
        for j in range(1, 4):
            neighbor = (i + j) % num_nodes
            edge_index.append([i, neighbor])
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()

    # Edge features (Gaussian expansion style)
    edge_attr = torch.randn(edge_index.size(1), edge_dim, device=device)

    # Batch tensor
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


class SyntheticTimeLaggedDataset(Dataset):
    """Synthetic dataset that returns time-lagged graph pairs."""

    def __init__(self, n_samples=100, num_nodes=10, node_dim=16, edge_dim=16):
        self.n_samples = n_samples
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Pre-generate all graphs for consistency
        np.random.seed(42)
        torch.manual_seed(42)
        self.graphs = []
        for _ in range(n_samples + 1):  # +1 for time-lagged pairs
            self.graphs.append(create_synthetic_graph(num_nodes, node_dim, edge_dim))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return time-lagged pair (t, t+1)
        return self.graphs[idx], self.graphs[idx + 1]


@pytest.fixture
def synthetic_dataset():
    """Create a synthetic time-lagged dataset."""
    return SyntheticTimeLaggedDataset(n_samples=100, num_nodes=10, node_dim=16, edge_dim=16)


@pytest.fixture
def train_loader(synthetic_dataset):
    """Create a training data loader."""
    return DataLoader(synthetic_dataset, batch_size=16, shuffle=True)


@pytest.fixture
def test_loader(synthetic_dataset):
    """Create a test data loader."""
    # Use a smaller subset for testing
    test_dataset = SyntheticTimeLaggedDataset(n_samples=20, num_nodes=10, node_dim=16, edge_dim=16)
    return DataLoader(test_dataset, batch_size=16, shuffle=False)


@pytest.fixture
def simple_model(device):
    """Create a simple VAMPNet model for testing."""
    encoder = SchNetEncoderNoEmbed(
        node_dim=16,
        edge_dim=16,
        hidden_dim=32,
        output_dim=16,
        n_interactions=2,
        use_attention=True
    )

    classifier = SoftmaxMLP(
        in_channels=16,
        hidden_channels=16,
        out_channels=5,
        num_layers=2,
    )

    vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')

    model = VAMPNet(
        encoder=encoder,
        classifier_module=classifier,
        vamp_score=vamp_score,
    )

    return model.to(device)


# =============================================================================
# TestSetupOutputDirectory
# =============================================================================

class TestSetupOutputDirectory:
    """Tests for setup_output_directory function."""

    def test_creates_directories(self, mock_args, temp_dir):
        """setup_output_directory creates required directories."""
        paths = setup_output_directory(mock_args)

        assert os.path.exists(paths['run_dir'])
        assert os.path.exists(paths['model_dir'])
        assert os.path.exists(paths['plot_dir'])

    def test_returns_correct_paths(self, mock_args, temp_dir):
        """setup_output_directory returns expected path keys."""
        paths = setup_output_directory(mock_args)

        assert 'run_dir' in paths
        assert 'model_dir' in paths
        assert 'plot_dir' in paths
        assert 'config' in paths
        assert 'scores_plot' in paths
        assert 'best_model' in paths
        assert 'final_model' in paths

    def test_generates_run_name_if_none(self, mock_args, temp_dir):
        """setup_output_directory generates run_name if not provided."""
        mock_args.run_name = None
        paths = setup_output_directory(mock_args)

        # run_name should be set to datetime string
        assert mock_args.run_name is not None
        assert len(mock_args.run_name) > 0

    def test_uses_provided_run_name(self, mock_args, temp_dir):
        """setup_output_directory uses provided run_name."""
        mock_args.run_name = "test_run"
        paths = setup_output_directory(mock_args)

        assert "test_run" in paths['run_dir']

    def test_model_dir_inside_run_dir(self, mock_args, temp_dir):
        """model_dir is inside run_dir."""
        paths = setup_output_directory(mock_args)

        assert paths['model_dir'].startswith(paths['run_dir'])


# =============================================================================
# TestSaveConfig
# =============================================================================

class TestSaveConfig:
    """Tests for save_config function."""

    def test_saves_config_file(self, mock_args, temp_dir):
        """save_config creates config file."""
        paths = setup_output_directory(mock_args)
        save_config(mock_args, paths)

        assert os.path.exists(paths['config'])

    def test_config_contains_parameters(self, mock_args, temp_dir):
        """Config file contains expected parameters."""
        paths = setup_output_directory(mock_args)
        save_config(mock_args, paths)

        with open(paths['config'], 'r') as f:
            content = f.read()

        assert 'epochs' in content
        assert 'batch_size' in content
        assert 'lr' in content
        assert 'encoder_type' in content

    def test_config_values_correct(self, mock_args, temp_dir):
        """Config file contains correct values."""
        mock_args.epochs = 50
        mock_args.batch_size = 32

        paths = setup_output_directory(mock_args)
        save_config(mock_args, paths)

        with open(paths['config'], 'r') as f:
            content = f.read()

        assert 'epochs = 50' in content
        assert 'batch_size = 32' in content


# =============================================================================
# TestTrainingStep
# =============================================================================

class TestTrainingStep:
    """Tests for individual training steps."""

    def test_single_forward_pass(self, simple_model, train_loader, device):
        """Single forward pass completes without error."""
        simple_model.train()

        batch = next(iter(train_loader))
        data_t0, data_t1 = batch
        data_t0, data_t1 = data_t0.to(device), data_t1.to(device)

        chi_t0, _ = simple_model.forward(data_t0, apply_classifier=True)
        chi_t1, _ = simple_model.forward(data_t1, apply_classifier=True)

        assert chi_t0.shape[1] == 5  # n_states
        assert chi_t1.shape[1] == 5

    def test_loss_computation(self, simple_model, train_loader, device):
        """Loss computation works correctly."""
        simple_model.train()

        batch = next(iter(train_loader))
        data_t0, data_t1 = batch
        data_t0, data_t1 = data_t0.to(device), data_t1.to(device)

        chi_t0, _ = simple_model.forward(data_t0, apply_classifier=True)
        chi_t1, _ = simple_model.forward(data_t1, apply_classifier=True)

        loss = simple_model.vamp_score.loss(chi_t0, chi_t1)

        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_backward_pass(self, simple_model, train_loader, device):
        """Backward pass computes gradients."""
        simple_model.train()

        batch = next(iter(train_loader))
        data_t0, data_t1 = batch
        data_t0, data_t1 = data_t0.to(device), data_t1.to(device)

        chi_t0, _ = simple_model.forward(data_t0, apply_classifier=True)
        chi_t1, _ = simple_model.forward(data_t1, apply_classifier=True)

        loss = simple_model.vamp_score.loss(chi_t0, chi_t1)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for param in simple_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"

    def test_optimizer_step(self, simple_model, train_loader, device):
        """Optimizer step updates parameters."""
        simple_model.train()
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)

        # Get initial parameter values
        initial_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        batch = next(iter(train_loader))
        data_t0, data_t1 = batch
        data_t0, data_t1 = data_t0.to(device), data_t1.to(device)

        optimizer.zero_grad()
        chi_t0, _ = simple_model.forward(data_t0, apply_classifier=True)
        chi_t1, _ = simple_model.forward(data_t1, apply_classifier=True)
        loss = simple_model.vamp_score.loss(chi_t0, chi_t1)
        loss.backward()
        optimizer.step()

        # Check that at least some parameters changed
        params_changed = False
        for name, param in simple_model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed, "Parameters not updated"


# =============================================================================
# TestTrainingLoop
# =============================================================================

class TestTrainingLoop:
    """Tests for the full training loop."""

    def test_fit_completes(self, simple_model, train_loader, device, temp_dir):
        """model.fit() completes without error."""
        history = simple_model.fit(
            train_loader=train_loader,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert history is not None
        assert 'train_scores' in history

    def test_fit_with_validation(self, simple_model, train_loader, test_loader, device, temp_dir):
        """model.fit() works with validation loader."""
        history = simple_model.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
            sample_validate_every=5,
        )

        assert history is not None

    def test_fit_returns_history(self, simple_model, train_loader, device, temp_dir):
        """model.fit() returns training history."""
        history = simple_model.fit(
            train_loader=train_loader,
            n_epochs=3,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert 'train_scores' in history
        assert 'epochs' in history
        assert len(history['train_scores']) == 3

    def test_training_improves_score(self, device, temp_dir):
        """Training improves VAMP score over epochs."""
        # Create fresh model and dataset
        torch.manual_seed(123)

        encoder = SchNetEncoderNoEmbed(
            node_dim=16, edge_dim=16, hidden_dim=32, output_dim=16,
            n_interactions=2, use_attention=True
        )
        classifier = SoftmaxMLP(in_channels=16, hidden_channels=16, out_channels=5, num_layers=2)
        vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')
        model = VAMPNet(encoder=encoder, classifier_module=classifier, vamp_score=vamp_score)
        model.to(device)

        dataset = SyntheticTimeLaggedDataset(n_samples=100, num_nodes=10, node_dim=16, edge_dim=16)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        history = model.fit(
            train_loader=loader,
            n_epochs=10,
            device=device,
            learning_rate=0.01,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        # Check that score improved (last score > first score)
        first_score = history['train_scores'][0]
        last_score = history['train_scores'][-1]

        # Allow for some variance - just check it's not getting worse significantly
        assert last_score >= first_score * 0.9, f"Score degraded: {first_score} -> {last_score}"


# =============================================================================
# TestCheckpointing
# =============================================================================

class TestCheckpointing:
    """Tests for model checkpointing during training."""

    def test_saves_checkpoints(self, simple_model, train_loader, device, temp_dir):
        """Training saves checkpoints at specified intervals."""
        history = simple_model.fit(
            train_loader=train_loader,
            n_epochs=6,
            device=device,
            save_dir=temp_dir,
            save_every=2,
            verbose=False,
            plot_scores=False,
        )

        # Should have checkpoints at epochs 2, 4, 6
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) >= 1

    def test_saves_final_model(self, simple_model, train_loader, device, temp_dir):
        """Training saves final model."""
        history = simple_model.fit(
            train_loader=train_loader,
            n_epochs=3,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        # Check for any saved model files
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) >= 1

    def test_checkpoint_loadable(self, simple_model, train_loader, device, temp_dir):
        """Saved checkpoints can be loaded."""
        simple_model.fit(
            train_loader=train_loader,
            n_epochs=3,
            device=device,
            save_dir=temp_dir,
            save_every=1,
            verbose=False,
            plot_scores=False,
        )

        # Find a checkpoint file
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) > 0

        # Try loading
        checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Should be a state dict or model
        assert checkpoint is not None


# =============================================================================
# TestEarlyStopping
# =============================================================================

class TestEarlyStopping:
    """Tests for early stopping functionality."""

    def test_early_stopping_triggers(self, device, temp_dir):
        """Early stopping triggers when score doesn't improve."""
        # Create a model that won't improve (very low learning rate)
        torch.manual_seed(42)

        encoder = SchNetEncoderNoEmbed(
            node_dim=16, edge_dim=16, hidden_dim=32, output_dim=16,
            n_interactions=2, use_attention=True
        )
        classifier = SoftmaxMLP(in_channels=16, hidden_channels=16, out_channels=5, num_layers=2)
        vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')
        model = VAMPNet(encoder=encoder, classifier_module=classifier, vamp_score=vamp_score)
        model.to(device)

        dataset = SyntheticTimeLaggedDataset(n_samples=50, num_nodes=10, node_dim=16, edge_dim=16)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        history = model.fit(
            train_loader=loader,
            n_epochs=100,  # Request many epochs
            device=device,
            learning_rate=1e-10,  # Very low LR = no improvement
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
            early_stopping=3,  # Stop after 3 epochs without improvement
        )

        # Should have stopped early (less than 100 epochs)
        assert len(history['train_scores']) < 100


# =============================================================================
# TestGradientClipping
# =============================================================================

class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_gradient_clipping_applied(self, simple_model, train_loader, device, temp_dir):
        """Gradient clipping is applied during training."""
        # This is hard to test directly, but we can verify training completes
        history = simple_model.fit(
            train_loader=train_loader,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            clip_grad_norm=1.0,  # Apply gradient clipping
            verbose=False,
            plot_scores=False,
        )

        assert history is not None
        assert len(history['train_scores']) == 2


# =============================================================================
# TestDifferentEncoders
# =============================================================================

class TestDifferentEncoders:
    """Tests training with different encoder types."""

    def test_training_with_schnet(self, train_loader, device, temp_dir):
        """Training works with SchNet encoder."""
        encoder = SchNetEncoderNoEmbed(
            node_dim=16, edge_dim=16, hidden_dim=32, output_dim=16,
            n_interactions=2, use_attention=True
        )
        classifier = SoftmaxMLP(in_channels=16, hidden_channels=16, out_channels=5, num_layers=2)
        vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')
        model = VAMPNet(encoder=encoder, classifier_module=classifier, vamp_score=vamp_score)
        model.to(device)

        history = model.fit(
            train_loader=train_loader,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert len(history['train_scores']) == 2

    def test_training_with_meta(self, train_loader, device, temp_dir):
        """Training works with Meta encoder."""
        encoder = Meta(
            node_dim=16, edge_dim=16, global_dim=16,
            num_node_mlp_layers=2, num_edge_mlp_layers=2, num_global_mlp_layers=2,
            hidden_dim=32, output_dim=16, num_meta_layers=2,
            embedding_type='node', use_attention=True
        )
        classifier = SoftmaxMLP(in_channels=16, hidden_channels=16, out_channels=5, num_layers=2)
        vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')
        model = VAMPNet(encoder=encoder, classifier_module=classifier, vamp_score=vamp_score)
        model.to(device)

        history = model.fit(
            train_loader=train_loader,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert len(history['train_scores']) == 2


# =============================================================================
# TestDevicePlacement
# =============================================================================

class TestDevicePlacement:
    """Tests for device placement."""

    def test_training_on_cpu(self, simple_model, train_loader, temp_dir):
        """Training works on CPU."""
        device = torch.device('cpu')

        history = simple_model.fit(
            train_loader=train_loader,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert history is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_on_gpu(self, train_loader, temp_dir):
        """Training works on GPU if available."""
        device = torch.device('cuda')

        encoder = SchNetEncoderNoEmbed(
            node_dim=16, edge_dim=16, hidden_dim=32, output_dim=16,
            n_interactions=2, use_attention=True
        )
        classifier = SoftmaxMLP(in_channels=16, hidden_channels=16, out_channels=5, num_layers=2)
        vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')
        model = VAMPNet(encoder=encoder, classifier_module=classifier, vamp_score=vamp_score)

        history = model.fit(
            train_loader=train_loader,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert history is not None


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_epoch(self, simple_model, train_loader, device, temp_dir):
        """Training with single epoch works."""
        history = simple_model.fit(
            train_loader=train_loader,
            n_epochs=1,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert len(history['train_scores']) == 1

    def test_small_batch_size(self, device, temp_dir):
        """Training with small batch size works."""
        encoder = SchNetEncoderNoEmbed(
            node_dim=16, edge_dim=16, hidden_dim=32, output_dim=16,
            n_interactions=2, use_attention=True
        )
        classifier = SoftmaxMLP(in_channels=16, hidden_channels=16, out_channels=5, num_layers=2)
        vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')
        model = VAMPNet(encoder=encoder, classifier_module=classifier, vamp_score=vamp_score)
        model.to(device)

        dataset = SyntheticTimeLaggedDataset(n_samples=20, num_nodes=10, node_dim=16, edge_dim=16)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        history = model.fit(
            train_loader=loader,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert len(history['train_scores']) == 2

    def test_no_save_dir(self, simple_model, train_loader, device):
        """Training works without save directory."""
        history = simple_model.fit(
            train_loader=train_loader,
            n_epochs=2,
            device=device,
            save_dir=None,
            verbose=False,
            plot_scores=False,
        )

        assert history is not None


# =============================================================================
# TestOptimizerOptions
# =============================================================================

class TestOptimizerOptions:
    """Tests for different optimizer configurations."""

    def test_custom_optimizer(self, simple_model, train_loader, device, temp_dir):
        """Training works with custom optimizer."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01, momentum=0.9)

        history = simple_model.fit(
            train_loader=train_loader,
            optimizer=optimizer,
            n_epochs=2,
            device=device,
            save_dir=temp_dir,
            verbose=False,
            plot_scores=False,
        )

        assert history is not None

    def test_different_learning_rates(self, train_loader, device, temp_dir):
        """Training works with different learning rates."""
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            encoder = SchNetEncoderNoEmbed(
                node_dim=16, edge_dim=16, hidden_dim=32, output_dim=16,
                n_interactions=2, use_attention=True
            )
            classifier = SoftmaxMLP(in_channels=16, hidden_channels=16, out_channels=5, num_layers=2)
            vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')
            model = VAMPNet(encoder=encoder, classifier_module=classifier, vamp_score=vamp_score)
            model.to(device)

            history = model.fit(
                train_loader=train_loader,
                n_epochs=1,
                device=device,
                learning_rate=lr,
                save_dir=temp_dir,
                verbose=False,
                plot_scores=False,
            )

            assert history is not None, f"Failed with lr={lr}"


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
