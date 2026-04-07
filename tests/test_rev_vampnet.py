"""
Unit tests for RevVAMPNet model class.

Tests verify:
1. Construction with different encoders (SchNet, GIN, ML3)
2. Forward pass produces valid softmax outputs
3. All parameters captured by optimizer
4. Transition matrix and stationary distribution properties
5. Save/load roundtrip
6. Training and evaluation

Run with: pytest tests/test_rev_vampnet.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import os
import tempfile

from pygv.vampnet.rev_vampnet import RevVAMPNet
from pygv.scores.reversible_score import ReversibleVAMPScore
from pygv.encoder.schnet import SchNetEncoderNoEmbed
from pygv.encoder.gin import GINEncoder
from pygv.encoder.ml3 import ML3Encoder
from pygv.classifier.SoftmaxMLP import SoftmaxMLP

from torch_geometric.data import Data, Batch


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def seed():
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def n_states():
    return 5


@pytest.fixture
def node_dim():
    return 16


@pytest.fixture
def edge_dim():
    return 16


@pytest.fixture
def hidden_dim():
    return 32


def _make_synthetic_batch(n_graphs=4, n_nodes=10, node_dim=16, edge_dim=16, device='cpu'):
    """Create a synthetic PyG batch for testing."""
    graphs = []
    for _ in range(n_graphs):
        # Random graph with some edges
        n_edges = n_nodes * 3
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        x = torch.randn(n_nodes, node_dim)
        edge_attr = torch.randn(n_edges, edge_dim)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    batch = Batch.from_data_list(graphs)
    return batch.to(device)


def _make_model(encoder_type='schnet', n_states=5, node_dim=16, edge_dim=16, hidden_dim=32):
    """Create a RevVAMPNet model with the given encoder type."""
    output_dim = hidden_dim

    if encoder_type == 'schnet':
        encoder = SchNetEncoderNoEmbed(
            node_dim=node_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, output_dim=output_dim,
            n_interactions=2
        )
    elif encoder_type == 'gin':
        encoder = GINEncoder(
            node_dim=node_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, output_dim=output_dim,
            n_interactions=2
        )
    elif encoder_type == 'ml3':
        encoder = ML3Encoder(
            node_dim=node_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, output_dim=output_dim,
            num_layers=2
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    classifier = SoftmaxMLP(
        in_channels=output_dim,
        hidden_channels=32,
        out_channels=n_states,
        num_layers=2
    )

    rev_score = ReversibleVAMPScore(n_states=n_states)

    model = RevVAMPNet(
        encoder=encoder,
        rev_score=rev_score,
        classifier_module=classifier,
        lag_time=1.0,
        training_jitter=1e-6
    )
    return model


# =============================================================================
# Tests
# =============================================================================

def test_construction_with_schnet(seed, n_states, node_dim, edge_dim, hidden_dim):
    """RevVAMPNet should construct with SchNet encoder."""
    model = _make_model('schnet', n_states, node_dim, edge_dim, hidden_dim)

    assert hasattr(model, 'encoder')
    assert hasattr(model, 'classifier_module')
    assert hasattr(model, 'rev_score')
    assert isinstance(model.rev_score, ReversibleVAMPScore)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"


def test_construction_with_gin(seed, n_states, node_dim, edge_dim, hidden_dim):
    """RevVAMPNet should construct with GIN encoder."""
    model = _make_model('gin', n_states, node_dim, edge_dim, hidden_dim)

    assert isinstance(model.encoder, GINEncoder)
    assert isinstance(model.rev_score, ReversibleVAMPScore)


def test_construction_with_ml3(seed, n_states, node_dim, edge_dim, hidden_dim):
    """RevVAMPNet should construct with ML3 encoder."""
    model = _make_model('ml3', n_states, node_dim, edge_dim, hidden_dim)

    assert isinstance(model.encoder, ML3Encoder)
    assert isinstance(model.rev_score, ReversibleVAMPScore)


def test_forward_produces_valid_softmax(seed, node_dim, edge_dim):
    """Forward pass should produce valid probability distributions."""
    model = _make_model('schnet', n_states=5, node_dim=node_dim, edge_dim=edge_dim)
    model.eval()

    batch = _make_synthetic_batch(n_graphs=4, n_nodes=10, node_dim=node_dim, edge_dim=edge_dim)

    with torch.no_grad():
        probs, _ = model.forward(batch, apply_classifier=True)

    # Check probabilities are valid
    assert (probs >= 0).all(), "Probabilities must be non-negative"
    assert (probs <= 1).all(), "Probabilities must be <= 1"
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"Probabilities must sum to 1 per sample, got {row_sums}"


def test_all_parameters_in_optimizer(seed):
    """All trainable parameters (encoder + classifier + rev_score) should be in optimizer."""
    model = _make_model('schnet')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Collect all parameter ids from optimizer
    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            opt_param_ids.add(id(p))

    # Check that rev_score parameters are included
    assert id(model.rev_score.log_stationary) in opt_param_ids, \
        "log_stationary must be in optimizer"
    assert id(model.rev_score.rate_matrix_weights) in opt_param_ids, \
        "rate_matrix_weights must be in optimizer"

    # Check that all model parameters are included
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert id(param) in opt_param_ids, \
                f"Parameter '{name}' not found in optimizer"


def test_get_transition_matrix_shape_and_properties(seed):
    """Transition matrix should be row-stochastic and satisfy detailed balance."""
    model = _make_model('schnet', n_states=5)

    K = model.get_transition_matrix()
    u = model.get_stationary_distribution()

    assert K.shape == (5, 5)
    assert (K >= 0).all(), "K must be non-negative"
    assert torch.allclose(K.sum(dim=1), torch.ones(5), atol=1e-5), "K must be row-stochastic"

    # Detailed balance
    for i in range(5):
        for j in range(5):
            assert torch.allclose(u[i] * K[i, j], u[j] * K[j, i], atol=1e-5), \
                f"Detailed balance violated at ({i},{j})"


def test_get_stationary_distribution(seed):
    """Stationary distribution should be a valid probability distribution."""
    model = _make_model('schnet', n_states=4)

    pi = model.get_stationary_distribution()

    assert pi.shape == (4,)
    assert (pi > 0).all(), "Stationary distribution must be positive"
    assert torch.allclose(pi.sum(), torch.tensor(1.0), atol=1e-6), "Must sum to 1"


def test_save_load_roundtrip(seed, node_dim, edge_dim):
    """Save and load should preserve the learned K and pi."""
    model = _make_model('schnet', n_states=3, node_dim=node_dim, edge_dim=edge_dim)

    # Do some forward passes to change parameters
    batch = _make_synthetic_batch(n_graphs=4, n_nodes=10, node_dim=node_dim, edge_dim=edge_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        chi, _ = model.forward(batch)
        # Just use a dummy loss to update params
        loss = chi.sum()
        loss.backward()
        optimizer.step()

    # Get original K and pi
    model.eval()
    with torch.no_grad():
        original_K = model.get_transition_matrix().clone()
        original_pi = model.get_stationary_distribution().clone()

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        model.save_complete_model(path)
        loaded_model = RevVAMPNet.load_complete_model(path)

    loaded_model.eval()
    with torch.no_grad():
        loaded_K = loaded_model.get_transition_matrix()
        loaded_pi = loaded_model.get_stationary_distribution()

    assert torch.allclose(original_K, loaded_K, atol=1e-6), "K should be preserved after save/load"
    assert torch.allclose(original_pi, loaded_pi, atol=1e-6), "pi should be preserved after save/load"


def test_short_training_run(seed, node_dim, edge_dim):
    """A short training run should complete without errors."""
    model = _make_model('schnet', n_states=3, node_dim=node_dim, edge_dim=edge_dim)

    # Create tiny synthetic dataset
    from torch.utils.data import DataLoader

    pairs = []
    for _ in range(50):
        g1 = _make_synthetic_batch(n_graphs=1, n_nodes=10, node_dim=node_dim, edge_dim=edge_dim)
        g2 = _make_synthetic_batch(n_graphs=1, n_nodes=10, node_dim=node_dim, edge_dim=edge_dim)
        # Convert single-graph batches to Data objects
        pairs.append((g1, g2))

    loader = DataLoader(pairs, batch_size=10, shuffle=True,
                        collate_fn=lambda batch: (
                            Batch.from_data_list([b[0] for b in batch]),
                            Batch.from_data_list([b[1] for b in batch])
                        ))

    with tempfile.TemporaryDirectory() as tmpdir:
        history = model.fit(
            train_loader=loader,
            n_epochs=5,
            device='cpu',
            save_dir=tmpdir,
            verbose=False,
            plot_scores=False,
        )

    assert len(history['train_scores']) == 5, "Should have 5 epoch scores"
    assert all(np.isfinite(s) for s in history['train_scores']), "All scores should be finite"


def test_evaluate_returns_finite(seed, node_dim, edge_dim):
    """Evaluate should return a finite value."""
    model = _make_model('schnet', n_states=3, node_dim=node_dim, edge_dim=edge_dim)

    from torch.utils.data import DataLoader

    pairs = []
    for _ in range(20):
        g1 = _make_synthetic_batch(n_graphs=1, n_nodes=10, node_dim=node_dim, edge_dim=edge_dim)
        g2 = _make_synthetic_batch(n_graphs=1, n_nodes=10, node_dim=node_dim, edge_dim=edge_dim)
        pairs.append((g1, g2))

    loader = DataLoader(pairs, batch_size=10, shuffle=False,
                        collate_fn=lambda batch: (
                            Batch.from_data_list([b[0] for b in batch]),
                            Batch.from_data_list([b[1] for b in batch])
                        ))

    result = model.evaluate(loader, device='cpu')
    assert result is not None, "Evaluate should return a value"
    assert np.isfinite(result), f"Evaluate should return finite, got {result}"
