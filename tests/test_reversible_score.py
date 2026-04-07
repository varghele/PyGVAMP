"""
Unit tests for ReversibleVAMPScore module.

Tests verify that:
1. Stationary distribution is valid (positive, sums to 1)
2. Rate matrix is symmetric and non-negative
3. Transition matrix is row-stochastic
4. Transition matrix satisfies detailed balance
5. Loss is finite, gradients flow
6. Loss decreases on simple system
7. Numerical stability with extreme inputs

Run with: pytest tests/test_reversible_score.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from pygv.scores.reversible_score import ReversibleVAMPScore


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
def score_module(seed):
    """Create a ReversibleVAMPScore with 5 states."""
    return ReversibleVAMPScore(n_states=5)


@pytest.fixture
def random_score_module(seed):
    """Create a ReversibleVAMPScore with random weights."""
    score = ReversibleVAMPScore(n_states=5)
    with torch.no_grad():
        score.log_stationary.copy_(torch.randn(5))
        score.rate_matrix_weights.copy_(torch.randn(5, 5))
    return score


# =============================================================================
# Tests
# =============================================================================

def test_stationary_is_valid_distribution(score_module):
    """Stationary distribution should be positive and sum to 1."""
    u = score_module.get_stationary_distribution()

    assert u.shape == (5,)
    assert (u > 0).all(), "Stationary distribution must be positive"
    assert torch.allclose(u.sum(), torch.tensor(1.0), atol=1e-6), "Stationary distribution must sum to 1"


def test_rate_matrix_is_symmetric(random_score_module):
    """Rate matrix should be symmetric."""
    S = random_score_module.get_rate_matrix()

    assert S.shape == (5, 5)
    assert torch.allclose(S, S.T, atol=1e-6), "Rate matrix must be symmetric"


def test_rate_matrix_is_nonnegative(random_score_module):
    """Rate matrix should be non-negative (softplus guarantees this)."""
    S = random_score_module.get_rate_matrix()

    assert (S >= 0).all(), "Rate matrix must be non-negative"


def test_transition_matrix_is_row_stochastic(random_score_module):
    """Transition matrix should have non-negative entries and rows summing to 1."""
    K = random_score_module.get_transition_matrix()

    assert K.shape == (5, 5)
    assert (K >= 0).all(), "Transition matrix must be non-negative"
    row_sums = K.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(5), atol=1e-5), \
        f"Rows must sum to 1, got {row_sums}"


def test_transition_matrix_satisfies_detailed_balance(random_score_module):
    """Transition matrix should satisfy u[i]*K[i,j] = u[j]*K[j,i]."""
    K = random_score_module.get_transition_matrix()
    u = random_score_module.get_stationary_distribution()

    n = K.shape[0]
    for i in range(n):
        for j in range(n):
            lhs = u[i] * K[i, j]
            rhs = u[j] * K[j, i]
            assert torch.allclose(lhs, rhs, atol=1e-5), \
                f"Detailed balance violated at ({i},{j}): {lhs.item():.6f} != {rhs.item():.6f}"


def test_loss_is_finite(seed):
    """Loss should be finite for valid softmax inputs."""
    score = ReversibleVAMPScore(n_states=3)

    chi_t0 = F.softmax(torch.randn(32, 3), dim=1)
    chi_t1 = F.softmax(torch.randn(32, 3), dim=1)

    loss = score.loss(chi_t0, chi_t1)

    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_loss_gradient_flows_to_all_parameters(seed):
    """Gradients should flow to log_stationary and rate_matrix_weights."""
    score = ReversibleVAMPScore(n_states=3)

    chi_t0 = F.softmax(torch.randn(32, 3), dim=1)
    chi_t1 = F.softmax(torch.randn(32, 3), dim=1)

    loss = score.loss(chi_t0, chi_t1)
    loss.backward()

    assert score.log_stationary.grad is not None, "Gradient must flow to log_stationary"
    assert not torch.all(score.log_stationary.grad == 0), "log_stationary gradient should not be all zeros"

    assert score.rate_matrix_weights.grad is not None, "Gradient must flow to rate_matrix_weights"
    assert not torch.all(score.rate_matrix_weights.grad == 0), "rate_matrix_weights gradient should not be all zeros"


def test_loss_gradient_flows_through_inputs(seed):
    """Gradients should flow through chi_t0 and chi_t1 to the encoder."""
    score = ReversibleVAMPScore(n_states=3)

    chi_t0 = F.softmax(torch.randn(32, 3), dim=1)
    chi_t0.requires_grad_(True)
    chi_t1 = F.softmax(torch.randn(32, 3), dim=1)
    chi_t1.requires_grad_(True)

    loss = score.loss(chi_t0, chi_t1)
    loss.backward()

    assert chi_t0.grad is not None, "Gradient must flow through chi_t0"
    assert chi_t1.grad is not None, "Gradient must flow through chi_t1"


def test_loss_decreases_on_simple_system(seed):
    """Loss should decrease when optimizing on a simple 2-state system."""
    score = ReversibleVAMPScore(n_states=2)
    optimizer = torch.optim.Adam(score.parameters(), lr=0.01)

    # Generate synthetic data: samples mostly transition between state 0 and 1
    # with known transition probabilities
    n_samples = 100
    chi_t0 = F.softmax(torch.randn(n_samples, 2) * 2, dim=1)
    chi_t1 = F.softmax(torch.randn(n_samples, 2) * 2, dim=1)

    # Record initial loss
    initial_loss = score.loss(chi_t0, chi_t1).item()

    # Optimize for 50 steps
    for _ in range(50):
        optimizer.zero_grad()
        loss = score.loss(chi_t0, chi_t1)
        loss.backward()
        optimizer.step()

    final_loss = score.loss(chi_t0, chi_t1).item()

    assert final_loss < initial_loss, \
        f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"


def test_numerical_stability_with_extreme_inputs(seed):
    """Loss should not produce NaN/Inf with near-zero or near-deterministic inputs."""
    score = ReversibleVAMPScore(n_states=3)

    # Near-zero probabilities
    chi_near_zero = torch.tensor([[0.999, 5e-4, 5e-4],
                                   [5e-4, 0.999, 5e-4],
                                   [5e-4, 5e-4, 0.999]])
    loss1 = score.loss(chi_near_zero, chi_near_zero)
    assert torch.isfinite(loss1), f"Loss should be finite for near-deterministic inputs, got {loss1.item()}"

    loss1.backward()
    assert score.log_stationary.grad is not None
    assert torch.isfinite(score.log_stationary.grad).all(), "Gradients should be finite"

    score.zero_grad()

    # Very small probabilities
    chi_small = torch.tensor([[1e-7, 1.0 - 2e-7, 1e-7],
                               [1e-7, 1e-7, 1.0 - 2e-7]])
    loss2 = score.loss(chi_small, chi_small)
    assert torch.isfinite(loss2), f"Loss should be finite for near-zero inputs, got {loss2.item()}"
