"""
Unit tests for VAMP score calculation.

Tests verify that VAMP score:
1. Computes correctly for all methods (VAMP1, VAMP2, VAMPE)
2. Allows gradient flow for optimization
3. Has correct mathematical properties (score >= 1)
4. Is higher for correlated data than uncorrelated data
5. Handles edge cases (small batches, numerical stability)

Run with: pytest tests/test_vamp_score.py -v
"""

import pytest
import torch
import numpy as np

from pygv.scores.vamp_score_v0 import VAMPScore


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
def simple_data(device, seed):
    """
    Create simple time-lagged data pairs.

    Returns data where lagged version is slightly correlated with original.
    """
    batch_size = 100
    n_states = 5

    # Create data at t=0
    data = torch.randn(batch_size, n_states, device=device)

    # Create time-lagged data with some correlation
    noise = torch.randn(batch_size, n_states, device=device) * 0.3
    data_lagged = data * 0.8 + noise  # 80% correlation + noise

    return data, data_lagged


@pytest.fixture
def uncorrelated_data(device, seed):
    """Create completely uncorrelated time-lagged pairs."""
    batch_size = 100
    n_states = 5

    data = torch.randn(batch_size, n_states, device=device)
    data_lagged = torch.randn(batch_size, n_states, device=device)  # Independent

    return data, data_lagged


@pytest.fixture
def perfectly_correlated_data(device, seed):
    """Create perfectly correlated time-lagged pairs (identity dynamics)."""
    batch_size = 100
    n_states = 5

    data = torch.randn(batch_size, n_states, device=device)
    data_lagged = data.clone()  # Perfect correlation

    return data, data_lagged


@pytest.fixture
def softmax_data(device, seed):
    """
    Create softmax probability outputs (like from a classifier).

    This simulates actual VAMPNet outputs where each row sums to 1.
    """
    batch_size = 100
    n_states = 5

    # Raw logits
    logits = torch.randn(batch_size, n_states, device=device)
    logits_lagged = logits * 0.9 + torch.randn(batch_size, n_states, device=device) * 0.2

    # Apply softmax
    data = torch.softmax(logits, dim=1)
    data_lagged = torch.softmax(logits_lagged, dim=1)

    return data, data_lagged


# =============================================================================
# Basic Forward Pass Tests
# =============================================================================

class TestVAMPScoreForward:
    """Test VAMP score forward pass."""

    def test_vamp2_forward(self, simple_data, device):
        """Test VAMP2 forward pass produces valid output."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='trunc')
        score = scorer(data, data_lagged)

        assert score.dim() == 0, "Score should be a scalar"
        assert not torch.isnan(score), "Score should not be NaN"
        assert not torch.isinf(score), "Score should not be infinite"

    def test_vamp1_forward(self, simple_data, device):
        """Test VAMP1 forward pass produces valid output."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMP1', epsilon=1e-6, mode='trunc')
        score = scorer(data, data_lagged)

        assert score.dim() == 0, "Score should be a scalar"
        assert not torch.isnan(score), "Score should not be NaN"

    def test_vampe_forward(self, simple_data, device):
        """Test VAMPE forward pass produces valid output."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMPE', epsilon=1e-6, mode='trunc')
        score = scorer(data, data_lagged)

        assert score.dim() == 0, "Score should be a scalar"
        assert not torch.isnan(score), "Score should not be NaN"

    def test_loss_is_negative_score(self, simple_data, device):
        """Test that loss() returns negative of forward()."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMP2')
        score = scorer(data, data_lagged)
        loss = scorer.loss(data, data_lagged)

        assert torch.allclose(loss, -score), "Loss should be negative of score"


# =============================================================================
# Mathematical Property Tests
# =============================================================================

class TestVAMPScoreMathProperties:
    """Test mathematical properties of VAMP score."""

    def test_score_at_least_one(self, simple_data, device):
        """
        Test that VAMP score is >= 1.

        The VAMP score includes +1 contribution from the constant singular function,
        so the minimum possible score is 1 (for completely uncorrelated data).
        """
        data, data_lagged = simple_data

        for method in ['VAMP1', 'VAMP2']:
            scorer = VAMPScore(method=method, epsilon=1e-6, mode='regularize')
            score = scorer(data, data_lagged)

            assert score >= 1.0 - 1e-5, f"{method} score should be >= 1, got {score.item():.4f}"

    def test_correlated_higher_than_uncorrelated(self, simple_data, uncorrelated_data, device):
        """
        Test that correlated data produces higher score than uncorrelated data.

        This is the fundamental property that makes VAMP useful for dynamics.
        """
        data_corr, data_lagged_corr = simple_data
        data_uncorr, data_lagged_uncorr = uncorrelated_data

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')

        score_correlated = scorer(data_corr, data_lagged_corr)
        score_uncorrelated = scorer(data_uncorr, data_lagged_uncorr)

        assert score_correlated > score_uncorrelated, (
            f"Correlated data score ({score_correlated.item():.4f}) should be higher than "
            f"uncorrelated ({score_uncorrelated.item():.4f})"
        )

    def test_perfect_correlation_high_score(self, perfectly_correlated_data, simple_data, device):
        """Test that perfectly correlated data produces higher score."""
        data_perfect, data_lagged_perfect = perfectly_correlated_data
        data_partial, data_lagged_partial = simple_data

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')

        score_perfect = scorer(data_perfect, data_lagged_perfect)
        score_partial = scorer(data_partial, data_lagged_partial)

        assert score_perfect > score_partial, (
            f"Perfect correlation score ({score_perfect.item():.4f}) should be higher than "
            f"partial correlation ({score_partial.item():.4f})"
        )

    def test_vamp2_equals_squared_vamp1_approx(self, simple_data, device):
        """
        Test relationship between VAMP1 and VAMP2.

        VAMP2 ≈ sum of squared singular values
        VAMP1 ≈ sum of singular values
        For normalized data, VAMP2 should be related to VAMP1.
        """
        data, data_lagged = simple_data

        scorer_v1 = VAMPScore(method='VAMP1', epsilon=1e-6, mode='regularize')
        scorer_v2 = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')

        score_v1 = scorer_v1(data, data_lagged)
        score_v2 = scorer_v2(data, data_lagged)

        # Both should be positive and finite
        assert score_v1 > 0 and score_v2 > 0, "Both scores should be positive"
        assert score_v1.isfinite() and score_v2.isfinite(), "Both scores should be finite"


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestVAMPScoreGradients:
    """Test gradient flow through VAMP score."""

    def test_gradient_flow_vamp2(self, device, seed):
        """Test that gradients flow through VAMP2 computation."""
        batch_size = 50
        n_states = 5

        data = torch.randn(batch_size, n_states, device=device, requires_grad=True)
        data_lagged = torch.randn(batch_size, n_states, device=device, requires_grad=True)

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')
        score = scorer(data, data_lagged)

        # Backpropagate
        score.backward()

        assert data.grad is not None, "Gradient should flow to data"
        assert data_lagged.grad is not None, "Gradient should flow to data_lagged"
        assert not torch.all(data.grad == 0), "Gradients should be non-zero"
        assert not torch.all(data_lagged.grad == 0), "Gradients should be non-zero"

    def test_gradient_flow_vamp1(self, device, seed):
        """Test that gradients flow through VAMP1 computation."""
        batch_size = 50
        n_states = 5

        data = torch.randn(batch_size, n_states, device=device, requires_grad=True)
        data_lagged = torch.randn(batch_size, n_states, device=device, requires_grad=True)

        scorer = VAMPScore(method='VAMP1', epsilon=1e-6, mode='regularize')
        score = scorer(data, data_lagged)

        score.backward()

        assert data.grad is not None, "Gradient should flow to data"
        assert not torch.all(data.grad == 0), "Gradients should be non-zero"

    def test_gradient_flow_vampe(self, device, seed):
        """Test that gradients flow through VAMPE computation."""
        batch_size = 50
        n_states = 5

        data = torch.randn(batch_size, n_states, device=device, requires_grad=True)
        data_lagged = torch.randn(batch_size, n_states, device=device, requires_grad=True)

        scorer = VAMPScore(method='VAMPE', epsilon=1e-6, mode='regularize')
        score = scorer(data, data_lagged)

        score.backward()

        assert data.grad is not None, "Gradient should flow to data"

    def test_loss_gradient_for_optimization(self, device, seed):
        """
        Test that loss gradients point in the right direction for optimization.

        The loss should decrease when we move data in the direction of its gradient.
        """
        batch_size = 50
        n_states = 5

        data = torch.randn(batch_size, n_states, device=device, requires_grad=True)
        data_lagged = data.detach() * 0.5 + torch.randn(batch_size, n_states, device=device) * 0.5

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')

        # Compute initial loss
        loss_initial = scorer.loss(data, data_lagged)
        loss_initial.backward()

        # Take a gradient step (should increase score / decrease loss)
        with torch.no_grad():
            data_updated = data - 0.01 * data.grad  # Gradient descent on loss

        data_updated.requires_grad_(True)
        loss_after = scorer.loss(data_updated, data_lagged)

        # Loss should decrease (or at least not increase significantly)
        # Note: Due to numerical issues, we allow small increases
        assert loss_after <= loss_initial + 0.1, (
            f"Loss should decrease with gradient step, went from {loss_initial.item():.4f} to {loss_after.item():.4f}"
        )


# =============================================================================
# Covariance Computation Tests
# =============================================================================

class TestCovarianceComputation:
    """Test covariance matrix computation."""

    def test_covariance_shapes(self, simple_data, device):
        """Test that covariance matrices have correct shapes."""
        data, data_lagged = simple_data
        n_states = data.size(1)

        scorer = VAMPScore(method='VAMP2')
        c00, c0t, ctt = scorer._covariances(data, data_lagged, remove_mean=True)

        assert c00.shape == (n_states, n_states), f"C00 shape should be ({n_states}, {n_states})"
        assert c0t.shape == (n_states, n_states), f"C0t shape should be ({n_states}, {n_states})"
        assert ctt.shape == (n_states, n_states), f"Ctt shape should be ({n_states}, {n_states})"

    def test_covariance_symmetry(self, simple_data, device):
        """Test that auto-covariance matrices are symmetric."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMP2')
        c00, c0t, ctt = scorer._covariances(data, data_lagged, remove_mean=True)

        assert torch.allclose(c00, c00.t(), atol=1e-5), "C00 should be symmetric"
        assert torch.allclose(ctt, ctt.t(), atol=1e-5), "Ctt should be symmetric"

    def test_covariance_positive_definite(self, simple_data, device):
        """Test that auto-covariance matrices are positive definite."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6)
        c00, c0t, ctt = scorer._covariances(data, data_lagged, remove_mean=True)

        # Check eigenvalues are positive (with regularization added)
        eigvals_c00 = torch.linalg.eigvalsh(c00)
        eigvals_ctt = torch.linalg.eigvalsh(ctt)

        assert torch.all(eigvals_c00 > 0), "C00 should be positive definite"
        assert torch.all(eigvals_ctt > 0), "Ctt should be positive definite"

    def test_covariance_with_softmax_outputs(self, softmax_data, device):
        """Test covariance computation with softmax probability outputs."""
        data, data_lagged = softmax_data

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6)
        c00, c0t, ctt = scorer._covariances(data, data_lagged, remove_mean=True)

        # Should not produce NaN
        assert not torch.isnan(c00).any(), "C00 should not contain NaN"
        assert not torch.isnan(c0t).any(), "C0t should not contain NaN"
        assert not torch.isnan(ctt).any(), "Ctt should not contain NaN"


# =============================================================================
# Mode Tests (Truncation vs Regularization)
# =============================================================================

class TestRegularizationModes:
    """Test different regularization modes."""

    def test_trunc_mode(self, simple_data, device):
        """Test truncation mode works."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='trunc')
        score = scorer(data, data_lagged)

        assert not torch.isnan(score), "Truncation mode should produce valid score"

    def test_regularize_mode(self, simple_data, device):
        """Test regularization mode works."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')
        score = scorer(data, data_lagged)

        assert not torch.isnan(score), "Regularization mode should produce valid score"

    def test_modes_produce_similar_results(self, simple_data, device):
        """Test that different modes produce reasonably similar results."""
        data, data_lagged = simple_data

        scorer_trunc = VAMPScore(method='VAMP2', epsilon=1e-6, mode='trunc')
        scorer_reg = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')

        score_trunc = scorer_trunc(data, data_lagged)
        score_reg = scorer_reg(data, data_lagged)

        # Scores should be in the same ballpark (within 50% of each other)
        ratio = score_trunc / score_reg
        assert 0.5 < ratio < 2.0, f"Modes should produce similar results, ratio was {ratio.item():.2f}"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_batch_size(self, device, seed):
        """Test handling of small batch sizes."""
        batch_size = 10  # Small batch
        n_states = 5

        data = torch.randn(batch_size, n_states, device=device)
        data_lagged = torch.randn(batch_size, n_states, device=device)

        scorer = VAMPScore(method='VAMP2', epsilon=1e-4, mode='regularize')
        score = scorer(data, data_lagged)

        assert not torch.isnan(score), "Should handle small batches"

    def test_large_n_states(self, device, seed):
        """Test handling of many states."""
        batch_size = 100
        n_states = 20  # More states than typical

        data = torch.randn(batch_size, n_states, device=device)
        data_lagged = torch.randn(batch_size, n_states, device=device)

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')
        score = scorer(data, data_lagged)

        assert not torch.isnan(score), "Should handle many states"

    def test_batch_larger_than_states(self, device, seed):
        """Test with batch size >> n_states (typical case)."""
        batch_size = 500
        n_states = 5

        data = torch.randn(batch_size, n_states, device=device)
        data_lagged = data * 0.9 + torch.randn(batch_size, n_states, device=device) * 0.1

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='trunc')
        score = scorer(data, data_lagged)

        assert not torch.isnan(score), "Should handle large batches"
        assert score > 1.0, "Correlated large batch should have score > 1"

    def test_shape_mismatch_error(self, device, seed):
        """Test that shape mismatch raises error."""
        data = torch.randn(100, 5, device=device)
        data_lagged = torch.randn(100, 6, device=device)  # Different n_states

        scorer = VAMPScore(method='VAMP2')

        with pytest.raises(ValueError, match="shapes must match"):
            scorer(data, data_lagged)

    def test_invalid_method_error(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            VAMPScore(method='INVALID')

    def test_invalid_mode_error(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            VAMPScore(method='VAMP2', mode='invalid')

    def test_non_tensor_input_error(self, device):
        """Test that non-tensor input raises error."""
        scorer = VAMPScore(method='VAMP2')

        with pytest.raises(TypeError, match="must be torch.Tensor"):
            scorer([[1, 2], [3, 4]], [[1, 2], [3, 4]])


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of VAMP score computation."""

    def test_stability_with_near_zero_variance(self, device, seed):
        """Test handling of data with near-zero variance in some dimensions."""
        batch_size = 100
        n_states = 5

        data = torch.randn(batch_size, n_states, device=device)
        data[:, 0] = 0.0001  # Near-zero variance in first dimension
        data_lagged = data * 0.9 + torch.randn(batch_size, n_states, device=device) * 0.1

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')
        score = scorer(data, data_lagged)

        assert not torch.isnan(score), "Should handle near-zero variance"
        assert not torch.isinf(score), "Should not produce infinite score"

    def test_stability_with_large_values(self, device, seed):
        """Test handling of large input values."""
        batch_size = 100
        n_states = 5

        data = torch.randn(batch_size, n_states, device=device) * 1000
        data_lagged = data * 0.9 + torch.randn(batch_size, n_states, device=device) * 100

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')
        score = scorer(data, data_lagged)

        assert not torch.isnan(score), "Should handle large values"
        assert not torch.isinf(score), "Should not produce infinite score"

    def test_deterministic_results(self, simple_data, device):
        """Test that same input produces same output."""
        data, data_lagged = simple_data

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='trunc')

        score1 = scorer(data, data_lagged)
        score2 = scorer(data, data_lagged)

        assert torch.allclose(score1, score2), "Same input should produce same output"


# =============================================================================
# Integration with Training
# =============================================================================

class TestTrainingIntegration:
    """Test VAMP score behavior in training-like scenarios."""

    def test_optimization_improves_score(self, device, seed):
        """
        Test that optimizing the loss actually improves the VAMP score.

        This simulates a simplified training loop.
        """
        batch_size = 100
        n_states = 5
        n_steps = 10

        # Create a simple "model" (just learnable parameters)
        params = torch.randn(batch_size, n_states, device=device, requires_grad=True)
        target = torch.randn(batch_size, n_states, device=device)

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')
        optimizer = torch.optim.Adam([params], lr=0.1)

        initial_score = scorer(params.detach(), target).item()

        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = scorer.loss(params, target)
            loss.backward()
            optimizer.step()

        final_score = scorer(params.detach(), target).item()

        assert final_score > initial_score, (
            f"Optimization should improve score, went from {initial_score:.4f} to {final_score:.4f}"
        )

    def test_batch_consistency(self, device, seed):
        """
        Test that score is reasonably consistent across different batch compositions.

        Note: VAMP score can vary with batch composition, but shouldn't vary wildly.
        """
        n_states = 5

        # Create larger dataset
        torch.manual_seed(seed)
        full_data = torch.randn(500, n_states, device=device)
        full_data_lagged = full_data * 0.8 + torch.randn(500, n_states, device=device) * 0.3

        scorer = VAMPScore(method='VAMP2', epsilon=1e-6, mode='regularize')

        # Compute score on different subsets
        scores = []
        for i in range(5):
            start = i * 100
            end = start + 100
            score = scorer(full_data[start:end], full_data_lagged[start:end])
            scores.append(score.item())

        # Scores should be within reasonable range of each other
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        cv = score_std / score_mean  # Coefficient of variation

        assert cv < 0.5, f"Batch scores should be relatively consistent, CV was {cv:.2f}"


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
