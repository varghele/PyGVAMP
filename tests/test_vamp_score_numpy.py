"""
VAMP score validation against NumPy reference implementation.

Validates that the torch VAMPScore module computes mathematically correct
covariance matrices, matrix inverse square roots, Koopman matrices, and
VAMP1/VAMP2/VAMPE scores by comparing against independent NumPy calculations.

Run with: pytest tests/test_vamp_score_numpy.py -v
"""

import pytest
import torch
import numpy as np
from scipy.linalg import sqrtm

from pygv.scores.vamp_score_v0 import VAMPScore


# =============================================================================
# NumPy reference implementations
# =============================================================================

def np_covariances(data, data_lagged, remove_mean=True):
    """
    Compute covariance matrices using NumPy.

    C00 = (1/(N-1)) * X^T @ X
    C0t = (1/(N-1)) * X^T @ Y
    Ctt = (1/(N-1)) * Y^T @ Y

    where X = data - mean(data), Y = data_lagged - mean(data_lagged)
    """
    X = data.copy()
    Y = data_lagged.copy()

    if remove_mean:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

    N = X.shape[0]
    c00 = (1.0 / (N - 1)) * X.T @ X
    c0t = (1.0 / (N - 1)) * X.T @ Y
    ctt = (1.0 / (N - 1)) * Y.T @ Y

    return c00, c0t, ctt


def np_sym_inverse_sqrt(mat, epsilon=1e-6, mode='trunc'):
    """
    Compute M^{-1/2} via eigendecomposition.

    1. Eigendecompose: M = V diag(λ) V^T
    2. Filter/regularize eigenvalues
    3. Return V diag(1/√λ) V^T
    """
    eigvals, eigvecs = np.linalg.eigh(mat)

    if mode == 'trunc':
        mask = eigvals > epsilon
        if not np.any(mask):
            # Fallback: keep the largest eigenvalue
            mask = np.zeros_like(eigvals, dtype=bool)
            mask[np.argmax(eigvals)] = True
            eigvals = np.array([epsilon])
        else:
            eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]
    elif mode == 'regularize':
        eigvals = np.abs(eigvals)
    elif mode == 'clamp':
        eigvals = np.maximum(eigvals, epsilon)

    diag_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    return eigvecs @ diag_inv_sqrt @ eigvecs.T


def np_koopman_matrix(data, data_lagged, epsilon=1e-6, mode='trunc'):
    """
    Compute the Koopman matrix: K = (C00^{-1/2} @ C0t @ Ctt^{-1/2})^T

    Note: The transpose matches the implementation in vamp_score_v0.py.
    """
    c00, c0t, ctt = np_covariances(data, data_lagged, remove_mean=True)

    # Add epsilon regularization to diagonals (matching _covariances())
    c00 += epsilon * np.eye(c00.shape[0])
    ctt += epsilon * np.eye(ctt.shape[0])

    c00_inv_sqrt = np_sym_inverse_sqrt(c00, epsilon=epsilon, mode=mode)
    ctt_inv_sqrt = np_sym_inverse_sqrt(ctt, epsilon=epsilon, mode=mode)

    koopman = (c00_inv_sqrt @ c0t @ ctt_inv_sqrt).T
    return koopman


def np_vamp2_score(data, data_lagged, epsilon=1e-6, mode='trunc'):
    """VAMP2 = ||K||_F^2 + 1"""
    K = np_koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
    return np.linalg.norm(K, 'fro') ** 2 + 1


def np_vamp1_score(data, data_lagged, epsilon=1e-6, mode='trunc'):
    """VAMP1 = ||K||_* + 1 (nuclear norm)"""
    K = np_koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
    return np.linalg.norm(K, 'nuc') + 1


def np_vampe_score(data, data_lagged, epsilon=1e-6, mode='trunc'):
    """
    VAMPE = tr(2 S U^T C0t V - S U^T C00 U S V^T Ctt V) + 1

    where K = U S V^T (SVD), and U,V are transformed by C00^{-1/2}, Ctt^{-1/2}.
    """
    c00, c0t, ctt = np_covariances(data, data_lagged, remove_mean=True)
    c00 += epsilon * np.eye(c00.shape[0])
    ctt += epsilon * np.eye(ctt.shape[0])

    c00_inv_sqrt = np_sym_inverse_sqrt(c00, epsilon=epsilon, mode=mode)
    ctt_inv_sqrt = np_sym_inverse_sqrt(ctt, epsilon=epsilon, mode=mode)

    koopman = (c00_inv_sqrt @ c0t @ ctt_inv_sqrt).T

    U, s, Vt = np.linalg.svd(koopman, full_matrices=False)
    V = Vt.T

    mask = s > epsilon
    U = c00_inv_sqrt @ U[:, mask]
    V = ctt_inv_sqrt @ V[:, mask]
    s = s[mask]

    S = np.diag(s)
    score = np.trace(
        2.0 * S @ U.T @ c0t @ V -
        S @ U.T @ c00 @ U @ S @ V.T @ ctt @ V
    )
    return score + 1


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def seed():
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def correlated_data(seed):
    """Data with known correlation structure."""
    N, d = 200, 5
    data_np = np.random.randn(N, d)
    data_lagged_np = data_np * 0.8 + np.random.randn(N, d) * 0.3
    return data_np, data_lagged_np


@pytest.fixture
def softmax_data(seed):
    """Softmax probability outputs (realistic VAMPNet scenario)."""
    N, d = 150, 4
    logits = np.random.randn(N, d)
    logits_lagged = logits * 0.9 + np.random.randn(N, d) * 0.2

    # Softmax
    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    return softmax(logits), softmax(logits_lagged)


@pytest.fixture
def identity_dynamics(seed):
    """Perfect correlation (identity Koopman operator)."""
    N, d = 100, 3
    data_np = np.random.randn(N, d)
    return data_np, data_np.copy()


# =============================================================================
# Test: Covariance matrices
# =============================================================================

class TestCovarianceValidation:
    """Validate covariance computation against NumPy."""

    @pytest.mark.parametrize("remove_mean", [True, False])
    def test_covariances_match_numpy(self, correlated_data, remove_mean):
        """C00, C0t, Ctt from torch should match NumPy within float32 tolerance."""
        data_np, data_lagged_np = correlated_data
        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP2', epsilon=1e-10)

        # Torch covariances
        c00_t, c0t_t, ctt_t = scorer._covariances(data_t, data_lagged_t, remove_mean=remove_mean)

        # NumPy covariances (without epsilon regularization — added inside _covariances)
        c00_np, c0t_np, ctt_np = np_covariances(data_np, data_lagged_np, remove_mean=remove_mean)
        # Add epsilon to match what _covariances does
        eps = 1e-10
        c00_np += eps * np.eye(c00_np.shape[0])
        ctt_np += eps * np.eye(ctt_np.shape[0])

        atol = 1e-10
        np.testing.assert_allclose(c00_t.numpy(), c00_np, atol=atol,
                                   err_msg="C00 mismatch")
        np.testing.assert_allclose(c0t_t.numpy(), c0t_np, atol=atol,
                                   err_msg="C0t mismatch")
        np.testing.assert_allclose(ctt_t.numpy(), ctt_np, atol=atol,
                                   err_msg="Ctt mismatch")

    def test_covariance_formula_manual(self, seed):
        """Verify covariance formula against manual calculation on tiny data."""
        # 3 samples, 2 features — small enough to verify by hand
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Y = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])

        # Mean-center
        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)

        # Expected covariances
        c00_expected = (1.0 / 2) * Xc.T @ Xc  # N-1 = 2
        c0t_expected = (1.0 / 2) * Xc.T @ Yc
        ctt_expected = (1.0 / 2) * Yc.T @ Yc

        data_t = torch.tensor(X, dtype=torch.float64)
        data_lagged_t = torch.tensor(Y, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP2', epsilon=0.0)
        c00_t, c0t_t, ctt_t = scorer._covariances(data_t, data_lagged_t, remove_mean=True)

        np.testing.assert_allclose(c00_t.numpy(), c00_expected, atol=1e-12)
        np.testing.assert_allclose(c0t_t.numpy(), c0t_expected, atol=1e-12)
        np.testing.assert_allclose(ctt_t.numpy(), ctt_expected, atol=1e-12)


# =============================================================================
# Test: Symmetric matrix inverse square root
# =============================================================================

class TestSymInverseValidation:
    """Validate matrix inverse (square root) against NumPy/SciPy."""

    @pytest.mark.parametrize("mode", ["trunc", "regularize"])
    def test_sym_inverse_sqrt_matches_numpy(self, correlated_data, mode):
        """M^{-1/2} from torch should match NumPy eigendecomposition result."""
        data_np, _ = correlated_data
        eps = 1e-6

        # Build a positive-definite matrix from data
        X = data_np - data_np.mean(axis=0)
        mat_np = (1.0 / (X.shape[0] - 1)) * X.T @ X + eps * np.eye(X.shape[1])

        mat_t = torch.tensor(mat_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode=mode)
        result_t = scorer._sym_inverse(mat_t, return_sqrt=True, epsilon=eps, mode=mode)

        result_np = np_sym_inverse_sqrt(mat_np, epsilon=eps, mode=mode)

        np.testing.assert_allclose(result_t.numpy(), result_np, atol=1e-6,
                                   err_msg=f"M^(-1/2) mismatch in {mode} mode")

    def test_inverse_sqrt_times_itself_gives_inverse(self, correlated_data):
        """Verify: M^{-1/2} @ M^{-1/2} ≈ M^{-1}"""
        data_np, _ = correlated_data
        eps = 1e-6

        X = data_np - data_np.mean(axis=0)
        mat_np = (1.0 / (X.shape[0] - 1)) * X.T @ X + eps * np.eye(X.shape[1])

        mat_t = torch.tensor(mat_np, dtype=torch.float64)
        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode='regularize')

        inv_sqrt = scorer._sym_inverse(mat_t, return_sqrt=True, epsilon=eps, mode='regularize')
        inv_from_sqrt = inv_sqrt @ inv_sqrt

        inv_direct = scorer._sym_inverse(mat_t, return_sqrt=False, epsilon=eps, mode='regularize')

        np.testing.assert_allclose(inv_from_sqrt.numpy(), inv_direct.numpy(), atol=1e-8,
                                   err_msg="M^{-1/2} @ M^{-1/2} should equal M^{-1}")

    def test_inverse_times_original_is_identity(self, correlated_data):
        """Verify: M^{-1} @ M ≈ I"""
        data_np, _ = correlated_data
        eps = 1e-6

        X = data_np - data_np.mean(axis=0)
        mat_np = (1.0 / (X.shape[0] - 1)) * X.T @ X + eps * np.eye(X.shape[1])

        mat_t = torch.tensor(mat_np, dtype=torch.float64)
        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode='regularize')

        inv = scorer._sym_inverse(mat_t, return_sqrt=False, epsilon=eps, mode='regularize')
        product = inv @ mat_t

        np.testing.assert_allclose(product.numpy(), np.eye(mat_np.shape[0]), atol=1e-5,
                                   err_msg="M^{-1} @ M should be identity")

    def test_inverse_sqrt_against_scipy(self, correlated_data):
        """Cross-check M^{-1/2} against scipy.linalg.sqrtm(M^{-1})."""
        data_np, _ = correlated_data
        eps = 1e-6

        X = data_np - data_np.mean(axis=0)
        mat_np = (1.0 / (X.shape[0] - 1)) * X.T @ X + eps * np.eye(X.shape[1])

        # SciPy reference
        mat_inv = np.linalg.inv(mat_np)
        scipy_inv_sqrt = np.real(sqrtm(mat_inv))

        # Torch implementation
        mat_t = torch.tensor(mat_np, dtype=torch.float64)
        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode='regularize')
        torch_inv_sqrt = scorer._sym_inverse(mat_t, return_sqrt=True, epsilon=eps, mode='regularize')

        # They may differ by sign of eigenvectors, so compare M^{-1/2} @ M @ M^{-1/2} = I
        # instead of element-wise
        product_scipy = scipy_inv_sqrt @ mat_np @ scipy_inv_sqrt
        product_torch = torch_inv_sqrt.numpy() @ mat_np @ torch_inv_sqrt.numpy()

        np.testing.assert_allclose(product_scipy, np.eye(mat_np.shape[0]), atol=1e-8)
        np.testing.assert_allclose(product_torch, np.eye(mat_np.shape[0]), atol=1e-5)


# =============================================================================
# Test: Koopman matrix
# =============================================================================

class TestKoopmanValidation:
    """Validate Koopman matrix computation against NumPy."""

    @pytest.mark.parametrize("mode", ["trunc", "regularize"])
    def test_koopman_matches_numpy(self, correlated_data, mode):
        """K = (C00^{-1/2} C0t Ctt^{-1/2})^T should match NumPy."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode=mode)
        K_torch = scorer._koopman_matrix(data_t, data_lagged_t, epsilon=eps, mode=mode)

        K_numpy = np_koopman_matrix(data_np, data_lagged_np, epsilon=eps, mode=mode)

        np.testing.assert_allclose(K_torch.numpy(), K_numpy, atol=1e-5,
                                   err_msg=f"Koopman matrix mismatch in {mode} mode")

    def test_koopman_identity_dynamics(self, identity_dynamics):
        """For X_t = X_{t+τ}, K should have singular values close to 1."""
        data_np, data_lagged_np = identity_dynamics
        eps = 1e-6

        K = np_koopman_matrix(data_np, data_lagged_np, epsilon=eps, mode='regularize')
        singular_values = np.linalg.svd(K, compute_uv=False)

        # With identity dynamics, singular values should be ≈ 1
        np.testing.assert_allclose(singular_values, np.ones_like(singular_values), atol=0.05,
                                   err_msg="Identity dynamics should give singular values ≈ 1")

    def test_koopman_singular_values_bounded(self, correlated_data):
        """Singular values of K should be in [0, 1] for well-behaved data."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        K = np_koopman_matrix(data_np, data_lagged_np, epsilon=eps, mode='regularize')
        singular_values = np.linalg.svd(K, compute_uv=False)

        assert np.all(singular_values >= -0.01), "Singular values should be non-negative"
        assert np.all(singular_values <= 1.05), "Singular values should be <= 1 (with tolerance)"


# =============================================================================
# Test: VAMP scores
# =============================================================================

class TestVAMPScoreValidation:
    """Validate VAMP1/VAMP2/VAMPE scores against NumPy reference."""

    @pytest.mark.parametrize("mode", ["trunc", "regularize"])
    def test_vamp2_matches_numpy(self, correlated_data, mode):
        """VAMP2 = ||K||_F^2 + 1 should match NumPy."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode=mode)
        score_torch = scorer(data_t, data_lagged_t).item()
        score_numpy = np_vamp2_score(data_np, data_lagged_np, epsilon=eps, mode=mode)

        np.testing.assert_allclose(score_torch, score_numpy, rtol=1e-4,
                                   err_msg=f"VAMP2 mismatch in {mode} mode")

    @pytest.mark.parametrize("mode", ["trunc", "regularize"])
    def test_vamp1_matches_numpy(self, correlated_data, mode):
        """VAMP1 = ||K||_* + 1 should match NumPy."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP1', epsilon=eps, mode=mode)
        score_torch = scorer(data_t, data_lagged_t).item()
        score_numpy = np_vamp1_score(data_np, data_lagged_np, epsilon=eps, mode=mode)

        np.testing.assert_allclose(score_torch, score_numpy, rtol=1e-4,
                                   err_msg=f"VAMP1 mismatch in {mode} mode")

    @pytest.mark.parametrize("mode", ["trunc", "regularize"])
    def test_vampe_matches_numpy(self, correlated_data, mode):
        """VAMPE score should match NumPy reference."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMPE', epsilon=eps, mode=mode)
        score_torch = scorer(data_t, data_lagged_t).item()
        score_numpy = np_vampe_score(data_np, data_lagged_np, epsilon=eps, mode=mode)

        np.testing.assert_allclose(score_torch, score_numpy, rtol=1e-5,
                                   err_msg=f"VAMPE mismatch in {mode} mode")

    def test_vamp2_with_softmax_data(self, softmax_data):
        """VAMP2 with realistic softmax outputs should match NumPy."""
        data_np, data_lagged_np = softmax_data
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode='regularize')
        score_torch = scorer(data_t, data_lagged_t).item()
        score_numpy = np_vamp2_score(data_np, data_lagged_np, epsilon=eps, mode='regularize')

        np.testing.assert_allclose(score_torch, score_numpy, rtol=1e-4,
                                   err_msg="VAMP2 mismatch with softmax data")

    def test_identity_dynamics_score(self, identity_dynamics):
        """Perfect correlation: VAMP2 ≈ d + 1 (d = number of features)."""
        data_np, data_lagged_np = identity_dynamics
        d = data_np.shape[1]
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode='regularize')
        score = scorer(data_t, data_lagged_t).item()

        # With identity dynamics, all d singular values ≈ 1, so VAMP2 ≈ d + 1
        np.testing.assert_allclose(score, d + 1, atol=0.2,
                                   err_msg=f"Identity dynamics: VAMP2 should be ≈ {d + 1}")

    def test_score_ordering_vamp2(self, correlated_data, seed):
        """VAMP2: perfect > correlated > uncorrelated."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        np.random.seed(seed + 100)
        uncorrelated = np.random.randn(*data_np.shape)

        score_corr = np_vamp2_score(data_np, data_lagged_np, epsilon=eps, mode='regularize')
        score_perfect = np_vamp2_score(data_np, data_np.copy(), epsilon=eps, mode='regularize')
        score_uncorr = np_vamp2_score(data_np, uncorrelated, epsilon=eps, mode='regularize')

        assert score_perfect > score_corr > score_uncorr, (
            f"Expected perfect ({score_perfect:.4f}) > corr ({score_corr:.4f}) > uncorr ({score_uncorr:.4f})"
        )


# =============================================================================
# Test: VAMPE specific properties
# =============================================================================

class TestVAMPEProperties:
    """Test VAMPE-specific mathematical properties."""

    def test_vampe_equals_vamp2_for_orthonormal_eigfunctions(self, correlated_data):
        """
        For well-conditioned data, VAMPE and VAMP2 should give similar scores.
        They are equivalent when the basis functions are optimal.
        """
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer_v2 = VAMPScore(method='VAMP2', epsilon=eps, mode='regularize')
        scorer_ve = VAMPScore(method='VAMPE', epsilon=eps, mode='regularize')

        score_v2 = scorer_v2(data_t, data_lagged_t).item()
        score_ve = scorer_ve(data_t, data_lagged_t).item()

        # VAMPE ≤ VAMP2 in general (VAMPE is a tighter bound)
        # They should be in the same ballpark
        assert abs(score_v2 - score_ve) / max(abs(score_v2), 1.0) < 0.5, (
            f"VAMP2 ({score_v2:.4f}) and VAMPE ({score_ve:.4f}) should be comparable"
        )

    def test_vampe_non_negative(self, correlated_data):
        """VAMPE score should be >= 1 (including +1 constant)."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMPE', epsilon=eps, mode='regularize')
        score = scorer(data_t, data_lagged_t).item()

        assert score >= 0.95, f"VAMPE score should be >= 1 (got {score:.4f})"


# =============================================================================
# Test: Float32 precision
# =============================================================================

class TestFloat32Precision:
    """Test that float32 (typical training dtype) gives acceptable accuracy."""

    def test_vamp2_float32_vs_float64(self, correlated_data):
        """VAMP2 in float32 should be close to float64 reference."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        data_f32 = torch.tensor(data_np, dtype=torch.float32)
        data_lagged_f32 = torch.tensor(data_lagged_np, dtype=torch.float32)
        data_f64 = torch.tensor(data_np, dtype=torch.float64)
        data_lagged_f64 = torch.tensor(data_lagged_np, dtype=torch.float64)

        scorer = VAMPScore(method='VAMP2', epsilon=eps, mode='regularize')

        score_f32 = scorer(data_f32, data_lagged_f32).item()
        score_f64 = scorer(data_f64, data_lagged_f64).item()

        np.testing.assert_allclose(score_f32, score_f64, rtol=1e-3,
                                   err_msg="Float32 should be within 0.1% of float64")

    def test_covariance_float32_vs_numpy(self, correlated_data):
        """Covariance matrices in float32 should be close to NumPy float64."""
        data_np, data_lagged_np = correlated_data
        eps = 1e-6

        data_t = torch.tensor(data_np, dtype=torch.float32)
        data_lagged_t = torch.tensor(data_lagged_np, dtype=torch.float32)

        scorer = VAMPScore(method='VAMP2', epsilon=eps)
        c00_t, c0t_t, ctt_t = scorer._covariances(data_t, data_lagged_t, remove_mean=True)

        c00_np, c0t_np, ctt_np = np_covariances(data_np, data_lagged_np, remove_mean=True)
        c00_np += eps * np.eye(c00_np.shape[0])
        ctt_np += eps * np.eye(ctt_np.shape[0])

        np.testing.assert_allclose(c00_t.numpy(), c00_np, atol=1e-4,
                                   err_msg="C00 float32 mismatch")
        np.testing.assert_allclose(c0t_t.numpy(), c0t_np, atol=1e-4,
                                   err_msg="C0t float32 mismatch")


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
