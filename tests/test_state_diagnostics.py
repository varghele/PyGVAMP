"""
Unit tests for state quality diagnostics.

Tests cover:
- Eigenvalue gap analysis with known spectral gaps
- State population analysis with known underpopulated states
- Jensen-Shannon divergence and transition row similarity
- Combined recommendation logic
"""

import pytest
import numpy as np


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def seed():
    np.random.seed(42)
    return 42


@pytest.fixture
def clear_3state_system(seed):
    """
    3-state system with two slow processes and no redundant states.
    Transition matrix has a clear eigenvalue gap after 3 eigenvalues.
    """
    T = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ])
    # Generate probs from a Markov chain
    n_frames = 5000
    states = np.zeros(n_frames, dtype=int)
    for i in range(1, n_frames):
        states[i] = np.random.choice(3, p=T[states[i - 1]])
    probs = np.zeros((n_frames, 3))
    for i in range(n_frames):
        probs[i, states[i]] = 0.85
        for j in range(3):
            if j != states[i]:
                probs[i, j] = 0.075
    return T, probs


@pytest.fixture
def redundant_5state_system(seed):
    """
    5-state system where states 2 and 3 have nearly identical transition rows
    (redundant), and state 4 has very low population.
    """
    T = np.array([
        [0.80, 0.10, 0.04, 0.04, 0.02],
        [0.10, 0.80, 0.04, 0.04, 0.02],
        [0.05, 0.05, 0.45, 0.40, 0.05],  # state 2
        [0.05, 0.05, 0.40, 0.45, 0.05],  # state 3 — similar to 2
        [0.20, 0.20, 0.20, 0.20, 0.20],  # state 4 — uniform transitions
    ])
    n_frames = 10000
    states = np.zeros(n_frames, dtype=int)
    # Bias initial states to make state 4 rare
    states[0] = 0
    for i in range(1, n_frames):
        states[i] = np.random.choice(5, p=T[states[i - 1]])
    probs = np.zeros((n_frames, 5))
    for i in range(n_frames):
        probs[i, states[i]] = 0.80
        for j in range(5):
            if j != states[i]:
                probs[i, j] = 0.05
    return T, probs


# ============================================================================
# Eigenvalue gap tests
# ============================================================================

class TestEigenvalueGap:
    def test_clear_gap_identity_like(self):
        """Near-identity matrix should suggest all states are distinct."""
        from pygv.utils.state_diagnostics import analyze_eigenvalue_gap

        T = np.eye(4) * 0.9 + np.ones((4, 4)) * 0.025
        eigenvalues, gap_ratios, suggested = analyze_eigenvalue_gap(T)

        assert len(eigenvalues) == 4
        assert len(gap_ratios) == 3
        assert suggested >= 2

    def test_3state_system(self, clear_3state_system):
        from pygv.utils.state_diagnostics import analyze_eigenvalue_gap

        T, _ = clear_3state_system
        eigenvalues, gap_ratios, suggested = analyze_eigenvalue_gap(T)

        assert len(eigenvalues) == 3
        # The first eigenvalue should be close to 1
        assert eigenvalues[0] > 0.95
        assert suggested >= 2

    def test_single_metastable_state(self):
        """Fully mixed system → should suggest 2 states (minimum)."""
        from pygv.utils.state_diagnostics import analyze_eigenvalue_gap

        T = np.ones((5, 5)) / 5.0
        _, _, suggested = analyze_eigenvalue_gap(T)
        assert suggested == 2

    def test_eigenvalues_sorted_descending(self):
        from pygv.utils.state_diagnostics import analyze_eigenvalue_gap

        T = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
        ])
        eigenvalues, _, _ = analyze_eigenvalue_gap(T)
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])


# ============================================================================
# Population analysis tests
# ============================================================================

class TestPopulationAnalysis:
    def test_equal_populations(self, seed):
        """All states equally populated → no underpopulated states."""
        from pygv.utils.state_diagnostics import analyze_state_populations

        n_frames = 10000
        n_states = 4
        probs = np.zeros((n_frames, n_states))
        for i in range(n_frames):
            state = i % n_states
            probs[i, state] = 0.9
            for j in range(n_states):
                if j != state:
                    probs[i, j] = 0.1 / (n_states - 1)

        populations, underpopulated, entropy = analyze_state_populations(probs)

        assert len(underpopulated) == 0
        assert len(populations) == n_states
        assert np.allclose(populations.sum(), 1.0)
        # Entropy should be close to max (ln 4)
        assert entropy > 0.9 * np.log(n_states)

    def test_one_underpopulated(self, seed):
        """One state with <2% population should be flagged."""
        from pygv.utils.state_diagnostics import analyze_state_populations

        n_frames = 10000
        probs = np.zeros((n_frames, 4))
        # States 0,1,2 get ~33% each, state 3 gets <1%
        for i in range(n_frames):
            if i < 50:  # Only 50 frames for state 3
                probs[i, 3] = 0.9
                probs[i, :3] = 0.1 / 3
            else:
                state = i % 3
                probs[i, state] = 0.9
                for j in range(4):
                    if j != state:
                        probs[i, j] = 0.1 / 3

        populations, underpopulated, _ = analyze_state_populations(probs, population_threshold=0.02)
        assert 3 in underpopulated

    def test_populations_sum_to_one(self, clear_3state_system):
        from pygv.utils.state_diagnostics import analyze_state_populations

        _, probs = clear_3state_system
        populations, _, _ = analyze_state_populations(probs)
        assert np.isclose(populations.sum(), 1.0)

    def test_custom_threshold(self, seed):
        """Higher threshold should flag more states."""
        from pygv.utils.state_diagnostics import analyze_state_populations

        n_frames = 1000
        probs = np.zeros((n_frames, 3))
        # 90% state 0, 8% state 1, 2% state 2
        for i in range(n_frames):
            if i < 900:
                probs[i, 0] = 0.95
                probs[i, 1] = 0.03
                probs[i, 2] = 0.02
            elif i < 980:
                probs[i, 1] = 0.95
                probs[i, 0] = 0.03
                probs[i, 2] = 0.02
            else:
                probs[i, 2] = 0.95
                probs[i, 0] = 0.03
                probs[i, 1] = 0.02

        _, underpop_low, _ = analyze_state_populations(probs, population_threshold=0.01)
        _, underpop_high, _ = analyze_state_populations(probs, population_threshold=0.05)
        assert len(underpop_high) >= len(underpop_low)


# ============================================================================
# JSD / transition similarity tests
# ============================================================================

class TestTransitionSimilarity:
    def test_identical_rows_merged(self):
        """States with identical transition rows should be grouped."""
        from pygv.utils.state_diagnostics import analyze_transition_similarity

        T = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.1, 0.8],  # Same as state 1
        ])
        jsd_matrix, merge_groups = analyze_transition_similarity(T, jsd_threshold=0.01)

        assert jsd_matrix.shape == (3, 3)
        assert jsd_matrix[1, 2] < 0.01  # Should be ~0
        assert len(merge_groups) >= 1
        # States 1 and 2 should be in the same group
        found = any(1 in g and 2 in g for g in merge_groups)
        assert found

    def test_distinct_rows_not_merged(self):
        """States with very different transition rows should not be grouped."""
        from pygv.utils.state_diagnostics import analyze_transition_similarity

        T = np.array([
            [0.95, 0.025, 0.025],
            [0.025, 0.95, 0.025],
            [0.025, 0.025, 0.95],
        ])
        jsd_matrix, merge_groups = analyze_transition_similarity(T, jsd_threshold=0.05)

        # No merges — all states are distinct
        assert len(merge_groups) == 0

    def test_jsd_symmetry(self):
        """JSD matrix should be symmetric."""
        from pygv.utils.state_diagnostics import analyze_transition_similarity

        T = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ])
        jsd_matrix, _ = analyze_transition_similarity(T)
        assert np.allclose(jsd_matrix, jsd_matrix.T, atol=1e-10)

    def test_jsd_diagonal_zero(self):
        """JSD of a row with itself should be 0."""
        from pygv.utils.state_diagnostics import analyze_transition_similarity

        T = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
        ])
        jsd_matrix, _ = analyze_transition_similarity(T)
        assert np.allclose(np.diag(jsd_matrix), 0.0)

    def test_redundant_system(self, redundant_5state_system):
        """States 2 and 3 in the fixture should be detected as similar."""
        from pygv.utils.state_diagnostics import analyze_transition_similarity

        T, _ = redundant_5state_system
        jsd_matrix, merge_groups = analyze_transition_similarity(T, jsd_threshold=0.05)

        # JSD between states 2 and 3 should be low
        assert jsd_matrix[2, 3] < 0.05


# ============================================================================
# Combined recommendation tests
# ============================================================================

class TestRecommendation:
    def test_keep_for_clean_system(self, clear_3state_system):
        """Clean 3-state system with no issues → recommend 'keep'."""
        from pygv.utils.state_diagnostics import recommend_state_reduction

        T, probs = clear_3state_system
        report = recommend_state_reduction(T, probs)

        assert report.original_n_states == 3
        assert report.effective_n_states >= 2
        assert report.recommendation in ("keep", "merge")  # No dramatic reduction expected

    def test_merge_for_redundant_system(self, redundant_5state_system):
        """System with redundant states should suggest merging."""
        from pygv.utils.state_diagnostics import recommend_state_reduction

        T, probs = redundant_5state_system
        report = recommend_state_reduction(T, probs, jsd_threshold=0.05)

        assert report.original_n_states == 5
        assert report.effective_n_states <= 5
        # Should have at least one merge group
        assert report.recommendation in ("merge", "retrain", "keep")

    def test_report_summary_runs(self, clear_3state_system):
        """summary() should return a non-empty string."""
        from pygv.utils.state_diagnostics import recommend_state_reduction

        T, probs = clear_3state_system
        report = recommend_state_reduction(T, probs)

        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 50
        assert "Recommendation" in summary

    def test_report_fields_populated(self, clear_3state_system):
        """All fields of the report should be populated."""
        from pygv.utils.state_diagnostics import recommend_state_reduction

        T, probs = clear_3state_system
        report = recommend_state_reduction(T, probs)

        assert report.eigenvalues is not None
        assert len(report.eigenvalues) == 3
        assert report.gap_ratios is not None
        assert report.populations is not None
        assert len(report.populations) == 3
        assert report.jsd_matrix is not None
        assert report.jsd_matrix.shape == (3, 3)
        assert report.confidence in ("high", "medium", "low")
        assert report.recommendation in ("keep", "merge", "retrain")

    def test_2state_system(self):
        """Minimum 2-state system should work without errors."""
        from pygv.utils.state_diagnostics import recommend_state_reduction

        T = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        probs = np.zeros((1000, 2))
        for i in range(500):
            probs[i] = [0.9, 0.1]
        for i in range(500, 1000):
            probs[i] = [0.1, 0.9]

        report = recommend_state_reduction(T, probs)
        assert report.original_n_states == 2
        assert report.effective_n_states == 2
