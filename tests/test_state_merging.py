"""
Unit tests for state merging via softmax output summing.

Tests cover:
- Softmax probability merging (sum correctness, mapping, edge cases)
- VAMP-2 score computation from probabilities
- Merge validation (quality drop check)
- High-level merge_and_validate pipeline
- Report serialization
"""

import pytest
import numpy as np
import os
import json


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def seed():
    np.random.seed(42)
    return 42


@pytest.fixture
def simple_5state_probs(seed):
    """
    5-state system with 1000 frames.
    States 2 and 3 are redundant (similar probabilities).
    """
    n_frames = 1000
    n_states = 5
    probs = np.zeros((n_frames, n_states))
    for i in range(n_frames):
        state = i % 3  # Cycle through states 0, 1, 2
        if state == 2:
            # Split probability between states 2 and 3 (redundant pair)
            probs[i, 2] = 0.40
            probs[i, 3] = 0.40
            probs[i, 0] = 0.08
            probs[i, 1] = 0.08
            probs[i, 4] = 0.04
        else:
            probs[i, state] = 0.80
            probs[i, :] += 0.04
            probs[i, state] = 0.80  # Override the += 0.04
            remaining = 0.20
            for j in range(n_states):
                if j != state:
                    probs[i, j] = remaining / (n_states - 1)
    return probs


@pytest.fixture
def markov_3state_probs(seed):
    """
    3-state Markov chain with clear kinetic separation.
    Returns probs and the transition matrix.
    """
    T = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ])
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
    return probs, T


# ============================================================================
# merge_states tests
# ============================================================================

class TestMergeStates:
    def test_no_merge_groups(self):
        """Empty merge groups should return the original probabilities."""
        from pygv.utils.state_merging import merge_states

        probs = np.random.dirichlet(np.ones(4), size=100)
        merged, mapping = merge_states(probs, [])

        assert merged.shape == probs.shape
        assert np.allclose(merged, probs)
        assert len(mapping) == 4

    def test_merge_two_states(self):
        """Merging states 1 and 2 should sum their columns."""
        from pygv.utils.state_merging import merge_states

        probs = np.array([
            [0.5, 0.2, 0.2, 0.1],
            [0.1, 0.4, 0.3, 0.2],
            [0.3, 0.1, 0.1, 0.5],
        ])
        merged, mapping = merge_states(probs, [{1, 2}])

        assert merged.shape == (3, 3)
        # State 0 → new state 0
        np.testing.assert_allclose(merged[:, 0], probs[:, 0])
        # States 1+2 → new state 1
        np.testing.assert_allclose(merged[:, 1], probs[:, 1] + probs[:, 2])
        # State 3 → new state 2
        np.testing.assert_allclose(merged[:, 2], probs[:, 3])

    def test_merged_probs_sum_to_one(self, simple_5state_probs):
        """Merged probabilities should still sum to 1 per frame."""
        from pygv.utils.state_merging import merge_states

        merged, _ = merge_states(simple_5state_probs, [{2, 3}])
        row_sums = merged.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_merge_reduces_state_count(self):
        """Merging 2 groups from 6 states should give 4 states."""
        from pygv.utils.state_merging import merge_states

        probs = np.random.dirichlet(np.ones(6), size=50)
        merged, mapping = merge_states(probs, [{0, 1}, {3, 4}])

        assert merged.shape[1] == 4
        assert len(mapping) == 4

    def test_state_mapping_covers_all_original(self):
        """Every original state should appear in exactly one mapping entry."""
        from pygv.utils.state_merging import merge_states

        probs = np.random.dirichlet(np.ones(5), size=20)
        _, mapping = merge_states(probs, [{1, 3}])

        all_original = set()
        for old_states in mapping.values():
            all_original.update(old_states)
        assert all_original == {0, 1, 2, 3, 4}

    def test_merge_three_states_into_one(self):
        """Merging 3 states into one group should work."""
        from pygv.utils.state_merging import merge_states

        probs = np.array([
            [0.2, 0.3, 0.1, 0.2, 0.2],
            [0.1, 0.1, 0.4, 0.3, 0.1],
        ])
        merged, mapping = merge_states(probs, [{1, 2, 3}])

        assert merged.shape == (2, 3)
        # States 1+2+3 should be summed
        np.testing.assert_allclose(
            merged[:, 1],
            probs[:, 1] + probs[:, 2] + probs[:, 3],
        )

    def test_multiple_merge_groups(self):
        """Two independent merge groups should both be applied."""
        from pygv.utils.state_merging import merge_states

        probs = np.random.dirichlet(np.ones(6), size=30)
        merged, mapping = merge_states(probs, [{0, 1}, {4, 5}])

        # 6 original - 2 absorbed = 4 merged
        assert merged.shape[1] == 4
        # Check both groups are present
        merged_sizes = [len(v) for v in mapping.values()]
        assert sorted(merged_sizes) == [1, 1, 2, 2]


# ============================================================================
# VAMP-2 computation tests
# ============================================================================

class TestVAMP2Computation:
    def test_vamp2_positive(self, markov_3state_probs):
        """VAMP-2 score should be positive for a proper Markov system."""
        from pygv.utils.state_merging import _compute_vamp2_from_probs

        probs, _ = markov_3state_probs
        score = _compute_vamp2_from_probs(probs, lag_frames=10)
        assert score > 1.0  # VAMP-2 >= 1 for normalized processes

    def test_vamp2_increases_with_signal(self):
        """VAMP-2 should be higher for correlated data than for random."""
        from pygv.utils.state_merging import _compute_vamp2_from_probs

        # Correlated: smooth state transitions
        n = 2000
        t = np.linspace(0, 10 * np.pi, n)
        corr_probs = np.column_stack([
            0.5 + 0.4 * np.sin(t),
            0.5 - 0.4 * np.sin(t),
        ])

        # Random: no temporal correlation
        np.random.seed(123)
        rand_probs = np.random.dirichlet(np.ones(2), size=n)

        score_corr = _compute_vamp2_from_probs(corr_probs, lag_frames=5)
        score_rand = _compute_vamp2_from_probs(rand_probs, lag_frames=5)
        assert score_corr > score_rand

    def test_vamp2_short_trajectory(self):
        """Trajectory shorter than lag should return 0."""
        from pygv.utils.state_merging import _compute_vamp2_from_probs

        probs = np.random.dirichlet(np.ones(3), size=5)
        score = _compute_vamp2_from_probs(probs, lag_frames=10)
        assert score == 0.0

    def test_vamp2_at_minimum(self):
        """For uncorrelated data, VAMP-2 should be low (near 1)."""
        from pygv.utils.state_merging import _compute_vamp2_from_probs

        np.random.seed(99)
        probs = np.random.dirichlet(np.ones(3), size=5000)
        score = _compute_vamp2_from_probs(probs, lag_frames=1)
        # Should be positive; for random data VAMP-2 is low
        assert score > 0


# ============================================================================
# validate_merge tests
# ============================================================================

class TestValidateMerge:
    def test_no_merge_passes(self, markov_3state_probs):
        """Keeping all states should have zero VAMP-2 drop."""
        from pygv.utils.state_merging import validate_merge

        probs, _ = markov_3state_probs
        v_orig, v_merged, drop, passed = validate_merge(
            original_probs=probs,
            merged_probs=probs,  # Same probs = no actual merge
            lag_time=1.0,
            stride=1,
            timestep=0.1,
        )
        assert v_orig > 0
        assert np.isclose(drop, 0.0, atol=1e-10)
        assert passed

    def test_large_merge_may_fail(self):
        """Drastic merging (5→2) of distinct states should show quality loss."""
        from pygv.utils.state_merging import validate_merge, merge_states

        np.random.seed(42)
        # 5 distinct states with clear separation
        T = np.eye(5) * 0.85 + np.ones((5, 5)) * 0.03
        n_frames = 5000
        states = np.zeros(n_frames, dtype=int)
        for i in range(1, n_frames):
            states[i] = np.random.choice(5, p=T[states[i - 1]])
        probs = np.zeros((n_frames, 5))
        for i in range(n_frames):
            probs[i, states[i]] = 0.90
            for j in range(5):
                if j != states[i]:
                    probs[i, j] = 0.025

        # Merge 3 distinct states into 1
        merged, _ = merge_states(probs, [{0, 1, 2}])

        v_orig, v_merged, drop, passed = validate_merge(
            original_probs=probs,
            merged_probs=merged,
            lag_time=1.0,
            stride=1,
            timestep=0.1,
            vamp2_drop_threshold=0.01,  # Very strict
        )
        # Merging distinct states should cause a noticeable drop
        assert drop > 0

    def test_threshold_controls_pass_fail(self, markov_3state_probs):
        """Strict threshold should fail more easily than lenient one."""
        from pygv.utils.state_merging import validate_merge, merge_states

        probs, _ = markov_3state_probs
        # Merge states 0 and 1
        merged, _ = merge_states(probs, [{0, 1}])

        _, _, drop_val, passed_lenient = validate_merge(
            probs, merged, lag_time=1.0, stride=1, timestep=0.1,
            vamp2_drop_threshold=0.99,
        )
        _, _, _, passed_strict = validate_merge(
            probs, merged, lag_time=1.0, stride=1, timestep=0.1,
            vamp2_drop_threshold=0.001,
        )
        # Lenient should always pass
        assert passed_lenient
        # If there's any drop at all, strict may fail
        if drop_val > 0.001:
            assert not passed_strict


# ============================================================================
# merge_and_validate tests
# ============================================================================

class TestMergeAndValidate:
    def test_basic_merge(self, simple_5state_probs):
        """merge_and_validate should return a properly structured MergeResult."""
        from pygv.utils.state_merging import merge_and_validate

        result = merge_and_validate(
            probs=simple_5state_probs,
            merge_groups=[{2, 3}],
            lag_time=1.0,
            stride=1,
            timestep=0.1,
        )

        assert result.original_n_states == 5
        assert result.merged_n_states == 4
        assert result.merged_probs.shape[1] == 4
        assert result.vamp2_original is not None
        assert result.vamp2_merged is not None
        assert result.vamp2_drop is not None

    def test_skip_validation(self, simple_5state_probs):
        """With validate=False, VAMP-2 fields should stay None."""
        from pygv.utils.state_merging import merge_and_validate

        result = merge_and_validate(
            probs=simple_5state_probs,
            merge_groups=[{2, 3}],
            lag_time=1.0,
            stride=1,
            timestep=0.1,
            validate=False,
        )

        assert result.merged_n_states == 4
        assert result.vamp2_original is None
        assert result.vamp2_merged is None
        assert result.vamp2_drop is None
        assert result.validation_passed is True

    def test_empty_merge_groups(self, simple_5state_probs):
        """Empty merge groups should return unchanged state count."""
        from pygv.utils.state_merging import merge_and_validate

        result = merge_and_validate(
            probs=simple_5state_probs,
            merge_groups=[],
            lag_time=1.0,
            stride=1,
            timestep=0.1,
        )

        assert result.original_n_states == 5
        assert result.merged_n_states == 5
        # No validation performed for empty merge groups
        assert result.vamp2_original is None

    def test_summary_string(self, simple_5state_probs):
        """summary() should produce a non-empty string."""
        from pygv.utils.state_merging import merge_and_validate

        result = merge_and_validate(
            probs=simple_5state_probs,
            merge_groups=[{2, 3}],
            lag_time=1.0,
            stride=1,
            timestep=0.1,
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Merging Result" in summary or "State Merging" in summary
        assert "5" in summary  # Original n_states


# ============================================================================
# save_merge_report tests
# ============================================================================

class TestSaveMergeReport:
    def test_save_and_load(self, tmp_path, markov_3state_probs):
        """Report should be saved as valid JSON and contain expected keys."""
        from pygv.utils.state_merging import merge_and_validate, save_merge_report
        from pygv.utils.state_diagnostics import recommend_state_reduction

        probs, T = markov_3state_probs
        report = recommend_state_reduction(T, probs)
        merge_result = merge_and_validate(
            probs=probs,
            merge_groups=report.merge_groups,
            lag_time=1.0,
            stride=1,
            timestep=0.1,
        )

        path = save_merge_report(report, merge_result, str(tmp_path), "test_protein")

        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)

        assert "diagnostics" in data
        assert data["diagnostics"]["original_n_states"] == 3
        assert "eigenvalues" in data["diagnostics"]
        assert "populations" in data["diagnostics"]
        assert "confidence" in data["diagnostics"]
        assert "recommendation" in data["diagnostics"]

    def test_save_without_merge(self, tmp_path, markov_3state_probs):
        """Should handle None merge_result gracefully."""
        from pygv.utils.state_merging import save_merge_report
        from pygv.utils.state_diagnostics import recommend_state_reduction

        probs, T = markov_3state_probs
        report = recommend_state_reduction(T, probs)

        path = save_merge_report(report, None, str(tmp_path), "test_protein")

        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)

        assert "diagnostics" in data
        assert "merge" not in data

    def test_save_with_merge_data(self, tmp_path, simple_5state_probs):
        """When merging was performed, merge section should be present."""
        from pygv.utils.state_merging import merge_and_validate, save_merge_report
        from pygv.utils.state_diagnostics import StateReductionReport

        merge_result = merge_and_validate(
            probs=simple_5state_probs,
            merge_groups=[{2, 3}],
            lag_time=1.0,
            stride=1,
            timestep=0.1,
        )

        # Create minimal report
        report = StateReductionReport(
            original_n_states=5,
            eigenvalues=np.array([1.0, 0.9, 0.8, 0.3, 0.1]),
            gap_ratios=np.array([1.1, 1.1, 2.7, 3.0]),
            populations=np.array([0.25, 0.25, 0.2, 0.2, 0.1]),
            merge_groups=[{2, 3}],
            effective_n_states=4,
        )

        path = save_merge_report(report, merge_result, str(tmp_path), "test")

        with open(path) as f:
            data = json.load(f)

        assert "merge" in data
        assert data["merge"]["original_n_states"] == 5
        assert data["merge"]["merged_n_states"] == 4
        assert "state_mapping" in data["merge"]
        assert "vamp2_original" in data["merge"]
        assert "validation_passed" in data["merge"]
