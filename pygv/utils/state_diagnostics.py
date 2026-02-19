"""
State quality diagnostics for VAMPNet analysis.

Provides three diagnostic signals to determine the effective number of metastable
states from a trained model's outputs:
1. Eigenvalue gap analysis of the Koopman operator
2. State population analysis (detection of underpopulated states)
3. Transition row similarity via Jensen-Shannon divergence (redundancy detection)

These diagnostics operate on softmax probabilities and the derived transition matrix,
requiring no retraining.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class StateReductionReport:
    """Report from combined state quality diagnostics."""

    # Required fields (no defaults) must come first
    original_n_states: int
    eigenvalues: np.ndarray = field(repr=False)
    gap_ratios: np.ndarray = field(repr=False)
    populations: np.ndarray = field(repr=False)

    # Fields with defaults
    eigenvalue_gap_suggestion: int = 0
    underpopulated_states: List[int] = field(default_factory=list)
    population_entropy: float = 0.0
    jsd_matrix: np.ndarray = field(repr=False, default=None)
    merge_groups: List[Set[int]] = field(default_factory=list)
    effective_n_states: int = 0
    confidence: str = "low"  # "high", "medium", "low"
    recommendation: str = "keep"  # "keep", "merge", "retrain"

    def summary(self) -> str:
        """Return a human-readable summary of the diagnostic results."""
        lines = [
            f"State Reduction Diagnostics (original: {self.original_n_states} states)",
            "=" * 60,
            "",
            f"Eigenvalue gap suggests: {self.eigenvalue_gap_suggestion} states",
            f"  Eigenvalues: {np.array2string(np.abs(self.eigenvalues), precision=4, separator=', ')}",
            f"  Gap ratios:  {np.array2string(self.gap_ratios, precision=3, separator=', ')}",
            "",
            f"Underpopulated states: {self.underpopulated_states or 'none'}",
            f"  Populations: {np.array2string(self.populations, precision=4, separator=', ')}",
            f"  Entropy:     {self.population_entropy:.3f} (max={np.log(self.original_n_states):.3f})",
            "",
            f"Merge groups (JSD-based): {[sorted(g) for g in self.merge_groups] or 'none'}",
            "",
            f"Effective n_states: {self.effective_n_states}",
            f"Confidence: {self.confidence}",
            f"Recommendation: {self.recommendation}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1a. Eigenvalue gap analysis
# ---------------------------------------------------------------------------

def analyze_eigenvalue_gap(
    transition_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute eigenvalues of the transition matrix and identify the spectral gap.

    The largest gap ratio after the stationary eigenvalue (|lambda_1| ~ 1) indicates
    the natural number of metastable states.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Row-stochastic transition matrix of shape (n_states, n_states).

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalue magnitudes sorted in descending order.
    gap_ratios : np.ndarray
        Ratio |lambda_i| / |lambda_{i+1}| for consecutive eigenvalue pairs.
        Length is n_states - 1.
    suggested_n_states : int
        Number of states suggested by the largest spectral gap (>= 2).
    """
    eigvals = np.linalg.eigvals(transition_matrix)
    magnitudes = np.sort(np.abs(eigvals))[::-1]

    # Compute gap ratios (avoid division by zero)
    gap_ratios = np.zeros(len(magnitudes) - 1)
    for i in range(len(gap_ratios)):
        denom = magnitudes[i + 1] if magnitudes[i + 1] > 1e-12 else 1e-12
        gap_ratios[i] = magnitudes[i] / denom

    # The largest gap after the stationary eigenvalue (index 0) suggests the
    # number of slow processes. If the gap is at index k, that means k+1
    # eigenvalues are "large" → k+1 states (including stationary).
    # We search from index 1 onwards (skip the trivial gap at index 0).
    if len(gap_ratios) > 1:
        # Index within gap_ratios[1:] that has the max gap
        search_gaps = gap_ratios[1:]
        best_gap_offset = int(np.argmax(search_gaps))
        suggested_n_states = best_gap_offset + 2  # +1 for 0-index, +1 because gap at k means k+1 states
    else:
        suggested_n_states = len(magnitudes)

    # Clamp to at least 2
    suggested_n_states = max(2, suggested_n_states)

    return magnitudes, gap_ratios, suggested_n_states


# ---------------------------------------------------------------------------
# 1b. State population analysis
# ---------------------------------------------------------------------------

def analyze_state_populations(
    probs: np.ndarray,
    population_threshold: float = 0.02,
) -> Tuple[np.ndarray, List[int], float]:
    """
    Identify underpopulated states that are likely transition-region artifacts.

    Parameters
    ----------
    probs : np.ndarray
        State probabilities of shape (n_frames, n_states).
    population_threshold : float
        Minimum fractional population for a state to be considered real.

    Returns
    -------
    populations : np.ndarray
        Fractional population of each state.
    underpopulated_states : list of int
        Indices of states below the population threshold.
    entropy : float
        Shannon entropy of the population distribution. Low entropy relative to
        max (log n_states) suggests a few dominant states; high entropy suggests
        over-segmentation into many equally sized states.
    """
    n_states = probs.shape[1]
    assignments = np.argmax(probs, axis=1)

    counts = np.bincount(assignments, minlength=n_states).astype(float)
    populations = counts / counts.sum()

    underpopulated = [i for i in range(n_states) if populations[i] < population_threshold]

    # Shannon entropy (use natural log to match max = ln(n_states))
    nonzero = populations[populations > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))

    return populations, underpopulated, entropy


# ---------------------------------------------------------------------------
# 1c. Transition row similarity (redundancy detection)
# ---------------------------------------------------------------------------

def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two probability distributions.

    Returns a value in [0, 1] (using base-2 logarithm).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Ensure valid distributions
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    def _kl(a, b):
        return np.sum(a * np.log2(a / b))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def analyze_transition_similarity(
    transition_matrix: np.ndarray,
    jsd_threshold: float = 0.05,
) -> Tuple[np.ndarray, List[Set[int]]]:
    """
    Identify kinetically redundant states by comparing transition probability rows.

    States whose transition rows have Jensen-Shannon divergence below the threshold
    are grouped as merge candidates via agglomerative clustering.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Row-stochastic transition matrix of shape (n_states, n_states).
    jsd_threshold : float
        Maximum JSD for two states to be considered redundant.

    Returns
    -------
    jsd_matrix : np.ndarray
        Pairwise JSD matrix of shape (n_states, n_states).
    merge_groups : list of set of int
        Groups of state indices that should be merged. Only groups with >= 2
        states are returned.
    """
    n_states = transition_matrix.shape[0]

    # Compute pairwise JSD matrix
    jsd_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i + 1, n_states):
            d = _jsd(transition_matrix[i], transition_matrix[j])
            jsd_matrix[i, j] = d
            jsd_matrix[j, i] = d

    # Agglomerative clustering on the JSD distance matrix
    if n_states < 2:
        return jsd_matrix, []

    condensed = squareform(jsd_matrix)
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=jsd_threshold, criterion='distance')

    # Build merge groups from cluster labels
    groups_dict = {}
    for state_idx, label in enumerate(cluster_labels):
        groups_dict.setdefault(label, set()).add(state_idx)

    # Only return groups with 2+ states (singletons aren't merges)
    merge_groups = [g for g in groups_dict.values() if len(g) >= 2]

    return jsd_matrix, merge_groups


# ---------------------------------------------------------------------------
# 1d. Combined recommendation
# ---------------------------------------------------------------------------

def recommend_state_reduction(
    transition_matrix: np.ndarray,
    probs: np.ndarray,
    population_threshold: float = 0.02,
    jsd_threshold: float = 0.05,
) -> StateReductionReport:
    """
    Run all three diagnostics and produce a combined state reduction recommendation.

    Decision logic:
    - If eigenvalue gap, population analysis, and redundancy detection all agree
      on a similar effective state count → high confidence
    - If effective_n_states >= 0.7 * original_n_states → recommend "merge"
    - If effective_n_states < 0.7 * original_n_states → recommend "retrain"
    - If no issues detected → recommend "keep"

    Parameters
    ----------
    transition_matrix : np.ndarray
        Row-stochastic transition matrix of shape (n_states, n_states).
    probs : np.ndarray
        State probabilities of shape (n_frames, n_states).
    population_threshold : float
        Minimum fractional population for a state to be considered real.
    jsd_threshold : float
        Maximum JSD for two states to be considered kinetically redundant.

    Returns
    -------
    StateReductionReport
        Full diagnostic report with recommendation.
    """
    n_states = transition_matrix.shape[0]

    # Run individual diagnostics
    eigenvalues, gap_ratios, eigenvalue_suggestion = analyze_eigenvalue_gap(transition_matrix)
    populations, underpopulated, entropy = analyze_state_populations(probs, population_threshold)
    jsd_matrix, merge_groups = analyze_transition_similarity(transition_matrix, jsd_threshold)

    # --- Compute effective n_states from each signal ---

    # From eigenvalue gap
    n_from_eigenvalues = eigenvalue_suggestion

    # From populations: original states minus underpopulated ones
    n_from_populations = n_states - len(underpopulated)

    # From JSD merge groups: original states minus the states that would be
    # absorbed into other states
    states_absorbed = sum(len(g) - 1 for g in merge_groups)
    n_from_jsd = n_states - states_absorbed

    # --- Combine signals ---
    # Use the median of the three estimates as the effective count
    estimates = [n_from_eigenvalues, n_from_populations, n_from_jsd]
    effective_n_states = int(np.median(estimates))
    effective_n_states = max(2, min(effective_n_states, n_states))

    # --- Assess confidence ---
    # If all three signals agree within ±1, high confidence
    spread = max(estimates) - min(estimates)
    if spread <= 1:
        confidence = "high"
    elif spread <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # --- Make recommendation ---
    if effective_n_states == n_states and not merge_groups and not underpopulated:
        recommendation = "keep"
    elif effective_n_states >= 0.7 * n_states:
        recommendation = "merge"
    else:
        recommendation = "retrain"

    return StateReductionReport(
        original_n_states=n_states,
        eigenvalues=eigenvalues,
        gap_ratios=gap_ratios,
        eigenvalue_gap_suggestion=n_from_eigenvalues,
        populations=populations,
        underpopulated_states=underpopulated,
        population_entropy=entropy,
        jsd_matrix=jsd_matrix,
        merge_groups=merge_groups,
        effective_n_states=effective_n_states,
        confidence=confidence,
        recommendation=recommendation,
    )
