"""
State merging via softmax output summing for VAMPNet analysis.

Given merge groups identified by state_diagnostics, this module:
1. Merges softmax probabilities by summing outputs for redundant states
2. Recomputes all derived quantities (transition matrix, assignments, populations)
3. Validates the merged model quality via VAMP-2 score comparison

Merging is appropriate when a small number of kinetically redundant states are
being combined (e.g., 10→7). For large reductions (e.g., 10→4), retraining is
recommended instead.
"""

import os
import json
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MergeResult:
    """Result of a state merging operation."""

    merged_probs: np.ndarray = field(repr=False)
    state_mapping: Dict[int, Set[int]] = field(default_factory=dict)
    original_n_states: int = 0
    merged_n_states: int = 0
    vamp2_original: Optional[float] = None
    vamp2_merged: Optional[float] = None
    vamp2_drop: Optional[float] = None
    validation_passed: bool = True

    def summary(self) -> str:
        lines = [
            f"State Merging Result",
            f"  Original states: {self.original_n_states}",
            f"  Merged states:   {self.merged_n_states}",
            f"  Mapping: {_format_mapping(self.state_mapping)}",
        ]
        if self.vamp2_original is not None:
            lines.append(f"  VAMP-2 original: {self.vamp2_original:.4f}")
            lines.append(f"  VAMP-2 merged:   {self.vamp2_merged:.4f}")
            lines.append(f"  VAMP-2 drop:     {self.vamp2_drop:.4f} ({self.vamp2_drop * 100:.1f}%)")
            lines.append(f"  Validation:      {'PASSED' if self.validation_passed else 'FAILED'}")
        return "\n".join(lines)


def _format_mapping(mapping: Dict[int, Set[int]]) -> str:
    """Format state mapping for display."""
    parts = []
    for new_idx in sorted(mapping.keys()):
        old_idxs = sorted(mapping[new_idx])
        if len(old_idxs) == 1:
            parts.append(f"S{old_idxs[0]}→S{new_idx}")
        else:
            merged_str = "+".join(f"S{i}" for i in old_idxs)
            parts.append(f"({merged_str})→S{new_idx}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 2a. Merge softmax probabilities
# ---------------------------------------------------------------------------

def merge_states(
    probs: np.ndarray,
    merge_groups: List[Set[int]],
) -> Tuple[np.ndarray, Dict[int, Set[int]]]:
    """
    Sum softmax outputs for states within each merge group.

    States not in any merge group keep their own (renumbered) index. States in a
    merge group share a single new index whose probability is the sum of the
    original probabilities.

    Parameters
    ----------
    probs : np.ndarray
        State probabilities of shape (n_frames, n_states_original).
    merge_groups : list of set of int
        Groups of original state indices to merge, e.g., [{2, 5}, {3, 7}].

    Returns
    -------
    merged_probs : np.ndarray
        Merged probabilities of shape (n_frames, n_states_merged).
    state_mapping : dict
        Mapping from new state index → set of original state indices.
    """
    n_states_original = probs.shape[1]

    # Determine which original states are merged
    merged_state_set = set()
    for group in merge_groups:
        merged_state_set.update(group)

    # Build the mapping: assign new indices
    # First, assign indices to merge groups (using the lowest original index as representative)
    state_mapping = {}
    new_idx = 0

    # Process all original states in order
    assigned = set()
    for orig_idx in range(n_states_original):
        if orig_idx in assigned:
            continue

        # Check if this state is part of a merge group
        group_for_this = None
        for group in merge_groups:
            if orig_idx in group:
                group_for_this = group
                break

        if group_for_this is not None:
            state_mapping[new_idx] = set(group_for_this)
            assigned.update(group_for_this)
        else:
            state_mapping[new_idx] = {orig_idx}
            assigned.add(orig_idx)

        new_idx += 1

    n_states_merged = len(state_mapping)

    # Build merged probabilities
    merged_probs = np.zeros((probs.shape[0], n_states_merged), dtype=probs.dtype)
    for new_state, old_states in state_mapping.items():
        for old_state in old_states:
            merged_probs[:, new_state] += probs[:, old_state]

    return merged_probs, state_mapping


# ---------------------------------------------------------------------------
# 2b. Recompute derived quantities
# ---------------------------------------------------------------------------

def recompute_transition_matrix(
    merged_probs: np.ndarray,
    lag_time: float,
    stride: int,
    timestep: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recompute the transition matrix from merged probabilities.

    Parameters
    ----------
    merged_probs : np.ndarray
        Merged state probabilities of shape (n_frames, n_states_merged).
    lag_time : float
        Lag time in nanoseconds.
    stride : int
        Stride used during frame extraction.
    timestep : float
        Trajectory timestep in nanoseconds.

    Returns
    -------
    transition_matrix : np.ndarray
    transition_matrix_no_self : np.ndarray
    """
    from pygv.utils.analysis import calculate_transition_matrices
    return calculate_transition_matrices(
        probs=merged_probs,
        lag_time=lag_time,
        stride=stride,
        timestep=timestep,
    )


# ---------------------------------------------------------------------------
# 2c. Quality validation via VAMP-2 score
# ---------------------------------------------------------------------------

def _compute_vamp2_from_probs(probs: np.ndarray, lag_frames: int) -> float:
    """
    Compute the VAMP-2 score from state probabilities using covariance matrices.

    VAMP-2 = ||C00^{-1/2} C0t Ctt^{-1/2}||^2_F + 1

    Parameters
    ----------
    probs : np.ndarray
        State probabilities of shape (n_frames, n_states).
    lag_frames : int
        Lag time in frames.

    Returns
    -------
    float
        VAMP-2 score.
    """
    if probs.shape[0] <= lag_frames:
        return 0.0

    x_0 = probs[:-lag_frames]
    x_t = probs[lag_frames:]

    n = x_0.shape[0]

    # Covariance matrices
    c00 = (x_0.T @ x_0) / (n - 1)
    c0t = (x_0.T @ x_t) / (n - 1)
    ctt = (x_t.T @ x_t) / (n - 1)

    # Regularize
    eps = 1e-6
    c00 += eps * np.eye(c00.shape[0])
    ctt += eps * np.eye(ctt.shape[0])

    # VAMP-2: trace(C00^{-1} C0t Ctt^{-1} Ct0) + 1
    c00_inv = np.linalg.inv(c00)
    ctt_inv = np.linalg.inv(ctt)

    M = c00_inv @ c0t @ ctt_inv @ c0t.T
    vamp2 = np.trace(M) + 1.0

    return float(vamp2)


def validate_merge(
    original_probs: np.ndarray,
    merged_probs: np.ndarray,
    lag_time: float,
    stride: int,
    timestep: float,
    vamp2_drop_threshold: float = 0.10,
) -> Tuple[float, float, float, bool]:
    """
    Validate a state merge by comparing VAMP-2 scores.

    Parameters
    ----------
    original_probs : np.ndarray
        Original state probabilities (n_frames, n_states_original).
    merged_probs : np.ndarray
        Merged state probabilities (n_frames, n_states_merged).
    lag_time : float
        Lag time in nanoseconds.
    stride : int
        Stride used during frame extraction.
    timestep : float
        Trajectory timestep in nanoseconds.
    vamp2_drop_threshold : float
        Maximum acceptable relative VAMP-2 drop.

    Returns
    -------
    vamp2_original : float
    vamp2_merged : float
    vamp2_drop : float
        Relative drop: (original - merged) / original. Positive means quality loss.
    passed : bool
        True if the drop is within the acceptable threshold.
    """
    effective_timestep = timestep * stride
    lag_frames = max(1, int(round(lag_time / effective_timestep)))

    vamp2_original = _compute_vamp2_from_probs(original_probs, lag_frames)
    vamp2_merged = _compute_vamp2_from_probs(merged_probs, lag_frames)

    if vamp2_original > 0:
        vamp2_drop = (vamp2_original - vamp2_merged) / vamp2_original
    else:
        vamp2_drop = 0.0

    passed = vamp2_drop <= vamp2_drop_threshold

    return vamp2_original, vamp2_merged, vamp2_drop, passed


# ---------------------------------------------------------------------------
# High-level merge + validate
# ---------------------------------------------------------------------------

def merge_and_validate(
    probs: np.ndarray,
    merge_groups: List[Set[int]],
    lag_time: float,
    stride: int,
    timestep: float,
    vamp2_drop_threshold: float = 0.10,
    validate: bool = True,
) -> MergeResult:
    """
    Merge states and optionally validate the result.

    Parameters
    ----------
    probs : np.ndarray
        Original state probabilities (n_frames, n_states_original).
    merge_groups : list of set of int
        Groups of state indices to merge.
    lag_time : float
        Lag time in nanoseconds.
    stride : int
        Stride used during frame extraction.
    timestep : float
        Trajectory timestep in nanoseconds.
    vamp2_drop_threshold : float
        Maximum acceptable relative VAMP-2 drop.
    validate : bool
        Whether to run VAMP-2 validation.

    Returns
    -------
    MergeResult
        Complete merge result with validation info.
    """
    merged_probs, state_mapping = merge_states(probs, merge_groups)

    result = MergeResult(
        merged_probs=merged_probs,
        state_mapping=state_mapping,
        original_n_states=probs.shape[1],
        merged_n_states=merged_probs.shape[1],
    )

    if validate and merge_groups:
        vamp2_original, vamp2_merged, vamp2_drop, passed = validate_merge(
            original_probs=probs,
            merged_probs=merged_probs,
            lag_time=lag_time,
            stride=stride,
            timestep=timestep,
            vamp2_drop_threshold=vamp2_drop_threshold,
        )
        result.vamp2_original = vamp2_original
        result.vamp2_merged = vamp2_merged
        result.vamp2_drop = vamp2_drop
        result.validation_passed = passed

    return result


def save_merge_report(
    report,  # StateReductionReport
    merge_result: Optional[MergeResult],
    save_dir: str,
    protein_name: str,
) -> str:
    """
    Save diagnostic and merge reports as JSON.

    Parameters
    ----------
    report : StateReductionReport
        Diagnostic report from recommend_state_reduction.
    merge_result : MergeResult or None
        Merge result, if merging was performed.
    save_dir : str
        Directory to save the report.
    protein_name : str
        Protein name for file naming.

    Returns
    -------
    str
        Path to the saved report.
    """
    os.makedirs(save_dir, exist_ok=True)

    data = {
        "diagnostics": {
            "original_n_states": report.original_n_states,
            "effective_n_states": report.effective_n_states,
            "eigenvalue_gap_suggestion": report.eigenvalue_gap_suggestion,
            "eigenvalues": np.abs(report.eigenvalues).tolist(),
            "gap_ratios": report.gap_ratios.tolist(),
            "populations": report.populations.tolist(),
            "underpopulated_states": report.underpopulated_states,
            "population_entropy": report.population_entropy,
            "merge_groups": [sorted(g) for g in report.merge_groups],
            "confidence": report.confidence,
            "recommendation": report.recommendation,
        },
    }

    if merge_result is not None:
        mapping_serializable = {
            str(k): sorted(v) for k, v in merge_result.state_mapping.items()
        }
        data["merge"] = {
            "original_n_states": merge_result.original_n_states,
            "merged_n_states": merge_result.merged_n_states,
            "state_mapping": mapping_serializable,
            "vamp2_original": merge_result.vamp2_original,
            "vamp2_merged": merge_result.vamp2_merged,
            "vamp2_drop": merge_result.vamp2_drop,
            "validation_passed": merge_result.validation_passed,
        }

    report_path = os.path.join(save_dir, f"{protein_name}_state_diagnostics.json")
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)

    return report_path