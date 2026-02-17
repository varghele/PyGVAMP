# State Merging & Optimal State Selection — Implementation Plan

## Problem Statement

When training VAMPNet models across multiple lag times, longer lag times often require fewer
metastable states than shorter ones. Currently, the pipeline uses the **same n_states for all
lag times**, leading to over-segmentation at long lag times — states that are kinetically
redundant (near-identical transition behavior) or sparsely populated (transition-region artifacts).

## Chosen Strategy: Diagnose from Single Training Run, Merge Where Needed

Train once with a generous n_states. In the analysis step, run lightweight diagnostics to
determine the **effective number of states**. Merge redundant states via softmax output
summing. Recommend retraining only when the gap between trained and effective n_states is large.

**Why this approach:**
- Zero additional training cost for the common case
- Diagnostics are cheap (matrix eigendecomposition, population counting)
- Softmax summing is a valid approximation *specifically for redundant states* — states that
  the network split artificially have near-identical learned basis functions, so summing
  introduces minimal error
- Retraining remains available as an optional follow-up for publication-quality results

**When softmax summing is adequate vs. when to retrain:**
- **Summing is fine:** Merging 2-3 states with similar transition rows (e.g., 10→7 states).
  These states were likely split by the softmax and have overlapping dynamics
- **Retrain instead:** When diagnostics suggest dramatically fewer states (e.g., 10→4).
  The network's learned basis functions are too far from optimal for the reduced state count

---

## Implementation Steps

### Step 1: State Quality Diagnostics Module

**New file:** `pygv/utils/state_diagnostics.py`

This module computes three diagnostic signals from a trained model's outputs. None of
these require retraining — they operate on the softmax probabilities and the derived
transition matrix.

#### 1a. Eigenvalue Gap Analysis

Compute the eigenvalues of the Koopman operator (transition matrix) and identify spectral
gaps. A large gap after eigenvalue *k* indicates *k* dynamically distinct slow processes,
suggesting *k+1* metastable states (including the stationary state).

```
Input:  transition_matrix (n_states x n_states)
Output: eigenvalues (sorted descending), suggested_n_states (int), gap_ratios (array)
```

**How it works:**
- Compute eigenvalues of the transition matrix: `eigenvalues = np.linalg.eig(T)[0]`
- Sort by magnitude (descending). The first eigenvalue is always ~1 (stationary)
- Compute gap ratios: `gap[i] = |λ_i| / |λ_{i+1}|`
- The largest gap ratio after the stationary eigenvalue suggests the natural number of
  metastable states
- This is the standard approach in MSM literature (Prinz et al., 2011)

**Already existing code to reuse:**
- `pygv/utils/its.py:estimate_koopman_op()` — computes the Koopman operator
- `pygv/utils/its.py:get_its()` — already does eigendecomposition but only extracts
  implied timescales, not gap ratios. Extend this

#### 1b. State Population Analysis

Identify states with very low population. States with <2-3% of total frames are likely
transition-region artifacts rather than genuine metastable basins.

```
Input:  probs (n_frames x n_states)
Output: populations (array), underpopulated_states (list of indices), population_threshold (float)
```

**How it works:**
- Hard-assign each frame to its most probable state: `assignments = np.argmax(probs, axis=1)`
- Count frames per state, normalize to get fractional populations
- Flag states below a configurable threshold (default: 2% of total frames)
- Also check the *entropy* of the population distribution — a uniform distribution across
  many states suggests over-segmentation, while a few dominant states with many tiny ones
  suggests the dominant states are real and the rest are artifacts

**Already existing code to reuse:**
- `pygv/utils/analysis.py:calculate_state_edge_attention_maps()` already computes
  `state_populations` — extract this into a standalone function

#### 1c. Transition Row Similarity (Redundancy Detection)

Two states are "kinetically redundant" if they transition to the same places with similar
probabilities. This directly identifies merge candidates.

```
Input:  transition_matrix (n_states x n_states)
Output: similarity_matrix (n_states x n_states), merge_candidates (list of tuples),
        suggested_merges (list of sets)
```

**How it works:**
- For each pair of states (i, j), compute the Jensen-Shannon divergence (JSD) between
  their transition probability rows. JSD is preferred over Euclidean distance because
  transition rows are probability distributions
- JSD ranges from 0 (identical) to 1 (completely different)
- States with JSD below a threshold (e.g., 0.05) are merge candidates
- Apply agglomerative clustering on the JSD distance matrix with the threshold to form
  merge groups
- This is more principled than visual inspection: it directly measures whether two states
  behave differently in terms of their dynamics

#### 1d. Combined Recommendation

Combine all three signals into a single recommendation:

```python
def recommend_state_reduction(
    transition_matrix,
    probs,
    population_threshold=0.02,
    jsd_threshold=0.05
) -> StateReductionReport:
    """
    Returns
    -------
    StateReductionReport with:
        - effective_n_states: int (recommended)
        - eigenvalue_gap_suggestion: int
        - underpopulated_states: list[int]
        - merge_groups: list[set[int]]  (e.g., [{2, 5}, {3, 7}])
        - confidence: str ("high" / "medium" / "low")
        - recommendation: str ("merge" / "retrain" / "keep")
    """
```

Decision logic:
- If eigenvalue gap, population analysis, and redundancy detection all agree → **high confidence**
- If effective_n_states >= 0.7 * trained_n_states → recommend **"merge"** (softmax summing)
- If effective_n_states < 0.7 * trained_n_states → recommend **"retrain"**
- If no issues detected → recommend **"keep"**

---

### Step 2: State Merging via Softmax Summing

**New file:** `pygv/utils/state_merging.py`

Given merge groups from Step 1, produce a reduced-state representation without retraining.

#### 2a. Merge Softmax Probabilities

```python
def merge_states(
    probs: np.ndarray,           # (n_frames, n_states_original)
    merge_groups: list[set[int]] # e.g., [{2, 5}, {3, 7}]
) -> tuple[np.ndarray, dict]:
    """
    Sum softmax outputs for states within each merge group.

    Returns
    -------
    merged_probs : np.ndarray of shape (n_frames, n_states_merged)
    state_mapping : dict mapping new_state_idx -> set of original_state_idxs
    """
```

**How it works:**
- Build a mapping from original state indices to new (merged) state indices
- States not in any merge group keep their own (renumbered) index
- States in a merge group share a single new index
- For each frame, sum the softmax probabilities of merged states:
  `merged_probs[:, new_idx] = sum(probs[:, old_idx] for old_idx in group)`
- The resulting probabilities still sum to 1 per frame (since we're only regrouping terms
  of a sum that already equals 1)

#### 2b. Recompute Derived Quantities

After merging, recompute everything downstream:

- **Transition matrix:** Use `calculate_transition_matrices()` on `merged_probs`
- **State assignments:** `np.argmax(merged_probs, axis=1)`
- **State populations:** From new assignments
- **Attention maps:** Re-aggregate `calculate_state_edge_attention_maps()` using new assignments
- **State structures:** Re-select representative structures using new assignments

This is essentially re-running the analysis step on merged probabilities — the model
inference (the expensive part) is not repeated.

#### 2c. Quality Validation

After merging, verify that the reduced model is still reasonable:

- Compute VAMP-2 score of the merged model (from the merged covariance matrices)
- Compare to the original VAMP-2 score — a large drop (>10%) indicates the merge was
  too aggressive
- Compute CK test for the merged model
- If validation fails, fall back to the unmerged model and recommend retraining

---

### Step 3: Integrate Diagnostics into the Analysis Pipeline

**Modified file:** `pygv/pipe/analysis.py`

Insert the diagnostic and merging steps into `run_analysis()`, after the existing
inference and before the plotting.

#### Current flow:
```
1. Load model, create dataset
2. analyze_vampnet_outputs() → probs, embeddings, attentions, edge_indices
3. Plot transition matrix
4. Calculate attention maps
5. Plot attention maps
6. Generate state structures
7. PyMOL visualizations
8. State network plot
9. Interactive report
```

#### New flow (additions marked with ►):
```
1.  Load model, create dataset
2.  analyze_vampnet_outputs() → probs, embeddings, attentions, edge_indices
3.  Plot transition matrix (original)
►4. Run state diagnostics → StateReductionReport
►5. If merge recommended:
      a. merge_states() → merged_probs, state_mapping
      b. Recompute transition matrix, assignments, populations
      c. Validate merged model (VAMP-2 comparison, CK test)
      d. If validation passes: use merged_probs for all downstream steps
      e. If validation fails: keep original, log warning
►6. Save diagnostic report (JSON + plots)
7.  Calculate attention maps (using final probs — merged or original)
8.  Plot attention maps
9.  Generate state structures
10. PyMOL visualizations
11. State network plot
►12. ITS analysis (already exists in its.py, just call it)
►13. CK test (already exists in ck.py, just call it)
14. Interactive report (with merged state info if applicable)
```

---

### Step 4: Diagnostic Visualizations

**Modified file:** `pygv/utils/plotting.py` (add new functions)

#### 4a. Eigenvalue Spectrum Plot

A bar chart of Koopman eigenvalues with the spectral gap highlighted. The gap location
is annotated with the suggested n_states.

```
eigenvalues: [1.0, 0.95, 0.91, 0.88, | 0.42, 0.35, 0.28, ...]
                                       ↑ gap here → 4 effective states
```

#### 4b. State Redundancy Heatmap

A heatmap of the JSD similarity matrix between all state pairs. Merge candidates are
highlighted with boxes or a dendrogram overlay showing the clustering.

#### 4c. Diagnostic Summary Plot

A single figure combining:
- Top-left: Eigenvalue spectrum with gap
- Top-right: State populations (bar chart, with threshold line)
- Bottom-left: JSD similarity heatmap
- Bottom-right: Before/after merge comparison (transition matrices side by side)

#### 4d. Merge Quality Report

If merging was performed:
- Original vs. merged transition matrix (side by side)
- VAMP-2 score comparison (original vs. merged)
- State mapping diagram (which original states merged into which new states)

---

### Step 5: Integrate ITS and CK Analysis into Pipeline

**Modified file:** `pygv/pipe/analysis.py`

The code for ITS and CK tests already exists (`pygv/utils/its.py`, `pygv/utils/ck.py`)
but is never called from the analysis pipeline. Add calls after the diagnostic step.

```python
# --- ITS Analysis ---
from pygv.utils.its import analyze_implied_timescales
analyze_implied_timescales(
    probs=final_probs,  # merged or original
    save_dir=paths['analysis_dir'],
    protein_name=args.protein_name,
    lag_times_ns=[...],  # derive from config
    stride=args.stride,
    timestep=inferred_timestep
)

# --- CK Test ---
from pygv.utils.ck import run_ck_analysis
run_ck_analysis(
    probs=final_probs,
    save_dir=paths['analysis_dir'],
    protein_name=args.protein_name,
    lag_times_ns=[...],
    stride=args.stride,
    timestep=inferred_timestep
)
```

These provide independent validation of the model quality and are useful whether or not
merging was performed.

---

### Step 6: Extend the Interactive Report

**Modified files:** `pygv/utils/interactive_report.py`, `pygviz/` templates

Add the diagnostic results to the interactive HTML report:

- **Diagnostic panel:** Show the eigenvalue spectrum, population chart, and
  merge recommendation alongside the existing transition matrix and embeddings
- **Before/after toggle:** If states were merged, allow switching between original and
  merged views
- **State mapping legend:** Show which original states were merged (color-coded)
- **ITS plot:** Embed the implied timescales plot
- **CK test results:** Embed the Chapman-Kolmogorov test plots

---

### Step 7: Configuration and CLI

**Modified files:** `pygv/config/base_config.py`, `pygv/pipe/args.py`

Add new configuration parameters:

```python
# State diagnostics / merging
auto_merge: bool = True                    # Enable automatic state merging
population_threshold: float = 0.02         # Min state population (fraction)
jsd_threshold: float = 0.05               # Max JSD for redundant states
merge_validation: bool = True              # Validate merged model quality
vamp2_drop_threshold: float = 0.10        # Max acceptable VAMP-2 drop after merge
```

CLI flags:

```
--no_auto_merge          Disable automatic state merging
--population_threshold   Minimum state population fraction (default: 0.02)
--jsd_threshold          JSD threshold for redundancy detection (default: 0.05)
--force_retrain          Always retrain instead of merging
```

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `pygv/utils/state_diagnostics.py` | **NEW** | Eigenvalue gap, populations, JSD redundancy, combined recommendation |
| `pygv/utils/state_merging.py` | **NEW** | Softmax summing, recompute derived quantities, quality validation |
| `pygv/pipe/analysis.py` | MODIFY | Insert diagnostics + merging + ITS/CK calls into pipeline |
| `pygv/utils/plotting.py` | MODIFY | Add eigenvalue spectrum, JSD heatmap, diagnostic summary, merge report plots |
| `pygv/utils/interactive_report.py` | MODIFY | Add diagnostic panel, before/after toggle, ITS/CK embeds |
| `pygv/config/base_config.py` | MODIFY | Add merging config parameters |
| `pygv/pipe/args.py` | MODIFY | Add CLI flags |
| `pygviz/md_visualizer/templates/` | MODIFY | New UI panels for diagnostics |
| `tests/test_state_diagnostics.py` | **NEW** | Unit tests for diagnostic functions |
| `tests/test_state_merging.py` | **NEW** | Unit tests for merging logic |

---

## Implementation Order

The steps are ordered so that each builds on the previous and can be tested independently:

```
Step 1 (state_diagnostics.py)      — core logic, no pipeline changes, fully testable in isolation
  │
Step 2 (state_merging.py)          — depends on Step 1 output, also testable in isolation
  │
Step 3 (analysis.py integration)   — wires Steps 1+2 into the pipeline
  │
Step 4 (plotting.py)               — visualization of Step 1+2 outputs
  │
Step 5 (ITS/CK integration)       — independent of Steps 1-4, can be done in parallel
  │
Step 6 (interactive report)        — depends on Steps 3-5 being done
  │
Step 7 (config/CLI)                — can be done anytime, but needed before Step 3 for configurability
```

**Suggested parallel tracks:**
- Track A: Steps 1 → 2 → 3 → 4 (diagnostic + merging core)
- Track B: Steps 5 + 7 (ITS/CK integration + config, independent)
- Track C: Step 6 (interactive report, after A and B converge)

---

## Testing Strategy

- **Unit tests** for `state_diagnostics.py`: Create synthetic transition matrices with known
  eigenvalue gaps, known redundant states, and known population distributions. Verify the
  diagnostics correctly identify the effective n_states
- **Unit tests** for `state_merging.py`: Merge known state groups, verify that merged probs
  sum to 1, that the state mapping is correct, and that derived quantities are consistent
- **Integration test**: Run the full analysis pipeline on test data with an intentionally
  over-segmented model (e.g., 10 states for a 3-state system). Verify that the diagnostics
  recommend ~3 states and that merging produces reasonable results
- **Regression test**: Verify that the pipeline with `auto_merge=False` produces identical
  output to the current pipeline (no behavioral changes for users who don't want merging)
