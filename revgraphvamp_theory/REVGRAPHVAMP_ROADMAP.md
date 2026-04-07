# RevGraphVAMP Implementation Roadmap

## Overview

This document describes the plan to add **RevGraphVAMP** (reversible GraphVAMPNet) support to PyGVAMP. The implementation adds a likelihood-based reversible training mode alongside the existing VAMP-score-based GraphVAMPNet, without modifying any existing code.

### Scientific Background

RevGraphVAMP (Huang et al., *Methods* 2024) extends GraphVAMPNet by incorporating physical constraints — reversibility (detailed balance) and stochasticity — into the learned transition matrix. This is based on the constrained deep Markov modeling framework from Mardt & Noé (MSML 2020).

The key idea: instead of learning unconstrained softmax outputs and estimating a transition matrix post-hoc, RevGraphVAMP jointly learns:
- **χ(x)**: State membership probabilities (softmax output from encoder + classifier — identical to standard GraphVAMPNet)
- **u**: The equilibrium/stationary distribution over states (a free learnable parameter)
- **S**: A symmetric rate matrix between states (a free learnable parameter)

From u and S, a transition matrix K is constructed that satisfies detailed balance by construction: π(i)·K(i,j) = π(j)·K(j,i). The model is trained by maximizing the log-likelihood of observed time-lagged transitions under this constrained K.

### Design Principles

1. **Leave existing VAMPNet untouched** — no modifications to `pygv/vampnet/vampnet.py`
2. **Code duplication is acceptable** — RevVAMPNet duplicates shared logic from VAMPNet rather than extracting a base class (refactor to shared base can happen later)
3. **All existing encoders work with both modes** — SchNet, GIN, ML3 are agnostic to the training objective
4. **Switchable via config/CLI** — a single `--reversible` flag toggles the mode
5. **Analysis pipeline detects model type** — automatically uses learned K for RevVAMPNet or count-based K for VAMPNet

---

## Phase 1: Core Reversible Score Module

**Goal**: Implement the `ReversibleVAMPScore` module that parameterizes and constructs a reversible stochastic transition matrix, and computes the negative log-likelihood loss.

### What gets built
- `pygv/scores/reversible_score.py` containing `ReversibleVAMPScore`
- Unit tests for the score module

### Technical details

The module maintains two learnable parameter sets:

**Stationary distribution u:**
```
raw weights w_u ∈ ℝ^k  →  u = softmax(w_u)  ∈ Δ^{k-1}
```
This guarantees u is always a valid probability distribution.

**Symmetric rate matrix S:**
```
raw weights W_S ∈ ℝ^{k×k}  →  S = softplus((W_S + W_S^T) / 2)
```
Symmetrization + softplus guarantees S is symmetric and non-negative.

**Transition matrix construction:**
```
K_ij = u_j · S_ij / Σ_l (u_l · S_il)    (row normalization)
```
This satisfies:
- Non-negativity: all entries ≥ 0 (from softplus on S and positivity of u)
- Stochasticity: rows sum to 1 (from normalization)
- Detailed balance: u_i·K_ij = u_j·K_ji (from symmetry of S)

**Loss function:**
```
L = -1/N Σ_t log( χ(x_t)^T · K · χ(x_{t+τ}) )
```
where χ(x_t) and χ(x_{t+τ}) are the softmax outputs for a time-lagged pair.

### Verification criteria
- K is row-stochastic (rows sum to 1)
- K satisfies detailed balance (u_i·K_ij = u_j·K_ji within numerical tolerance)
- All entries of K are non-negative
- Gradients flow through K back to u, S, and the encoder parameters
- Loss decreases on a simple two-state system

---

## Phase 2: RevVAMPNet Model Class

**Goal**: Create a new `RevVAMPNet` class that uses the reversible score module for training while reusing the same encoder/classifier architecture.

### What gets built
- `pygv/vampnet/rev_vampnet.py` containing `RevVAMPNet`
- Unit tests for the model class

### Architecture

```
Input Graph → [Embedding MLP] → Encoder (SchNet/GIN/ML3) → Classifier (SoftmaxMLP) → Softmax → χ(x)
                                                                                                  ↓
                                                                                    ReversibleVAMPScore
                                                                                    (contains u, S, builds K)
                                                                                                  ↓
                                                                                    -log P(transitions)
```

The forward pass through embedding → encoder → classifier is identical to VAMPNet. The difference is only in how the loss is computed and what additional parameters exist.

### Key methods

| Method | Description |
|--------|-------------|
| `forward()` | Same as VAMPNet — returns softmax probabilities and features |
| `fit()` | Training loop using reversible likelihood loss instead of VAMP score |
| `get_transition_matrix()` | Returns the learned K directly from the reversible score module |
| `get_stationary_distribution()` | Returns the learned π/u |
| `get_attention()` | Same as VAMPNet — delegates to encoder |
| `save_complete_model()` | Same pattern as VAMPNet |
| `load_complete_model()` | Same pattern as VAMPNet |
| `evaluate()` | Computes average negative log-likelihood on a validation set |

### Important implementation notes

- The `ReversibleVAMPScore` parameters (u, S) **must** be included in the optimizer. Since they're registered as `nn.Parameter` inside a submodule of RevVAMPNet, `model.parameters()` will pick them up automatically — but this should be verified in tests.
- The training history should log negative log-likelihood (not VAMP score) so plots remain interpretable.
- `fit()` should still support `save_every`, `early_stopping`, gradient clipping, and all other training features from the existing VAMPNet.

---

## Phase 3: Configuration and CLI

**Goal**: Wire RevVAMPNet into the configuration system and CLI so users can enable it with `--reversible`.

### What gets modified
- `pygv/config/base_config.py` — add reversibility fields
- `pygv/args/args_train.py` — add `--reversible` flag
- `pygv/args/args_pipeline.py` — propagate the flag
- `pygv/pipe/args.py` — add `--reversible` to pipeline args
- All preset files (small/medium/large) — add default `reversible = False`

### New config fields
```python
# In BaseConfig
reversible: bool = False           # Use RevGraphVAMP (reversible likelihood loss)
rev_symmetry_reg: float = 0.0      # Optional regularization strength for S symmetry
```

### CLI addition
```bash
python run_pipeline.py --preset medium_schnet --reversible --traj_dir ... --top ...
```

---

## Phase 4: Training Pipeline Integration

**Goal**: Make `create_model()` and `run_training()` in `pygv/pipe/training.py` support RevVAMPNet.

### What gets modified
- `pygv/pipe/training.py` — branch in `create_model()` and adjust `run_training()` to handle both model types

### Logic in `create_model()`
```python
if args.reversible:
    from pygv.vampnet.rev_vampnet import RevVAMPNet
    rev_score = ReversibleVAMPScore(n_states=args.n_states, epsilon=args.vamp_epsilon)
    model = RevVAMPNet(
        encoder=encoder,
        rev_score=rev_score,
        classifier_module=classifier,
        embedding_module=embedding_module,
        ...
    )
else:
    # Existing VAMPNet creation (unchanged)
    model = VAMPNet(...)
```

### Training loop considerations
- `run_training()` calls `model.fit()` which is polymorphic — VAMPNet.fit() uses VAMP loss, RevVAMPNet.fit() uses likelihood loss
- The ITS and CK analyses at the end of training should work with both — they operate on the softmax probabilities which have the same shape regardless of model type
- Score plots will show negative log-likelihood instead of VAMP score for RevVAMPNet — the plotting function should handle this (axis label change)

---

## Phase 5: Analysis Pipeline Integration

**Goal**: Make the analysis pipeline detect which model type was used and extract the transition matrix accordingly.

### What gets modified
- `pygv/pipe/analysis.py` — detect model type in `run_analysis()`
- `pygv/utils/analysis.py` — add method to extract learned K from RevVAMPNet

### Detection logic
```python
from pygv.vampnet.rev_vampnet import RevVAMPNet

if isinstance(model, RevVAMPNet):
    # Extract the learned transition matrix directly
    learned_K = model.get_transition_matrix().detach().cpu().numpy()
    learned_pi = model.get_stationary_distribution().detach().cpu().numpy()
    # Use learned_K for all downstream analysis
else:
    # Estimate from counts (existing behavior)
    learned_K, _ = calculate_transition_matrices(probs, lag_time, stride, timestep)
```

### What changes in the analysis output
- When RevVAMPNet is detected, the analysis saves both the learned K and the count-estimated K for comparison
- A new comparison plot shows the element-wise difference between learned and estimated K
- The learned stationary distribution is saved alongside state populations
- All downstream analyses (ITS, CK, state network, TPT if added later) use the learned K

### What stays the same
- Attention extraction — identical for both model types
- State structure generation — operates on softmax probabilities, unchanged
- Embedding extraction — identical
- PyMOL visualization — unchanged

---

## Phase 6: Tests

**Goal**: Comprehensive test coverage for all new components.

### Test files to create

**`tests/test_reversible_score.py`**
- Test that K is row-stochastic
- Test that K satisfies detailed balance
- Test that all entries of K are non-negative
- Test gradient flow through K to u and S parameters
- Test loss computation on known simple systems
- Test numerical stability with edge cases (very small/large states)
- Test that u converges to known stationary distribution on a simple system

**`tests/test_rev_vampnet.py`**
- Test model construction with each encoder type (SchNet, GIN, ML3)
- Test that forward pass produces valid softmax outputs
- Test that all parameters (encoder + classifier + u + S) are in optimizer
- Test save/load round-trip preserves learned K
- Test that `get_transition_matrix()` returns correct shape and properties
- Test that `get_stationary_distribution()` returns valid distribution
- Test short training run (few epochs) on synthetic data — loss should decrease
- Test that `evaluate()` returns finite values

**`tests/test_rev_integration.py`**
- Integration test: full pipeline with `--reversible` flag on synthetic/tiny data
- Test that analysis correctly detects RevVAMPNet and extracts learned K
- Test that training + analysis produces expected output files

---

## Phase 7: Documentation and Examples

**Goal**: Update documentation so users know how to use RevGraphVAMP.

### What gets updated
- `README.md` — add section on reversible mode
- `CODEBASE_SUMMARY.md` — update architecture description and component list
- Docstrings in all new files

### Example usage in README
```bash
# Standard GraphVAMPNet (unchanged)
python run_pipeline.py --preset medium_schnet --traj_dir ./data --top topology.pdb

# RevGraphVAMPNet (new)
python run_pipeline.py --preset medium_schnet --reversible --traj_dir ./data --top topology.pdb
```

---

## Dependency Graph

```
Phase 1 (ReversibleVAMPScore)
    ↓
Phase 2 (RevVAMPNet)
    ↓
Phase 3 (Config/CLI) ←── can be done in parallel with Phase 2
    ↓
Phase 4 (Training pipeline)
    ↓
Phase 5 (Analysis pipeline)
    ↓
Phase 6 (Tests) ←── tests for Phases 1-2 should be written alongside them
    ↓
Phase 7 (Documentation)
```

Phases 1 and 2 are the core scientific contribution. Phases 3-5 are integration work. Phase 6 should be done incrementally (write tests for each phase as you go). Phase 7 is polish.

---

## Files Created (New)

| File | Description |
|------|-------------|
| `pygv/scores/reversible_score.py` | ReversibleVAMPScore module |
| `pygv/vampnet/rev_vampnet.py` | RevVAMPNet model class |
| `tests/test_reversible_score.py` | Score module tests |
| `tests/test_rev_vampnet.py` | Model class tests |
| `tests/test_rev_integration.py` | Integration tests |

## Files Modified

| File | Change |
|------|--------|
| `pygv/config/base_config.py` | Add `reversible` and `rev_symmetry_reg` fields |
| `pygv/args/args_train.py` | Add `--reversible` argument |
| `pygv/args/args_pipeline.py` | Propagate reversible flag |
| `pygv/pipe/args.py` | Add `--reversible` to pipeline CLI |
| `pygv/pipe/training.py` | Branch in `create_model()` for RevVAMPNet |
| `pygv/pipe/analysis.py` | Detect model type, extract learned K |
| `pygv/utils/analysis.py` | Add `extract_learned_transition_matrix()` utility |
| `pygv/scores/__init__.py` | Export ReversibleVAMPScore |
| `pygv/vampnet/__init__.py` | Export RevVAMPNet |
| `pygv/utils/plotting.py` | Handle NLL vs VAMP score axis labels in training plots |
| `README.md` | Document reversible mode |
| `CODEBASE_SUMMARY.md` | Update architecture description |

---

*Last updated: 2026-04-07*
