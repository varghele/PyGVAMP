# CLAUDE.md — RevGraphVAMP Implementation

## Context

You are adding RevGraphVAMP (reversible GraphVAMPNet) to the PyGVAMP codebase. Read `REVGRAPHVAMP_ROADMAP.md` for the full scientific background and design rationale.

**Critical rule: Do NOT modify `pygv/vampnet/vampnet.py` or any existing test files.** The existing VAMPNet class stays untouched. RevVAMPNet is a separate, new class that duplicates shared logic where needed.

Before starting any step, read the relevant existing files to understand current patterns, naming conventions, and code style. Match them exactly.

---

## Step 1: Implement ReversibleVAMPScore

**Read first:**
- `pygv/scores/vamp_score_v0.py` (understand the existing score interface)
- `pygv/scores/__init__.py`

**Create `pygv/scores/reversible_score.py`:**

```python
class ReversibleVAMPScore(nn.Module):
```

This module has:

**Learnable parameters:**
- `self.log_stationary` — `nn.Parameter(torch.zeros(n_states))` — unconstrained weights for the stationary distribution u
- `self.rate_matrix_weights` — `nn.Parameter(torch.zeros(n_states, n_states))` — unconstrained weights for the symmetric rate matrix S

**Methods to implement:**

`get_stationary_distribution() -> torch.Tensor`:
- Returns `torch.softmax(self.log_stationary, dim=0)` — shape `(n_states,)`

`get_rate_matrix() -> torch.Tensor`:
- Symmetrize: `W = (self.rate_matrix_weights + self.rate_matrix_weights.T) / 2`
- Apply softplus: `S = F.softplus(W)`
- Return S — shape `(n_states, n_states)`

`get_transition_matrix() -> torch.Tensor`:
- Get u from `get_stationary_distribution()`
- Get S from `get_rate_matrix()`
- Compute `K_unnorm[i,j] = S[i,j] * u[j]` — this is `S * u.unsqueeze(0)`
- Row-normalize: `K = K_unnorm / (K_unnorm.sum(dim=1, keepdim=True) + self.epsilon)`
- Return K — shape `(n_states, n_states)`

`forward(chi_t0: torch.Tensor, chi_t1: torch.Tensor) -> torch.Tensor`:
- Get K from `get_transition_matrix()`
- Compute transition probabilities: `p = (chi_t0 @ K) * chi_t1` then `p = p.sum(dim=1)` — shape `(batch,)`
- This computes χ(x_t)^T · K · χ(x_{t+τ}) for each pair
- Return negative log-likelihood: `-torch.log(torch.clamp(p, min=self.epsilon)).mean()`

`loss(chi_t0: torch.Tensor, chi_t1: torch.Tensor) -> torch.Tensor`:
- Just calls `self.forward(chi_t0, chi_t1)` — kept for API consistency with VAMPScore

**Constructor parameters:**
- `n_states: int` — number of metastable states
- `epsilon: float = 1e-6` — numerical stability constant

**After creating the file**, update `pygv/scores/__init__.py` to export `ReversibleVAMPScore`.

---

## Step 2: Write tests for ReversibleVAMPScore

**Read first:**
- `tests/` directory — look at existing test files for patterns and conventions
- The test .md file in the tests directory if it exists

**Create `tests/test_reversible_score.py`:**

Write these test functions:

`test_stationary_is_valid_distribution`:
- Create ReversibleVAMPScore with n_states=5
- Call `get_stationary_distribution()`
- Assert all values > 0
- Assert sum ≈ 1.0 (within 1e-6)

`test_rate_matrix_is_symmetric`:
- Create score module, set random weights
- Call `get_rate_matrix()`
- Assert `S == S.T` within tolerance

`test_rate_matrix_is_nonnegative`:
- Create score module, set random weights (including negative values)
- Call `get_rate_matrix()`
- Assert all entries >= 0

`test_transition_matrix_is_row_stochastic`:
- Create score module
- Call `get_transition_matrix()`
- Assert all entries >= 0
- Assert each row sums to 1.0 within tolerance

`test_transition_matrix_satisfies_detailed_balance`:
- Create score module
- Get K and u
- For all i,j: assert `u[i] * K[i,j] ≈ u[j] * K[j,i]` within 1e-5

`test_loss_is_finite`:
- Create score module with n_states=3
- Create random softmax outputs: `chi_t0 = F.softmax(torch.randn(32, 3), dim=1)`
- Compute loss
- Assert loss is finite and not NaN

`test_loss_gradient_flows_to_all_parameters`:
- Create score module
- Compute loss on random data
- Call `loss.backward()`
- Assert `score.log_stationary.grad` is not None and not all zeros
- Assert `score.rate_matrix_weights.grad` is not None and not all zeros

`test_loss_gradient_flows_through_inputs`:
- Create random `chi_t0` and `chi_t1` with `requires_grad=True`
- Compute loss
- Call `loss.backward()`
- Assert `chi_t0.grad` is not None — this confirms gradients reach the encoder

`test_loss_decreases_on_simple_system`:
- Create a 2-state system with known transition matrix
- Generate synthetic softmax data consistent with this system
- Run 50 optimization steps on the score module's parameters
- Assert final loss < initial loss

`test_numerical_stability_with_extreme_inputs`:
- Test with near-zero softmax outputs (e.g., one state has probability ~1e-7)
- Test with near-deterministic assignments (one state has probability ~0.999)
- Assert no NaN or Inf in loss or gradients

---

## Step 3: Implement RevVAMPNet

**Read first:**
- `pygv/vampnet/vampnet.py` — read the entire file carefully, understand every method
- `pygv/vampnet/__init__.py`

**Create `pygv/vampnet/rev_vampnet.py`:**

```python
class RevVAMPNet(nn.Module):
```

This class duplicates the structure of VAMPNet but uses ReversibleVAMPScore for training.

**Constructor — same parameters as VAMPNet plus:**
- `rev_score: ReversibleVAMPScore` — the reversible score module (replaces `vamp_score`)
- Accept all the same embedding/classifier parameters as VAMPNet

**Copy these methods from VAMPNet with no changes:**
- `forward()` — the graph-to-probabilities pass is identical
- `get_attention()` — delegates to encoder, identical
- `get_embeddings()` — identical
- `save_complete_model()` — identical (torch.save(self, ...))
- `load_complete_model()` — identical (torch.load)
- `get_config()` — add `'reversible': True` to the returned dict
- `quick_evaluate()` — same structure, but compute NLL instead of VAMP score

**New methods (not in VAMPNet):**

`get_transition_matrix() -> torch.Tensor`:
- Returns `self.rev_score.get_transition_matrix()`

`get_stationary_distribution() -> torch.Tensor`:
- Returns `self.rev_score.get_stationary_distribution()`

**Modified methods:**

`fit()` — Copy the training loop from VAMPNet.fit() but change:
1. The loss computation: replace `self.vamp_score.loss(chi_t0, chi_t1)` with `self.rev_score.loss(chi_t0, chi_t1)`
2. The score logging: log negative log-likelihood values (lower is better, not higher like VAMP)
3. The "best model" tracking: best model has **lowest** NLL (not highest VAMP score), so flip the comparison
4. The plot title: indicate this is RevGraphVAMPNet training

`evaluate()` — Same structure as VAMPNet.evaluate() but compute average NLL instead of VAMP score. Return **negative** NLL so that higher = better (consistent with the best-model tracking). Or return the raw NLL and handle the sign flip in fit(). Pick one convention and be consistent.

`save()` and `load()` — Copy from VAMPNet, but also save/load the rev_score config (n_states, epsilon). The `save_complete_model`/`load_complete_model` approach (which saves the entire object) handles this automatically.

**After creating the file**, update `pygv/vampnet/__init__.py` to export `RevVAMPNet`.

---

## Step 4: Write tests for RevVAMPNet

**Create `tests/test_rev_vampnet.py`:**

`test_construction_with_schnet`:
- Create a SchNetEncoderNoEmbed, SoftmaxMLP classifier, ReversibleVAMPScore
- Construct RevVAMPNet
- Assert model has encoder, classifier_module, rev_score attributes
- Assert total parameter count is reasonable

`test_construction_with_gin`:
- Same as above but with GINEncoder

`test_construction_with_ml3`:
- Same as above but with ML3Encoder

`test_forward_produces_valid_softmax`:
- Create model, create a synthetic PyG batch (use torch_geometric.data.Data with random x, edge_index, edge_attr, batch)
- Call model.forward()
- Assert output probabilities sum to 1 per sample
- Assert all values in [0, 1]

`test_all_parameters_in_optimizer`:
- Create model
- Create optimizer with `model.parameters()`
- Collect all parameter names from the optimizer
- Assert that parameters from encoder, classifier, rev_score (log_stationary, rate_matrix_weights) are all present

`test_get_transition_matrix_shape_and_properties`:
- Create model with n_states=5
- Call `model.get_transition_matrix()`
- Assert shape is (5, 5)
- Assert row-stochastic
- Assert detailed balance with `model.get_stationary_distribution()`

`test_get_stationary_distribution`:
- Create model with n_states=4
- Call `model.get_stationary_distribution()`
- Assert shape is (4,)
- Assert valid probability distribution

`test_save_load_roundtrip`:
- Create model, do a few forward passes to change parameters from init
- Save with `save_complete_model()`
- Load with `load_complete_model()`
- Assert `get_transition_matrix()` matches between original and loaded
- Assert `get_stationary_distribution()` matches

`test_short_training_run`:
- Create model with small dimensions
- Create a tiny synthetic dataset (50 time-lagged graph pairs)
- Run fit() for 5 epochs
- Assert training completes without error
- Assert loss history has 5 entries
- Assert final loss is finite

`test_evaluate_returns_finite`:
- Create model, create a small test loader
- Call evaluate()
- Assert result is finite

---

## Step 5: Add config fields

**Read first:**
- `pygv/config/base_config.py`
- `pygv/config/model_configs/schnet.py` (and others) to see inheritance pattern
- `pygv/config/presets/small.py`, `medium.py`, `large.py`

**Modify `pygv/config/base_config.py`:**
Add these fields to the `BaseConfig` dataclass (place them after the existing `vamp_epsilon` / numerical stability section):

```python
# Reversibility constraints
reversible: bool = False                    # Use RevGraphVAMP (reversible likelihood loss)
```

No changes needed in the model_configs or presets — they inherit from BaseConfig and `reversible` defaults to `False`.

---

## Step 6: Add CLI arguments

**Read first:**
- `pygv/args/args_train.py`
- `pygv/args/args_pipeline.py`
- `pygv/pipe/args.py`

**Modify `pygv/args/args_train.py`:**
In the `add_training_args()` function, add:
```python
train_group.add_argument('--reversible', action='store_true',
                         help='Use RevGraphVAMP (reversible likelihood-based training)')
```

**Modify `pygv/pipe/args.py`:**
In `parse_pipeline_args()`, add to the "Training overrides" section:
```python
parser.add_argument('--reversible', action='store_true',
                    help='Use RevGraphVAMP (reversible likelihood-based training)')
```

**Modify `pygv/pipe/master_pipeline.py`:**
In the `main()` function, after the existing training override block, add:
```python
if args.reversible:
    config.reversible = True
```

Also ensure `_create_train_args()` in `PipelineOrchestrator` passes through the `reversible` field — it already copies all config fields via `args = argparse.Namespace(**self.config.to_dict())`, so this should work automatically. Verify this.

---

## Step 7: Integrate into training pipeline

**Read first:**
- `pygv/pipe/training.py` — read `create_model()` and `run_training()` carefully

**Modify `pygv/pipe/training.py`:**

At the top, add import:
```python
from pygv.vampnet.rev_vampnet import RevVAMPNet
from pygv.scores.reversible_score import ReversibleVAMPScore
```

In `create_model()`, after the existing model creation block (after the `model = VAMPNet(...)` call), add a branch:

```python
if getattr(args, 'reversible', False):
    # Create reversible score module
    rev_score = ReversibleVAMPScore(
        n_states=args.n_states,
        epsilon=args.vamp_epsilon
    )
    
    model = RevVAMPNet(
        embedding_module=embedding_module,
        encoder=encoder,
        rev_score=rev_score,
        classifier_module=classifier,
        lag_time=args.lag_time,
        training_jitter=args.training_jitter
    )
else:
    # Existing VAMPNet creation (the code that's already there)
    model = VAMPNet(...)
```

Make sure the `else` branch wraps the existing VAMPNet creation code. Do not delete or modify the existing VAMPNet creation — just wrap it in the else.

Apply `init_for_vamp(model)` after both branches (it already exists after the current VAMPNet creation).

In `run_training()`, the call to `model.fit()` should work polymorphically — both VAMPNet and RevVAMPNet have a `fit()` method with the same signature. Verify that the return value (history dict) has compatible keys.

**Modify `pygv/utils/plotting.py`:**
In `plot_vamp_scores()`, the y-axis label currently says "VAMP Score". This should adapt:
- Check if the history values are negative log-likelihood (they'll be positive and decreasing) vs VAMP scores (positive and increasing)
- The simplest approach: accept an optional `ylabel` parameter and pass it from the training loop. Or check if the history dict contains a `'score_type'` key.

---

## Step 8: Integrate into analysis pipeline

**Read first:**
- `pygv/pipe/analysis.py` — read `run_analysis()` carefully
- `pygv/utils/analysis.py` — read `calculate_transition_matrices()`

**Modify `pygv/utils/analysis.py`:**

Add a new function:

```python
def extract_learned_transition_matrix(model) -> tuple:
    """
    Extract the learned transition matrix and stationary distribution
    from a RevVAMPNet model.
    
    Parameters
    ----------
    model : RevVAMPNet
        Trained reversible VAMPNet model
        
    Returns
    -------
    tuple (transition_matrix, stationary_distribution)
        Both as numpy arrays
    """
    from pygv.vampnet.rev_vampnet import RevVAMPNet
    
    if not isinstance(model, RevVAMPNet):
        raise TypeError("Model must be a RevVAMPNet instance")
    
    model.eval()
    with torch.no_grad():
        K = model.get_transition_matrix().cpu().numpy()
        pi = model.get_stationary_distribution().cpu().numpy()
    
    return K, pi
```

**Modify `pygv/pipe/analysis.py`:**

In `run_analysis()`, after the model is loaded and the probs are computed, add model type detection:

```python
from pygv.vampnet.rev_vampnet import RevVAMPNet

is_reversible = isinstance(model, RevVAMPNet)
```

Then in the transition matrix section (Step 3 in the existing pipeline), branch:

```python
if is_reversible:
    from pygv.utils.analysis import extract_learned_transition_matrix
    learned_K, learned_pi = extract_learned_transition_matrix(model)
    original_transition_matrix = learned_K
    
    # Also compute count-based estimate for comparison
    count_based_K, _ = calculate_transition_matrices(
        probs=probs, lag_time=args.lag_time, stride=args.stride, timestep=inferred_timestep
    )
    
    # Save comparison
    np.save(os.path.join(paths['analysis_dir'], f"{args.protein_name}_learned_K.npy"), learned_K)
    np.save(os.path.join(paths['analysis_dir'], f"{args.protein_name}_learned_pi.npy"), learned_pi)
    np.save(os.path.join(paths['analysis_dir'], f"{args.protein_name}_count_based_K.npy"), count_based_K)
    
    print(f"Learned stationary distribution: {learned_pi}")
    print(f"Max |learned_K - count_K|: {np.max(np.abs(learned_K - count_based_K)):.6f}")
else:
    original_transition_matrix, _ = calculate_transition_matrices(
        probs=probs, lag_time=args.lag_time, stride=args.stride, timestep=inferred_timestep
    )
```

The rest of the analysis pipeline uses `original_transition_matrix` which now holds the learned K for RevVAMPNet, so state diagnostics, ITS, CK tests, and network plots all automatically use the correct matrix.

Add `is_reversible` to the results dict returned by `run_analysis()`.

---

## Step 9: Write integration tests

**Create `tests/test_rev_integration.py`:**

These tests verify the full pipeline works end-to-end.

`test_create_rev_model_via_training_pipeline`:
- Create a mock args namespace with `reversible=True` and small model dimensions
- Call `create_model(args)`
- Assert returned model is an instance of RevVAMPNet
- Assert model has `get_transition_matrix` method

`test_create_standard_model_unchanged`:
- Create a mock args namespace with `reversible=False`
- Call `create_model(args)`
- Assert returned model is an instance of VAMPNet (not RevVAMPNet)

`test_analysis_detects_rev_model`:
- Create a RevVAMPNet model
- Assert `isinstance(model, RevVAMPNet)` is True
- Call `extract_learned_transition_matrix(model)`
- Assert returned K is row-stochastic and satisfies detailed balance

`test_analysis_detects_standard_model`:
- Create a standard VAMPNet model
- Assert `isinstance(model, RevVAMPNet)` is False

`test_config_reversible_field`:
- Create BaseConfig
- Assert `config.reversible` is False by default
- Set `config.reversible = True`
- Convert to dict and back
- Assert field preserved

---

## Step 10: Update documentation

**Modify `README.md`:**

Add a section after the existing "2. Training" section:

```markdown
### 2.1 Reversible GraphVAMPNet (RevGraphVAMP)

To train with physical constraints (reversibility and detailed balance), 
add the `--reversible` flag:

\`\`\`bash
python run_training.py \
    --reversible \
    --protein_name "your_protein" \
    --top "topology.pdb" \
    --traj_dir "trajectories/" \
    ... (other arguments same as standard training)
\`\`\`

This uses a likelihood-based loss function instead of VAMP-2, and learns 
a transition matrix that satisfies detailed balance by construction. 
Recommended for:
- Non-equilibrium MD trajectories
- Quantitative kinetic analysis (rates, mean first passage times)
- Transition path theory calculations
```

**Modify `CODEBASE_SUMMARY.md`:**

In the "Core Components" section, add RevVAMPNet to the VAMPNet model description. In the "VAMP Score" section, add the reversible score. Update the directory tree to include the new files.

**Update `pygv/scores/__init__.py`:**
```python
from pygv.scores.vamp_score_v0 import VAMPScore
from pygv.scores.reversible_score import ReversibleVAMPScore

__all__ = ['VAMPScore', 'ReversibleVAMPScore']
```

**Update `pygv/vampnet/__init__.py`:**
```python
from pygv.vampnet.vampnet import VAMPNet
from pygv.vampnet.rev_vampnet import RevVAMPNet

__all__ = ['VAMPNet', 'RevVAMPNet']
```

---

## Verification Checklist

After completing all steps, verify:

- [ ] `python -m pytest tests/test_reversible_score.py -v` passes
- [ ] `python -m pytest tests/test_rev_vampnet.py -v` passes
- [ ] `python -m pytest tests/test_rev_integration.py -v` passes
- [ ] Existing tests still pass: `python -m pytest tests/ -v --ignore=tests/test_reversible_score.py --ignore=tests/test_rev_vampnet.py --ignore=tests/test_rev_integration.py`
- [ ] `pygv/vampnet/vampnet.py` has zero modifications (check with git diff)
- [ ] Creating a model with `--reversible` produces a RevVAMPNet instance
- [ ] Creating a model without `--reversible` produces a VAMPNet instance (unchanged behavior)
- [ ] RevVAMPNet's `get_transition_matrix()` returns a valid row-stochastic matrix satisfying detailed balance
- [ ] The analysis pipeline detects RevVAMPNet and extracts the learned K
