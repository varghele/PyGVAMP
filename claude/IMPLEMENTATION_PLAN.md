# IMPLEMENTATION_PLAN.md — Auto-Stride, Warm-Starting, Retrain Policy

## Context

You are extending the existing PyGVAMP codebase with three features that work together to make multi-lag-time experiments tractable. Most infrastructure already exists — your job is to add three specific features and integrate them cleanly without breaking what's there.

**Critical rules:**
- Read the relevant files before writing any code. The codebase has established patterns; match them.
- Do not refactor unrelated code. Stay strictly in scope.
- All three new features must be opt-in via CLI flags, with defaults matching current behavior. The one exception is the retrain policy change (see Phase 4).
- After each phase, run the tests for that phase before starting the next.
- If you're uncertain about a design decision that wasn't covered here, stop and ask rather than guessing.

---

## Phase 1: Reconnaissance and Reporting

**Goal:** Build a written understanding of the existing codebase so subsequent phases can make informed implementation decisions. This phase does not write production code.

**Files to read (at minimum):**
- `pygv/pipe/master_pipeline.py` — overall pipeline orchestration
- `pygv/pipe/training.py` — model creation and training loop
- `pygv/pipe/analysis.py` — analysis pipeline including state diagnostics
- `pygv/utils/state_diagnostics.py` (or wherever state-count recommendations are computed)
- `pygv/dataset/` (whatever the dataset module is called) — how stride is currently handled
- `pygv/vampnet/vampnet.py` and `pygv/vampnet/rev_vampnet.py` — model classes
- `pygv/config/base_config.py` — configuration structure
- Any existing retrain-handling code (search for `retrain`, `max_retrain`, or similar)

**Deliverable — a written report at `/tmp/phase1_report.md` covering:**

1. **State diagnostics:** How are state-count recommendations computed? Can recommended k grow (e.g., from 8 → 10) or only shrink? Where exactly does the recommendation come from in the code?

2. **Model reconstruction on retrain:** When the pipeline currently retrains with a new k, what happens? Is the entire model (encoder + classifier) rebuilt from scratch? Where in the code does this happen? What state is currently preserved vs. discarded?

3. **Stride handling:** Is stride a preprocessing-time parameter (baked into the cached `.npy` arrays) or a runtime parameter (applied by the dataset's `__getitem__` or DataLoader)? Where in the code is stride applied? Is it possible to change stride per lag time without recomputing the preprocessed cache?

4. **Dataset loading model:** Does the dataset load the full preprocessed cache into memory at init, or lazy-load slices? How are time-lagged pairs (x_t, x_{t+τ}) constructed?

5. **Reversible model integration:** How does `RevVAMPNet` differ from `VAMPNet` in its model creation path? Does the retrain logic already support both, or only `VAMPNet`?

6. **Existing retrain loop:** Where is the retrain max iterations currently set? How is the loop structured? What triggers a retrain?

7. **BatchNorm running stats:** Are BN running stats currently saved/loaded with the model checkpoint? If a model is rebuilt mid-pipeline, what happens to BN stats?

**STOP after Phase 1 and present the report to the user before proceeding.** The user will confirm the findings and may adjust the implementation plan based on what you found. Do not start Phase 2 until explicitly approved.

---

## Phase 2: Auto-Stride

**Goal:** Add a feature that automatically computes an appropriate stride per lag time, so multi-lag-time experiments don't waste compute on near-duplicate frames at long lag times.

**Read first:**
- Phase 1's findings on stride handling
- `pygv/config/base_config.py`
- The dataset module Phase 1 identified
- The pipeline file that handles multi-lag-time runs (likely `pygv/pipe/master_pipeline.py` or a sweep handler)

**Specification:**

The stride formula is:
```
stride = max(1, floor(τ / (10 × frame_dt)))
```
where `frame_dt` is the time delta between consecutive frames in the preprocessed cache (not the raw trajectory). For example: if `frame_dt = 200 ps` and `τ = 20 ns`, then `stride = max(1, floor(20000 / 2000)) = 10`.

**CLI flag:** Add `--auto_stride` (boolean, default `False`). When False, the pipeline uses the user-specified stride exactly as it does today.

**Behavior when on:**
- Stride is recomputed per lag time inside the multi-lag pipeline. If the pipeline runs τ = 1, 5, 10, 20 ns, each gets its own stride.
- Within a single lag time (including all retrain rounds at that lag time), stride is fixed. Retrain does not change stride.
- The pipeline must log the computed stride for each lag time at INFO level: `Auto-stride: τ=20.0ns, frame_dt=200ps, stride=10, effective frames=N`.
- The effective number of training pairs after stride should also be logged: `Pairs per epoch: N`.

**`frame_dt` source:** This must be discoverable from the preprocessing metadata or computable from the trajectory. Phase 1 should have identified where this lives. If it's not currently stored, add it to the preprocessing output or read it from the trajectory at pipeline start. Don't hardcode it.

**Where to apply the stride:**
- If stride is currently a runtime parameter (per Phase 1), the implementation is straightforward: pass the auto-computed value to the dataset constructor instead of the user value.
- If stride is currently a preprocessing parameter, you need to either (a) add a runtime stride that subsamples on top of the preprocessed cache, or (b) refactor stride to be runtime. Option (a) is preferred for minimal disruption — it means the auto-stride value is applied as additional subsampling on top of any preprocessing-level stride. This requires the auto-stride value to be ≥ the preprocessing-level stride. Add a sanity check.

**Edge cases to handle:**
- If the auto-computed stride is less than the preprocessing-level stride, log a warning and use the preprocessing-level stride: "Auto-stride requested 1, preprocessing already at stride 5, using 5."
- If the resulting dataset has fewer than 1000 training pairs, log a warning: "Auto-stride at τ=Xns produces only N pairs, may have insufficient statistical power."
- If `frame_dt` cannot be determined, fail with a clear error message rather than guessing.

**Tests to write at `tests/test_auto_stride.py`:**
- `test_stride_computation_correctness`: Verify the formula returns expected values for several (τ, frame_dt) combinations including edge cases (τ very small, τ very large).
- `test_stride_minimum_one`: Verify stride is never 0 even for very small τ.
- `test_stride_per_lag_time`: Verify a multi-lag pipeline assigns different strides to different lag times.
- `test_stride_constant_within_lag`: Verify retrains at the same lag time use the same stride.
- `test_stride_off_by_default`: Verify pipeline behavior is unchanged when `--auto_stride` is not specified.
- `test_preprocessing_stride_floor`: Verify the auto-stride respects the preprocessing-level stride as a floor.

**Verification before moving to Phase 3:**
- All Phase 2 tests pass
- A manual end-to-end run with `--auto_stride` on a small Aβ42 subset across 2 lag times completes without error and logs the computed strides correctly
- A manual run without `--auto_stride` on the same data produces identical results to the pre-Phase-2 behavior (regression check)

---

## Phase 3: Warm-Starting

**Goal:** When a retrain is triggered with a new state count k, preserve the encoder/embedding weights and BN running stats, swap only the classifier head with new k dimensionality, and reinitialize the optimizer fully.

**Read first:**
- Phase 1's findings on model reconstruction and BN handling
- `pygv/vampnet/vampnet.py` and `pygv/vampnet/rev_vampnet.py`
- The retrain loop identified in Phase 1
- The classifier module (likely `SoftmaxMLP` based on the codebase)

**Specification:**

**When warm-starting fires:** Only when the diagnostic recommends a *new* k that differs from the current model's k. If the same k is recommended (which is the convergence signal), no retraining happens — the loop terminates per Phase 4.

**What gets preserved:**
- All encoder weights (SchNet/GIN/ML3)
- Embedding MLP weights
- BatchNorm running statistics in the encoder and embedding modules
- The reversible score module's parameters (`u`, `S`) if using `RevVAMPNet` — these depend on n_states, see special handling below

**What gets replaced:**
- The classifier head (its output dimension changes from old_k to new_k)
- Any BN layers *inside the classifier* (their dimensions change with k)

**What gets reinitialized:**
- The optimizer entirely (Adam momentum state for old classifier weights is meaningless)
- Learning rate schedule reset to start

**Special handling for RevVAMPNet:**
The reversible score module's `log_stationary` is shape `(n_states,)` and `rate_matrix_weights` is shape `(n_states, n_states)`. Both depend on k. When k changes, these must be reinitialized. The user's intent is to preserve only the encoder representation; the reversible parameters relearn from scratch. Document this in the docstring.

**CLI flag:** Add `--warm_start_retrains` (boolean, default `False`). When False, retrains rebuild the model from scratch (current behavior). When True, retrains use the warm-start path. Phase 4 will likely flip the default to True, but in Phase 3 keep it False.

**Implementation approach:**

Add a method to the model class (both `VAMPNet` and `RevVAMPNet`):
```python
def warm_restart_with_new_k(self, new_k: int) -> None:
    """
    Replace the classifier head with one sized for new_k states.
    Preserves encoder, embedding, and BN running stats.
    
    For RevVAMPNet, also reinitializes the reversible score module
    (log_stationary and rate_matrix_weights) to the new size.
    
    Note: The optimizer must be recreated by the caller after this
    method returns, because parameter references have changed.
    """
```

In the pipeline retrain loop, branch on the `warm_start_retrains` flag:
```python
if args.warm_start_retrains and isinstance(model, (VAMPNet, RevVAMPNet)):
    model.warm_restart_with_new_k(new_k)
    optimizer = create_optimizer(model, args)  # full reinit
    scheduler = create_scheduler(optimizer, args)  # if applicable
else:
    # Existing rebuild path (unchanged)
    model = create_model(args, n_states=new_k)
    optimizer = create_optimizer(model, args)
```

**Tests to write at `tests/test_warm_start.py`:**
- `test_warm_restart_preserves_encoder_weights`: Save encoder weights before warm-restart, call warm-restart, verify encoder weights are byte-identical after.
- `test_warm_restart_replaces_classifier`: Verify classifier output dim changes from old_k to new_k.
- `test_warm_restart_preserves_bn_running_stats`: Verify encoder's BN running_mean/running_var are unchanged after warm-restart.
- `test_warm_restart_revvampnet_resets_reversible`: For RevVAMPNet, verify log_stationary and rate_matrix_weights are reset to fresh init (not preserving old values).
- `test_warm_restart_optimizer_reinit_required`: Verify the model.parameters() iterator includes the new classifier parameters (so a fresh optimizer picks them up correctly).
- `test_warm_restart_off_by_default`: Verify pipeline rebuilds from scratch when flag is not set.
- `test_warm_restart_short_training_converges`: Train a small model, warm-restart with reduced k, train 10 more epochs, verify final VAMP-2 is reasonable (not catastrophically broken).

**Verification before moving to Phase 4:**
- All Phase 3 tests pass
- An end-to-end run with `--warm_start_retrains` on small Aβ42 data with a forced retrain (e.g., diagnostic recommends k=8 from k=10) completes successfully and produces a reasonable final VAMP-2 score
- The warm-started run uses meaningfully less wall time than a from-scratch retrain (verify with timing)
- An existing test suite still passes (regression check)

---

## Phase 4: Retrain Policy

**Goal:** Replace the current `max_retrain=2` policy with `max_retrain=5` and add early termination when the same k is recommended in two consecutive rounds.

**Read first:**
- Phase 1's findings on the existing retrain loop
- `pygv/config/base_config.py` for retrain config
- The pipeline file containing the retrain loop

**Specification:**

**New default behavior:**
- `max_retrains = 5` (previously 2)
- Early termination when the diagnostic recommends the same k in two consecutive rounds
- If `max_retrains` exhausted without convergence: use the last model, log a clear warning

**Why this policy works:**
- With warm-starting (Phase 3), retrains are cheap (~5–10 min each instead of full training time)
- Most pipelines converge in 2–3 rounds; the cap of 5 protects against pathological oscillation
- "Same k twice" is the natural convergence signal because the diagnostic has stabilized
- "Use last model" on exhaustion is correct because the last model has the smallest k recommended (typically) which is what the diagnostic is pushing toward; there's no benefit to going back to a "better val score" earlier model that had more states

**CLI flags / config:**
- `max_retrains: int = 5` (configurable in base_config, was likely 2 before)
- `--max_retrains` CLI override
- `convergence_check: bool = True` (defaults on) — allows users to disable early termination if they want exactly N retrains for some reason
- `--no_convergence_check` CLI flag to disable

**Default for `--warm_start_retrains`:** Flip to True. With max_retrains=5 and warm-starting, the from-scratch rebuild mode would be unbearably slow. Users who want the old behavior can pass `--no_warm_start_retrains` (add this flag).

**Implementation:**

```python
def run_retrain_loop(model, args, ...):
    k_history = [model.n_states]
    best_model_state = None  # in case we want to track this for some reason
    
    for retrain_round in range(args.max_retrains):
        # Run training (warm-started or from scratch per phase 3 logic)
        train_one_round(model, ...)
        
        # Run diagnostics
        recommended_k = run_diagnostics(model, ...)
        k_history.append(recommended_k)
        
        if recommended_k == model.n_states:
            logger.info(f"Convergence: same k={recommended_k} as current. Stopping retrains.")
            break
        
        if args.convergence_check and len(k_history) >= 3:
            if k_history[-1] == k_history[-2]:
                logger.info(f"Convergence: same k={recommended_k} recommended twice. Stopping retrains.")
                break
        
        # Apply warm-start or full rebuild per Phase 3
        if args.warm_start_retrains:
            model.warm_restart_with_new_k(recommended_k)
            optimizer = create_optimizer(model, args)
        else:
            model = create_model(args, n_states=recommended_k)
            optimizer = create_optimizer(model, args)
    
    else:
        # Loop completed without break
        logger.warning(
            f"Retrain loop exhausted ({args.max_retrains} rounds) without convergence. "
            f"k history: {k_history}. Using last trained model with k={model.n_states}."
        )
    
    return model
```

**Adapt the above to the actual code structure** found in Phase 1. The pseudocode is illustrative.

**Tests to write at `tests/test_retrain_policy.py`:**
- `test_terminate_on_convergence`: Mock the diagnostic to recommend k=8 then k=8, verify loop terminates after 2 rounds.
- `test_continue_on_no_convergence`: Mock the diagnostic to recommend k=10 → 8 → 7 → 6, verify loop continues until either max or convergence.
- `test_max_retrains_cap`: Mock the diagnostic to always recommend a different k, verify loop stops at max_retrains and logs warning.
- `test_max_retrains_uses_last_model`: After exhaustion, verify the returned model has the k from the final retrain attempt.
- `test_default_max_is_five`: Verify the default value in config is 5.
- `test_convergence_check_can_be_disabled`: With `--no_convergence_check`, verify loop runs all max_retrains even if k stabilizes.
- `test_warm_start_default_on`: Verify warm_start_retrains defaults to True after Phase 4.

**Verification before moving to Phase 5:**
- All Phase 4 tests pass
- An end-to-end run with default settings (warm_start=True, max_retrains=5, convergence_check=True) on small Aβ42 data converges and terminates correctly
- Wall time per retrain round is meaningfully shorter than Phase 3 baseline (warm-starting is being used)
- Old behavior is recoverable: `--no_warm_start_retrains --max_retrains 2 --no_convergence_check` reproduces pre-Phase-4 behavior

---

## Phase 5: Integration Tests

**Goal:** Verify that all three features compose correctly end-to-end on real (small) data.

**Create `tests/test_phase5_integration.py`:**

`test_full_multi_lag_pipeline_with_all_features`:
- Use a small Aβ42 subset (~10k frames after preprocessing)
- Run multi-lag pipeline with `--auto_stride --warm_start_retrains` and default max_retrains=5
- Lag times: [1.0, 5.0, 10.0] ns
- Verify: pipeline completes, each lag time has a different stride logged, retrain converges in ≤5 rounds for each lag time, output files exist for each lag time

`test_features_can_be_individually_disabled`:
- Run with `--no_warm_start_retrains` only (auto_stride still on)
- Run with auto_stride off only (warm_start still on)
- Run with both off
- Verify all three configurations complete and produce reasonable outputs

`test_reversible_model_with_all_features`:
- Same as `test_full_multi_lag_pipeline_with_all_features` but with `--reversible`
- Verify warm-starting correctly handles the reversible score module

`test_backward_compatibility`:
- Run a pre-existing config (one that worked before these phases) with no new flags
- Verify the result matches what the pre-Phase-2 codebase would have produced (within numerical tolerance for any stochastic differences)

**Verification before moving to Phase 6:**
- All Phase 5 tests pass
- Existing test suite still passes
- Manual end-to-end on real Aβ42 with all features on completes within expected wall-time budget (~30 min for a 3-lag-time, k-converging-in-2-rounds run)

---

## Phase 6: Documentation Update

**Goal:** Update the codebase documentation to reflect the three new features and the new retrain defaults.

**Files to update:**

`README.md`: Add a section on the three new features with example invocations:
```bash
# Old behavior (single lag time, manual stride, fixed max_retrains=2)
python run_pipeline.py --traj_dir ... --top ... --lag_time 5.0 --n_states 8

# New: multi-lag with auto-stride and warm-start (recommended for production)
python run_pipeline.py --traj_dir ... --top ... \
    --lag_times 1.0 5.0 10.0 20.0 \
    --auto_stride --warm_start_retrains \
    --max_retrains 5

# To reproduce old behavior exactly
python run_pipeline.py --traj_dir ... --top ... --lag_time 5.0 --n_states 8 \
    --no_warm_start_retrains --max_retrains 2 --no_convergence_check
```

`CODEBASE_SUMMARY.md`: Update the components list with:
- Auto-stride logic (file location)
- Warm-start methods on VAMPNet and RevVAMPNet
- New retrain policy

Add docstrings to:
- `warm_restart_with_new_k` on both model classes
- The auto-stride computation function
- The retrain loop function

**No further verification beyond docs being readable and the example commands being syntactically correct.**

---

## Final Verification Checklist

Before declaring the implementation complete:

- [ ] Phase 1 report exists and was reviewed
- [ ] All phase test files exist and pass: `pytest tests/test_auto_stride.py tests/test_warm_start.py tests/test_retrain_policy.py tests/test_phase5_integration.py`
- [ ] Existing test suite still passes
- [ ] Three end-to-end runs succeed:
  - All features off (backward compat)
  - All features on (production mode)
  - All features on with reversible model
- [ ] Documentation updated (README, CODEBASE_SUMMARY, docstrings)
- [ ] No unrelated code was modified (`git diff` review)

## Things NOT to do

- Do not implement experiment-launching scripts. The user will write those separately.
- Do not modify the encoder code (SchNet, GIN, ML3). They are out of scope.
- Do not change the VAMP score implementation. Out of scope.
- Do not refactor the dataset module beyond what's needed for auto-stride.
- Do not change the analysis pipeline beyond logging the auto-stride values.
- Do not "improve" the diagnostics logic. Only consume its output.

If you find a bug in code you're not modifying, log it as a comment in your final report but do not fix it. Scope discipline is more valuable than opportunistic fixes.
