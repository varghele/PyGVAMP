# Phase 1 Reconnaissance Report — Auto-Stride / Warm-Start / Retrain Policy

Scope: read-only investigation of the existing PyGVAMP codebase at `/home/vi/PycharmProjects/PyGVAMP` to answer seven questions that drive Phases 2–5 of the implementation plan. All citations are in `file:line` form against the current `main` working tree.

---

## 1. State diagnostics — how recommended k is computed

**Where the recommendation comes from:**
- The entry point consumed by the pipeline is `recommend_state_reduction(...)` in `pygv/utils/state_diagnostics.py:248-334`.
- It returns a `StateReductionReport` (`pygv/utils/state_diagnostics.py:21-61`). Two fields matter downstream:
  - `recommendation: str` — `"keep"` or `"retrain"` (`pygv/utils/state_diagnostics.py:316-319`).
  - `effective_n_states: int` — this is the value the retrain loop actually uses.
- The analysis pipeline builds the report at `pygv/pipe/analysis.py:305-310` and the retrain loop reads `report.effective_n_states` at `pygv/pipe/master_pipeline.py:458` and again at `pygv/pipe/master_pipeline.py:520`.

**How `effective_n_states` is computed** (`pygv/utils/state_diagnostics.py:279-303`):

```
n_from_eigenvalues = analyze_eigenvalue_gap(K)     # bounded by clamp below
n_from_populations = n_states - len(underpopulated)  # <= n_states
n_from_jsd         = n_states - states_absorbed      # <= n_states
effective_n_states = int(np.median([n_from_eig, n_from_pop, n_from_jsd]))
effective_n_states = max(2, min(effective_n_states, n_states))   # line 303
```

**Can recommended k grow (e.g. 8 → 10)?** **No.** Two of the three signals (`n_from_populations`, `n_from_jsd`) are constructed as `n_states - X` where `X >= 0`, so they are bounded above by the current `n_states`. The median of three numbers, two of which are `<= n_states`, is itself `<= n_states`. The explicit clamp `min(effective_n_states, n_states)` on `pygv/utils/state_diagnostics.py:303` enforces this hard. So k can only **shrink or stay the same**. There is no code path by which this function recommends a larger k.

Side note: `analyze_eigenvalue_gap` alone (`pygv/utils/state_diagnostics.py:68-116`) could in principle return any `n_states`-compatible value, but it is combined by median with the two monotonically-non-increasing signals and then clamped — so the combined recommendation cannot exceed the incoming `n_states`.

**Where retrain is triggered:**
- `pygv/pipe/master_pipeline.py:457` — collects experiments with `recommendation == "retrain"`.
- `pygv/utils/state_diagnostics.py:316-319` — the only place `recommendation` is set. It's `"keep"` only if `effective_n_states == n_states AND merge_groups == [] AND underpopulated == []`; otherwise `"retrain"`. That means even when `effective_n_states == n_states`, the recommendation can still be `"retrain"` if there are any merge groups or any underpopulated states. (Whether this is intended is worth noting — see Ambiguities.)

---

## 2. Model reconstruction on retrain

**Short answer:** the model is rebuilt entirely from scratch. No state is preserved between the pre-retrain model and the retrained model; they share nothing.

**Trace:**
- Retrain loop is `PipelineOrchestrator._run_retrain_loop` at `pygv/pipe/master_pipeline.py:432-524`.
- Inside the loop, for each pending experiment, it constructs fresh `train_args` (`pygv/pipe/master_pipeline.py:489-491`), then calls `run_training(train_args)` (`pygv/pipe/master_pipeline.py:493`).
- `run_training` (`pygv/pipe/training.py:405-535`) calls `create_model(args)` at `pygv/pipe/training.py:444` with the new `args.n_states` value.
- `create_model` (`pygv/pipe/training.py:177-328`) instantiates a completely fresh encoder, classifier, optional embedding MLP, vamp/rev score module, and then `VAMPNet(...)` (line 312) or `RevVAMPNet(...)` (line 303) — all newly-constructed objects. `init_for_vamp(model, method='kaiming_normal')` (line 322) re-initialises weights.
- The previous model object goes out of scope when its analysis results are overwritten; there is no reference to it by the new model.

**What state is preserved vs discarded on retrain:**
- Discarded: encoder weights, embedding MLP weights, classifier weights, BatchNorm running stats anywhere in the model, optimizer state, LR-scheduler state, reversible score parameters (`log_stationary`, `rate_matrix_weights`).
- Preserved: only the dataset (cached at `dataset_path`) and the experiment directory structure. A new timestamped training-run subdir is created underneath `dirs['training']/{retrained_exp}` (`pygv/pipe/master_pipeline.py:485-486`) and a new `best_model.pt` is written into it.

This is the behavior Phase 3's warm-starting is intended to replace (opt-in) — the current rebuild path is what you'd fall back to when `--warm_start_retrains` is False.

---

## 3. Stride handling

**Short answer:** stride is currently a **preprocessing-time parameter**. It is baked into the cached `.pkl` and into the in-memory `self.frames` array at load time. It is **not** re-applied inside `__getitem__` and not by the DataLoader. Varying stride per lag time would require either recomputing the cache or subsampling on top of it.

**Where stride is applied:**
- `pygv/dataset/vampnet_dataset.py:176` — inside `_process_trajectories()`:
  ```python
  traj = traj[::self.stride]
  coords = traj.xyz[:, self.atom_indices, :]
  self.frames.extend(coords)
  ```
  Frames are stored post-stride. The subsequent `coords`, `self.frames`, `n_frames`, `atom_indices`, `trajectory_boundaries`, and `t0/t1_indices` all treat the strided sequence as "the data".
- `pygv/dataset/vampnet_dataset.py:607` — `effective_timestep = timestep * self.stride` is used to compute `self.lag_frames = int(lag_time_ps / effective_timestep)` at line 619. So the time-lagged-pair index offset is computed against the strided timeline.
- `pygv/dataset/vampnet_dataset.py:241-260` — `_create_time_lagged_pairs` uses `self.lag_frames` directly as the integer frame-index offset; `__getitem__` at line 445-453 just indexes into `self.frames` using the precomputed `t0_indices`/`t1_indices`. No stride appears in `__getitem__`.

**Cache key includes stride:**
- `pygv/dataset/vampnet_dataset.py:478` — cache filename is `f"vampnet_data_{traj_hash}_lag{lag_time}_nn{nn}_str{stride}_{cont_flag}.pkl"`. A different stride -> different cache file -> full reprocess.
- The cache also includes `lag_time` in the filename. Combined with stride, this means per-lag-time stride changes at minimum force a fresh pickle per (lag, stride) combination. (`pygv/dataset/vampnet_dataset.py:473-522`)

**Could stride be varied per lag time without recomputing the cache?**
Not directly: `self.frames` is already strided and `self.lag_frames` is computed from the strided timeline. Phase 2's Option (a) (runtime subsampling on top of the preprocessing-level stride) is the viable approach — you'd keep `self.frames` as is and subsample the `t0_indices`/`t1_indices` arrays (or equivalently re-derive them with a larger lag in the strided timeline). The plan's constraint that "auto-stride must be >= preprocessing-level stride" is what makes this safe.

**Is `frame_dt` stored anywhere?** **No.** It's inferred at runtime from trajectory `time[1] - time[0]` in three places:
- `pygv/dataset/vampnet_dataset.py:567-625` — `_infer_timestep()`, returns picoseconds.
- `pygv/pipe/master_pipeline.py:228-244` — inside `_calculate_optimal_stride` (for `--hurry` mode).
- `pygv/pipe/master_pipeline.py:692-755` — `validate_lag_times`.

The preprocessing output (`dataset_stats.json`, `pygv/pipe/preparation.py:267-274`) records `lag_time` and `stride` but **not** the raw `frame_dt`. This means:
1. The effective `frame_dt` of the preprocessed cache (= raw timestep × preprocessing stride) is not persisted.
2. Phase 2 needs to either (a) write the raw per-frame timestep in ps into the prep metadata, or (b) re-infer at pipeline start from the trajectory itself.

Worth noting: the `timestep` config field (`pygv/config/base_config.py:23`) is a **user override in nanoseconds**, not the discovered value. The discovered value only lives in local variables inside `_calculate_optimal_stride` and `validate_lag_times` and is not surfaced to Phase 2 consumers.

---

## 4. Dataset loading model

**Class:** `VAMPNetDataset` at `pygv/dataset/vampnet_dataset.py:21`.

**Loads full preprocessed cache into memory at init, lazy-builds graphs per-item:**
- On `__init__` the dataset either loads `self.frames` (raw xyz coordinates, shape `(n_frames, n_atoms, 3)`) from the pickled cache (`pygv/dataset/vampnet_dataset.py:524-565`, `_load_from_cache`), or processes every trajectory file with `md.load` and concatenates into memory (`pygv/dataset/vampnet_dataset.py:155-203`, `_process_trajectories`). Either way the full coord array lives in RAM as `np.ndarray`.
- Graphs (edge_index, edge_attr, node features) are built on-demand in `_create_graph_from_frame` (`pygv/dataset/vampnet_dataset.py:365-439`) called from `__getitem__`. Graph construction is per-frame — k-NN edges, Gaussian expansion — computed from the stored coordinate tensor. So it's "eager for frames, lazy for graphs".
- Optional `precompute_graphs(max_graphs=...)` at `pygv/dataset/vampnet_dataset.py:689-726` materializes graphs up front and replaces `__getitem__` via a bound override; this path is not wired into `create_dataset_and_loader` in the training pipeline.

**Time-lagged pair construction:**
- Constructed during init in `_create_time_lagged_pairs` (`pygv/dataset/vampnet_dataset.py:233-274`).
- Stores integer index lists `self.t0_indices` and `self.t1_indices` (both length = number of pairs). No actual data duplication — just index pairs.
- Two modes:
  - `continuous=True` (default): `t0 = range(0, n_frames - lag_frames)`, `t1 = range(lag_frames, n_frames)` (`pygv/dataset/vampnet_dataset.py:241-242`). Pairs cross trajectory-file boundaries.
  - `continuous=False`: pairs only within each trajectory (`pygv/dataset/vampnet_dataset.py:244-260`). Uses `self.trajectory_boundaries` to stay in-bounds.
- `__getitem__(idx)` at `pygv/dataset/vampnet_dataset.py:445-453` looks up `t0_indices[idx]` / `t1_indices[idx]` and builds the two graphs on the fly.
- Number of pairs: `len(self.t0_indices)` — exposed via `__len__` (`pygv/dataset/vampnet_dataset.py:441-443`).

**Frames-only view:** `get_frames_dataset(return_pairs=False)` at `pygv/dataset/vampnet_dataset.py:728-762` wraps the dataset so each item is a single graph (for inference/analysis).

---

## 5. Reversible model integration — RevVAMPNet vs VAMPNet, score parameter names

**Construction path differences** (`pygv/pipe/training.py:298-319`):
- Both go through `create_model(args)`. Branch at `pygv/pipe/training.py:298`:
  - If `args.reversible`: creates `ReversibleVAMPScore(n_states=args.n_states, epsilon=args.vamp_epsilon)` (`pygv/pipe/training.py:299-302`) then `RevVAMPNet(embedding_module, encoder, rev_score, classifier_module, lag_time, training_jitter)` (`pygv/pipe/training.py:303-310`).
  - Else: creates `VAMPScore(epsilon=args.vamp_epsilon, mode='regularize')` (`pygv/pipe/training.py:247`) then `VAMPNet(embedding_module, encoder, vamp_score, classifier_module, lag_time, training_jitter)` (`pygv/pipe/training.py:312-319`).
- Both classes use the same encoder / embedding MLP / classifier instances; they are allocated identically above the branch. The classifier in `create_model` (`pygv/pipe/training.py:249-281`) is a `SoftmaxMLP` with `out_channels=args.n_states`.

**Does retrain logic handle both?** The existing `_run_retrain_loop` does not care — it calls `run_training` which calls `create_model` which already branches on `args.reversible`. So the current from-scratch retrain mode handles both models identically by virtue of full-rebuild. **But** any warm-start path (Phase 3) must explicitly handle `RevVAMPNet`'s extra score module (see below).

**Reversible score parameters depending on n_states** (`pygv/scores/reversible_score.py:23-31`):
```python
class ReversibleVAMPScore(nn.Module):
    def __init__(self, n_states: int, epsilon: float = 1e-6):
        super().__init__()
        self.n_states = n_states
        self.epsilon = epsilon
        self.log_stationary       = nn.Parameter(torch.zeros(n_states))               # (n_states,)
        self.rate_matrix_weights  = nn.Parameter(torch.zeros(n_states, n_states))     # (n_states, n_states)
```

**Exact field names in actual code:**
- `log_stationary` — `pygv/scores/reversible_score.py:30`, shape `(n_states,)`. Used by `get_stationary_distribution` via softmax (`pygv/scores/reversible_score.py:33-42`).
- `rate_matrix_weights` — `pygv/scores/reversible_score.py:31`, shape `(n_states, n_states)`. Symmetrised + softplus'd inside `get_rate_matrix` (`pygv/scores/reversible_score.py:44-55`).

**So the plan's exact naming matches the code** — good news. No translation needed for Phase 3. On `RevVAMPNet` the score module lives at `self.rev_score` (`pygv/vampnet/rev_vampnet.py:65`); not at `self.vamp_score` as in `VAMPNet`. Both names are used consistently throughout.

Also note: `ReversibleVAMPScore` is not `nn.Module`-nested inside the classifier — it's a peer of the encoder/classifier. When k changes, one cannot just "replace the classifier"; the score module itself must also be replaced or resized. The cleanest is to construct a fresh `ReversibleVAMPScore(n_states=new_k, epsilon=...)` and assign it to `model.rev_score`.

**Accessors that also depend on n_states:**
- `get_transition_matrix()` at `pygv/scores/reversible_score.py:57-98` — uses `self.n_states` (line 83). So `self.n_states` must also be updated, not just the parameter shapes.

---

## 6. Existing retrain loop

**Location:** `PipelineOrchestrator._run_retrain_loop` at `pygv/pipe/master_pipeline.py:432-524`.

**Where max_retrain is set (value + file:line):**
- Method signature: `def _run_retrain_loop(self, dirs, dataset_path, trained_models, analysis_results, max_retrain=2):` at `pygv/pipe/master_pipeline.py:432-433`. **Default is 2**, set via parameter default.
- Caller passes no override: `pygv/pipe/master_pipeline.py:585` — `self._run_retrain_loop(dirs, dataset_path, trained_models, analysis_results)`. So it's effectively hard-coded 2 from the user's perspective — there is no CLI flag or config field.
- `BaseConfig` has no `max_retrain` / `max_retrains` field (searched `pygv/config/base_config.py`). Phase 4 will need to add one.

**Loop structure** (pseudocoded — real at `pygv/pipe/master_pipeline.py:452-524`):
```
pending = [(exp_name, effective_n_states, iteration=0)
           for each experiment whose diagnostic recommends "retrain"]

while pending:
    exp_name, new_n_states, iteration = pending.pop(0)
    # parse lag_time from exp_name via regex
    # build retrained_exp name: lag{lag}ns_{new_n_states}states_retrained[_N]
    # run_training(train_args with n_states=new_n_states) -> new best_model.pt
    # run_analysis(...) -> new report
    if report.recommendation == "retrain" and iteration + 1 < max_retrain:
        pending.append((retrained_exp, report.effective_n_states, iteration + 1))
    elif report.recommendation == "retrain":
        print("max iterations reached")
```

**What triggers a retrain vs terminates:**
- **Initial trigger** (`pygv/pipe/master_pipeline.py:453-458`): populated from the FIRST-pass analysis of experiments that recommend `"retrain"`. These are the only experiments that get queued. An experiment whose initial diagnostic is `"keep"` never enters the loop.
- **Continuation** (`pygv/pipe/master_pipeline.py:513-520`): after each retrained experiment, if its new analysis still says `"retrain"` AND `iteration + 1 < max_retrain`, re-queue with incremented iteration.
- **Termination** (`pygv/pipe/master_pipeline.py:521-524`): the loop's natural `while pending:` exits when nothing is queued, i.e. when either (a) the diagnostic no longer recommends retrain on the latest model, or (b) max_retrain was hit. On (b), a message is printed — but no clear "warning" logging level, just a `print`.

**No convergence check on same-k.** The loop just checks `recommendation == "retrain"`. Even if `effective_n_states == model.n_states` (possible when `merge_groups` or `underpopulated` is nonempty — see Q1 discussion), the loop would still trigger a retrain at the same k. Phase 4's "same k twice" convergence signal would catch this, but it is not currently present.

**The `ab42_red_discovery.sh` and similar files** under `cluster_scripts/` are submission scripts; they invoke `run_pipeline.py`, they do not contain any retrain logic.

---

## 7. BatchNorm — running stats persistence

**Are there BN layers in the encoder/embedding?** Yes, almost certainly — in the SchNet encoder via `torch_geometric.nn.models.MLP`'s default `norm="batch_norm"`.

- `torch_geometric.nn.models.MLP.__init__` default is `norm="batch_norm"` (verified at `/opt/software/pygvamp/1.0.0/conda_env/lib/python3.12/site-packages/torch_geometric/nn/models/mlp.py:90`).
- In `pygv/encoder/schnet.py`, several MLPs are constructed without passing `norm=...`, i.e. using the default:
  - `SchNetEncoderNoEmbed.output_network` MLP at `pygv/encoder/schnet.py:242-249` — no `norm=` kwarg -> default `batch_norm`.
  - `GCNInteraction.output_layer` MLP at `pygv/encoder/schnet.py:184-190` — no `norm=` kwarg -> default `batch_norm`. This lives inside each of the `n_interactions` interaction blocks.
- One MLP in the encoder does explicitly disable BN:
  - `CFConv.filter_network` at `pygv/encoder/schnet.py:49-57` passes `norm=None`.
- GIN encoder (`pygv/encoder/gin.py:87`) explicitly passes `norm=None` — no BN.
- ML3 encoder (`pygv/encoder/ml3.py:385, 397`) explicitly passes `norm=None` — no BN.
- Meta and MetaAtt encoders (`pygv/encoder/meta.py:26,44,53,75` and `pygv/encoder/meta_att.py:26,113,123,161`) hardcode `norm="BatchNorm"` in their internal MLPs — very heavy BN presence.

**Embedding MLP:** created in `create_model` at `pygv/pipe/training.py:283-295` via `MLP(...)` with `norm=args.embedding_norm`. The default CLI value of `--embedding_norm` is `None` (`pygv/args/args_train.py:33-35`) and the config default `embedding_norm: Optional[str] = None` (`pygv/config/base_config.py:96`). **In the default config, the embedding MLP has no BN.** But a user can opt in by passing `--embedding_norm BatchNorm`.

**Classifier BN:** `clf_norm` defaults to `"batch_norm"` in the config (`pygv/config/base_config.py:103`), although the CLI arg default is `None` (`pygv/args/args_train.py:53-55`). Either way, BN-in-classifier is the user-intended configuration. In the warm-start path, this BN must be rebuilt (new k → different classifier dimensions → new BN dimensions).

**Preserving BN running stats across a mid-pipeline rebuild:** currently **nothing is preserved**. On retrain, `create_model` builds brand new nn.Modules and `init_for_vamp` freshly initialises them (`pygv/pipe/training.py:322`). `init_for_vamp` does touch BN layers (`pygv/utils/nn_utils.py:122-129`) — it re-initialises weight=1, bias=small-nonzero, which overwrites any prior buffer/running state. No code loads running stats from the previous model; the previous model object is simply dropped.

**Are BN running stats saved with the checkpoint?**
- `VAMPNet.save_complete_model` at `pygv/vampnet/vampnet.py:584-601` uses `torch.save(self, filepath)` — this pickles the full `nn.Module` including buffers. BN `running_mean` / `running_var` / `num_batches_tracked` are registered buffers and are pickled. **So the running stats are saved with the checkpoint on disk**, and reloading via `torch.load(...)` / `load_complete_model` (`pygv/vampnet/vampnet.py:603-629`) restores them.
- `VAMPNet.save` at `pygv/vampnet/vampnet.py:374-463` saves `self.state_dict()` (line 394), which also includes BN buffers. Same for `RevVAMPNet.save` (`pygv/vampnet/rev_vampnet.py:318-392`) and `RevVAMPNet.save_complete_model` (`pygv/vampnet/rev_vampnet.py:394-405`).

**However** — the retrain loop never reads these from disk. The retrain path doesn't touch the previous checkpoint at all; it just calls `run_training` with new args and builds everything from scratch. So while the running stats *exist* on disk, they are discarded in practice during retrain.

For Phase 3's warm-restart, the plan is to preserve the live `nn.Module` object (which already has the running stats in its buffers) and only replace the classifier head. This is feasible without any checkpoint I/O: the encoder and (optional) embedding-MLP nn.Module references remain alive inside the model; reassigning `model.classifier_module = SoftmaxMLP(...)` with the new out_channels leaves encoder BN buffers untouched. For `RevVAMPNet`, replacing `model.rev_score` with a fresh `ReversibleVAMPScore(n_states=new_k)` is the corresponding step — and crucially the rev_score has no BN of its own (just `log_stationary` and `rate_matrix_weights` as raw `nn.Parameter`s), so there's nothing extra to worry about on that module.

---

## Ambiguities / bugs noticed (not fixed)

Per the plan's "Things NOT to do": I have not fixed any of these. Logged here for visibility.

1. **`recommendation="retrain"` can fire even when `effective_n_states == n_states`.** `pygv/utils/state_diagnostics.py:316-319` sets `recommendation="keep"` only when `effective_n_states == n_states AND not merge_groups AND not underpopulated`. If the median estimate lands back at `n_states` but one of the two other flags is nonempty, the loop retrains at the same k, which is guaranteed to produce the same (or a symmetrically similar) report — potential oscillation. Phase 4's "same k twice" convergence check would handle this, but it's a real anomaly today.

2. **Cache can be silently stale.** `_load_from_cache` at `pygv/dataset/vampnet_dataset.py:552-560` detects config mismatch (e.g. different lag_time) and prints a warning but proceeds with the cached data anyway. This means a user who re-runs with a different `--stride` could load an old-stride cache if the hashing of trajectory files happens to be the same but stride was not part of the stored file's identity match. Actually the cache filename already embeds `str{stride}` (`pygv/dataset/vampnet_dataset.py:478`) so the filename mismatch would just result in "no cache found" + reprocess — so this is probably fine in the filename-embedded-keys case, but the warning-without-failing codepath is a footgun worth noting.

3. **`dataset._infer_timestep` is called in `pygv/pipe/analysis.py:252` every time the analysis runs.** This is fine functionally but means the trajectory file is opened a second time purely to recompute a value that was already computed at dataset init. Minor perf issue, not a correctness one.

4. **`--continuous` flag not wired into master pipeline.** `args.no_continuous` is read at `pygv/pipe/master_pipeline.py:890-891`, but `args.continuous` never gets set explicitly when `no_continuous` is False — instead the config's default `continuous=True` is used. Works in practice but inconsistent with the rest of the arg-override pattern. Additionally `_create_train_args` does not set `continuous` on the args namespace, it relies on `continuous` coming from the config dict spread (`pygv/pipe/master_pipeline.py:305`). Fine, just easy to misread.

5. **`_calculate_optimal_stride` scans down from stride=50 for compatibility** (`pygv/pipe/master_pipeline.py:250-259`). This is fine, but note that it stops at the first match rather than picking an optimum — which means with lag_time=10 and raw timestep=1, it returns stride=50 (effective 50, lag 10 => incompatible), 49, ... stopping at the first divisor. For many (lag, timestep) combos this gives a much coarser stride than the user likely expected. Relevant context for Phase 2: the auto-stride formula `max(1, floor(tau / (10 * frame_dt)))` is a simpler, more predictable model.

6. **`save` method in `VAMPNet` (not `save_complete_model`) has a bug in `classifier_config['hidden_channels']`** at `pygv/vampnet/vampnet.py:431-434`: `mlp_hidden_channels = getattr(self.classifier_module.mlp, 'hidden_channels', None)` — but `MLP.hidden_channels` is not a scalar; it's a property. Same in `RevVAMPNet.save` at `pygv/vampnet/rev_vampnet.py:368-370` and the corresponding `get_config` methods at `pygv/vampnet/vampnet.py:667-668` and `pygv/vampnet/rev_vampnet.py:465-466`. This path isn't exercised by `save_complete_model` (which pickles the whole model), but `save` + `load` would produce odd values. Non-blocking for Phases 2–5.

7. **`_create_analysis_args` in master_pipeline** (`pygv/pipe/master_pipeline.py:316-363`) sets `args.stride = self.config.stride` — the SAME stride used at training time. For Phase 2's auto-stride, the training-time stride for each lag time must be made available to analysis too (otherwise the analysis pipeline uses the wrong effective timestep in e.g. `calculate_transition_matrices`). Something to be aware of for Phase 2 implementation.

8. **Recent commits out of scope per instructions:** the main() return-value fix (`run_complete_pipeline` no longer returns a dict through main), the concat-and-score eval fix in `VAMPNet.quick_evaluate` / `evaluate` and `RevVAMPNet.quick_evaluate` / `evaluate`, and the `--lr_schedule {none,cosine}` plumbing (`pygv/pipe/training.py:344-352`, base_config lines 37-38). Noted here only so Phase 3 doesn't undo them by accident when touching the training module.

---

## Summary table for Phase 2-4 consumers

| Topic | Current state | Phase that touches it |
|---|---|---|
| Stride applied at | Preprocessing (`pygv/dataset/vampnet_dataset.py:176`) | Phase 2 (auto-stride) |
| Stride cached in filename | Yes (`pygv/dataset/vampnet_dataset.py:478`) | Phase 2 |
| `frame_dt` stored | No — only inferred at runtime | Phase 2 must add |
| Retrain loop | `pygv/pipe/master_pipeline.py:432-524` | Phase 3 + 4 |
| `max_retrain` | Hardcoded 2 (kwarg default at `pygv/pipe/master_pipeline.py:433`) | Phase 4 |
| Retrain model construction | Full rebuild via `create_model` | Phase 3 (warm-start) |
| RevVAMPNet score params | `log_stationary (n,)`, `rate_matrix_weights (n,n)` at `pygv/scores/reversible_score.py:30-31` | Phase 3 (reinit on k-change) |
| Encoder BN | Present in SchNet (via default MLP norm), Meta/MetaAtt (hardcoded); absent in GIN/ML3 | Phase 3 (preserve running stats) |
| Classifier BN | Present if `clf_norm="batch_norm"` (config default) | Phase 3 (rebuild with new dim) |
| Diagnostic recommends grow-k? | No — strictly shrink or equal (`pygv/utils/state_diagnostics.py:303`) | Phase 3 must not assume grow is possible |
