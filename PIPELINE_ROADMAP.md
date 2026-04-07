# PyGVAMP Pipeline Roadmap

Tracks the current state, remaining work, and technical debt for the PyGVAMP pipeline.

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| VAMPNetDataset | Done | MD→PyG graphs with caching, continuous/non-continuous modes |
| VAMPNet Model | Done | Embedding + Encoder + Classifier |
| RevVAMPNet Model | Done | Reversible variant with likelihood-based loss and detailed balance |
| SchNet Encoder | Done | With attention mechanism |
| GIN Encoder | Done | Parallel attention, WL-expressive |
| ML3 Encoder | Done | Spectral convolutions, parallel attention, gaussian/spectral edge modes |
| Meta Encoder | Experimental | Works but unstable, not production-ready |
| VAMP Score | Done | VAMP1, VAMP2, VAMPE modes; validated against NumPy reference |
| Reversible Score | Done | Learnable stationary distribution + symmetric rate matrix → detailed balance |
| SoftmaxMLP Classifier | Done | |
| Master Pipeline | Done | 3-phase orchestration with dry_run, resume, validate_only |
| Preparation Phase | Done | Dataset creation + optional state discovery |
| Training Phase | Done | Multi-lag, multi-state grid with checkpointing; standard and reversible modes |
| Analysis Phase | Done | Transitions, attention maps, structures, ITS, CK tests; RevVAMPNet detection |
| Configuration System | Done | BaseConfig + encoder configs + 16 presets (small/medium/large) |
| CLI | Done | `--reversible`, `--batch_size`, all pipeline/training args wired through |
| Visualization | Done | Static plots (2133L) + interactive HTML viewer (Three.js) |
| SLURM Templates | Done | Standard + reversible pipeline scripts, multi-run paper scripts |
| Paper Analysis | Done | Multi-run aggregation with Hungarian state matching, mean + 95% CI |
| Tests | Done | 529+ unit tests, 7 integration tests (SchNet/GIN/ML3, .xtc/.dcd) |

---

## Pipeline Execution

### Phase 0: Configuration
- Parse CLI args → load preset/model config → create experiment directory structure
- Supports `--dry_run`, `--resume`, `--validate_only`, `--skip_preparation`

### Phase 1: Preparation (`preparation.py`)
1. Validate and convert topology to PDB
2. Find trajectory files (.xtc, .dcd, .trr)
3. Process trajectories (load, select atoms, apply stride)
4. Build k-NN graphs with Gaussian-expanded edge features
5. Create time-lagged pairs (continuous or boundary-aware)
6. Optional: Graph2Vec state discovery to recommend n_states

### Phase 2: Training (`training.py`)
For each `(lag_time, n_states)` combination:
1. Create dataset and DataLoaders (80/20 train/val split)
2. Auto-infer feature dimensions from dataset
3. Build model:
   - **Standard**: Embedding → Encoder → Classifier → VAMPScore (VAMP-2 loss, higher is better)
   - **Reversible** (`--reversible`): Embedding → Encoder → Classifier → ReversibleVAMPScore (NLL loss, lower is better)
4. Train with AdamW, gradient clipping, early stopping
5. Post-training: CK test + ITS analysis

### Phase 3: Analysis (`analysis.py`)
For each trained model:
1. Load best checkpoint, run inference on all frames
2. Detect model type (VAMPNet vs RevVAMPNet)
3. State assignment, transition matrices:
   - **Standard**: Count-based transition matrix from state assignments
   - **Reversible**: Extract learned K and π from RevVAMPNet (satisfies detailed balance); also compute count-based K for comparison
4. Per-state attention map aggregation
5. Representative structure extraction (PDB)
6. Visualization: networks, heatmaps, ensembles

### Output Directory Structure
```
exp_{protein_name}_{timestamp}/
├── config.yaml
├── pipeline_summary.json
├── logs/
├── preparation/
│   └── prep_{timestamp}/
│       ├── topology.pdb, prep_config.json, dataset_stats.json
│       ├── cache/
│       └── state_discovery/    (if --discover_states)
├── training/
│   └── lag{X}ns_{Y}states/
│       ├── best_model.pt, final_model.pt, config.txt
│       ├── vamp_scores.png, models/, plots/{ck_test,its}/
│       └── state_probs.npy, embeddings.npy
└── analysis/
    └── lag{X}ns_{Y}states/
        ├── transition_matrix.npy, state_probs.npy
        ├── learned_transition_matrix.npy   (reversible only)
        ├── stationary_distribution.npy     (reversible only)
        ├── state_structures/State_N/*.pdb
        ├── attention_maps/, attention_structures/
        └── visualizations/
```

### Multi-Run Paper Experiments (`for_publication/`)
```
paper_experiments/
├── {protein}/                        # standard runs
│   └── lag{X}ns/run_{YY}/           # 10 runs per lag time
└── {protein}_rev/                    # reversible runs
    └── lag{X}ns/run_{YY}/
```
- `paper_runs_standard.sh` / `paper_runs_reversible.sh`: SLURM 2D array jobs (lag_time × run_index)
- `paper_analysis.py`: Aggregates runs with Hungarian state matching, computes mean + 95% CI (t-distribution), publication-quality plots (ITS, CK, populations)

---

## Remaining Work

### Next Up
- **Multi-lag edge cases (1.4)**: Verify lag time compatibility checks, stride+lag combinations, different lag times producing different models
- **Gradient flow & numerical stability (2.2)**: Document gradient workarounds, add monitoring/logging, investigate NaN root causes, auto learning rate adjustment
- **CI/CD (2.3)**: Set up automated testing pipeline

### Future
- **Documentation (4.1)**: Complete Sphinx docs, tutorial notebooks, example configs, hyperparameter guidelines
- **Ensemble methods (3.4)**: Model selection and ensemble support (multi-run statistical validation now available via `for_publication/`)

---

## Technical Debt

### Medium Priority

| Issue | Location | Description |
|-------|----------|-------------|
| One-hot node features | `vampnet_dataset.py` | N×N matrix for N atoms; learned embeddings exist but had issues |
| NaN masking | `vampnet.py` | NaN outputs replaced with zeros instead of fixing root cause |
| Batch size sensitivity | `vamp_score_v0.py` | VAMP score varies with batch composition |
| ~~BatchNorm single-sample~~ | ~~training pipeline~~ | ~~`drop_last=True` not set in DataLoader~~ — fixed, train/val loaders now drop incomplete last batches |

### Low Priority

| Issue | Location | Description |
|-------|----------|-------------|
| Inconsistent naming | Various | Mix of `n_states` and `n_states_list` (compatibility pattern in master_pipeline) |
| Pre-analysis duplication | `training.py` | Quick CK/ITS after training duplicates analysis phase (intentional: validation vs comprehensive) |
| Type hints | Various | Incomplete type annotations |
| Logging | Various | Print statements instead of `logging` module |
| Encoder abstraction | Various | No common `BaseEncoder` abstract class |

---

## Completed Items

<details>
<summary>Click to expand completed work</summary>

### Critical Path (Phase 1)
- End-to-end pipeline validated with automated integration tests (SchNet+XTC, GIN+DCD, ML3+XTC)
- VAMP score validated against NumPy reference (25 tests: covariances, Koopman, VAMP1/2/E)
- All high-priority tech debt fixed (CUDA hardcoding, broken imports, None encoder, device inconsistency, missing presets)

### Features
- ML3 encoder rewritten with SpectConvWithAttention, parallel attention, spectral/gaussian edge modes
- GIN encoder with parallel attention preserving WL expressiveness
- Non-continuous trajectory support with boundary tracking
- Automatic n_states selection (eigenvalue gap, population, JSD analysis)
- Interactive HTML report generation (Three.js)
- Complete preset system (small/medium/large × 4 encoder types)
- CLI: --dry_run, --resume, --validate_only, --skip_preparation, --reversible, --batch_size

### RevGraphVAMP (Reversible)
- RevVAMPNet model with likelihood-based loss and learnable transition matrix satisfying detailed balance
- ReversibleVAMPScore with off-diagonal construction (symmetric rate matrix × stationary distribution)
- Full pipeline integration: `--reversible` flag in CLI, training branching, analysis auto-detection
- SLURM templates for both standard and reversible modes
- Multi-run paper analysis tooling (`for_publication/`) with Hungarian state matching and CI aggregation

### Code Quality
- Hardcoded epsilon values moved to BaseConfig (`vamp_epsilon`, `training_jitter`, `edge_norm_eps`)
- Repository cleanup (legacy code, dead functions, build artifacts, duplicate datasets)
- 529+ unit tests, 7 integration tests
- Unused imports removed, empty files cleaned up

</details>

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Asymmetric k-NN graphs | Intentional — edges are directional |
| One-hot node features | Learned embeddings had issues; one-hot is the starting point |
| Full lag×states grid | Behavior on different timescales is important |
| Parallel attention | Additive pathway preserves WL expressiveness (paper-worthy) |
| Pre-analysis duplication | Training does quick validation; analysis does comprehensive output |
| RevVAMPNet as separate class | Keeps VAMPNet unchanged; polymorphic fit()/evaluate() API |
| Off-diagonal K construction | Row-normalization breaks detailed balance; off-diag + diag enforces it by construction |

## Open Questions

1. **Memory constraints**: For large trajectories, should the pipeline support streaming/chunked processing?
2. **Validation strategy**: Should CK/ITS run during training (current) or only in analysis phase?
3. **RevVAMPNet hyperparameters**: Optimal initialization for rate matrix weights and stationary distribution?

---

*Last updated: 2026-04-07*
