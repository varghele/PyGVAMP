# PyGVAMP Pipeline Roadmap

Tracks the current state, remaining work, and technical debt for the PyGVAMP pipeline.

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| VAMPNetDataset | Done | MD→PyG graphs with caching, continuous/non-continuous modes |
| VAMPNet Model | Done | Embedding + Encoder + Classifier |
| SchNet Encoder | Done | With attention mechanism |
| GIN Encoder | Done | Parallel attention, WL-expressive |
| ML3 Encoder | Done | Spectral convolutions, parallel attention, gaussian/spectral edge modes |
| Meta Encoder | Experimental | Works but unstable, not production-ready |
| VAMP Score | Done | VAMP1, VAMP2, VAMPE modes; validated against NumPy reference |
| SoftmaxMLP Classifier | Done | |
| Master Pipeline | Done | 3-phase orchestration with dry_run, resume, validate_only |
| Preparation Phase | Done | Dataset creation + optional state discovery |
| Training Phase | Done | Multi-lag, multi-state grid with checkpointing |
| Analysis Phase | Done | Transitions, attention maps, structures, ITS, CK tests |
| Configuration System | Done | BaseConfig + encoder configs + 16 presets (small/medium/large) |
| Visualization | Done | Static plots (2133L) + interactive HTML viewer (Three.js) |
| Tests | Done | 504+ unit tests, 7 integration tests (SchNet/GIN/ML3, .xtc/.dcd) |

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
3. Build model: Embedding → Encoder → Classifier → VAMPScore
4. Train with AdamW, VAMP-2 loss, gradient clipping, early stopping
5. Post-training: CK test + ITS analysis

### Phase 3: Analysis (`analysis.py`)
For each trained model:
1. Load best checkpoint, run inference on all frames
2. State assignment, transition matrices
3. Per-state attention map aggregation
4. Representative structure extraction (PDB)
5. Visualization: networks, heatmaps, ensembles

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
        ├── state_structures/State_N/*.pdb
        ├── attention_maps/, attention_structures/
        └── visualizations/
```

---

## Remaining Work

### Next Up
- **Multi-lag edge cases (1.4)**: Verify lag time compatibility checks, stride+lag combinations, different lag times producing different models
- **Gradient flow & numerical stability (2.2)**: Document gradient workarounds, add monitoring/logging, investigate NaN root causes, auto learning rate adjustment
- **CI/CD (2.3)**: Set up automated testing pipeline

### Future
- **Documentation (4.1)**: Complete Sphinx docs, tutorial notebooks, example configs, hyperparameter guidelines
- **Cross-validation & ensemble (3.4)**: Model selection and ensemble support

---

## Technical Debt

### Medium Priority

| Issue | Location | Description |
|-------|----------|-------------|
| One-hot node features | `vampnet_dataset.py` | N×N matrix for N atoms; learned embeddings exist but had issues |
| NaN masking | `vampnet.py` | NaN outputs replaced with zeros instead of fixing root cause |
| Batch size sensitivity | `vamp_score_v0.py` | VAMP score varies with batch composition |
| BatchNorm single-sample | training pipeline | `drop_last=True` not set in DataLoader; last batch can have 1 sample |

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
- CLI: --dry_run, --resume, --validate_only, --skip_preparation

### Code Quality
- Hardcoded epsilon values moved to BaseConfig (`vamp_epsilon`, `training_jitter`, `edge_norm_eps`)
- Repository cleanup (legacy code, dead functions, build artifacts, duplicate datasets)
- 504+ unit tests, 7 integration tests
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

## Open Questions

1. **Memory constraints**: For large trajectories, should the pipeline support streaming/chunked processing?
2. **Validation strategy**: Should CK/ITS run during training (current) or only in analysis phase?

---

*Last updated: 2026-04-07*
