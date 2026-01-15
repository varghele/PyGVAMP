# PyGVAMP Pipeline Roadmap

This document outlines the current state of the PyGVAMP pipeline, steps needed to complete it, and technical debt that should be addressed.

---

## Pipeline Overview

PyGVAMP is a refactored implementation of GraphVAMPNets for analyzing molecular dynamics (MD) trajectories using the Variational Approach for Markov Processes (VAMP). The project achieves ~50x speedup compared to the original implementation through PyTorch Geometric architecture.

### What VAMP/VAMPNets Does

1. **Input**: MD trajectory files (.xtc, .dcd) + topology (.pdb)
2. **Process**: Converts molecular structures to k-NN graphs, learns latent representations via message-passing neural networks
3. **Output**: Discrete state assignments, transition probabilities, and slow collective variable identification

---

## Pipeline Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PyGVAMP MASTER PIPELINE                            │
│                           (master_pipeline.py)                                  │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 │
┌─────────────────────────────────────┐               │
│   CONFIGURATION LOADING             │               │
│   ─────────────────────────         │               │
│   • Presets (YAML/JSON)             │               │
│   • CLI argument overrides          │               │
│   • BaseConfig / SchNetConfig       │               │
└─────────────────┬───────────────────┘               │
                  │                                   │
                  ▼                                   │
┌═══════════════════════════════════════════════════════════════════════════════┐
║                          PHASE 1: PREPARATION                                  ║
║                         (preparation.py)                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐ ║
║  │ Trajectory Files │    │  Topology File   │    │    Configuration         │ ║
║  │   (.xtc/.dcd)    │    │     (.pdb)       │    │  (selection, stride,     │ ║
║  └────────┬─────────┘    └────────┬─────────┘    │   lag_time, n_neighbors) │ ║
║           │                       │              └────────────┬─────────────┘ ║
║           └───────────┬───────────┘                           │               ║
║                       ▼                                       │               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                    VAMPNetDataset                                       │  ║
║  │   ┌─────────────────────────────────────────────────────────────────┐   │  ║
║  │   │ 1. Load trajectory frames (MDTraj)                              │   │  ║
║  │   │ 2. Select atoms (e.g., "name CA" for alpha carbons)             │   │  ║
║  │   │ 3. Apply stride (every Nth frame)                               │   │  ║
║  │   │ 4. Calculate pairwise distances                                 │   │  ║
║  │   │ 5. Build k-NN graphs for each frame                            │   │  ║
║  │   │ 6. Gaussian expansion of edge distances                        │   │  ║
║  │   │ 7. Create time-lagged pairs (t=0, t=lag)                       │   │  ║
║  │   └─────────────────────────────────────────────────────────────────┘   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                       │                                                        ║
║                       ▼                                                        ║
║           ┌───────────────────────┐                                            ║
║           │   Cached Dataset      │ ──► Hash-based caching for reuse           ║
║           │   (PyG Data objects)  │                                            ║
║           └───────────────────────┘                                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
                                     ▼
┌═══════════════════════════════════════════════════════════════════════════════┐
║                          PHASE 2: TRAINING                                     ║
║                         (training.py)                                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                    GRID SEARCH OVER HYPERPARAMETERS                     │  ║
║  │                                                                         │  ║
║  │    for lag_time in [10, 20, 50, 100] ns:                               │  ║
║  │        for n_states in [3, 5, 7, 10]:                                  │  ║
║  │            → Create experiment directory                               │  ║
║  │            → Train model                                               │  ║
║  │            → Save best checkpoint                                      │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         VAMPNet MODEL                                   │  ║
║  │                                                                         │  ║
║  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐  │  ║
║  │  │  Embedding  │ → │   Encoder   │ → │ Classifier  │ → │  Softmax   │  │  ║
║  │  │   (MLP)     │   │  (SchNet/   │   │  (MLP)      │   │  Output    │  │  ║
║  │  │  optional   │   │   Meta)     │   │             │   │            │  │  ║
║  │  └─────────────┘   └─────────────┘   └─────────────┘   └────────────┘  │  ║
║  │                                                                         │  ║
║  │  Input: PyG Graph (node_feat, edge_index, edge_attr)                   │  ║
║  │  Output: State probabilities [batch_size, n_states]                    │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                       TRAINING LOOP                                     │  ║
║  │                                                                         │  ║
║  │   for epoch in range(n_epochs):                                        │  ║
║  │       for (x_t0, x_t1) in train_loader:   # time-lagged pairs          │  ║
║  │           χ_t0 = model(x_t0)              # state probs at t=0         │  ║
║  │           χ_t1 = model(x_t1)              # state probs at t=lag       │  ║
║  │           loss = -VAMP_score(χ_t0, χ_t1)  # maximize VAMP score        │  ║
║  │           loss.backward()                                              │  ║
║  │           optimizer.step()                                             │  ║
║  │                                                                         │  ║
║  │       validate(test_loader)                                            │  ║
║  │       early_stopping_check()                                           │  ║
║  │       save_checkpoint_if_best()                                        │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                       VAMP SCORE CALCULATION                            │  ║
║  │                                                                         │  ║
║  │   C₀₀ = E[χ(t)ᵀ χ(t)]           # instantaneous covariance             │  ║
║  │   C₀ₜ = E[χ(t)ᵀ χ(t+τ)]         # cross-covariance                     │  ║
║  │   Cₜₜ = E[χ(t+τ)ᵀ χ(t+τ)]       # lagged covariance                    │  ║
║  │                                                                         │  ║
║  │   VAMP-2 = ||C₀₀^(-1/2) C₀ₜ Cₜₜ^(-1/2)||²_F + 1                        │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║                        ┌───────────────────────┐                               ║
║                        │  Trained Models       │                               ║
║                        │  (best_model.pt)      │                               ║
║                        └───────────────────────┘                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
                                     ▼
┌═══════════════════════════════════════════════════════════════════════════════┐
║                          PHASE 3: ANALYSIS                                     ║
║                         (analysis.py)                                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                      POST-TRAINING ANALYSIS                             │  ║
║  │                                                                         │  ║
║  │  1. STATE PROBABILITIES                                                │  ║
║  │     • Run inference on all frames                                      │  ║
║  │     • Assign each frame to most probable state                         │  ║
║  │                                                                         │  ║
║  │  2. TRANSITION ANALYSIS                                                │  ║
║  │     • Calculate transition probability matrices                        │  ║
║  │     • Plot transition networks (state graph)                           │  ║
║  │                                                                         │  ║
║  │  3. ATTENTION MAPS                                                     │  ║
║  │     • Extract encoder attention weights                                │  ║
║  │     • Compute per-state residue-residue attention                      │  ║
║  │     • Identify key interactions per state                              │  ║
║  │                                                                         │  ║
║  │  4. STATE STRUCTURES                                                   │  ║
║  │     • Extract representative PDB structures per state                  │  ║
║  │     • Generate ensemble PDBs                                           │  ║
║  │     • Color by attention values                                        │  ║
║  │                                                                         │  ║
║  │  5. IMPLIED TIMESCALES (ITS)                                           │  ║
║  │     • Calculate: ITS = -τ / ln|λ|                                      │  ║
║  │     • Plot ITS vs lag time                                             │  ║
║  │     • Identify slow processes                                          │  ║
║  │                                                                         │  ║
║  │  6. CHAPMAN-KOLMOGOROV TESTS                                           │  ║
║  │     • Validate Markov assumption                                       │  ║
║  │     • Compare predicted vs observed transitions                        │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                     │                                          ║
║                                     ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         OUTPUT ARTIFACTS                                │  ║
║  │                                                                         │  ║
║  │  analysis/                                                             │  ║
║  │  ├── transition_matrix.png                                             │  ║
║  │  ├── state_network.png                                                 │  ║
║  │  ├── attention_maps/                                                   │  ║
║  │  │   ├── state_0_attention.png                                         │  ║
║  │  │   ├── state_1_attention.png                                         │  ║
║  │  │   └── ...                                                           │  ║
║  │  ├── structures/                                                       │  ║
║  │  │   ├── state_0_ensemble.pdb                                          │  ║
║  │  │   └── ...                                                           │  ║
║  │  ├── its_plot.png                                                      │  ║
║  │  └── ck_test.png                                                       │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │      pipeline_summary.json     │
                    │  ────────────────────────────  │
                    │  • Configuration used          │
                    │  • Models trained              │
                    │  • Analysis completed          │
                    │  • Timestamp                   │
                    └────────────────────────────────┘
```

---

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| VAMPNetDataset | ✅ Complete | MD→PyG graphs with caching |
| VAMPNet Model | ✅ Complete | Embedding + Encoder + Classifier |
| SchNet Encoder | ✅ Complete | With attention mechanism |
| Meta Encoder | ⚠️ Partial | Config WIP, encoder exists |
| ML3 Encoder | ⚠️ Separate | Working independently, needs pipeline integration |
| VAMP Score | ✅ Complete | VAMP1, VAMP2, VAMPE modes |
| SoftmaxMLP Classifier | ✅ Complete | |
| Master Pipeline | ✅ Complete | 3-phase orchestration |
| Preparation Phase | ✅ Complete | Dataset creation |
| Training Phase | ✅ Complete | Multi-lag, multi-state grid |
| Analysis Phase | ✅ Complete | All visualizations |
| Caching | ✅ Complete | Hash-based dataset caching |
| ITS Analysis | ✅ Complete | Implied timescales |
| CK Tests | ✅ Complete | Chapman-Kolmogorov |
| Visualization | ✅ Complete | 2133 lines in plotting.py |
| Documentation | ⚠️ Minimal | Sphinx setup exists |
| Tests | ⚠️ Incomplete | Tests exist but not in CI |

---

## Steps to Complete the Pipeline

### Phase 1: Critical Path (Required for Publication)

#### 1.1 Validate End-to-End Pipeline Execution
- [ ] Run complete pipeline on test dataset
- [ ] Verify all phases execute without errors
- [ ] Confirm all expected outputs are generated
- [ ] Test with multiple trajectory formats (.xtc, .dcd)

#### 1.2 Fix Known Issues
- [ ] Address hardcoded `model.to('cuda')` in `training.py:268` (should use device variable)
- [ ] Remove unused `from pymol.querying import distance` import in `training.py:10`
- [ ] Handle case when encoder is `None` for ML3 type in `create_model()`
- [ ] Add proper error handling when model loading fails in analysis

#### 1.3 Verify VAMP Score Calculation
- [ ] Validate covariance matrix calculations against reference implementation
- [ ] Test eigenvalue truncation/regularization strategies
- [ ] Confirm score maximization (higher = better) throughout codebase

#### 1.4 Test Multi-Lag Pipeline
- [ ] Verify lag time compatibility check with trajectory timestep
- [ ] Test stride + lag_time combination edge cases
- [ ] Validate that different lag times produce different models

### Phase 2: Robustness & Quality

#### 2.1 Error Handling
- [ ] Add input validation for trajectory files (existence, format)
- [ ] Add configuration validation (compatible parameters)
- [ ] Improve error messages with actionable guidance
- [ ] Add graceful failure modes with partial results saved

#### 2.2 Gradient Flow & Numerical Stability
- [ ] Review and document gradient workarounds (jitter, skip connections)
- [ ] Add gradient monitoring/logging during training
- [ ] Investigate and fix root causes of NaN outputs
- [ ] Add automatic learning rate adjustment on gradient explosion

#### 2.3 Testing
- [ ] Create integration tests for full pipeline
- [ ] Add unit tests for VAMP score calculation
- [ ] Add unit tests for dataset creation
- [ ] Clean up existing tests in `testdata/` and `area51_testing_grounds/`
- [ ] Set up CI/CD with automated testing

### Phase 3: Feature Completion

#### 3.1 Complete Encoder Options
- [ ] Integrate ML3 encoder into pipeline (working separately, needs integration)
- [ ] Complete Meta encoder configuration (MetaConfig)
- [ ] Add encoder selection validation

#### 3.2 Trajectory Handling
- [ ] Add `continuous` flag for trajectory loading (currently all trajectories concatenated)
- [ ] Implement non-continuous trajectory support (treat each file as separate trajectory)
- [ ] Add trajectory boundary detection and handling

#### 3.3 State Count Selection
- [ ] Implement automatic n_states selection method
- [ ] Ensure comparable state counts across different lag times
- [ ] Add state count validation/recommendation system

#### 3.4 Enhanced Analysis
- [ ] Add cross-validation for model selection
- [ ] Add ensemble model support

#### 3.5 Visualization Improvements
- [ ] Add interactive plots (Plotly/Bokeh option)
- [ ] Add comparison plots across lag times
- [ ] Add trajectory projection onto learned states
- [ ] Add video/animation generation for state transitions

### Phase 4: Documentation & Usability

#### 4.1 Documentation
- [ ] Complete Sphinx documentation
- [ ] Add tutorial notebooks (Jupyter)
- [ ] Add example configurations for common proteins
- [ ] Document hyperparameter tuning guidelines

#### 4.2 Report Generation
- [ ] Integrate HTML report generation (code exists, needs integration)
- [ ] Combine all analysis outputs into single shareable HTML file
- [ ] Add interactive elements to HTML report (collapsible sections, tooltips)

#### 4.3 CLI Improvements
- [ ] Add `--dry-run` option to preview pipeline
- [ ] Add `--resume` option to continue failed runs
- [ ] Add progress persistence for long-running jobs
- [ ] Add `--validate-only` mode for configuration checking

---

## Technical Debt

### High Priority (Blocking Issues)

| Issue | Location | Description | Suggested Fix |
|-------|----------|-------------|---------------|
| Hardcoded CUDA | `training.py:268` | `model.to('cuda')` ignores `--cpu` flag | Use `device` variable from args |
| Unused import | `training.py:10` | `from pymol.querying import distance` never used | Remove import |
| None encoder | `training.py:194` | ML3 encoder returns `None`, causes crash | Import and instantiate `GNNML3` from `pygv/encoder/ml3.py` |
| Device inconsistency | `training.py:268,276` | Model moved to CUDA before device is determined | Move `model.to(device)` after device determination |
| Broken imports | `pygv/config/__init__.py:4` | Imports `MetaConfig`, `ML3Config` which are commented out in `base_config.py` | Either uncomment configs or remove from imports |
| Missing preset files | `pygv/config/presets/` | `medium.py` and `large.py` imported but don't exist | Create these files or update imports |

### Medium Priority (Code Quality)

| Issue | Location | Description | Suggested Fix |
|-------|----------|-------------|---------------|
| Dual dataset files | `vampnet_dataset.py`, `vampnet_dataset_with_AA.py` | Two nearly identical files; AA encoding is also accessible via `use_amino_acid_encoding` flag in `_create_graph_from_frame()` | Consolidate into single file with encoding flag; remove duplicate |
| One-hot node features | `vampnet_dataset.py` | Creates N×N matrix for N atoms (memory inefficient) | Learned embeddings exist but had issues; document tradeoffs |
| Magic numbers | Various | Epsilon values, thresholds hardcoded | Move to configuration |
| NaN masking | `vampnet.py` | NaN outputs replaced with zeros | Fix root cause of NaN generation |
| Commented code | `analysis.py:266-272` | TODO comment with dead code | Remove or implement |
| Batch size sensitivity | `vamp_score_v0.py` | VAMP score varies with batch composition | Document limitation or implement batch-invariant version |

### Low Priority (Improvements)

| Issue | Location | Description | Suggested Fix |
|-------|----------|-------------|---------------|
| WIP configs | `base_config.py:135-169` | MetaConfig and ML3Config commented out | Complete implementations |
| Empty viz module | `viz/` | Directory exists but is empty | Either populate or remove |
| Inconsistent naming | Various | Mix of `n_states` and `n_states_list` | Standardize parameter names |
| Redundant analysis | `training.py` | Pre-analysis duplicates analysis phase | Consider consolidating (see details below) |
| Type hints | Various | Incomplete type annotations | Add comprehensive type hints |

### Pre-Analysis Duplication Details

**Location of duplication:**

In `training.py:373-438` (after training completes):
```python
# Line 375-382: analyze_vampnet_outputs()
# Line 418-426: run_ck_analysis()
# Line 430-437: analyze_implied_timescales()
```

In `analysis.py:212-326` (Analysis phase):
```python
# Line 212-218: analyze_vampnet_outputs()
# Line 223-228: plot_transition_probabilities()
# Plus all attention/structure analysis (lines 230-326)
```

**Decision needed:** Should training.py do quick validation (CK/ITS only) while analysis.py does comprehensive output? Or consolidate all analysis to one location?

### Planned Features (Not Yet Implemented)

| Feature | Description | Priority |
|---------|-------------|----------|
| Non-continuous trajectories | Add `continuous` flag to handle trajectories that aren't continuous. Currently all MD files are concatenated as one continuous trajectory. When disabled, skip time-lagged pairs that would cross trajectory boundaries. | High |
| Automatic n_states selection | Find correct/comparable number of states for each timescale | High |
| Comparable state counts | Ensure consistent state definitions across different lag times | High |
| Dataset encoding flag | Clean up dual dataset system (one-hot vs amino acid encoding). Add flag to select encoding type instead of having separate datasets/methods. See `vampnet_dataset_with_AA.py` and `_create_graph_from_frame()` `use_amino_acid_encoding` parameter. | High |
| Complete preset system | Add missing preset files (`medium.py`, `large.py`) and uncomment MetaConfig/ML3Config in `base_config.py`. Currently `__init__.py` imports them but they're commented out (will cause ImportError). | Medium |
| ML3 pipeline integration | Integrate working ML3 encoder (`pygv/encoder/ml3.py` - GNNML3 class) into training pipeline. Currently `training.py:194` returns `None` for ML3. | Medium |
| HTML report generation | Combine all analysis outputs into single HTML file for sharing | Medium |
| Unit test cleanup | Existing tests need cleanup and CI integration | Medium |

### Architecture Considerations

| Issue | Description | Suggested Approach |
|-------|-------------|-------------------|
| Encoder abstraction | No common base class for encoders | Create `BaseEncoder` abstract class |
| Config validation | No schema validation for configs | Add Pydantic or dataclass validation |
| Logging | Print statements instead of proper logging | Replace with `logging` module |
| Progress tracking | Manual print statements | Use unified progress tracking |

---

## Recommended Priority Order

1. **Week 1**: Fix high-priority technical debt (especially CUDA hardcoding)
2. **Week 2**: Run and validate end-to-end pipeline on test data
3. **Week 3**: Add critical error handling and input validation
4. **Week 4**: Complete ML3/Meta encoder implementations
5. **Ongoing**: Add tests incrementally as issues are fixed

---

## Design Decisions (Resolved)

These questions have been clarified:

| Question | Decision |
|----------|----------|
| **Graph Bidirectionality** | Asymmetric k-NN graphs are **intentional** - edges should NOT automatically be bidirectional |
| **Node Features** | One-hot encoding is the starting point; learned embeddings had issues previously |
| **Multiple Lag Times** | Always train the **full grid** - behavior on different timescales is important |
| **Encoder Priority** | **ML3** has priority (already working independently, needs pipeline integration) |

## Open Questions

1. **Validation Strategy**: Should CK test and ITS analysis run during training (current) or only in analysis phase? (See Pre-Analysis Duplication Details above)

2. **Memory Constraints**: For large trajectories, should the pipeline support streaming/chunked processing?

### n_states Selection Methods (Suggestions to Investigate)

Since graph2vec didn't work well for determining optimal state count, here are alternative approaches to consider:

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **VAMP-2 Score Comparison** | Train models with different n_states, compare validation VAMP-2 scores. Optimal n_states maximizes score without overfitting. | Direct optimization target; theoretically grounded | Requires training multiple models |
| **Eigenvalue Gap Analysis** | Analyze VAMP eigenvalues: large gap after k eigenvalues suggests k slow processes (states) | Fast; no retraining needed; MSM-standard approach | May not align with interpretable states |
| **ITS Plateau Count** | Count number of implied timescales that plateau (converge) across lag times | Physically meaningful; validated approach | Requires multiple lag time runs |
| **Silhouette Score** | Cluster state assignments, compute silhouette score for different k values | Measures cluster quality | May not capture kinetic relevance |
| **Cross-Validation** | Hold out trajectories, measure state assignment consistency | Tests generalization | Computationally expensive |
| **Bayesian Model Selection** | Use BIC/AIC to penalize model complexity | Automatic complexity penalty | Requires likelihood formulation |

**Recommended approach**: Start with **eigenvalue gap analysis** (fast, standard in MSM literature) combined with **ITS plateau counting** (physically interpretable). Use VAMP-2 score comparison for final validation.

---

## File Reference

Key files for understanding the pipeline:

| File | Lines | Purpose |
|------|-------|---------|
| `pygv/pipe/master_pipeline.py` | 431 | Pipeline orchestration |
| `pygv/pipe/preparation.py` | 197 | Data preparation |
| `pygv/pipe/training.py` | 449 | Model training |
| `pygv/pipe/analysis.py` | 352 | Post-training analysis |
| `pygv/dataset/vampnet_dataset.py` | 693 | MD trajectory → PyG graphs (main) |
| `pygv/dataset/vampnet_dataset_with_AA.py` | 759 | MD trajectory → PyG graphs (with AA encoding - duplicate) |
| `pygv/vampnet/vampnet.py` | 1184 | VAMPNet model |
| `pygv/scores/vamp_score_v0.py` | 341 | VAMP loss calculation |
| `pygv/encoder/schnet_wo_embed_v2.py` | 300+ | SchNet encoder |
| `pygv/encoder/ml3.py` | 414 | ML3/GNNML3 encoder (working, needs integration) |
| `pygv/encoder/meta_att.py` | - | Meta encoder |
| `pygv/config/base_config.py` | 169 | Configuration classes |
| `pygv/config/__init__.py` | 99 | Config registry and presets |
| `pygv/config/presets/small.py` | 45 | Small preset configurations |
| `pygv/utils/plotting.py` | 2133 | Visualization utilities |
| `run_pipeline.py` | 12 | Entry point |

---

*Generated: 2025-12-31*
