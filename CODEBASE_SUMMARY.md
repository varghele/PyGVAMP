# PyGVAMP Codebase Summary

## Project Overview

**PyGVAMP** (PyTorch Geometric VAMP) is a high-performance implementation of GraphVAMPNets for analyzing molecular dynamics (MD) trajectories. It achieves ~50x speedup compared to the original implementation.

### What It Does
1. **Input**: MD trajectory files (.xtc, .dcd) + topology (.pdb)
2. **Process**: Converts molecular structures to k-NN graphs, learns latent representations via message-passing neural networks
3. **Output**: Discrete state assignments, transition probabilities, slow collective variable identification

### Key Scientific Concepts
- **VAMP (Variational Approach for Markov Processes)**: Mathematical framework for analyzing time-series dynamics
- **VAMP Score**: Optimization objective that captures slow dynamics (higher = better)
- **Implied Timescales (ITS)**: Kinetic timescales extracted from eigenvalues
- **Chapman-Kolmogorov (CK) Tests**: Validates Markov model assumptions

---

## Directory Structure

```
PyGVAMP/
├── pygv/                        # Main package
│   ├── pipe/                    # Pipeline orchestration
│   │   ├── master_pipeline.py   # Entry point, 3-phase orchestration
│   │   ├── preparation.py       # Phase 1: Data preparation
│   │   ├── training.py          # Phase 2: Model training
│   │   ├── analysis.py          # Phase 3: Post-training analysis
│   │   └── args.py              # Pipeline CLI argument parsing
│   ├── dataset/
│   │   └── vampnet_dataset.py   # MD→PyG graphs (k-NN, Gaussian expansion, time-lagged pairs, caching)
│   ├── vampnet/
│   │   ├── vampnet.py           # VAMPNet model (Embedding → Encoder → Classifier → Softmax)
│   │   └── rev_vampnet.py       # RevVAMPNet — reversible likelihood-based training with detailed balance
│   ├── encoder/                 # Message-passing encoders
│   │   ├── schnet.py            # SchNet (default) — continuous-filter convolutions
│   │   ├── gin.py               # GIN — parallel attention preserving WL expressiveness
│   │   ├── ml3.py               # ML3 — spectral convolutions with 3-WL expressivity
│   │   ├── meta.py              # Meta (experimental)
│   │   ├── meta_att.py          # Meta with attention (experimental)
│   │   └── gat.py               # GAT-based (experimental)
│   ├── classifier/
│   │   └── SoftmaxMLP.py        # State classification (embedding → probabilities)
│   ├── scores/
│   │   ├── vamp_score_v0.py     # VAMP1/VAMP2/VAMPE/VAMPCE loss functions
│   │   └── reversible_score.py  # ReversibleVAMPScore — likelihood loss with detailed balance
│   ├── config/
│   │   ├── base_config.py       # BaseConfig dataclass (all shared parameters)
│   │   ├── model_configs/       # SchNetConfig, GINConfig, ML3Config, MetaConfig
│   │   └── presets/             # Small/Medium/Large presets for all encoder types
│   ├── args/                    # CLI argument definitions
│   ├── clustering/              # Graph2Vec + state discovery
│   ├── utils/                   # Utilities (plotting, analysis, CK, ITS, state diagnostics)
│   └── visualization/           # Interactive HTML visualizer (Three.js)
│
├── tests/                       # 504+ unit tests, 7 integration tests
├── cluster_scripts/             # SLURM job scripts
├── run_pipeline.py              # Entry point
└── PIPELINE_ROADMAP.md          # Roadmap, status, remaining work
```

---

## Core Components

### 1. VAMPNetDataset (`pygv/dataset/vampnet_dataset.py`)
Converts MD trajectories to PyTorch Geometric graphs:
- Loads frames with MDTraj, applies atom selection and stride
- Builds asymmetric k-NN graphs per frame
- Gaussian expansion of edge distances
- Creates time-lagged pairs (t=0, t=lag) with continuous/non-continuous modes
- Hash-based caching for reuse

### 2. VAMPNet Model (`pygv/vampnet/vampnet.py`)
```
Input Graph → [Embedding MLP] → Encoder (SchNet/GIN/ML3) → Classifier (SoftmaxMLP) → State Probabilities
```

### 2b. RevVAMPNet Model (`pygv/vampnet/rev_vampnet.py`)
Same architecture as VAMPNet but trained with likelihood-based loss and a learned
transition matrix satisfying detailed balance. Enabled via `--reversible` flag.

### 3. Encoders
| Encoder | File | Status | Key Feature |
|---------|------|--------|-------------|
| SchNet | `schnet.py` | Production | Continuous-filter convolutions with attention |
| GIN | `gin.py` | Production | Parallel attention preserving 1-WL expressiveness |
| ML3 | `ml3.py` | Production | Spectral convolutions, 3-WL expressivity, gaussian/spectral edge modes |
| Meta | `meta.py` | Experimental | Graph meta-learning |

### 4. VAMP Score (`pygv/scores/vamp_score_v0.py`)
```
C₀₀ = E[χ(t)ᵀ χ(t)]           # instantaneous covariance
C₀ₜ = E[χ(t)ᵀ χ(t+τ)]         # cross-covariance
VAMP-2 = ‖C₀₀^{-1/2} C₀ₜ Cₜₜ^{-1/2}‖²_F + 1
```

### 4b. Reversible Score (`pygv/scores/reversible_score.py`)
Learns stationary distribution u and symmetric rate matrix S. Constructs
transition matrix K satisfying detailed balance. Loss = negative log-likelihood
of observed transitions under K.

### 5. Pipeline Phases
1. **Preparation** (`preparation.py`): Load trajectories → Convert to graphs → Cache. Writes `dataset_stats.json` including the raw trajectory `frame_dt_ps` (consumed by auto-stride).
2. **Training** (`training.py`): Grid search over lag times × n_states → Train → Save best. Accepts an optional ``pre_built_model`` for warm-start retrains.
3. **Analysis** (`analysis.py`): Inference → States → Transitions → Attention → Structures → Plots
4. **Automatic retrain loop** (`master_pipeline._run_retrain_loop`): Phase 3b, optional. Triggered by the diagnostic's `recommendation="retrain"`; terminates on same-k convergence, exhaustion (`max_retrains`), or `"keep"`.

### 6. Auto-Stride (`master_pipeline._compute_auto_stride`)
Per-lag-time runtime subsampling. Formula: `stride = max(1, floor(τ / (10 · cache_frame_dt)))`, applied on top of the preprocessing-level stride without re-caching. Opt-in via `--auto_stride`; requires `--timestep` when no prepared dataset with persisted `frame_dt_ps` exists. The dataset's runtime subsample lives at `VAMPNetDataset.runtime_stride` (lines 140-147 in `vampnet_dataset.py`, applied after the time-lagged pair construction).

### 7. Warm-Start Retrains
`VAMPNet.warm_restart_with_new_k(new_k)` and `RevVAMPNet.warm_restart_with_new_k(new_k)` swap only the classifier head (and, for `RevVAMPNet`, the reversible score module — `log_stationary` and `rate_matrix_weights` are reinitialised because both depend on `n_states`; `epsilon` is preserved). Encoder + embedding weights and BatchNorm running statistics are kept in place. The caller must recreate the optimizer (and any LR scheduler) after the restart because parameter references have changed. Opt-in via `--warm_start_retrains` (default after Phase 4: ON). `SoftmaxMLP` exposes its construction hyperparameters as attributes so the restart can rebuild a same-shape, different-k classifier without external plumbing.

---

## Configuration

All configuration lives in `pygv/config/base_config.py` (shared params) and `pygv/config/model_configs/` (encoder-specific). Key groups:

- **Data**: `traj_dir`, `top`, `file_pattern`, `selection`, `stride`, `lag_time`, `continuous`
- **Graph**: `n_neighbors`, `gaussian_expansion_dim`
- **Model**: `encoder_type`, `n_states`
- **Training**: `epochs`, `batch_size`, `lr`, `weight_decay`, `clip_grad`, `lr_schedule`, `lr_min`
- **Numerical stability**: `vamp_epsilon`, `training_jitter`, `edge_norm_eps`
- **Embedding/Classifier**: `use_embedding`, `clf_hidden_dim`, `clf_num_layers`, etc.
- **Multi-lag policy**: `auto_stride`, `warm_start_retrains`, `max_retrains`, `convergence_check`

---

## Entry Points

```bash
# Full pipeline (recommended)
python run_pipeline.py --traj_dir /path/to/trajectories --top /path/to/topology.pdb \
    --preset medium_schnet --lag_times 10 20 50 --n_states 3 5 7

# Dry run (preview config without executing)
python run_pipeline.py ... --dry_run

# Resume from existing experiment
python run_pipeline.py ... --resume /path/to/experiment

# Validate configuration only
python run_pipeline.py ... --validate_only
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Asymmetric k-NN graphs | Intentional — edges are directional |
| One-hot node features | Learned embeddings had issues; one-hot is the starting point |
| Full lag_time × n_states grid | Behavior on different timescales matters |
| Parallel attention (GIN/ML3) | Additive pathway preserves WL expressiveness while adding learned re-weighting |

---

*Last updated: 2026-04-21*
