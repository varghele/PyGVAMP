# PyGVAMP Codebase Summary & Claude Context Primer

## Quick Reference Prompt for Claude

Copy and paste this prompt at the start of a new Claude session:

---

**START PROMPT:**

```
I'm working on PyGVAMP, a PyTorch Geometric implementation of GraphVAMPNets for molecular dynamics analysis. Please read these files to understand the codebase:

1. /home/iwe81/PycharmProjects/PyGVAMP/CODEBASE_SUMMARY.md (this file)
2. /home/iwe81/PycharmProjects/PyGVAMP/PIPELINE_ROADMAP.md (technical debt & roadmap)

Key context:
- PyGVAMP converts MD trajectories (.xtc/.dcd + .pdb topology) into k-NN graphs
- Trains VAMPNet models to learn slow collective variables via VAMP score optimization
- 3-phase pipeline: Preparation → Training → Analysis
- Main package: pygv/ (production code)
- Visualization: pygviz/ (new interactive HTML visualizer)
- Testing areas: area51_testing_grounds/, area52/, testdata/

Current state (as of last session):
- Core pipeline is functional (training, analysis work)
- Known issues: hardcoded CUDA in training.py:268, ML3 encoder not integrated
- Technical debt documented in PIPELINE_ROADMAP.md

[Add your specific task here]
```

**END PROMPT**

---

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
├── pygv/                     # MAIN PACKAGE - Production pipeline
│   ├── pipe/                 # Pipeline orchestration
│   │   ├── master_pipeline.py    # Entry point (431 lines)
│   │   ├── preparation.py        # Phase 1: Data prep
│   │   ├── training.py           # Phase 2: Model training
│   │   └── analysis.py           # Phase 3: Post-analysis
│   ├── dataset/              # Data handling
│   │   ├── vampnet_dataset.py    # Main: MD→PyG graphs (692 lines)
│   │   └── vampnet_dataset_with_AA.py  # Amino acid variant (duplicate)
│   ├── vampnet/              # Model architecture
│   │   └── vampnet.py            # VAMPNet model (1184 lines)
│   ├── encoder/              # Message-passing encoders
│   │   ├── schnet_wo_embed_v2.py # SchNet (DEFAULT)
│   │   ├── ml3.py                # ML3 (working, needs integration)
│   │   ├── meta_att.py           # Meta with attention
│   │   └── gat.py                # GAT-based
│   ├── classifier/           # State classification
│   │   └── SoftmaxMLP.py
│   ├── scores/               # VAMP loss calculation
│   │   └── vamp_score_v0.py      # VAMP1/2/E/CE (341 lines)
│   ├── config/               # Configuration management
│   │   ├── base_config.py        # BaseConfig dataclass
│   │   └── presets/small.py
│   ├── args/                 # CLI argument parsing
│   └── utils/                # Utilities
│       ├── plotting.py           # Visualization (2133 lines!)
│       ├── analysis.py           # Analysis utils (1048 lines)
│       ├── ck.py                 # Chapman-Kolmogorov tests
│       └── its.py                # Implied timescales
│
├── pygviz/                   # NEW - Interactive visualization
│   └── md_visualizer/        # Web-based 3D visualizer
│       ├── visualizer.py         # MDTrajectoryVisualizer class
│       ├── data_handler.py       # Data validation
│       └── templates/            # HTML/JS/CSS assets
│
├── area51_testing_grounds/   # Graph2vec experiments
├── area52/                   # User-friendly training scripts
│   ├── train.py                  # Simple training entry
│   └── anly.py                   # Simple analysis entry
├── cluster_scripts/          # SLURM cluster scripts
├── testdata/                 # Test/debug scripts (50+ files)
│
├── run_pipeline.py           # Main entry point
├── PIPELINE_ROADMAP.md       # Technical debt & roadmap
├── README.md                 # Installation & usage
└── requirements.txt          # Dependencies
```

---

## Core Components

### 1. VAMPNetDataset (`pygv/dataset/vampnet_dataset.py`)
Converts MD trajectories to PyTorch Geometric graphs:
- Loads frames with MDTraj
- Builds k-NN graphs for each frame
- Gaussian expansion of edge distances
- Creates time-lagged pairs (t=0, t=lag)
- Hash-based caching for reuse

### 2. VAMPNet Model (`pygv/vampnet/vampnet.py`)
Neural network architecture:
```
Input Graph → [Embedding MLP] → Encoder (SchNet/Meta/ML3) → Classifier → Softmax → State Probabilities
```

### 3. VAMP Score (`pygv/scores/vamp_score_v0.py`)
Loss function that maximizes VAMP-2 score:
```
C₀₀ = E[χ(t)ᵀ χ(t)]           # instantaneous covariance
C₀ₜ = E[χ(t)ᵀ χ(t+τ)]         # cross-covariance
VAMP-2 = ||C₀₀^(-1/2) C₀ₜ Cₜₜ^(-1/2)||²_F + 1
```

### 4. Pipeline Phases
1. **Preparation** (`preparation.py`): Load trajectories → Convert to graphs → Cache
2. **Training** (`training.py`): Grid search over lag times × n_states → Train → Save best
3. **Analysis** (`analysis.py`): Inference → States → Transitions → Attention → Plots

---

## Configuration (BaseConfig)

Key parameters in `pygv/config/base_config.py`:

```python
# Data
traj_dir, top, file_pattern     # Input files
selection, stride, lag_time     # Processing

# Graph
n_neighbors, gaussian_expansion_dim

# Model
encoder_type: "schnet" | "meta" | "ml3"
n_states: int                   # Number of output states

# Training
epochs, batch_size, lr, weight_decay
```

---

## Entry Points

```bash
# Full pipeline
python run_pipeline.py --protein_name ATR --traj_dir ~/data --top ~/data/prot.pdb --lag_time 20 --n_states 5

# Training only (modify create_test_args() first)
python area52/train.py

# Analysis only (set base_output_dir first)
python area52/anly.py

# SLURM cluster
sbatch cluster_scripts/atr.sh
```

---

## Known Issues (High Priority)

| Issue | Location | Description |
|-------|----------|-------------|
| Hardcoded CUDA | `training.py:268` | Uses `model.to('cuda')` instead of device variable |
| Unused import | `training.py:10` | `from pymol.querying import distance` never used |
| ML3 not integrated | `training.py:194` | Returns `None` for ML3 encoder |
| Broken imports | `config/__init__.py:4` | Imports MetaConfig/ML3Config which are commented out |
| Missing presets | `config/presets/` | `medium.py`, `large.py` don't exist |

---

## Planned Features

1. **Non-continuous trajectories**: Handle trajectory boundaries correctly
2. **Automatic n_states selection**: Via eigenvalue gap or ITS plateau counting
3. **ML3 encoder integration**: Working code exists in `pygv/encoder/ml3.py`
4. **HTML report generation**: Interactive analysis reports
5. **Complete Meta/ML3 configs**: Currently commented out

---

## File Line Counts (Key Files)

| File | Lines | Purpose |
|------|-------|---------|
| `vampnet.py` | 1184 | Full VAMPNet model |
| `vampnet_dataset.py` | 692 | Dataset creation |
| `plotting.py` | 2133 | All visualizations |
| `analysis.py` (utils) | 1048 | Analysis utilities |
| `master_pipeline.py` | 431 | Pipeline orchestration |
| `training.py` | 449 | Training loop |
| `vamp_score_v0.py` | 341 | VAMP loss |

---

## Design Decisions (Resolved)

| Question | Decision |
|----------|----------|
| Graph Bidirectionality | Asymmetric k-NN graphs are **intentional** |
| Node Features | One-hot encoding (learned embeddings had issues) |
| Multiple Lag Times | Always train **full grid** |
| Encoder Priority | **ML3** has priority (needs integration) |

---

## Development Workflow

1. **Fix bugs**: Address PIPELINE_ROADMAP.md high-priority issues first
2. **Test**: Run `python run_pipeline.py` with test data to verify
3. **Develop**: Add features incrementally, test each
4. **Document**: Update this file and PIPELINE_ROADMAP.md as changes are made

---

*Last updated: 2026-01-15*
