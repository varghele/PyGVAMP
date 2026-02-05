# PyGVAMP Pipeline Execution Guide

This document explains step-by-step what happens when you run the PyGVAMP pipeline.

---

## Overview

The pipeline has **3 main phases** plus initialization:

```
Phase 0: Configuration & Setup
    ↓
Phase 1: Data Preparation (trajectories → graphs)
    ↓
Phase 2: Training (for each lag_time × n_states combination)
    ↓
Phase 3: Analysis (visualization & structure extraction)
```

---

## Phase 0: Initialization & Configuration

### Entry Point
**File:** `pygv/pipe/master_pipeline.py`

### What Happens

1. **Parse command-line arguments** (`pygv/pipe/args.py`)
   - Required: `--traj_dir`, `--top`
   - Optional: `--preset`, `--model`, `--lag_times`, `--n_states`, etc.

2. **Load configuration** (in order of precedence):
   - Preset config (if `--preset` specified) → `pygv/config/presets/`
   - Model config (if `--model` specified) → `pygv/config/model_configs/`
   - Base config defaults → `pygv/config/base_config.py`

3. **Create experiment directory structure**:
   ```
   exp_{protein_name}_{timestamp}/
   ├── config.yaml
   ├── preparation/
   ├── training/
   ├── analysis/
   ├── cache/
   └── logs/
   ```

### Key Parameters Set Here

| Parameter | Default | Description |
|-----------|---------|-------------|
| `traj_dir` | (required) | Directory with trajectory files |
| `top` | (required) | Topology file (.pdb, .mae) |
| `selection` | "name CA" | Atom selection (MDTraj syntax) |
| `lag_times` | [20.0] | Lag times in nanoseconds |
| `n_states` | [5] | Number of Markov states |
| `stride` | 10 | Frame skip interval |
| `n_neighbors` | 4 | k-NN graph neighbors |
| `encoder_type` | "schnet" | Encoder architecture |
| `discover_states` | False | Run Graph2Vec + clustering |
| `g2v_embedding_dim` | 64 | Graph2Vec embedding dimension |
| `g2v_max_degree` | 2 | Graph2Vec WL iteration depth |
| `g2v_epochs` | 50 | Graph2Vec training epochs |
| `min_states` | 2 | Min clusters to test |
| `max_states` | 15 | Max clusters to test |

---

## Phase 1: Data Preparation

### Entry Point
**File:** `pygv/pipe/preparation.py`

### Step-by-Step Process

#### Step 1.1: Convert and Save Topology as PDB
```
save_topology_as_pdb(topology_file, selection, output_path)
```
- Load topology file (supports .pdb, .mae, .gro, .psf, etc.)
- Apply atom selection (e.g., "name CA")
- Save selected atoms as standardized PDB file
- **Output:** `topology.pdb` in preparation directory

**Why this matters:**
- Ensures topology is valid and loadable
- Creates standardized format for visualization tools
- Preserves only the atoms used in training

#### Step 1.2: Find Trajectory Files
```
find_trajectory_files(traj_dir, file_pattern="*.xtc", recursive=True)
```
- Searches `traj_dir` for trajectory files
- Supports: `.xtc`, `.dcd`, `.trr`, etc.
- Returns sorted list of file paths

#### Step 1.3: Process Trajectories
```
VAMPNetDataset._process_trajectories()
```
For each trajectory file:
1. Load with MDTraj
2. Apply atom selection
3. Apply stride (keep every Nth frame)
4. Extract coordinates
5. Track trajectory boundaries (for non-continuous mode)

**Output:** `self.frames` - numpy array of shape `(n_frames, n_atoms, 3)`

#### Step 1.4: Determine Distance Range
```
VAMPNetDataset._determine_distance_range()
```
- Sample random frames
- Calculate pairwise atom distances
- Find min/max distances for Gaussian expansion normalization

#### Step 1.5: Create Time-Lagged Pairs
```
VAMPNetDataset._create_time_lagged_pairs()
```
- Calculate `lag_frames = int(lag_time_ns / timestep_ns)`
- **Continuous mode** (default): pairs can cross trajectory boundaries
- **Non-continuous mode**: pairs only within same trajectory

**Output:** `(t0_indices, t1_indices)` - pairs of frame indices

#### Step 1.6: Save Dataset Statistics
```
save_config(args, paths) → prep_config.json
create_and_analyze_dataset() → dataset_stats.json
```

#### Step 1.7: State Discovery (Optional)

If `--discover_states` is enabled:

```
run_state_discovery(dataset, args, paths)
```

Uses Graph2Vec + clustering to recommend optimal `n_states`:
1. Train Graph2Vec on all frames to get graph embeddings
2. Run KMeans clustering for k = min_states to max_states
3. Calculate silhouette scores and inertia (elbow method)
4. Recommend k with highest silhouette score
5. Generate visualizations and save results

**Why this matters:**
- Helps determine the natural number of metastable states
- Provides visual verification of clustering quality
- Avoids guesswork when choosing `n_states` for training

### Outputs from Phase 1

| File | Contents |
|------|----------|
| `topology.pdb` | Topology converted to PDB (selected atoms only) |
| `prep_config.json` | All preparation parameters |
| `dataset_stats.json` | Dataset statistics (n_samples, n_atoms, etc.) |
| `cache/*.pkl` | Cached dataset (if caching enabled) |
| `state_discovery/` | State discovery results (if `--discover_states`) |

#### State Discovery Outputs (if enabled)

| File | Contents |
|------|----------|
| `state_discovery/embeddings.npy` | Graph2Vec embeddings (n_frames × embedding_dim) |
| `state_discovery/cluster_labels.npy` | Cluster labels for recommended k |
| `state_discovery/elbow_plot.png` | Inertia vs k with elbow detection |
| `state_discovery/silhouette_plot.png` | Silhouette score vs k |
| `state_discovery/tsne_embeddings.png` | 2D t-SNE visualization |
| `state_discovery/umap_embeddings.png` | 2D UMAP visualization (if umap-learn installed) |
| `state_discovery/cluster_sizes.png` | Cluster population distribution |
| `state_discovery/discovery_summary.json` | Metrics and recommendation |

### Graph Structure Created

Each frame becomes a graph with:
- **Nodes:** One per selected atom
- **Edges:** k-nearest neighbors (asymmetric)
- **Node features:** One-hot encoding or amino acid properties
- **Edge features:** Gaussian-expanded distances

---

## Phase 2: Training

### Entry Point
**File:** `pygv/pipe/training.py`

### Iteration Structure
```python
for lag_time in config.lag_times:
    for n_states in config.n_states_list:
        # Train one model
```

### Step-by-Step Process (per model)

#### Step 2.1: Setup Output Directory
```
training/lag{lag_time}ns_{n_states}states/
├── models/
├── plots/
├── config.txt
├── best_model.pt
└── final_model.pt
```

#### Step 2.2: Create DataLoaders
```python
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
```

#### Step 2.3: Create Model Architecture
```
create_model(args) → VAMPNet
```

**Model components:**

1. **Encoder** (graph → embedding):
   - SchNet: Continuous-filter convolutions with interactions
   - Meta: Graph meta-learning encoder
   - ML3: (not yet implemented)

2. **VAMPScore**: Calculates VAMP loss (VAMP1, VAMP2, or VAMPE)

3. **Classifier** (optional, if n_states > 0):
   - SoftmaxMLP: embedding → state probabilities

**Assembly:**
```python
model = VAMPNet(
    encoder=encoder,
    vamp_score=vamp_score,
    classifier_module=classifier,
    lag_time=lag_time
)
```

#### Step 2.4: Training Loop
```python
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

for epoch in range(n_epochs):
    for (batch_t0, batch_t1) in train_loader:
        # Forward pass
        out_t0 = model(batch_t0)  # state probabilities
        out_t1 = model(batch_t1)

        # VAMP loss (negative VAMP score)
        loss = vamp_score.loss(out_t0, out_t1)

        # Backward pass
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation
    val_score = validate(model, test_loader)

    # Save best model
    if val_score > best_score:
        save_checkpoint(model, "best_model.pt")
```

#### Step 2.5: Post-Training Analysis
```python
# Get model predictions on all frames
probs, embeddings, attentions, edge_indices = analyze_vampnet_outputs(model, loader)

# Chapman-Kolmogorov test (validates Markov assumption)
run_ck_analysis(probs, lag_times=[lag_time], steps=10)

# Implied timescales (relaxation times)
analyze_implied_timescales(probs, lag_times=range(1, max_tau))
```

### Outputs from Phase 2

| File | Contents |
|------|----------|
| `best_model.pt` | Best model weights (by validation score) |
| `final_model.pt` | Final epoch model weights |
| `config.txt` | Training configuration |
| `vamp_scores.png` | Training/validation score curves |
| `models/epoch_*.pt` | Periodic checkpoints |
| `plots/ck_test/*.png` | Chapman-Kolmogorov test plots |
| `plots/its/*.png` | Implied timescales plots |
| `state_probs.npy` | Predicted state probabilities |
| `embeddings.npy` | Encoder feature vectors |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `batch_size` | 32 | Batch size |
| `lr` | 0.001 | Learning rate |
| `weight_decay` | 0.0001 | L2 regularization |
| `val_split` | 0.2 | Validation fraction |
| `save_every` | 10 | Checkpoint frequency |
| `clip_grad` | 1.0 | Gradient clipping norm |

---

## Phase 3: Analysis

### Entry Point
**File:** `pygv/pipe/analysis.py`

### Step-by-Step Process (per trained model)

#### Step 3.1: Load Trained Model
```python
model = VAMPNet.load_complete_model("best_model.pt")
model.eval()
```

#### Step 3.2: Run Inference
```python
# Create frame-by-frame dataset (not pairs)
frames_dataset = dataset.get_frames_dataset(return_pairs=False)
loader = DataLoader(frames_dataset, shuffle=False)

# Get predictions for all frames
probs, embeddings, attentions, edge_indices = analyze_vampnet_outputs(model, loader)
```

#### Step 3.3: Transition Matrix Analysis
```python
plot_transition_probabilities(probs, save_dir)
```
- Calculate transition matrix from state assignments
- Plot heatmap of state-to-state transitions
- Plot time series of state occupations

#### Step 3.4: Attention Map Analysis
```python
state_attention_maps, state_populations = calculate_state_edge_attention_maps(
    attentions, edge_indices, probs, n_states
)
plot_state_edge_attention_maps(state_attention_maps, ...)
plot_state_attention_weights(state_attention_maps, ...)
```
- Aggregate edge attentions by state
- Create residue-level attention summaries
- Identify important interactions per state

#### Step 3.5: Representative Structure Extraction
```python
generate_state_structures(
    probs=probs,
    trajectory_file=traj_file,
    topology_file=top_file,
    n_structures=10,        # structures per state
    prob_threshold=0.7      # minimum state probability
)
```
For each state:
1. Find frames with highest probability for that state
2. Extract top N structures
3. Save as PDB files

#### Step 3.6: Visualization
```python
visualize_state_ensemble(...)           # Overlay structures per state
save_attention_colored_structures(...)  # Color by attention importance
visualize_attention_ensemble(...)       # Attention-colored overlays
plot_state_network(...)                 # Transition network diagram
```

### Outputs from Phase 3

| File/Directory | Contents |
|----------------|----------|
| `transition_matrix.npy` | State transition probabilities |
| `state_probs.npy` | Frame-by-frame state probabilities |
| `state_structures/State_N/` | Representative PDB structures |
| `attention_maps/*.png` | Edge and residue attention heatmaps |
| `attention_structures/*.pdb` | Attention-colored structures |
| `visualizations/state_ensemble_*.png` | State overlay images |
| `visualizations/state_network.png` | Transition network diagram |

---

## Final Output Structure

```
exp_{protein_name}_{timestamp}/
├── config.yaml                    # Complete configuration
├── pipeline_summary.json          # Execution summary
│
├── preparation/
│   └── prep_{timestamp}/
│       ├── topology.pdb           # Topology (selected atoms only)
│       ├── prep_config.json       # Preparation parameters
│       ├── dataset_stats.json     # Dataset statistics
│       ├── cache/                 # Cached dataset files
│       └── state_discovery/       # (if --discover_states)
│           ├── embeddings.npy     # Graph2Vec embeddings
│           ├── cluster_labels.npy # Recommended k labels
│           ├── elbow_plot.png     # Elbow method plot
│           ├── silhouette_plot.png
│           ├── tsne_embeddings.png
│           ├── cluster_sizes.png
│           └── discovery_summary.json
│
├── training/
│   ├── lag10ns_5states/
│   │   ├── best_model.pt          # Best model weights
│   │   ├── final_model.pt         # Final model weights
│   │   ├── config.txt             # Training config
│   │   ├── vamp_scores.png        # Score curves
│   │   ├── models/                # Epoch checkpoints
│   │   └── plots/
│   │       ├── ck_test/           # Chapman-Kolmogorov plots
│   │       └── its/               # Implied timescales plots
│   │
│   └── lag20ns_7states/           # Another experiment
│       └── ...
│
├── analysis/
│   ├── lag10ns_5states/
│   │   ├── transition_matrix.npy
│   │   ├── state_probs.npy
│   │   ├── state_structures/
│   │   │   ├── State_1/*.pdb
│   │   │   └── State_N/*.pdb
│   │   ├── attention_maps/
│   │   ├── attention_structures/
│   │   └── visualizations/
│   │
│   └── lag20ns_7states/
│       └── ...
│
└── logs/                          # Execution logs
```

---

## Quick Reference: What Each Phase Produces

| Phase | Input | Output | Time |
|-------|-------|--------|------|
| **0. Init** | CLI args | Config + directories | Seconds |
| **1. Prep** | Trajectories | Graph dataset | Minutes-Hours |
| **2. Train** | Dataset | Trained models | Hours-Days |
| **3. Analysis** | Models | Visualizations + structures | Minutes |

---

## Data Flow Diagram

```
Trajectory Files (.xtc/.dcd)
        │
        ▼
┌───────────────────────────────────────────┐
│  PHASE 1: PREPARATION                      │
│  ┌─────────────────────────────────────┐  │
│  │ Load with MDTraj                    │  │
│  │ Apply selection ("name CA")         │  │
│  │ Apply stride (every Nth frame)      │  │
│  │ Build k-NN graphs                   │  │
│  │ Expand edge distances (Gaussian)    │  │
│  │ Create time-lagged pairs            │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
        │
        ▼
   VAMPNetDataset
   (graph_t0, graph_t1) pairs
        │
        ▼
┌───────────────────────────────────────────┐
│  PHASE 2: TRAINING                         │
│  ┌─────────────────────────────────────┐  │
│  │ Split: 80% train / 20% validation   │  │
│  │                                     │  │
│  │ Model Architecture:                 │  │
│  │   Graph → Encoder → Embedding       │  │
│  │   Embedding → Classifier → States   │  │
│  │                                     │  │
│  │ Training Loop:                      │  │
│  │   VAMP Loss = -VAMPScore(t0, t1)    │  │
│  │   Backprop + AdamW optimizer        │  │
│  │   Save best model by val score      │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
        │
        ▼
   Trained VAMPNet Model
   (encoder + classifier weights)
        │
        ▼
┌───────────────────────────────────────────┐
│  PHASE 3: ANALYSIS                         │
│  ┌─────────────────────────────────────┐  │
│  │ Load model + run inference          │  │
│  │ Extract state probabilities         │  │
│  │ Calculate transition matrices       │  │
│  │ Aggregate attention by state        │  │
│  │ Extract representative structures   │  │
│  │ Generate visualizations             │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
        │
        ▼
   Results: Structures, Plots, Matrices
```

---

## Common Issues Checklist

### Before Running
- [ ] Trajectory files exist in `traj_dir`
- [ ] Topology file is valid and matches trajectories
- [ ] Atom selection returns atoms (test with MDTraj first)
- [ ] Lag time is compatible with trajectory length
- [ ] Sufficient disk space for outputs

### During Preparation
- [ ] Trajectories load without errors
- [ ] Selected atoms > 0
- [ ] Lag frames < trajectory length
- [ ] Cache directory writable (if caching)

### During Training
- [ ] VAMP score is increasing (not stuck at 1.0)
- [ ] Gradients are not exploding (NaN loss)
- [ ] GPU memory not exhausted (reduce batch_size)
- [ ] Checkpoints being saved

### During Analysis
- [ ] Model loads successfully
- [ ] State probabilities sum to 1.0
- [ ] Transition matrix is valid (rows sum to 1)
- [ ] Structures are chemically reasonable

---

## Useful Commands

```bash
# Run full pipeline
python -m pygv.pipe.master_pipeline \
    --traj_dir /path/to/trajectories \
    --top /path/to/topology.pdb \
    --preset medium_schnet \
    --lag_times 10 20 50 \
    --n_states 3 5 7 \
    --cache

# Run preparation with state discovery
python -m pygv.pipe.preparation \
    --traj_dir /path/to/trajectories \
    --top /path/to/topology.pdb \
    --discover_states \
    --g2v_epochs 20 \
    --max_states 10

# Skip preparation (use cached dataset)
python -m pygv.pipe.master_pipeline ... --skip_preparation

# Only run analysis on existing models
python -m pygv.pipe.master_pipeline ... --only_analysis

# Resume from checkpoint
python -m pygv.pipe.master_pipeline ... --resume /path/to/experiment
```

---

*Last updated: 2026-02-05*
