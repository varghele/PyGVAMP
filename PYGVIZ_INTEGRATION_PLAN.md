# Plan: Integrate pygviz Interactive HTML Report into Analysis Pipeline

## Context

The `pygviz` package (`pygviz/md_visualizer/`) is a standalone interactive HTML visualizer for MD trajectory analysis. It produces a self-contained HTML file with 3D embedding plots (Three.js), protein structure viewer with attention coloring (3Dmol.js), transition matrix heatmap (D3.js), and state legend — all interactive in the browser.

Currently pygviz is disconnected from the pipeline. The goal is to wire it into the end of the analysis phase so an interactive HTML report is automatically generated after every analysis run.

## Data Gap Analysis

| pygviz needs | Analysis pipeline has | Transformation needed |
|---|---|---|
| `embeddings` (n_frames, 2) | `embeddings` (n_frames, embedding_dim) | UMAP/t-SNE reduction to 2D |
| `state_assignments` (n_frames,) int | `probs` (n_frames, n_states) float | `np.argmax(probs, axis=1)` |
| `frame_indices` (n_frames,) | not stored | `np.arange(n_frames)` |
| `transition_matrix` (n_states, n_states) | computed by `calculate_transition_matrices()` | reuse existing function |
| `attention_values` (n_frames, n_residues) | `edge_attentions` list of (n_edges,) per frame | aggregate incoming edges per target node |
| `pdb_path` | `args.top` | pass directly |

## Implementation Steps

### Step 1: Create `pygv/utils/interactive_report.py` (new file)

Three functions:

**`reduce_embeddings_to_2d(embeddings, method='umap', random_state=42)`**
- Input: `(n_frames, embedding_dim)` -> Output: `(n_frames, 2)`
- Try UMAP first (matching pattern from `pygv/clustering/state_discovery.py:167-173`), fall back to t-SNE
- UMAP: `n_components=2, n_neighbors=min(15, n-1), min_dist=0.1`
- t-SNE: `n_components=2, perplexity=min(30, n-1), max_iter=1000`

**`aggregate_edge_attention_to_residue(edge_attentions, edge_indices, n_nodes)`**
- Input: lists of per-frame `(n_edges,)` arrays + `(2, n_edges)` index arrays
- Output: `(n_frames, n_nodes)` — mean incoming edge attention per target node
- For each frame: scatter-add attention to target nodes, divide by edge count
- Handle `None` entries (no attention) by filling with zeros

**`generate_interactive_report(probs, embeddings, edge_attentions, edge_indices, topology_file, save_dir, protein_name, lag_time, stride, timestep, n_nodes)`**
- Orchestrates: imports pygviz (with ImportError guard), calls the two helper functions, computes state assignments + transition matrix (reuses `calculate_transition_matrices` from `pygv/utils/analysis.py`), creates `MDTrajectoryVisualizer`, calls `add_timescale()`, `set_protein_structure()`, `generate()`
- Returns output path or `None` if pygviz unavailable

### Step 2: Modify `pygv/pipe/analysis.py`

Insert call after line 326 (after `plot_state_network`, before `analyze_trajectories`):

```python
from pygv.utils.interactive_report import generate_interactive_report

# ... at line ~327:
print("Generating interactive HTML report...")
try:
    report_path = generate_interactive_report(
        probs=probs,
        embeddings=embeddings,
        edge_attentions=attentions,
        edge_indices=edge_indices,
        topology_file=args.top,
        save_dir=paths['analysis_dir'],
        protein_name=args.protein_name,
        lag_time=args.lag_time,
        stride=args.stride,
        timestep=inferred_timestep,
        n_nodes=len(residue_indices),
    )
    if report_path:
        print(f"Interactive report saved to: {report_path}")
except Exception as e:
    print(f"Warning: Could not generate interactive report: {e}")
```

All variables (`probs`, `embeddings`, `attentions`, `edge_indices`, `residue_indices`, `inferred_timestep`) are already in scope at that point in `run_analysis()`.

### Step 3: Create `tests/test_interactive_report.py`

- Test `reduce_embeddings_to_2d`: output shape, UMAP and t-SNE paths, NaN handling, small datasets
- Test `aggregate_edge_attention_to_residue`: output shape, None handling, single/multiple edge averaging
- Test `generate_interactive_report`: returns None when pygviz missing, correct shapes passed to visualizer

## Files to modify/create

| File | Action |
|---|---|
| `pygv/utils/interactive_report.py` | **Create** — data transforms + pygviz orchestration |
| `pygv/pipe/analysis.py` | **Modify** — add import + call at line ~327 |
| `tests/test_interactive_report.py` | **Create** — unit tests |

## Error handling

- pygviz not installed: `ImportError` caught, returns `None`, prints warning
- UMAP not installed: falls back to t-SNE (sklearn always available)
- Any pygviz error: outer `try/except` in analysis.py prevents pipeline crash
- No attention data: fills with zeros, visualization still works (uniform coloring)

## Verification

1. Run `pytest tests/test_interactive_report.py` for unit tests
2. Run `pytest tests/` to verify no regressions
3. If trajectory data is available, run a full analysis and open the generated HTML in browser
