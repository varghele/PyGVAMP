"""
Interactive HTML report generation for VAMPNet analysis.

Bridges the analysis pipeline outputs to the pygviz interactive visualizer,
producing a self-contained HTML file with 3D embedding plots, protein structure
viewer with attention coloring, transition matrix heatmap, and state legend.
"""

import os
import numpy as np
from typing import Optional, List


def reduce_embeddings_to_2d(embeddings: np.ndarray,
                            method: str = 'umap',
                            random_state: int = 42) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2D for visualization.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings array of shape (n_frames, embedding_dim)
    method : str
        Reduction method: 'umap' (default, falls back to 'tsne') or 'tsne'
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        2D embeddings of shape (n_frames, 2)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

    n_samples = embeddings.shape[0]

    # Already 2D — return as-is
    if embeddings.shape[1] == 2:
        return embeddings.copy()

    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, n_samples - 1),
                min_dist=0.1,
                random_state=random_state,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("  UMAP not installed, falling back to t-SNE.")
            method = 'tsne'

    # t-SNE fallback
    from sklearn.manifold import TSNE
    reducer = TSNE(
        n_components=2,
        perplexity=min(30.0, float(n_samples - 1)),
        max_iter=1000,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def aggregate_edge_attention_to_residue(edge_attentions: List,
                                        edge_indices: List,
                                        n_nodes: int) -> np.ndarray:
    """
    Aggregate per-edge attention values into per-residue (per-node) attention.

    For each frame, computes the mean incoming-edge attention for every target node.

    Parameters
    ----------
    edge_attentions : list
        List of per-frame attention arrays, each of shape (n_edges,).
        Entries may be None (no attention available).
    edge_indices : list
        List of per-frame edge index arrays, each of shape (2, n_edges).
        Entries may be None.
    n_nodes : int
        Number of nodes (residues) in the graph.

    Returns
    -------
    np.ndarray
        Residue-level attention of shape (n_frames, n_nodes).
    """
    n_frames = len(edge_attentions)
    result = np.zeros((n_frames, n_nodes), dtype=np.float32)

    for i in range(n_frames):
        att = edge_attentions[i]
        idx = edge_indices[i]

        if att is None or idx is None:
            continue

        att = np.asarray(att, dtype=np.float32).ravel()
        idx = np.asarray(idx)

        if idx.shape[0] != 2 or len(att) != idx.shape[1]:
            continue

        targets = idx[1]  # target (destination) nodes

        # Accumulate attention per target node
        node_sum = np.zeros(n_nodes, dtype=np.float64)
        node_count = np.zeros(n_nodes, dtype=np.float64)

        for e in range(len(att)):
            t = int(targets[e])
            if 0 <= t < n_nodes:
                node_sum[t] += att[e]
                node_count[t] += 1

        # Mean attention per node (zero where no edges)
        mask = node_count > 0
        node_sum[mask] /= node_count[mask]
        result[i] = node_sum.astype(np.float32)

    return result


def generate_interactive_report(
        probs: np.ndarray,
        embeddings: np.ndarray,
        edge_attentions: list,
        edge_indices: list,
        topology_file: str,
        save_dir: str,
        protein_name: str = "protein",
        lag_time: float = 1.0,
        stride: int = 1,
        timestep: float = 0.001,
        n_nodes: Optional[int] = None,
) -> Optional[str]:
    """
    Generate an interactive HTML report using pygviz.

    Orchestrates dimensionality reduction, attention aggregation, transition
    matrix computation, and pygviz visualizer calls.

    Parameters
    ----------
    probs : np.ndarray
        State probabilities of shape (n_frames, n_states).
    embeddings : np.ndarray
        Graph embeddings of shape (n_frames, embedding_dim).
    edge_attentions : list
        Per-frame edge attention arrays.
    edge_indices : list
        Per-frame edge index arrays.
    topology_file : str
        Path to PDB/topology file for protein structure.
    save_dir : str
        Directory to write the output HTML file.
    protein_name : str
        Name of the protein (used in filenames and titles).
    lag_time : float
        Lag time in nanoseconds.
    stride : int
        Stride used during frame extraction.
    timestep : float
        Trajectory timestep in nanoseconds.
    n_nodes : int, optional
        Number of graph nodes (residues). If None, inferred from edge_indices.

    Returns
    -------
    str or None
        Path to the generated HTML file, or None if pygviz is not available.
    """
    try:
        from pygviz.md_visualizer import MDTrajectoryVisualizer
    except ImportError:
        print("  pygviz not available — skipping interactive report.")
        return None

    from pygv.utils.analysis import calculate_transition_matrices

    # --- Infer n_nodes if not provided ---
    if n_nodes is None:
        for idx in edge_indices:
            if idx is not None:
                n_nodes = int(np.max(idx)) + 1
                break
        if n_nodes is None:
            print("  Could not infer n_nodes from edge_indices — skipping interactive report.")
            return None

    # --- 1. Reduce embeddings to 2D ---
    embeddings_2d = reduce_embeddings_to_2d(embeddings)

    # --- 2. State assignments ---
    state_assignments = np.argmax(probs, axis=1).astype(np.int32)

    # --- 3. Frame indices ---
    frame_indices = np.arange(len(probs), dtype=np.int32)

    # --- 4. Transition matrix ---
    transition_matrix, _ = calculate_transition_matrices(
        probs=probs,
        lag_time=lag_time,
        stride=stride,
        timestep=timestep,
    )

    # --- 5. Aggregate edge attention to residue-level ---
    attention_values = aggregate_edge_attention_to_residue(
        edge_attentions=edge_attentions,
        edge_indices=edge_indices,
        n_nodes=n_nodes,
    )

    # --- 6. Build visualizer ---
    viz = MDTrajectoryVisualizer()

    viz.add_timescale(
        lagtime=int(lag_time),
        embeddings=embeddings_2d,
        frame_indices=frame_indices,
        state_assignments=state_assignments,
        transition_matrix=transition_matrix,
        attention_values=attention_values,
    )

    viz.set_protein_structure(pdb_path=topology_file)

    # --- 7. Generate HTML ---
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{protein_name}_interactive_report.html")

    viz.generate(
        output_path=output_path,
        title=f"{protein_name} — Interactive VAMPNet Analysis",
    )

    return output_path
