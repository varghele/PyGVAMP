"""
Interactive HTML report generation for VAMPNet analysis.

Bridges the analysis pipeline outputs to the pygviz interactive visualizer,
producing a self-contained HTML file with 3D embedding plots, protein structure
viewer with attention coloring, transition matrix heatmap, and state legend.
"""

import os
import re
import glob
import json
import base64
import numpy as np
from typing import Optional, List, Tuple, Dict


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


def _superpose_pdb_onto_reference(ref_pdb_str: str, mobile_pdb_str: str) -> str:
    """
    Superpose a mobile PDB structure onto a reference, preserving B-factors.

    Uses MDTraj for optimal rigid-body alignment (Kabsch algorithm), then
    writes the aligned coordinates back into the original PDB text so that
    B-factor values (used for attention coloring) are preserved exactly.

    Parameters
    ----------
    ref_pdb_str : str
        PDB file contents of the reference (average) structure.
    mobile_pdb_str : str
        PDB file contents of the structure to align.

    Returns
    -------
    str
        PDB text with coordinates from the superposed structure and
        all other fields (including B-factors) unchanged.
    """
    try:
        import mdtraj as md
        import tempfile
    except ImportError:
        return mobile_pdb_str  # can't align without mdtraj

    try:
        # Write to temp files for MDTraj loading
        with tempfile.NamedTemporaryFile(
                suffix='.pdb', mode='w', delete=False) as f:
            f.write(ref_pdb_str)
            ref_path = f.name
        with tempfile.NamedTemporaryFile(
                suffix='.pdb', mode='w', delete=False) as f:
            f.write(mobile_pdb_str)
            mob_path = f.name

        ref = md.load(ref_path)
        mob = md.load(mob_path)
        os.unlink(ref_path)
        os.unlink(mob_path)

        # Superpose mobile onto reference
        mob.superpose(ref)

        # Patch aligned coordinates back into the original PDB text,
        # preserving B-factors, occupancy, and all other columns.
        aligned_xyz = mob.xyz[0] * 10.0  # nm → Å
        atom_idx = 0
        new_lines = []
        for line in mobile_pdb_str.split('\n'):
            if line.startswith(('ATOM', 'HETATM')) and atom_idx < len(aligned_xyz):
                x, y, z = aligned_xyz[atom_idx]
                # PDB columns 31-54 hold x, y, z (each 8.3f)
                line = (line[:30]
                        + f'{x:8.3f}{y:8.3f}{z:8.3f}'
                        + line[54:])
                atom_idx += 1
            new_lines.append(line)
        return '\n'.join(new_lines)

    except Exception as e:
        import warnings
        warnings.warn(f"Could not superpose PDB structure: {e}")
        return mobile_pdb_str


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
        from pygv.visualization import MDTrajectoryVisualizer
    except ImportError:
        print("  pygv.visualization not available — skipping interactive report.")
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


def subsample_frames(
        probs: np.ndarray,
        embeddings: np.ndarray,
        edge_attentions: list,
        edge_indices: list,
        max_frames: int = 5000,
        random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list, list, np.ndarray]:
    """
    Stratified subsampling by state assignment to preserve state distribution.

    Parameters
    ----------
    probs : np.ndarray
        State probabilities of shape (n_frames, n_states).
    embeddings : np.ndarray
        Embeddings of shape (n_frames, embedding_dim).
    edge_attentions : list
        Per-frame edge attention arrays (length n_frames).
    edge_indices : list
        Per-frame edge index arrays (length n_frames).
    max_frames : int
        Maximum number of frames to keep.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (probs, embeddings, edge_attentions, edge_indices, selected_indices)
        subsampled, where selected_indices maps subsampled points back to
        original frame indices.
    """
    n_frames = len(probs)
    if n_frames <= max_frames:
        return probs, embeddings, edge_attentions, edge_indices, np.arange(n_frames)

    rng = np.random.RandomState(random_state)
    state_assignments = np.argmax(probs, axis=1)
    unique_states = np.unique(state_assignments)

    selected_indices = []
    for state in unique_states:
        state_mask = np.where(state_assignments == state)[0]
        # Proportional allocation
        n_sample = max(1, int(round(len(state_mask) / n_frames * max_frames)))
        n_sample = min(n_sample, len(state_mask))
        chosen = rng.choice(state_mask, size=n_sample, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.sort(np.concatenate(selected_indices))

    # Trim to max_frames if rounding caused overshoot
    if len(selected_indices) > max_frames:
        selected_indices = rng.choice(selected_indices, size=max_frames, replace=False)
        selected_indices = np.sort(selected_indices)

    sub_probs = probs[selected_indices]
    sub_embeddings = embeddings[selected_indices]
    sub_attentions = [edge_attentions[i] for i in selected_indices]
    sub_indices = [edge_indices[i] for i in selected_indices]

    return sub_probs, sub_embeddings, sub_attentions, sub_indices, selected_indices


def _load_analysis_artifacts(analysis_dir: str) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], list, list
]:
    """
    Load saved analysis artifacts from an analysis subdirectory.

    Returns
    -------
    tuple
        (probs, embeddings, edge_attentions, edge_indices) or Nones on failure.
    """
    probs_path = os.path.join(analysis_dir, 'transformed_traj.npz')
    emb_path = os.path.join(analysis_dir, 'embeddings.npz')

    if not os.path.isfile(probs_path) or not os.path.isfile(emb_path):
        return None, None, [], []

    probs_data = np.load(probs_path)
    probs = probs_data[probs_data.files[0]]

    emb_data = np.load(emb_path)
    embeddings = emb_data[emb_data.files[0]]

    # Load edge attentions
    att_dir = os.path.join(analysis_dir, 'edge_attentions')
    idx_dir = os.path.join(analysis_dir, 'edge_indices')

    n_frames = len(probs)
    edge_attentions = [None] * n_frames
    edge_indices = [None] * n_frames

    if os.path.isdir(att_dir):
        for fname in sorted(os.listdir(att_dir)):
            if fname.startswith('attention_') and fname.endswith('.npy'):
                frame_num = int(fname.replace('attention_', '').replace('.npy', ''))
                if frame_num < n_frames:
                    edge_attentions[frame_num] = np.load(os.path.join(att_dir, fname))

    if os.path.isdir(idx_dir):
        for fname in sorted(os.listdir(idx_dir)):
            if fname.startswith('edge_indices_') and fname.endswith('.npy'):
                frame_num = int(fname.replace('edge_indices_', '').replace('.npy', ''))
                if frame_num < n_frames:
                    edge_indices[frame_num] = np.load(os.path.join(idx_dir, fname))

    return probs, embeddings, edge_attentions, edge_indices


def _image_to_base64(path: str) -> Optional[str]:
    """Read an image file and return a base64-encoded data URI string."""
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode('ascii')
    ext = os.path.splitext(path)[1].lower()
    mime = 'image/png' if ext == '.png' else 'image/jpeg'
    return f"data:{mime};base64,{encoded}"


def _load_diagnostic_data(
    analysis_subdir: str,
    protein_name: str,
) -> Dict:
    """
    Load diagnostic report JSON and diagnostic plot images as base64.

    Parameters
    ----------
    analysis_subdir : str
        Path to the analysis subdirectory (e.g., analysis/lag10ns_5states/).
    protein_name : str
        Protein name used in filenames.

    Returns
    -------
    dict
        Diagnostic metadata with keys: 'report', 'plots'.
        Empty dict if no diagnostics are found.
    """
    # Load JSON report
    json_path = os.path.join(analysis_subdir, f"{protein_name}_state_diagnostics.json")
    if not os.path.isfile(json_path):
        return {}

    with open(json_path) as f:
        report_data = json.load(f)

    # Load diagnostic plots as base64
    plots = {}
    plot_files = {
        'diagnostic_summary': f"{protein_name}_diagnostic_summary.png",
        'eigenvalue_spectrum': f"{protein_name}_eigenvalue_spectrum.png",
        'jsd_heatmap': f"{protein_name}_jsd_heatmap.png",
        'implied_timescales': f"{protein_name}_implied_timescales.png",
    }
    for key, filename in plot_files.items():
        img = _image_to_base64(os.path.join(analysis_subdir, filename))
        if img:
            plots[key] = img

    # CK test plot lives in ck_analysis subfolder
    ck_dir = os.path.join(analysis_subdir, 'ck_analysis')
    if os.path.isdir(ck_dir):
        ck_img = _image_to_base64(
            os.path.join(ck_dir, f"{protein_name}_ck_test_comparison.png"))
        if ck_img:
            plots['ck_test'] = ck_img

    return {
        'report': report_data,
        'plots': plots,
    }


def _create_pdb_template(frame, topology_file):
    """
    Create PDB template string from a single MDTraj frame.

    Parameters
    ----------
    frame : mdtraj.Trajectory
        Single-frame trajectory to use as template.
    topology_file : str
        Path to topology file (unused but kept for clarity).

    Returns
    -------
    str
        PDB text used as template for coordinate patching.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
        tmp_path = f.name
    frame.save_pdb(tmp_path)
    with open(tmp_path) as f:
        pdb_text = f.read()
    os.unlink(tmp_path)
    return pdb_text


def generate_merged_interactive_report(
        experiment_dir: str,
        topology_file: str,
        protein_name: str = "protein",
        max_frames: int = 5000,
        stride: int = 1,
        timestep: float = 0.001,
        traj_dir: str = None,
        file_pattern: str = "*.xtc",
        selection: str = None,
        training_selection: str = None,
) -> Optional[str]:
    """
    Generate a single interactive HTML report combining all lag times.

    Scans the experiment's analysis/ subdirectories, loads saved artifacts,
    subsamples frames, and produces one merged HTML with a timescale per lag time.

    Parameters
    ----------
    experiment_dir : str
        Root experiment directory (contains analysis/ subdirectory).
    topology_file : str
        Path to PDB/topology file for protein structure.
    protein_name : str
        Name of the protein (used in filename and title).
    max_frames : int
        Maximum frames per lag time (stratified subsampling).
    stride : int
        Stride used during frame extraction.
    timestep : float
        Trajectory timestep in nanoseconds.
    traj_dir : str, optional
        Path to trajectory directory. If None, reads from config.yaml.
    file_pattern : str
        Glob pattern for trajectory files (default: '*.xtc').
    selection : str, optional
        MDTraj atom selection string for visualization (e.g. 'protein').
        Controls which atoms are loaded for the 3D viewer.
    training_selection : str, optional
        MDTraj atom selection string used during training (e.g. 'name CA').
        Used to compute the residue mapping between attention values
        (one per training-selection residue) and the visualization PDB
        residue numbering. If None, defaults to ``selection``.

    Returns
    -------
    str or None
        Path to the generated HTML file, or None if generation failed.
    """
    try:
        from pygv.visualization import MDTrajectoryVisualizer
    except ImportError:
        print("  pygv.visualization not available — skipping merged interactive report.")
        return None

    from pygv.utils.analysis import calculate_transition_matrices

    analysis_root = os.path.join(experiment_dir, 'analysis')
    if not os.path.isdir(analysis_root):
        print(f"  No analysis directory found at {analysis_root}")
        return None

    # Discover analysis subdirectories matching lag<X>ns_<Y>states[_retrained]
    lag_pattern = re.compile(r'lag(\d+(?:\.\d+)?)ns_(\d+)states(_retrained)?')
    all_entries = []
    for name in sorted(os.listdir(analysis_root)):
        match = lag_pattern.match(name)
        if match and os.path.isdir(os.path.join(analysis_root, name)):
            lag_time = float(match.group(1))
            n_states = int(match.group(2))
            retrained = match.group(3) is not None
            full_path = os.path.join(analysis_root, name)
            mtime = os.path.getmtime(full_path)
            all_entries.append((lag_time, n_states, full_path, retrained, mtime))

    # For each lag time, find the final (best) run:
    # - If retrained versions exist, the newest retrained dir is "final"
    # - If no retrained versions, the original is "final"
    # All other entries for that lag time are superseded.
    from collections import defaultdict
    lag_groups = defaultdict(list)
    for entry in all_entries:
        lag_groups[entry[0]].append(entry)

    subdirs = []
    for lag_time, entries in lag_groups.items():
        retrained_entries = [e for e in entries if e[3]]
        if retrained_entries:
            # Pick only the newest retrained entry (final model)
            final = max(retrained_entries, key=lambda e: e[4])
            subdirs.append((final[0], final[1], final[2], True))
        else:
            # No retrained — original is final
            for e in entries:
                subdirs.append((e[0], e[1], e[2], True))

    # Sort by lag time
    subdirs.sort(key=lambda x: x[0])

    if not subdirs:
        print("  No analysis subdirectories found matching lag*ns_*states pattern.")
        return None

    viz = MDTrajectoryVisualizer()

    # --- Config.yaml fallback for traj_dir ---
    if traj_dir is None:
        config_path = os.path.join(experiment_dir, 'config.yaml')
        if os.path.isfile(config_path):
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                traj_dir = cfg.get('traj_dir')
                file_pattern = cfg.get('file_pattern', file_pattern)
                if selection is None:
                    selection = cfg.get('selection')
                if training_selection is None:
                    training_selection = cfg.get('selection')
            except Exception:
                pass

    # If no separate training_selection, fall back to selection
    if training_selection is None:
        training_selection = selection

    # --- Compute residue mapping (training selection residues → resSeq) ---
    # When the visualization uses full protein but attention is per training-selection
    # residue, we need to tell the JS frontend which resSeq each attention index maps to.
    residue_mapping = None
    if training_selection and topology_file:
        try:
            import mdtraj as md
            full_top = md.load_topology(topology_file)
            sel_indices = full_top.select(training_selection)
            # Get unique resSeq values in order of first appearance
            seen = set()
            resseq_list = []
            for ai in sel_indices:
                rs = full_top.atom(ai).residue.resSeq
                if rs not in seen:
                    seen.add(rs)
                    resseq_list.append(rs)
            residue_mapping = resseq_list
            print(f"  Residue mapping: {len(residue_mapping)} training residues → resSeq values")
        except Exception as e:
            print(f"  Warning: could not compute residue mapping: {e}")

    # --- Load trajectory frames for per-frame protein structure viewer ---
    frame_coords = None
    pdb_template = None
    if traj_dir and topology_file:
        try:
            import mdtraj as md
            from pygv.utils.pipe_utils import find_trajectory_files
            traj_files = find_trajectory_files(traj_dir, file_pattern)
            if traj_files:
                traj = md.load(traj_files, top=topology_file, stride=stride)
                if selection:
                    atom_indices = traj.topology.select(selection)
                    traj = traj.atom_slice(atom_indices)
                pdb_template = _create_pdb_template(traj[0], topology_file)
                # frame_coords shape: (n_frames, n_atoms, 3) in Ångströms
                frame_coords = traj.xyz * 10.0  # nm → Å
                print(f"  Loaded trajectory: {frame_coords.shape[0]} frames, "
                      f"{frame_coords.shape[1]} atoms for per-frame structures.")
        except Exception as e:
            print(f"  Warning: could not load trajectory for frame structures: {e}")

    # --- Load preparation-phase data (Graph2Vec embeddings) ---
    prep_dir = os.path.join(experiment_dir, 'preparation')
    if os.path.isdir(prep_dir):
        prep_runs = sorted(glob.glob(os.path.join(prep_dir, 'prep_*', 'state_discovery')))
        if prep_runs:
            sd_dir = prep_runs[-1]
            summary_path = os.path.join(sd_dir, 'discovery_summary.json')
            labels_path = os.path.join(sd_dir, 'cluster_labels.npy')

            if os.path.isfile(summary_path) and os.path.isfile(labels_path):
                try:
                    with open(summary_path) as f:
                        discovery_summary = json.load(f)

                    chosen_source = discovery_summary.get('chosen_source', 'tsne_2')
                    emb_path = os.path.join(sd_dir, f'{chosen_source}_embeddings.npy')
                    if not os.path.isfile(emb_path):
                        emb_path = os.path.join(sd_dir, 'tsne_2_embeddings.npy')

                    if os.path.isfile(emb_path):
                        prep_embeddings = np.load(emb_path)
                        prep_labels = np.load(labels_path)
                        viz.set_prep_data(prep_embeddings, prep_labels, discovery_summary)
                        print(f"  Loaded prep data: {len(prep_embeddings)} frames, "
                              f"{int(np.max(prep_labels)) + 1} clusters, source={chosen_source}")
                    else:
                        print(f"  Prep embeddings file not found, skipping prep data.")
                except Exception as e:
                    print(f"  Warning: failed to load prep data: {e}")

    timescales_added = 0

    for lag_time, n_states, subdir, is_final in subdirs:
        print(f"  Loading artifacts from {os.path.basename(subdir)}...")
        probs, embeddings, edge_attentions, edge_indices = _load_analysis_artifacts(subdir)

        if probs is None or embeddings is None:
            print(f"    Skipping — missing artifacts.")
            continue

        # Subsample frames
        probs, embeddings, edge_attentions, edge_indices, selected_indices = subsample_frames(
            probs, embeddings, edge_attentions, edge_indices,
            max_frames=max_frames,
        )

        # Infer n_nodes from edge indices
        n_nodes = None
        for idx in edge_indices:
            if idx is not None:
                n_nodes = int(np.max(idx)) + 1
                break
        if n_nodes is None:
            print(f"    Skipping — could not infer n_nodes.")
            continue

        # Reduce embeddings to 2D
        embeddings_2d = reduce_embeddings_to_2d(embeddings)

        # State assignments and frame indices
        state_assignments = np.argmax(probs, axis=1).astype(np.int32)
        frame_indices = np.arange(len(probs), dtype=np.int32)

        # Transition matrix
        transition_matrix, _ = calculate_transition_matrices(
            probs=probs,
            lag_time=lag_time,
            stride=stride,
            timestep=timestep,
        )

        # Aggregate edge attention to residue-level
        attention_values = aggregate_edge_attention_to_residue(
            edge_attentions=edge_attentions,
            edge_indices=edge_indices,
            n_nodes=n_nodes,
        )

        # Load pre-computed state structure PDBs (attention-colored)
        state_structures = {}
        for state_idx in range(n_states):
            state_num = state_idx + 1
            att_dir = os.path.join(subdir, f'state_{state_num}_attention')
            avg_pdb = os.path.join(
                att_dir,
                f'{protein_name}_state_{state_num}_average_attention.pdb')
            rep_pdbs = glob.glob(os.path.join(
                att_dir,
                f'{protein_name}_state_{state_num}_rank_*_attention.pdb'))
            # Sort by rank number (numeric) to get closest-to-average first
            def _rank_key(p):
                m = re.search(r'rank_(\d+)_', os.path.basename(p))
                return int(m.group(1)) if m else 999
            rep_pdbs.sort(key=_rank_key)

            entry = {'average': None, 'representatives': []}
            if os.path.isfile(avg_pdb):
                with open(avg_pdb) as f:
                    entry['average'] = f.read()
            for pdb_path in rep_pdbs[:3]:
                with open(pdb_path) as f:
                    rep_str = f.read()
                # Superpose representative onto the average for proper overlay
                if entry['average']:
                    rep_str = _superpose_pdb_onto_reference(
                        entry['average'], rep_str)
                entry['representatives'].append(rep_str)
            state_structures[state_idx] = entry

        # Load diagnostic data if available
        diagnostics = _load_diagnostic_data(subdir, protein_name)
        metadata = {'is_final': is_final}
        if diagnostics:
            metadata['diagnostics'] = diagnostics

        viz.add_timescale(
            lagtime=int(lag_time),
            embeddings=embeddings_2d,
            frame_indices=frame_indices,
            state_assignments=state_assignments,
            transition_matrix=transition_matrix,
            attention_values=attention_values,
            state_structures=state_structures,
            metadata=metadata,
            trajectory_frame_indices=selected_indices,
        )
        timescales_added += 1
        n_loaded = sum(1 for s in state_structures.values() if s['average'])
        n_diag_plots = len(diagnostics.get('plots', {}))
        diag_info = f", {n_diag_plots} diagnostic plots" if n_diag_plots else ""
        print(f"    Added timescale: lag={lag_time}ns, {len(probs)} frames (subsampled), {n_loaded}/{n_states} state structures{diag_info}.")

    if timescales_added == 0:
        print("  No timescales could be added — skipping report generation.")
        return None

    # Pass per-frame coordinates if trajectory was loaded
    if frame_coords is not None and pdb_template is not None:
        viz.set_frame_coordinates(frame_coords, pdb_template)

    # Use the selection-sliced PDB template if available, otherwise fall back
    # to the full topology file.
    if pdb_template is not None:
        viz.set_protein_structure(pdb_string=pdb_template)
    else:
        viz.set_protein_structure(pdb_path=topology_file)

    # Pass residue mapping so the JS frontend can map attention indices to resSeq
    if residue_mapping is not None:
        viz.set_residue_mapping(residue_mapping)

    output_path = os.path.join(experiment_dir, f"{protein_name}_interactive_report.html")
    viz.generate(
        output_path=output_path,
        title=f"{protein_name} — Interactive VAMPNet Analysis",
    )

    return output_path
