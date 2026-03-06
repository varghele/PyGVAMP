#!/usr/bin/env python
"""
Compare G2Vec (preparation) states with GraphVAMP (training) states.

Loads a completed pipeline experiment directory and computes overlap metrics
between the unsupervised G2Vec cluster labels and the per-lag-time GraphVAMP
state assignments.  Produces:

1. Confusion matrices (G2Vec vs GraphVAMP for each lag time)
2. Contingency-based metrics (adjusted mutual information, adjusted Rand index)
3. Best permutation mapping from GraphVAMP states → G2Vec master states
4. Cross-lag-time consistency matrix (do different lag times agree once mapped?)
5. Per-state Jaccard overlap between G2Vec and each lag time
6. Summary JSON with all numbers

Usage:
    python experimental_area/compare_states.py /path/to/exp_dir [--save_dir ./state_comparison]
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_g2vec_labels(exp_dir: str) -> np.ndarray:
    """Find and load G2Vec cluster_labels.npy from the preparation phase."""
    pattern = os.path.join(exp_dir, "preparation", "*", "state_discovery", "cluster_labels.npy")
    matches = sorted(glob(pattern))
    if not matches:
        sys.exit(f"No cluster_labels.npy found under {exp_dir}/preparation/*/state_discovery/")
    path = matches[-1]  # latest prep run
    labels = np.load(path)
    print(f"G2Vec labels loaded from {path}  ({len(labels)} frames, {len(np.unique(labels))} states)")
    return labels


def find_training_probs(exp_dir: str) -> dict:
    """Find all transformed_traj.npz files under training/, keyed by experiment name."""
    results = {}
    training_dir = os.path.join(exp_dir, "training")
    if not os.path.isdir(training_dir):
        sys.exit(f"No training/ directory found in {exp_dir}")

    for exp_name in sorted(os.listdir(training_dir)):
        exp_path = os.path.join(training_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        # Find transformed_traj.npz (may be nested under a timestamp dir)
        pattern = os.path.join(exp_path, "**", "transformed_traj.npz")
        matches = sorted(glob(pattern, recursive=True))
        if matches:
            data = np.load(matches[-1])
            probs = data[data.files[0]]
            states = np.argmax(probs, axis=1)
            results[exp_name] = {
                'probs': probs,
                'states': states,
                'path': matches[-1],
            }
            print(f"  {exp_name}: {probs.shape[0]} frames, {probs.shape[1]} states  ({matches[-1]})")

    if not results:
        sys.exit(f"No transformed_traj.npz found under {training_dir}")
    return results


def best_permutation_mapping(ref_labels: np.ndarray, pred_labels: np.ndarray):
    """
    Find the permutation of pred_labels that maximises overlap with ref_labels.

    Uses the Hungarian algorithm on the confusion matrix.

    Returns
    -------
    mapping : dict
        pred_state -> ref_state
    accuracy : float
        Fraction of frames where mapped pred matches ref
    mapped_labels : np.ndarray
        pred_labels remapped to ref label space
    """
    n_ref = len(np.unique(ref_labels))
    n_pred = len(np.unique(pred_labels))
    n_classes = max(n_ref, n_pred)

    # Build cost matrix (negative overlap = we want to maximise)
    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for r, p in zip(ref_labels, pred_labels):
        if r < n_classes and p < n_classes:
            cost[p, r] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)  # maximise
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    mapped = np.array([mapping.get(int(s), -1) for s in pred_labels])
    accuracy = np.mean(mapped == ref_labels)
    return mapping, accuracy, mapped


def jaccard_per_state(ref: np.ndarray, pred: np.ndarray, n_states: int) -> dict:
    """Jaccard index per state after optimal mapping."""
    scores = {}
    for s in range(n_states):
        ref_set = set(np.where(ref == s)[0])
        pred_set = set(np.where(pred == s)[0])
        if len(ref_set) == 0 and len(pred_set) == 0:
            scores[s] = 1.0
        else:
            inter = len(ref_set & pred_set)
            union = len(ref_set | pred_set)
            scores[s] = inter / union if union > 0 else 0.0
    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion(ref, pred, title, save_path, ref_label="G2Vec", pred_label="GraphVAMP"):
    n = max(ref.max(), pred.max()) + 1
    cm = confusion_matrix(ref, pred, labels=list(range(n)))
    # Normalise rows (ref states) to fractions
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    im0 = axes[0].imshow(cm, cmap='Blues', aspect='auto')
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_xlabel(f"{pred_label} state")
    axes[0].set_ylabel(f"{ref_label} state")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Normalised
    im1 = axes[1].imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(f"{title} (row-normalised)")
    axes[1].set_xlabel(f"{pred_label} state")
    axes[1].set_ylabel(f"{ref_label} state")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center', fontsize=7)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_jaccard(jaccard_dict, save_path):
    """Bar chart of per-state Jaccard for each lag time."""
    exp_names = list(jaccard_dict.keys())
    if not exp_names:
        return

    n_states = max(len(v) for v in jaccard_dict.values())
    x = np.arange(n_states)
    width = 0.8 / len(exp_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(exp_names):
        scores = jaccard_dict[name]
        vals = [scores.get(s, 0.0) for s in range(n_states)]
        ax.bar(x + i * width, vals, width, label=name)

    ax.set_xlabel("G2Vec master state")
    ax.set_ylabel("Jaccard index")
    ax.set_title("Per-state Jaccard overlap (G2Vec vs mapped GraphVAMP)")
    ax.set_xticks(x + width * (len(exp_names) - 1) / 2)
    ax.set_xticklabels([str(s) for s in range(n_states)])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cross_lag_agreement(mapped_states_dict, n_frames, save_path):
    """Heatmap showing pairwise agreement between lag times (after mapping to G2Vec space)."""
    names = list(mapped_states_dict.keys())
    n = len(names)
    if n < 2:
        return

    agreement = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agreement[i, j] = np.mean(mapped_states_dict[names[i]] == mapped_states_dict[names[j]])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(agreement, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Cross-lag agreement (mapped to G2Vec master states)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agreement[i, j]:.2f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_state_timeline(g2v_labels, mapped_states_dict, save_path, max_frames=5000):
    """Timeline comparing G2Vec and mapped GraphVAMP states."""
    n_plots = 1 + len(mapped_states_dict)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    n = min(len(g2v_labels), max_frames)
    x = np.arange(n)

    axes[0].scatter(x, g2v_labels[:n], c=g2v_labels[:n], cmap='tab10', s=0.5, alpha=0.5)
    axes[0].set_ylabel("G2Vec")
    axes[0].set_title("State assignments over frames")

    for i, (name, mapped) in enumerate(mapped_states_dict.items(), 1):
        axes[i].scatter(x, mapped[:n], c=mapped[:n], cmap='tab10', s=0.5, alpha=0.5)
        axes[i].set_ylabel(name)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare G2Vec and GraphVAMP states")
    parser.add_argument("exp_dir", type=str, help="Path to a completed pipeline experiment directory")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory (default: <exp_dir>/state_comparison)")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    save_dir = args.save_dir or os.path.join(exp_dir, "state_comparison")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")
    print(f"Output directory:     {save_dir}\n")

    # --- Load data ---
    g2v_labels = find_g2vec_labels(exp_dir)
    print()
    print("Loading GraphVAMP training outputs:")
    training_data = find_training_probs(exp_dir)

    # --- Compare each lag time against G2Vec ---
    summary = {
        'g2v_n_frames': int(len(g2v_labels)),
        'g2v_n_states': int(len(np.unique(g2v_labels))),
        'experiments': {},
    }

    mapped_states_all = {}
    jaccard_all = {}

    for exp_name, data in training_data.items():
        gv_states = data['states']
        n_frames = min(len(g2v_labels), len(gv_states))

        if n_frames != len(g2v_labels):
            print(f"\n  WARNING: frame count mismatch for {exp_name}: "
                  f"G2Vec={len(g2v_labels)}, GraphVAMP={len(gv_states)}. Using first {n_frames}.")

        g2v = g2v_labels[:n_frames]
        gv = gv_states[:n_frames]

        print(f"\n--- {exp_name} ---")

        # Clustering metrics (label-permutation invariant)
        ami = adjusted_mutual_info_score(g2v, gv)
        nmi = normalized_mutual_info_score(g2v, gv)
        ari = adjusted_rand_score(g2v, gv)
        print(f"  Adjusted Mutual Information: {ami:.4f}")
        print(f"  Normalised Mutual Information: {nmi:.4f}")
        print(f"  Adjusted Rand Index:          {ari:.4f}")

        # Best permutation mapping
        mapping, accuracy, mapped = best_permutation_mapping(g2v, gv)
        mapped_states_all[exp_name] = mapped
        print(f"  Best-permutation accuracy:    {accuracy:.4f}")
        print(f"  Mapping (GraphVAMP → G2Vec):  {mapping}")

        # Per-state Jaccard
        n_master = len(np.unique(g2v))
        jacc = jaccard_per_state(g2v, mapped, n_master)
        jaccard_all[exp_name] = jacc
        for s in sorted(jacc.keys()):
            print(f"    State {s} Jaccard: {jacc[s]:.4f}")

        # Confusion matrix plot
        plot_confusion(
            g2v, gv,
            title=exp_name,
            save_path=os.path.join(save_dir, f"confusion_{exp_name}.png"),
        )

        # Mapped confusion (should be more diagonal)
        plot_confusion(
            g2v, mapped,
            title=f"{exp_name} (mapped)",
            save_path=os.path.join(save_dir, f"confusion_mapped_{exp_name}.png"),
            pred_label="Mapped GraphVAMP",
        )

        summary['experiments'][exp_name] = {
            'n_frames': int(n_frames),
            'n_graphvamp_states': int(data['probs'].shape[1]),
            'ami': float(ami),
            'nmi': float(nmi),
            'ari': float(ari),
            'best_permutation_accuracy': float(accuracy),
            'mapping_graphvamp_to_g2vec': {str(k): int(v) for k, v in mapping.items()},
            'jaccard_per_state': {str(k): float(v) for k, v in jacc.items()},
            'mean_jaccard': float(np.mean(list(jacc.values()))),
        }

    # --- Cross-lag consistency ---
    if len(mapped_states_all) >= 2:
        plot_cross_lag_agreement(
            mapped_states_all, len(g2v_labels),
            save_path=os.path.join(save_dir, "cross_lag_agreement.png"),
        )

    # --- Jaccard bar chart ---
    plot_jaccard(jaccard_all, save_path=os.path.join(save_dir, "jaccard_per_state.png"))

    # --- Timeline ---
    plot_state_timeline(
        g2v_labels, mapped_states_all,
        save_path=os.path.join(save_dir, "state_timeline.png"),
    )

    # --- Save summary ---
    summary_path = os.path.join(save_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {save_dir}")
    print(f"  - confusion_*.png          (raw and mapped confusion matrices)")
    print(f"  - cross_lag_agreement.png  (pairwise agreement after mapping)")
    print(f"  - jaccard_per_state.png    (per-state overlap)")
    print(f"  - state_timeline.png       (frame-by-frame comparison)")
    print(f"  - comparison_summary.json  (all metrics)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
