#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper Analysis — Aggregate multiple independent runs for publication figures.

This script discovers completed pipeline runs, matches states across runs,
and produces publication-ready plots with mean + 95% confidence intervals
for implied timescales (ITS), Chapman-Kolmogorov (CK) tests, and state
populations.

Expected directory structure (produced by paper_runs_*.sh):
    paper_experiments/protein_name/
        lag5ns/
            run_00/exp_.../analysis/lag5ns_Xstates/
            run_01/exp_.../analysis/lag5ns_Xstates/
            ...
        lag10ns/
            run_00/exp_.../analysis/lag10ns_Xstates/
            ...

Usage:
    python for_publication/paper_analysis.py \\
        --experiment_dir ./paper_experiments/my_protein \\
        --output_dir ./paper_figures \\
        --protein_name my_protein \\
        --stride 10 \\
        --timestep 0.001
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from scipy.optimize import linear_sum_assignment

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pygv.utils.ck import estimate_koopman_op, get_ck_test
from pygv.utils.its import get_its


# =============================================================================
# Discovery
# =============================================================================

def discover_runs(experiment_dir):
    """
    Discover completed pipeline runs grouped by lag time.

    Parameters
    ----------
    experiment_dir : str
        Path to the experiment directory (e.g., paper_experiments/my_protein)

    Returns
    -------
    dict
        {lag_time_str: [list of analysis directory paths]}
    """
    experiment_dir = Path(experiment_dir)
    runs_by_lag = {}

    # Look for lag*/run_*/ structure
    for lag_dir in sorted(experiment_dir.glob("lag*ns")):
        lag_str = lag_dir.name  # e.g., "lag10ns"

        analysis_dirs = []
        for run_dir in sorted(lag_dir.glob("run_*")):
            # Find analysis subdirectory (inside the timestamped exp dir)
            # Structure: run_XX/exp_.../analysis/lagXns_Ystates/
            candidates = list(run_dir.glob("exp_*/analysis/*/"))
            if candidates:
                # Use the first analysis dir found
                analysis_dirs.append(str(sorted(candidates)[-1]))
            else:
                # Maybe the analysis dir is directly in run_dir
                direct = list(run_dir.glob("analysis/*/"))
                if direct:
                    analysis_dirs.append(str(sorted(direct)[-1]))

        if analysis_dirs:
            runs_by_lag[lag_str] = analysis_dirs
            print(f"  {lag_str}: {len(analysis_dirs)} runs found")
        else:
            print(f"  {lag_str}: no completed runs found")

    return runs_by_lag


def load_run_data(analysis_dir, protein_name):
    """
    Load analysis outputs from a single run.

    Returns
    -------
    dict with keys: probs, its_data, transition_matrix, or None values if missing
    """
    analysis_dir = Path(analysis_dir)
    data = {}

    # State probabilities
    probs_path = analysis_dir / "transformed_traj.npz"
    if probs_path.exists():
        npz = np.load(str(probs_path))
        # The key might be 'arr_0' or 'probs'
        key = list(npz.keys())[0]
        data['probs'] = npz[key]
    else:
        print(f"  Warning: {probs_path} not found")
        data['probs'] = None

    # ITS data
    its_path = analysis_dir / "implied_timescales" / f"{protein_name}_its_data.npz"
    if its_path.exists():
        its_npz = np.load(str(its_path))
        data['its'] = its_npz['its']            # (n_states-1, n_lags)
        data['its_lag_times'] = its_npz['lag_times']
    else:
        data['its'] = None
        data['its_lag_times'] = None

    # Transition matrix (try CSV first, then npy)
    K_csv = list(analysis_dir.glob(f"{protein_name}_transition_matrix_all_lag*ns.csv"))
    if K_csv:
        data['transition_matrix'] = np.loadtxt(str(K_csv[0]), delimiter=',')
    else:
        data['transition_matrix'] = None

    # RevVAMPNet: learned K and pi
    learned_K_path = analysis_dir / f"{protein_name}_learned_K.npy"
    learned_pi_path = analysis_dir / f"{protein_name}_learned_pi.npy"
    if learned_K_path.exists():
        data['learned_K'] = np.load(str(learned_K_path))
        data['learned_pi'] = np.load(str(learned_pi_path))
    else:
        data['learned_K'] = None
        data['learned_pi'] = None

    return data


# =============================================================================
# State matching
# =============================================================================

def match_states(probs_reference, probs_target):
    """
    Find the permutation that best aligns target states to reference states.

    Uses the Hungarian algorithm on the overlap matrix between hard state
    assignments from two runs.

    Parameters
    ----------
    probs_reference : np.ndarray
        State probabilities from reference run, shape (n_frames, n_states)
    probs_target : np.ndarray
        State probabilities from target run, shape (n_frames, n_states)

    Returns
    -------
    np.ndarray
        Permutation indices — permutation[i] is the target state that
        corresponds to reference state i
    """
    n_states = probs_reference.shape[1]
    n_frames = min(len(probs_reference), len(probs_target))

    ref_assignments = np.argmax(probs_reference[:n_frames], axis=1)
    tgt_assignments = np.argmax(probs_target[:n_frames], axis=1)

    # Build overlap matrix: overlap[i,j] = number of frames where
    # reference assigns state i AND target assigns state j
    overlap = np.zeros((n_states, n_states))
    for i in range(n_states):
        ref_mask = ref_assignments == i
        for j in range(n_states):
            overlap[i, j] = np.sum(ref_mask & (tgt_assignments == j))

    # Hungarian algorithm to maximize overlap (minimize negative overlap)
    row_ind, col_ind = linear_sum_assignment(-overlap)

    # col_ind[i] = target state that matches reference state i
    permutation = col_ind
    return permutation


def apply_permutation_to_probs(probs, permutation):
    """Reorder columns of probs according to permutation."""
    return probs[:, permutation]


def apply_permutation_to_its(its, permutation):
    """Reorder ITS rows according to permutation (excluding stationary)."""
    # ITS has shape (n_states-1, n_lags), corresponding to eigenvalues 1..n_states-1
    # After state permutation, eigenvalues change order.
    # The safest approach: sort ITS rows by magnitude (slowest first) per lag,
    # which is how they're typically reported anyway.
    return np.sort(its, axis=0)[::-1]


def apply_permutation_to_matrix(K, permutation):
    """Reorder rows and columns of transition matrix."""
    return K[np.ix_(permutation, permutation)]


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_its(all_its, confidence=0.95):
    """
    Compute mean and CI for implied timescales across runs.

    Parameters
    ----------
    all_its : list of np.ndarray
        Each array has shape (n_states-1, n_lags)
    confidence : float
        Confidence level (default 0.95 for 95% CI)

    Returns
    -------
    dict with 'mean', 'lower', 'upper', each shape (n_states-1, n_lags)
    """
    stacked = np.stack(all_its, axis=0)  # (n_runs, n_states-1, n_lags)
    n_runs = stacked.shape[0]

    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0, ddof=1)

    # t-distribution critical value approximated by z for n>=10
    from scipy.stats import t as t_dist
    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha / 2, df=n_runs - 1)

    margin = t_crit * std / np.sqrt(n_runs)

    return {
        'mean': mean,
        'lower': mean - margin,
        'upper': mean + margin,
        'std': std,
        'n_runs': n_runs,
    }


def aggregate_ck(all_predicted, all_estimated, confidence=0.95):
    """
    Compute mean and CI for CK test results across runs.

    Parameters
    ----------
    all_predicted : list of np.ndarray
        Each shape (n_states, n_states, steps)
    all_estimated : list of np.ndarray
        Each shape (n_states, n_states, steps)

    Returns
    -------
    dict with 'pred_mean', 'pred_lower', 'pred_upper', 'est_mean', etc.
    """
    from scipy.stats import t as t_dist

    pred_stack = np.stack(all_predicted, axis=0)  # (n_runs, n_states, n_states, steps)
    est_stack = np.stack(all_estimated, axis=0)

    n_runs = pred_stack.shape[0]
    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha / 2, df=max(1, n_runs - 1))

    def _stats(arr):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1)
        margin = t_crit * std / np.sqrt(n_runs)
        return mean, mean - margin, mean + margin

    pred_mean, pred_lower, pred_upper = _stats(pred_stack)
    est_mean, est_lower, est_upper = _stats(est_stack)

    return {
        'pred_mean': pred_mean, 'pred_lower': pred_lower, 'pred_upper': pred_upper,
        'est_mean': est_mean, 'est_lower': est_lower, 'est_upper': est_upper,
        'n_runs': n_runs,
    }


def aggregate_populations(all_probs, confidence=0.95):
    """
    Compute mean and CI for state populations across runs.

    Parameters
    ----------
    all_probs : list of np.ndarray
        Each shape (n_frames, n_states)

    Returns
    -------
    dict with 'mean', 'lower', 'upper', each shape (n_states,)
    """
    from scipy.stats import t as t_dist

    populations = []
    for probs in all_probs:
        assignments = np.argmax(probs, axis=1)
        n_states = probs.shape[1]
        pops = np.array([np.mean(assignments == s) for s in range(n_states)])
        populations.append(pops)

    stacked = np.stack(populations, axis=0)  # (n_runs, n_states)
    n_runs = stacked.shape[0]

    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1)

    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha / 2, df=max(1, n_runs - 1))
    margin = t_crit * std / np.sqrt(n_runs)

    return {
        'mean': mean,
        'lower': mean - margin,
        'upper': mean + margin,
        'std': std,
        'n_runs': n_runs,
    }


# =============================================================================
# Plotting — publication quality
# =============================================================================

def setup_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def plot_its_with_ci(its_stats, lag_times, save_path, protein_name,
                     n_processes=None, figsize=(8, 6)):
    """
    Plot implied timescales with confidence intervals.

    Parameters
    ----------
    its_stats : dict
        Output from aggregate_its()
    lag_times : array-like
        Lag times in nanoseconds
    save_path : str
        Path to save the figure
    protein_name : str
        Protein name for title
    n_processes : int, optional
        Number of slowest processes to plot (default: all)
    """
    mean = its_stats['mean']
    lower = its_stats['lower']
    upper = its_stats['upper']

    n_total = mean.shape[0]
    if n_processes is None:
        n_processes = n_total
    n_processes = min(n_processes, n_total)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 0.85, n_processes))

    for i in range(n_processes):
        ax.plot(lag_times, mean[i], color=colors[i], linewidth=1.5,
                label=f"Process {i + 1}")
        ax.fill_between(lag_times, lower[i], upper[i],
                        color=colors[i], alpha=0.2)

    # Reference line: ITS = lag time
    lag_arr = np.array(lag_times)
    ax.plot(lag_arr, lag_arr, 'k--', alpha=0.4, linewidth=1, label=r'$t_i = \tau$')

    ax.set_xlabel(r'Lag time $\tau$ (ns)')
    ax.set_ylabel('Implied timescale (ns)')
    ax.set_title(f'{protein_name} — Implied Timescales '
                 f'(n={its_stats["n_runs"]} runs, 95% CI)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_ck_with_ci(ck_stats, steps, tau_ns, save_path, protein_name,
                    figsize_per_state=2.5):
    """
    Plot Chapman-Kolmogorov test with confidence intervals.

    Parameters
    ----------
    ck_stats : dict
        Output from aggregate_ck()
    steps : int
        Number of CK steps
    tau_ns : float
        Base lag time in nanoseconds
    save_path : str
        Path to save the figure
    protein_name : str
        Protein name for title
    """
    pred_mean = ck_stats['pred_mean']
    pred_lower = ck_stats['pred_lower']
    pred_upper = ck_stats['pred_upper']
    est_mean = ck_stats['est_mean']
    est_lower = ck_stats['est_lower']
    est_upper = ck_stats['est_upper']

    n_states = pred_mean.shape[0]
    time_axis = np.arange(steps) * tau_ns

    fig, axes = plt.subplots(n_states, n_states,
                             figsize=(figsize_per_state * n_states,
                                      figsize_per_state * n_states),
                             sharex=True, sharey=True)

    for i in range(n_states):
        for j in range(n_states):
            ax = axes[i, j] if n_states > 1 else axes

            # Predicted (solid line with CI)
            ax.plot(time_axis, pred_mean[i, j, :], 'b-', linewidth=1.2,
                    label='Predicted' if (i == 0 and j == 0) else None)
            ax.fill_between(time_axis, pred_lower[i, j, :], pred_upper[i, j, :],
                            color='blue', alpha=0.15)

            # Estimated (dashed line with CI)
            ax.plot(time_axis, est_mean[i, j, :], 'r--', linewidth=1.2,
                    label='Estimated' if (i == 0 and j == 0) else None)
            ax.fill_between(time_axis, est_lower[i, j, :], est_upper[i, j, :],
                            color='red', alpha=0.15)

            if i == n_states - 1:
                ax.set_xlabel(r'$n \cdot \tau$ (ns)')
            if j == 0:
                ax.set_ylabel(f'P(S{i+1}→)')
            ax.set_title(f'S{i+1}→S{j+1}', fontsize=9)
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f'{protein_name} — Chapman-Kolmogorov Test '
                 f'(τ={tau_ns} ns, n={ck_stats["n_runs"]} runs, 95% CI)',
                 fontsize=14, y=1.02)

    if n_states > 1:
        axes[0, 0].legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_populations_with_ci(pop_stats, save_path, protein_name, lag_str,
                             figsize=(8, 5)):
    """
    Plot state populations as bar chart with confidence intervals.
    """
    mean = pop_stats['mean']
    lower = pop_stats['lower']
    upper = pop_stats['upper']
    n_states = len(mean)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_states)
    colors = plt.cm.viridis(np.linspace(0, 0.85, n_states))

    yerr_lower = mean - lower
    yerr_upper = upper - mean
    yerr = np.array([yerr_lower, yerr_upper])

    bars = ax.bar(x, mean, yerr=yerr, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add percentage labels
    for i, (bar, m) in enumerate(zip(bars, mean)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yerr_upper[i] + 0.005,
                f'{m:.1%}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('State')
    ax.set_ylabel('Population fraction')
    ax.set_title(f'{protein_name} — State Populations ({lag_str}, '
                 f'n={pop_stats["n_runs"]} runs, 95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i + 1}' for i in range(n_states)])
    ax.set_ylim(0, min(1.0, max(upper) * 1.3))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_score_convergence(all_histories, save_path, protein_name, lag_str,
                           figsize=(8, 5)):
    """
    Plot training score convergence across runs with CI.
    """
    # Find common epoch count (minimum across runs)
    min_epochs = min(len(h) for h in all_histories)
    stacked = np.array([h[:min_epochs] for h in all_histories])  # (n_runs, epochs)

    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1)
    n_runs = stacked.shape[0]

    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(0.975, df=max(1, n_runs - 1))
    margin = t_crit * std / np.sqrt(n_runs)

    epochs = np.arange(1, min_epochs + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, mean, 'b-', linewidth=1.5, label='Mean')
    ax.fill_between(epochs, mean - margin, mean + margin,
                    color='blue', alpha=0.2, label='95% CI')

    # Individual runs as thin lines
    for h in all_histories:
        ax.plot(np.arange(1, len(h) + 1), h, color='gray', alpha=0.15, linewidth=0.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(f'{protein_name} — Training Convergence ({lag_str}, '
                 f'n={n_runs} runs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Main analysis pipeline
# =============================================================================

def analyze_lag_group(lag_str, analysis_dirs, output_dir, protein_name,
                      stride, timestep, ck_steps=10):
    """
    Aggregate and plot results for all runs at one lag time.
    """
    # Parse lag time from string
    lag_ns = float(lag_str.replace('lag', '').replace('ns', ''))
    effective_timestep = timestep * stride

    print(f"\n{'='*60}")
    print(f"Analyzing {lag_str} ({len(analysis_dirs)} runs)")
    print(f"{'='*60}")

    # Load all runs
    all_data = []
    for i, adir in enumerate(analysis_dirs):
        print(f"  Loading run {i}: {adir}")
        data = load_run_data(adir, protein_name)
        if data['probs'] is not None:
            all_data.append(data)
        else:
            print(f"  Skipping run {i} (no state probabilities found)")

    if len(all_data) < 2:
        print(f"  ERROR: Need at least 2 complete runs, found {len(all_data)}. Skipping.")
        return

    n_runs = len(all_data)
    n_states = all_data[0]['probs'].shape[1]
    print(f"  {n_runs} runs loaded, {n_states} states")

    # ---- State matching (use run 0 as reference) ----
    print("  Matching states across runs...")
    reference_probs = all_data[0]['probs']

    permutations = [np.arange(n_states)]  # identity for reference
    for i in range(1, n_runs):
        perm = match_states(reference_probs, all_data[i]['probs'])
        permutations.append(perm)
        print(f"    Run {i} permutation: {perm}")

    # Apply permutations to probs
    aligned_probs = []
    for i, data in enumerate(all_data):
        aligned_probs.append(apply_permutation_to_probs(data['probs'], permutations[i]))

    # ---- State populations ----
    print("  Computing state populations...")
    pop_stats = aggregate_populations(aligned_probs)
    plot_populations_with_ci(
        pop_stats,
        os.path.join(output_dir, f"{protein_name}_{lag_str}_populations.png"),
        protein_name, lag_str
    )

    # ---- ITS ----
    # Recompute ITS from probs for consistency (all runs use same lag range)
    print("  Computing implied timescales...")
    max_its_lag = min(5 * lag_ns, (len(reference_probs) * effective_timestep) / 3)
    its_lag_times = np.linspace(effective_timestep, max_its_lag, 15).tolist()

    all_its = []
    for i, probs in enumerate(aligned_probs):
        try:
            its_arr, _ = get_its(probs, its_lag_times, stride=stride, timestep=timestep)
            # Sort by magnitude (slowest first) for consistent ordering
            its_arr = np.sort(its_arr, axis=0)[::-1]
            # Replace non-finite values with NaN
            its_arr[~np.isfinite(its_arr)] = np.nan
            all_its.append(its_arr)
        except Exception as e:
            print(f"    Warning: ITS computation failed for run {i}: {e}")

    if len(all_its) >= 2:
        its_stats = aggregate_its(all_its)
        plot_its_with_ci(
            its_stats, its_lag_times,
            os.path.join(output_dir, f"{protein_name}_{lag_str}_its.png"),
            protein_name
        )
    else:
        print("  Not enough successful ITS computations for CI.")

    # ---- CK test ----
    print("  Computing Chapman-Kolmogorov tests...")
    lag_frames = max(1, int(round(lag_ns / effective_timestep)))

    all_ck_pred = []
    all_ck_est = []
    for i, probs in enumerate(aligned_probs):
        try:
            predicted, estimated = get_ck_test(probs, steps=ck_steps, tau=lag_frames)
            # Apply state permutation to CK arrays
            perm = permutations[i]
            predicted = predicted[np.ix_(perm, perm, range(ck_steps))]
            estimated = estimated[np.ix_(perm, perm, range(ck_steps))]
            all_ck_pred.append(predicted)
            all_ck_est.append(estimated)
        except Exception as e:
            print(f"    Warning: CK test failed for run {i}: {e}")

    if len(all_ck_pred) >= 2:
        ck_stats = aggregate_ck(all_ck_pred, all_ck_est)
        plot_ck_with_ci(
            ck_stats, ck_steps, lag_ns,
            os.path.join(output_dir, f"{protein_name}_{lag_str}_ck.png"),
            protein_name
        )
    else:
        print("  Not enough successful CK computations for CI.")

    # ---- Save numerical results ----
    results_path = os.path.join(output_dir, f"{protein_name}_{lag_str}_results.npz")
    save_dict = {
        'n_runs': n_runs,
        'n_states': n_states,
        'lag_ns': lag_ns,
        'pop_mean': pop_stats['mean'],
        'pop_std': pop_stats['std'],
    }
    if len(all_its) >= 2:
        save_dict['its_mean'] = its_stats['mean']
        save_dict['its_std'] = its_stats['std']
        save_dict['its_lag_times'] = np.array(its_lag_times)
    np.savez(results_path, **save_dict)
    print(f"  Saved numerical results: {results_path}")

    # ---- Summary ----
    print(f"\n  Summary for {lag_str}:")
    print(f"    Runs:        {n_runs}")
    print(f"    States:      {n_states}")
    print(f"    Populations: {' '.join(f'{p:.1%}' for p in pop_stats['mean'])}")
    if len(all_its) >= 2:
        print(f"    Slowest ITS: {its_stats['mean'][0, -1]:.2f} "
              f"± {its_stats['std'][0, -1]:.2f} ns (at τ={its_lag_times[-1]:.1f} ns)")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate paper runs and produce publication figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python for_publication/paper_analysis.py \\
      --experiment_dir ./paper_experiments/my_protein \\
      --output_dir ./paper_figures \\
      --protein_name my_protein \\
      --stride 10 \\
      --timestep 0.001
        """
    )

    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory '
                             '(e.g., paper_experiments/my_protein)')
    parser.add_argument('--output_dir', type=str, default='./paper_figures',
                        help='Output directory for publication figures')
    parser.add_argument('--protein_name', type=str, default='protein',
                        help='Protein name for plot labels and file names')
    parser.add_argument('--stride', type=int, default=10,
                        help='Frame stride used during training')
    parser.add_argument('--timestep', type=float, default=0.001,
                        help='Trajectory timestep in nanoseconds')
    parser.add_argument('--ck_steps', type=int, default=10,
                        help='Number of Chapman-Kolmogorov steps')

    args = parser.parse_args()

    setup_publication_style()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Paper Analysis — Aggregating Multi-Run Results")
    print("=" * 60)
    print(f"Experiment dir: {args.experiment_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Protein:        {args.protein_name}")
    print(f"Stride:         {args.stride}")
    print(f"Timestep:       {args.timestep} ns")

    # Discover runs
    print("\nDiscovering runs...")
    runs_by_lag = discover_runs(args.experiment_dir)

    if not runs_by_lag:
        print("\nERROR: No completed runs found. Check your experiment directory.")
        print(f"Expected structure: {args.experiment_dir}/lagXns/run_YY/exp_.../analysis/...")
        sys.exit(1)

    # Analyze each lag time group
    for lag_str, analysis_dirs in sorted(runs_by_lag.items()):
        analyze_lag_group(
            lag_str=lag_str,
            analysis_dirs=analysis_dirs,
            output_dir=args.output_dir,
            protein_name=args.protein_name,
            stride=args.stride,
            timestep=args.timestep,
            ck_steps=args.ck_steps,
        )

    print(f"\n{'='*60}")
    print(f"Paper analysis complete. Figures saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
