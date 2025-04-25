import os
import re

import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from matplotlib.colors import ListedColormap
from matplotlib.image import imread
from pygv.utils.analysis import calculate_transition_matrices
import mdtraj as md

import subprocess
from pygv.utils.analysis import calculate_transition_matrices


def plot_vamp_scores(scores, save_path=None, smoothing=None, title="VAMPNet Training Performance"):
    """
    Plot the VAMP score curve from training.

    Parameters:
    -----------
    scores : list
        List of VAMP scores across epochs
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    smoothing : int, optional
        Window size for smoothing the curve. If None, no smoothing is applied.
    title : str, default="VAMPNet Training Performance"
        Title for the plot

    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axis objects for further customization
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert negative losses to positive VAMP scores if needed
    vamp_scores = [-score if score < 0 else score for score in scores]

    # Create x-axis (epochs)
    epochs = np.arange(1, len(vamp_scores) + 1)

    # Plot raw VAMP scores
    ax.plot(epochs, vamp_scores, 'b-', alpha=0.3, label='Raw VAMP Score')

    # Apply smoothing if specified
    if smoothing is not None and smoothing > 1:
        # Simple moving average
        kernel_size = min(smoothing, len(vamp_scores))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_scores = np.convolve(vamp_scores, kernel, mode='valid')

        # Adjust x-axis for the convolution
        smooth_epochs = epochs[kernel_size - 1:]
        if len(smooth_epochs) == len(smoothed_scores) - 1:
            smooth_epochs = epochs[kernel_size - 1:len(smoothed_scores) + kernel_size - 1]

        ax.plot(smooth_epochs, smoothed_scores, 'r-', linewidth=2,
                label=f'Smoothed (window={smoothing})')

    # Add min/max markers
    max_score = max(vamp_scores)
    max_epoch = vamp_scores.index(max_score) + 1
    ax.plot(max_epoch, max_score, 'ro', markersize=8)
    ax.annotate(f'Max: {max_score:.4f}',
                xy=(max_epoch, max_score),
                xytext=(max_epoch + 1, max_score),
                fontsize=10)

    # Add styling
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('VAMP Score', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend if smoothing was applied
    if smoothing is not None:
        ax.legend()

    # Show final VAMP score
    final_score = vamp_scores[-1]
    ax.annotate(f'Final: {final_score:.4f}',
                xy=(len(vamp_scores), final_score),
                xytext=(len(vamp_scores) - 5, final_score),
                fontsize=10)

    # Formatting
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()

    return fig, ax


def plot_transition_probabilities(probs: np.ndarray,
                                  save_dir: str,
                                  protein_name: str,
                                  lag_time: float,
                                  stride: int,
                                  timestep: float,
                                  cmap_name: str = 'YlOrRd',
                                  fig_size: tuple[int, int] = (10, 8),
                                  font_size: int = 10):
    """
    Plot both standard and non-self transition probability matrices.

    Parameters
    ----------
    probs : np.ndarray
        State probability trajectory with shape [n_frames, n_states]
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein for file naming
    lag_time : float, optional
        Lag time for transition matrix calculation in nanoseconds
    stride : int, optional
        Stride used when extracting frames from trajectory
    timestep : float, optional
        Trajectory timestep in nanoseconds
    cmap_name : str, optional
        Matplotlib colormap name for the heatmaps
    fig_size : tuple[int, int], optional
        Figure size in inches
    font_size : int, optional
        Font size for annotations

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing (transition_matrix, transition_matrix_without_self_transitions)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Calculate transition matrices with proper lag time conversion
    trans_matrix, trans_matrix_no_self = calculate_transition_matrices(
        probs,
        lag_time=lag_time,
        stride=stride,
        timestep=timestep
    )

    # Get number of states
    n_states = trans_matrix.shape[0]

    # Create state labels for the axes
    state_labels = [f'State {i + 1}' for i in range(n_states)]

    # Get the standard colormap for regular transitions
    cmap = plt.cm.get_cmap(cmap_name)

    # Create custom colormap for non-self transitions with black diagonal
    colors = [cmap(i) for i in range(cmap.N)]
    custom_cmap_no_self = ListedColormap(['black'] + colors[1:])

    # Plot both matrices
    for matrix, suffix, use_custom_cmap in [
        (trans_matrix, "all", False),
        (trans_matrix_no_self, "no_self", True)
    ]:
        fig, ax = plt.subplots(figsize=fig_size)

        # Create visualization matrix for non-self transitions
        if suffix == "no_self":
            # Create visualization matrix with black diagonal
            viz_matrix = matrix.copy()
            np.fill_diagonal(viz_matrix, -1)  # -1 will be colored black
            used_cmap = custom_cmap_no_self
            vmin, vmax = -1, 1
        else:
            viz_matrix = matrix
            used_cmap = cmap
            vmin, vmax = 0, 1

        # Create heatmap
        im = ax.imshow(viz_matrix, cmap=used_cmap, vmin=vmin, vmax=vmax, aspect='equal')

        # Add colorbar (exclude -1 from colorbar in no_self plot)
        if suffix == "no_self":
            # Create a separate colorbar that doesn't include the -1 value
            norm = plt.Normalize(0, 1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax)
        else:
            cbar = plt.colorbar(im, ax=ax)

        cbar.set_label('Transition Probability')

        # Add text annotations
        for i in range(n_states):
            for j in range(n_states):
                # Choose text color based on background darkness
                if suffix == "no_self" and i == j:
                    text_color = 'white'  # White text on black diagonal
                else:
                    # Use white text on dark backgrounds, black text on light backgrounds
                    text_color = 'black' if matrix[i, j] < 0.5 else 'white'

                # Add the probability value as text
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                               ha='center', va='center', color=text_color,
                               fontsize=font_size)

        # Customize ticks and labels
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
        ax.set_xticklabels(state_labels)
        ax.set_yticklabels(state_labels)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add labels and title
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')

        # Calculate effective lag time in frames for title
        effective_timestep = timestep * stride  # ns
        lag_frames = int(round(lag_time / effective_timestep))

        title = f"State Transition Probabilities - Lag {lag_time} ns ({lag_frames} frames)"
        if suffix == "no_self":
            title = f"Non-Self {title}"
        ax.set_title(f'{title}\n{protein_name}')

        # Adjust layout
        plt.tight_layout()

        # Save outputs
        plot_path = os.path.join(save_dir, f"{protein_name}_transition_matrix_{suffix}_lag{lag_time}ns.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save to CSV
        csv_path = os.path.join(save_dir, f"{protein_name}_transition_matrix_{suffix}_lag{lag_time}ns.csv")
        pd.DataFrame(matrix, index=state_labels, columns=state_labels).to_csv(csv_path)

        print(f"Saved {suffix} transition matrix plot to: {plot_path}")
        print(f"Saved {suffix} transition matrix to: {csv_path}")

    return trans_matrix, trans_matrix_no_self


def plot_state_attention_maps_old(attention_maps, states_order, n_states, state_populations, save_path=None, n_atoms=None):
    """
    Plot node-level attention maps for each state individually and in a combined figure.

    Parameters
    ----------
    attention_maps : np.ndarray
        Attention values for each state [n_states, n_atoms]
    states_order : np.ndarray
        Order of states by population
    n_states : int
        Number of states
    state_populations : np.ndarray
        Population of each state
    save_path : str, optional
        Base path to save the figures
    n_atoms : int, optional
        Number of atoms (if None, will be inferred from attention_maps)
    """
    plt.style.use('default')
    plt.rcParams['axes.linewidth'] = 1.0

    # Determine number of atoms from the attention maps
    if n_atoms is None:
        n_atoms = attention_maps.shape[1]

    # Create x-axis for plotting
    x_range = np.arange(n_atoms)

    # Find global min and max for consistent colorbar
    vmin = np.min(attention_maps)
    vmax = np.max(attention_maps)

    # Create individual plots for each state
    for i in range(n_states):
        state_idx = states_order[i]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Plot attention as a bar chart
        bars = ax.bar(x_range, attention_maps[state_idx], width=0.8,
                      color=plt.cm.viridis(
                          (attention_maps[state_idx] - vmin) / (vmax - vmin + 1e-10)))

        # Add horizontal grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set axis labels and title
        ax.set_xlabel('Atom Index', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(f'State {state_idx + 1} Attention Profile\nPopulation: {state_populations[state_idx]:.1%}',
                     fontsize=14, pad=10)

        # Set x-ticks at reasonable intervals
        if n_atoms > 50:
            tick_interval = max(1, n_atoms // 20)
            ax.set_xticks(x_range[::tick_interval])
            ax.set_xticklabels([f"{x + 1}" for x in x_range[::tick_interval]], fontsize=8)
        else:
            ax.set_xticks(x_range)
            ax.set_xticklabels([f"{x + 1}" for x in x_range], fontsize=8)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Weight', fontsize=10)

        plt.tight_layout()

        # Save individual state plot
        if save_path:
            state_save_path = save_path.replace('.png', f'_state_{state_idx + 1}.png')
            plt.savefig(state_save_path, bbox_inches='tight')
            print(f"Saved state {state_idx + 1} plot to: {state_save_path}")

        plt.close()

    # Create combined plot
    # Calculate number of rows and columns needed
    n_cols = int(np.ceil(np.sqrt(n_states)))
    n_rows = int(np.ceil(n_states / n_cols))

    # Create combined figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=150)
    if n_states > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot each state
    for i in range(n_states):
        state_idx = states_order[i]
        ax = axes[i]

        # Plot attention as a bar chart
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        bars = ax.bar(x_range, attention_maps[state_idx], width=0.8,
                      color=plt.cm.viridis(norm(attention_maps[state_idx])))

        # Add horizontal grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set axis labels and title
        ax.set_xlabel('Atom Index', fontsize=10)
        ax.set_ylabel('Attention Weight', fontsize=10)
        ax.set_title(f'State {state_idx + 1}\nPopulation: {state_populations[state_idx]:.1%}',
                     fontsize=12, pad=10)

        # Set x-ticks at reasonable intervals
        if n_atoms > 50:
            tick_interval = max(1, n_atoms // 10)
            ax.set_xticks(x_range[::tick_interval])
            ax.set_xticklabels([f"{x + 1}" for x in x_range[::tick_interval]], fontsize=8)
        else:
            ax.set_xticks(x_range)
            ax.set_xticklabels([f"{x + 1}" for x in x_range], fontsize=8)

    # Remove empty subplots
    for i in range(n_states, len(axes)):
        axes[i].remove()

    # Add shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=12)

    fig.suptitle('Attention Profiles by State', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Save combined plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved combined plot to: {save_path}")

    plt.close()


def plot_state_edge_attention_maps(
        state_attention_maps: np.ndarray,
        state_populations: np.ndarray,
        save_dir: str = None,
        protein_name: str = "Protein",
        cmap_name: str = 'viridis',
        figsize: tuple = (10, 10),
        threshold: float = None,
        focus_atom: int = None
):
    """
    Plot edge-level attention maps for each state.

    Parameters
    ----------
    state_attention_maps : np.ndarray
        Attention matrices for each state [n_states, n_atoms, n_atoms]
    state_populations : np.ndarray
        Population of each state [n_states]
    save_dir : str, optional
        Directory to save the plots to
    protein_name : str, optional
        Name of the protein for plot titles
    cmap_name : str, optional
        Matplotlib colormap name for heatmaps
    figsize : tuple, optional
        Figure size for the combined plot
    threshold : float, optional
        Minimum attention value to display (None for no threshold)
    focus_atom : int, optional
        If provided, highlight connections to/from this atom
    """
    plt.style.use('default')
    plt.rcParams['axes.linewidth'] = 1.0

    # Get dimensions
    n_states = len(state_attention_maps)
    n_atoms = state_attention_maps.shape[1]

    # Order states by population (highest first)
    states_order = np.argsort(-state_populations)

    # Calculate global min and max for consistent colorbar
    vmin = np.min([m.min() for m in state_attention_maps])
    vmax = np.max([m.max() for m in state_attention_maps])

    # Apply threshold if specified
    if threshold is not None:
        vmin = max(vmin, threshold)

    # Create individual heatmaps for each state
    for i, state_idx in enumerate(states_order):
        # Get the attention matrix for this state
        attention_matrix = state_attention_maps[state_idx]

        # Apply threshold if specified
        if threshold is not None:
            display_matrix = np.copy(attention_matrix)
            display_matrix[display_matrix < threshold] = 0
        else:
            display_matrix = attention_matrix

        # Plot as heatmap
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Create the heatmap
        im = ax.imshow(
            display_matrix,
            cmap=plt.cm.get_cmap(cmap_name),
            vmin=vmin,
            vmax=vmax,
            aspect='equal'
        )

        # Add gridlines
        ax.set_xticks(np.arange(-.5, n_atoms, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_atoms, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5, alpha=0.3)

        # Highlight specific atom if requested
        if focus_atom is not None and 0 <= focus_atom < n_atoms:
            # Highlight row (connections from focus atom)
            ax.axhline(y=focus_atom, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
            # Highlight column (connections to focus atom)
            ax.axvline(x=focus_atom, color='r', linestyle='-', linewidth=1.0, alpha=0.7)

        # Add title and labels
        plt.title(f"State {state_idx + 1} Edge Attention Map\nPopulation: {state_populations[state_idx]:.1%}")
        plt.xlabel("Target Atom")
        plt.ylabel("Source Atom")

        # Set reasonable tick frequency
        if n_atoms > 30:
            tick_interval = max(1, n_atoms // 15)
            ticks = np.arange(0, n_atoms, tick_interval)
            plt.xticks(ticks, [str(t + 1) for t in ticks])
            plt.yticks(ticks, [str(t + 1) for t in ticks])
        else:
            plt.xticks(np.arange(n_atoms), [str(i + 1) for i in range(n_atoms)])
            plt.yticks(np.arange(n_atoms), [str(i + 1) for i in range(n_atoms)])

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Attention Weight')

        # Save individual plot if save_dir is provided
        if save_dir:
            individual_path = os.path.join(save_dir, f"{protein_name}_state_{state_idx + 1}_attention.png")
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            print(f"Saved state {state_idx + 1} attention map to: {individual_path}")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    # Create combined plot
    n_cols = min(3, n_states)
    n_rows = (n_states + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten() if n_states > 1 else [axes]

    for i, state_idx in enumerate(states_order):
        ax = axes[i]

        # Get the attention matrix for this state
        attention_matrix = state_attention_maps[state_idx]

        # Apply threshold if specified
        if threshold is not None:
            display_matrix = np.copy(attention_matrix)
            display_matrix[display_matrix < threshold] = 0
        else:
            display_matrix = attention_matrix

        # Create heatmap
        im = ax.imshow(
            display_matrix,
            cmap=plt.cm.get_cmap(cmap_name),
            vmin=vmin,
            vmax=vmax,
            aspect='equal'
        )

        # Add gridlines (for small matrices)
        if n_atoms <= 30:
            ax.set_xticks(np.arange(-.5, n_atoms, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n_atoms, 1), minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=0.3, alpha=0.2)

        # Highlight specific atom if requested
        if focus_atom is not None and 0 <= focus_atom < n_atoms:
            ax.axhline(y=focus_atom, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
            ax.axvline(x=focus_atom, color='r', linestyle='-', linewidth=1.0, alpha=0.7)

        # Add state information
        ax.set_title(f"State {state_idx + 1}\nPop: {state_populations[state_idx]:.1%}")

        # Set reasonable tick frequency for smaller plots
        if n_atoms > 30:
            tick_interval = max(1, n_atoms // 10)
            ticks = np.arange(0, n_atoms, tick_interval)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t + 1) for t in ticks], rotation=90, fontsize=8)
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=8)
        else:
            ax.set_xticks(np.arange(0, n_atoms, 2))
            ax.set_xticklabels([str(i + 1) for i in range(0, n_atoms, 2)], rotation=90, fontsize=8)
            ax.set_yticks(np.arange(0, n_atoms, 2))
            ax.set_yticklabels([str(i + 1) for i in range(0, n_atoms, 2)], fontsize=8)

        # Add labels only to the first column and last row
        if i % n_cols == 0:
            ax.set_ylabel("Source Atom")
        if i >= (n_rows - 1) * n_cols or i == n_states - 1:
            ax.set_xlabel("Target Atom")

    # Remove empty subplots
    for i in range(n_states, len(axes)):
        fig.delaxes(axes[i])

    # Add a colorbar on the right side
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Weight')

    # Add title
    fig.suptitle(f"{protein_name}: State Edge Attention Maps", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Save combined figure if save_dir is provided
    if save_dir:
        combined_path = os.path.join(save_dir, f"{protein_name}_all_states_attention.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined attention maps to: {combined_path}")
        plt.close()
    else:
        plt.show()


def plot_state_attention_weights(
        state_attention_maps: np.ndarray,
        topology_file: str,
        save_dir: str = None,
        protein_name: str = "protein",
        plot_sum_direction: str = "target",
        cmap_name: str = "viridis",
        figsize: tuple = (12, 6)
):
    """
    Plot average attention weights for residues across different states.

    Parameters
    ----------
    state_attention_maps : np.ndarray
        Attention maps for each state [n_states, n_atoms, n_atoms]
    topology_file : str
        Path to topology file (PDB or similar) to get residue names
    save_dir : str, optional
        Directory to save the figures
    protein_name : str, optional
        Name of the protein for plot titles and filenames
    plot_sum_direction : str, optional
        Direction to sum attention weights: "source", "target", or "both"
    cmap_name : str, optional
        Matplotlib colormap name for heatmap
    figsize : tuple, optional
        Figure size in inches
    """

    def minmax_scale(x, axis=None):
        """Manual implementation of MinMaxScaler"""
        x_min = np.min(x, axis=axis, keepdims=True)
        x_max = np.max(x, axis=axis, keepdims=True)
        # Handle the case where all values are the same
        if np.all(x_max == x_min):
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + 1e-10)  # Add epsilon to avoid division by zero

    # Load topology to get residue information
    try:
        top = md.load(topology_file).topology
    except Exception as e:
        print(f"Error loading topology file: {str(e)}")
        print("Using atom indices instead of residue names")
        top = None

    # Get dimensions from the attention maps
    n_states, n_atoms, _ = state_attention_maps.shape
    print(f"Processing attention maps: {n_states} states, {n_atoms} atoms")

    # Get atom-to-residue mapping and residue names if topology is available
    atom_to_res = {}
    res_names = []

    if top is not None:
        # Map atoms to residues
        for atom in top.atoms:
            res = atom.residue
            res_idx = res.index
            res_name = f"{res.name}{res.resSeq}"

            # Store atom to residue mapping
            atom_to_res[atom.index] = res_idx

            # Add residue name if not already in list
            if res_idx >= len(res_names):
                res_names.append(res_name)

        n_residues = len(res_names)
        print(f"Found {n_residues} residues in topology")

        # Create residue-level attention maps
        residue_attention_maps = np.zeros((n_states, n_residues, n_residues))

        # Map atom-level attention to residue-level
        print("Mapping atom attention to residue attention...")
        for atom_i in range(min(n_atoms, len(atom_to_res))):
            if atom_i in atom_to_res:
                res_i = atom_to_res[atom_i]
                if res_i < n_residues:
                    for atom_j in range(min(n_atoms, len(atom_to_res))):
                        if atom_j in atom_to_res:
                            res_j = atom_to_res[atom_j]
                            if res_j < n_residues:
                                residue_attention_maps[:, res_i, res_j] += state_attention_maps[:, atom_i, atom_j]

        # Use residue-level maps and names for plotting
        attention_maps = residue_attention_maps
        labels = res_names
        n_entities = n_residues
        entity_name = "Residue"
    else:
        # Use atom-level maps and indices for plotting
        attention_maps = state_attention_maps
        labels = [str(i) for i in range(1, n_atoms + 1)]
        n_entities = n_atoms
        entity_name = "Atom"

    # Calculate attention scores by summing in specified direction
    scores = np.zeros((n_states, n_entities))

    if plot_sum_direction == "source":
        # Sum along source dimension (how much attention this residue pays to others)
        scores = np.sum(attention_maps, axis=2)
    elif plot_sum_direction == "target":
        # Sum along target dimension (how much attention this residue receives)
        scores = np.sum(attention_maps, axis=1)
    else:  # "both"
        # Sum total attention involvement
        scores_from = np.sum(attention_maps, axis=2)
        scores_to = np.sum(attention_maps, axis=1)
        scores = scores_from + scores_to

    # Scale scores to [0, 1] for each state
    scaled_scores = np.zeros_like(scores)
    for i in range(n_states):
        scaled_scores[i] = minmax_scale(scores[i])

    # Create a figure with square cells by adjusting the figure dimensions
    # Calculate aspect ratio to make cells square
    aspect_ratio = n_states / n_entities

    # Adjust figure size to make cells square
    if aspect_ratio < 1:
        # More entities than states: adjust height
        fig_width = figsize[0]
        fig_height = figsize[0] * aspect_ratio
    else:
        # More states than entities: adjust width
        fig_height = figsize[1]
        fig_width = figsize[1] / aspect_ratio

    # Create the plot with adjusted dimensions
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)

    # Plot heatmap with square cells
    im = ax.imshow(scaled_scores, cmap=plt.cm.get_cmap(cmap_name), aspect='equal')

    # Set ticks and labels
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels([f"{i + 1}" for i in range(n_states)], fontweight='bold')

    # Set x-ticks (showing a subset if too many)
    if n_entities > 30:
        tick_interval = max(1, n_entities // 30)
        tick_positions = np.arange(0, n_entities, tick_interval)
        tick_labels = [labels[i] for i in tick_positions]
    else:
        tick_positions = np.arange(n_entities)
        tick_labels = labels

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontweight='bold', rotation=90)

    # Add axis labels
    ax.set_xlabel(entity_name, fontweight='bold')
    ax.set_ylabel('State', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Attention', fontweight='bold')

    # Add grid lines for better readability
    ax.set_xticks(np.arange(-0.5, n_entities, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_states, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)

    # Add title
    direction_text = "from" if plot_sum_direction == "source" else "to" if plot_sum_direction == "target" else "total for"
    plt.title(f"{protein_name}: Attention {direction_text} Each {entity_name} by State", fontweight='bold')

    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"{protein_name}_{entity_name.lower()}_attention_{plot_sum_direction}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {entity_name.lower()} attention plot to: {save_path}")

    return fig, ax


def plot_all_residue_attention_directions(
        state_attention_maps: np.ndarray,
        topology_file: str,
        save_dir: str = None,
        protein_name: str = "protein"
):
    """
    Plot residue attention weights in all directions (source, target, both).

    Parameters
    ----------
    state_attention_maps : np.ndarray
        Attention maps for each state [n_states, n_atoms, n_atoms]
    topology_file : str
        Path to topology file (PDB or similar) to get residue names
    save_dir : str, optional
        Directory to save the figures
    protein_name : str, optional
        Name of the protein for plot titles and filenames
    """
    # Plot attention in all directions
    directions = ["source", "target", "both"]
    descriptions = {
        "source": "Attention FROM Residues (Sum of Outgoing Attention)",
        "target": "Attention TO Residues (Sum of Incoming Attention)",
        "both": "Total Attention Involvement by Residue"
    }

    for direction in directions:
        fig, ax = plot_state_attention_weights(
            state_attention_maps=state_attention_maps,
            topology_file=topology_file,
            save_dir=save_dir,
            protein_name=protein_name,
            plot_sum_direction=direction
        )

        # Update title with more descriptive text
        ax.set_title(f"{protein_name}: {descriptions[direction]}", fontweight='bold')

        # Re-save with updated title if save_dir provided
        if save_dir:
            save_path = os.path.join(save_dir, f"{protein_name}_residue_attention_{direction}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)


def visualize_state_ensemble(
        state_structures: dict,
        save_dir: str,
        protein_name: str,
        ray_opaque_background: str = "off",
        use_transparency: bool = True,
        image_size: tuple = (1200, 1200)
):
    """
    Create PyMOL visualizations of state ensembles from multiple angles.

    Parameters
    ----------
    state_structures : dict
        Dictionary mapping state numbers to lists of PDB file paths
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein for file naming
    ray_opaque_background : str, optional
        Whether to use opaque background for ray-traced images ("on" or "off")
    use_transparency : bool, optional
        Whether to apply transparency to lower-ranked structures
    image_size : tuple, optional
        Size of output images (width, height) in pixels
    """
    # Define viewing angles (front, side, top)
    views = {
        'front': (0, 0, 0),
        'side': (90, 0, 0),
        'top': (0, 90, 0),
        'iso': (30, 30, 0)
    }

    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a log file for PyMOL output
    pymol_log = os.path.join(save_dir, f"{protein_name}_pymol.log")

    # Try importing PyMOL
    try:
        import pymol
        from pymol import cmd
        pymol_imported = True
        print("Successfully imported PyMOL")
    except ImportError:
        pymol_imported = False
        print("WARNING: Could not import PyMOL Python module.")
        print("Will attempt to run PyMOL via command line instead.")

    if pymol_imported:
        try:
            # Initialize PyMOL in headless mode
            pymol.finish_launching(['pymol', '-qc'])

            for state_num, structures in state_structures.items():
                state_dir = os.path.join(save_dir, f"state_{state_num + 1}")
                img_dir = os.path.join(state_dir, "images")
                os.makedirs(img_dir, exist_ok=True)

                print(f"Processing state {state_num + 1} with {len(structures)} structures...")

                # Initialize for each state
                cmd.reinitialize()

                # Set up visualization parameters
                cmd.bg_color("white")
                cmd.set("ray_opaque_background", ray_opaque_background)
                cmd.set("cartoon_fancy_helices", 1)
                cmd.set("cartoon_transparency", 0)
                cmd.set("ray_shadows", 0)

                # Load and process structures
                for i, pdb_file in enumerate(structures):
                    try:
                        # Extract distance or RMSD from filename if available
                        if '_dist_' in pdb_file:
                            dist = float(pdb_file.split('_dist_')[-1].replace('.pdb', ''))
                        elif '_rmsd_' in pdb_file:
                            dist = float(pdb_file.split('_rmsd_')[-1].split('_')[0])
                        else:
                            dist = i  # Use index as a fallback

                        # Determine opacity based on rank
                        opacity = 1.0
                        if use_transparency:
                            opacity = 1.0 - (i / len(structures) * 0.8)
                            opacity = max(0.2, opacity)  # Ensure minimum visibility

                        # Name structure based on state and rank
                        name = f"state_{state_num + 1}_rank_{i}"

                        # Load structure
                        cmd.load(pdb_file, name)
                        cmd.show_as("cartoon", name)

                        # Apply transparency
                        cmd.set("transparency", 1 - opacity, name)

                        # Color by secondary structure
                        cmd.color("marine", f"{name} and ss h")
                        cmd.color("forest", f"{name} and ss s")
                        cmd.color("wheat", f"{name} and ss l")

                        # Align to first structure
                        if i > 0:
                            cmd.align(name, f"state_{state_num + 1}_rank_0")

                    except Exception as e:
                        print(f"Error processing structure {pdb_file}: {str(e)}")

                # Save combined state
                try:
                    combined_pdb = os.path.join(state_dir, f"{protein_name}_state_{state_num + 1}_ensemble.pdb")
                    cmd.save(combined_pdb, f"state_{state_num + 1}_rank_*")
                    print(f"Saved combined ensemble to {combined_pdb}")
                except Exception as e:
                    print(f"Error saving combined ensemble: {str(e)}")

                # Generate images from different angles
                for view_name, (x, y, z) in views.items():
                    try:
                        # Reset orientation
                        cmd.reset()

                        # Apply the view rotation
                        cmd.rotate('x', x)
                        cmd.rotate('y', y)
                        cmd.rotate('z', z)

                        # Center and zoom
                        cmd.center()
                        cmd.zoom()

                        # Create file path
                        img_file = os.path.join(img_dir, f"{protein_name}_state_{state_num + 1}_{view_name}.png")

                        # Render image
                        cmd.ray(image_size[0], image_size[1])
                        cmd.png(img_file, dpi=300, ray=1)

                        if os.path.exists(img_file):
                            print(f"Saved {view_name} view to {img_file}")
                        else:
                            print(f"Failed to save {view_name} view")
                    except Exception as e:
                        print(f"Error generating {view_name} view: {str(e)}")

                # Clean up this state
                cmd.delete("all")

            # Clean up PyMOL session without killing the process
            cmd.delete("all")
            cmd.reinitialize()

        except Exception as e:
            print(f"ERROR using PyMOL Python API: {str(e)}")
            pymol_imported = False
            print("Falling back to command-line approach")

    # If Python API failed or wasn't available, use command line approach
    if not pymol_imported:
        print("Using command-line approach for PyMOL visualization")

        # Create a temporary PyMOL script
        temp_script_path = os.path.join(save_dir, f"{protein_name}_pymol_script.py")

        with open(temp_script_path, 'w') as script:
            script.write("from pymol import cmd\n\n")

            for state_num, structures in state_structures.items():
                # Create directories
                state_dir = os.path.join(save_dir, f"state_{state_num + 1}")
                img_dir = os.path.join(state_dir, "images")
                script.write(f"import os\nos.makedirs('{img_dir}', exist_ok=True)\n\n")

                script.write(f"# Processing state {state_num + 1}\n")
                script.write("cmd.reinitialize()\n")

                # Set up visualization parameters
                script.write("cmd.bg_color('white')\n")
                script.write(f"cmd.set('ray_opaque_background', '{ray_opaque_background}')\n")
                script.write("cmd.set('cartoon_fancy_helices', 1)\n")
                script.write("cmd.set('cartoon_transparency', 0)\n")
                script.write("cmd.set('ray_shadows', 0)\n\n")

                # Load and process structures
                for i, pdb_file in enumerate(structures):
                    # Determine opacity
                    opacity = 1.0
                    if use_transparency:
                        opacity = 1.0 - (i / len(structures) * 0.8)
                        opacity = max(0.2, opacity)

                    # Name structure
                    name = f"state_{state_num + 1}_rank_{i}"

                    script.write(f"# Load structure {i}\n")
                    script.write(f"try:\n")
                    script.write(f"    cmd.load(r'{os.path.abspath(pdb_file)}', '{name}')\n")
                    script.write(f"    cmd.show_as('cartoon', '{name}')\n")
                    script.write(f"    cmd.set('transparency', {1 - opacity}, '{name}')\n")
                    script.write(f"    cmd.color('marine', '{name} and ss h')\n")
                    script.write(f"    cmd.color('forest', '{name} and ss s')\n")
                    script.write(f"    cmd.color('wheat', '{name} and ss l')\n")

                    if i > 0:
                        script.write(f"    cmd.align('{name}', 'state_{state_num + 1}_rank_0')\n")

                    script.write(f"except Exception as e:\n")
                    script.write(f"    print(f'Error processing {pdb_file}: {{str(e)}}')\n\n")

                # Save combined state
                combined_pdb = os.path.join(state_dir, f"{protein_name}_state_{state_num + 1}_ensemble.pdb")
                script.write(f"# Save combined ensemble\n")
                script.write(f"try:\n")
                script.write(f"    cmd.save(r'{os.path.abspath(combined_pdb)}', 'state_{state_num + 1}_rank_*')\n")
                script.write(f"    print('Saved combined ensemble to {combined_pdb}')\n")
                script.write(f"except Exception as e:\n")
                script.write(f"    print(f'Error saving combined ensemble: {{str(e)}}')\n\n")

                # Generate images from different angles
                for view_name, (x, y, z) in views.items():
                    img_file = os.path.join(img_dir, f"{protein_name}_state_{state_num + 1}_{view_name}.png")

                    script.write(f"# Generate {view_name} view\n")
                    script.write(f"try:\n")
                    script.write(f"    cmd.reset()\n")
                    script.write(f"    cmd.rotate('x', {x})\n")
                    script.write(f"    cmd.rotate('y', {y})\n")
                    script.write(f"    cmd.rotate('z', {z})\n")
                    script.write(f"    cmd.center()\n")
                    script.write(f"    cmd.zoom()\n")
                    script.write(f"    cmd.ray({image_size[0]}, {image_size[1]})\n")
                    script.write(f"    cmd.png(r'{os.path.abspath(img_file)}', dpi=300, ray=1)\n")
                    script.write(f"    print('Saved {view_name} view to {img_file}')\n")
                    script.write(f"except Exception as e:\n")
                    script.write(f"    print(f'Error generating {view_name} view: {{str(e)}}')\n\n")

                # Clean up this state
                script.write("cmd.delete('all')\n\n")

            # Clean up PyMOL session without killing the process
            script.write("cmd.delete('all')\n")
            script.write("cmd.reinitialize()\n")
            script.write("cmd.quit()\n")

        print(f"Created PyMOL script at {temp_script_path}")

        # Execute the script with PyMOL
        if pymol_executable:
            print(f"Running PyMOL with script...")
            try:
                with open(pymol_log, 'w') as log:
                    subprocess.run([pymol_executable, '-qc', temp_script_path],
                                   stdout=log, stderr=log, check=True)
                print(f"PyMOL execution completed. See log at {pymol_log}")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: PyMOL execution failed with return code {e.returncode}")
        else:
            print("ERROR: PyMOL executable not found, cannot generate visualizations.")
            print(f"Please run this script manually with PyMOL: {temp_script_path}")

    print("Visualization process completed")


def visualize_attention_ensemble(
        state_structures: dict,
        state_attention_maps: np.ndarray,
        save_dir: str,
        protein_name: str
):
    """
    Create PyMOL visualizations of existing state structures colored by attention.

    Parameters
    ----------
    state_structures : dict
        Dictionary mapping state numbers to lists of PDB file paths
    state_attention_maps : np.ndarray
        Attention maps for each state [n_states, n_atoms, n_atoms]
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein for file naming
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define viewing angles (front, side, top)
    views = {
        'front': (0, 0, 0),
        'side': (90, 0, 0),
        'top': (0, 90, 0),
        'iso': (30, 30, 0)
    }

    # Define a simple scaling function
    def scale(x):
        """Scale array to range [0,1]"""
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        # Handle case where all values are the same
        return np.zeros_like(x)

    # Calculate scaled attention scores for each state
    state_residue_attention = {}
    for state in range(len(state_attention_maps)):
        scores = scale(state_attention_maps[state].sum(axis=0))
        state_residue_attention[state] = scores

    # Try to import PyMOL
    try:
        import pymol
        from pymol import cmd
        pymol_imported = True
        print("Successfully imported PyMOL")
    except ImportError:
        pymol_imported = False
        print("WARNING: Could not import PyMOL Python API.")
        print("Will create PyMOL script files instead that you can run manually.")

    # Create a function to generate PyMOL script for a state
    def generate_pymol_script(state_num, structures, state_dir):
        """Generate a PyMOL script for the given state"""
        script_path = os.path.join(state_dir, f"{protein_name}_state_{state_num + 1}_attention_view.pml")

        with open(script_path, 'w') as script:
            script.write("# PyMOL script for visualizing attention-colored structures\n")
            script.write("reinitialize\n")
            script.write("bg_color white\n")
            script.write("set ray_opaque_background, off\n")
            script.write("set cartoon_fancy_helices, 1\n")
            script.write("set cartoon_transparency, 0\n")
            script.write("set ray_shadows, 0\n\n")

            # Load structures and apply attention coloring
            for i, pdb_file in enumerate(structures):
                if i == 0:
                    opacity = 1.0  # First structure fully opaque
                else:
                    # Exponential decay starting from 0.7
                    decay_rate = 1.5  # Adjust this value to control decay speed
                    opacity = 0.3 * np.exp(-decay_rate * (i - 1))

                name = f"state_{state_num + 1}_rank_{i}"
                script.write(f"# Load structure {i}\n")
                script.write(f"load {os.path.abspath(pdb_file)}, {name}\n")
                script.write(f"show cartoon, {name}\n")
                script.write(f"set transparency, {1 - opacity}, {name}\n\n")

                # Apply attention values as B-factors
                script.write(f"# Apply attention values as B-factors\n")
                for res_idx, attention in enumerate(state_residue_attention[state_num]):
                    b_factor = attention * 100
                    script.write(f"alter {name} and resi {res_idx + 1}, b={b_factor}\n")

                # Color by B-factor (attention values)
                script.write(f"spectrum b, blue_white_red, {name}\n")

                if i > 0:
                    script.write(f"align {name}, state_{state_num + 1}_rank_0\n")

                script.write("\n")

            # Save combined state
            combined_pdb = os.path.join(state_dir, f"{protein_name}_state_{state_num + 1}_attention_ensemble.pdb")
            script.write(f"# Save combined ensemble\n")
            script.write(f"save {os.path.abspath(combined_pdb)}, state_{state_num + 1}_rank_*\n\n")

            # Generate images from different views
            img_dir = os.path.join(state_dir, "attention_images")
            script.write(f"# Create image directory\n")
            script.write(f"import os\nos.makedirs(r'{img_dir}', exist_ok=True)\n\n")

            script.write(f"# Generate images from different angles\n")
            for view_name, (x, y, z) in views.items():
                img_file = os.path.join(img_dir, f"{protein_name}_state_{state_num + 1}_{view_name}_attention.png")

                script.write(f"# {view_name} view\n")
                script.write(f"reset\n")
                script.write(f"rotate x, {x}\n")
                script.write(f"rotate y, {y}\n")
                script.write(f"rotate z, {z}\n")
                script.write(f"center\n")
                script.write(f"zoom\n")
                script.write(f"ray 1200, 1200\n")
                script.write(f"png {os.path.abspath(img_file)}, dpi=300, ray=1\n\n")

            script.write("# Clean up\n")
            script.write("reinitialize\n")
            script.write("# End of script\n")

        return script_path

    # Process each state
    pymol_scripts = []

    if pymol_imported:
        # Initialize PyMOL in headless mode
        try:
            pymol.finish_launching(['pymol', '-qc'])
        except Exception as e:
            print(f"ERROR: Failed to initialize PyMOL in headless mode: {str(e)}")
            pymol_imported = False  # Fall back to script generation

    for state_num, structures in state_structures.items():
        print(f"Processing state {state_num + 1} with attention coloring...")

        state_dir = os.path.join(save_dir, f"state_{state_num + 1}_attention")
        img_dir = os.path.join(state_dir, "attention_images")
        os.makedirs(img_dir, exist_ok=True)

        # Generate script for this state
        script_path = generate_pymol_script(state_num, structures, state_dir)
        pymol_scripts.append(script_path)

        # If PyMOL API is available, execute the visualization
        if pymol_imported:
            try:
                # Initialize for each state
                cmd.reinitialize()

                # Set up visualization parameters
                cmd.bg_color("white")
                cmd.set("ray_opaque_background", "off")
                cmd.set("cartoon_fancy_helices", 1)
                cmd.set("cartoon_transparency", 0)
                cmd.set("ray_shadows", 0)

                # Load existing structures and apply attention coloring
                for i, pdb_file in enumerate(structures):
                    if i == 0:
                        opacity = 1.0  # First structure fully opaque
                    else:
                        # Exponential decay starting from 0.7
                        decay_rate = 1.5  # Adjust this value to control decay speed
                        opacity = 0.3 * np.exp(-decay_rate * (i - 1))
                    name = f"state_{state_num + 1}_rank_{i}"

                    # Load structure
                    cmd.load(pdb_file, name)
                    cmd.show_as("cartoon", name)
                    cmd.set("transparency", 1 - opacity, name)

                    # Apply attention values as B-factors
                    for res_idx, attention in enumerate(state_residue_attention[state_num]):
                        b_factor = attention * 100
                        cmd.alter(f"{name} and resi {res_idx + 1}", f"b={b_factor}")

                    # Color by B-factor (attention values)
                    cmd.spectrum("b", "blue_white_red", name)

                    if i > 0:
                        cmd.align(name, f"state_{state_num + 1}_rank_0")

                # Save combined state
                combined_pdb = os.path.join(state_dir, f"{protein_name}_state_{state_num + 1}_attention_ensemble.pdb")
                cmd.save(combined_pdb, f"state_{state_num + 1}_rank_*")

                # Generate images from different angles
                for view_name, (x, y, z) in views.items():
                    cmd.reset()
                    cmd.rotate('x', x)
                    cmd.rotate('y', y)
                    cmd.rotate('z', z)
                    cmd.center()
                    cmd.zoom()

                    img_file = os.path.join(img_dir, f"{protein_name}_state_{state_num + 1}_{view_name}_attention.png")
                    cmd.ray(1200, 1200)
                    cmd.png(img_file, dpi=300, ray=1)
                    print(f"Saved {view_name} attention view to {img_file}")

                cmd.delete("all")

            except Exception as e:
                print(f"Error processing state {state_num + 1} with PyMOL API: {str(e)}")
                print(f"You can run the generated script manually: {script_path}")
        else:
            print(f"Generated PyMOL script for state {state_num + 1}: {script_path}")

    # Clean up PyMOL session if used
    if pymol_imported:
        try:
            cmd.delete("all")
            cmd.reinitialize()
        except:
            pass

    # Create a combined script that runs all state scripts
    if not pymol_imported:
        master_script_path = os.path.join(save_dir, f"{protein_name}_run_all_attention_visualizations.pml")
        with open(master_script_path, 'w') as master_script:
            master_script.write("# Master script to run all state attention visualizations\n\n")
            for script_path in pymol_scripts:
                master_script.write(f"@{os.path.abspath(script_path)}\n")

        print(f"Created master PyMOL script: {master_script_path}")
        print("To visualize all states, run PyMOL and execute:")
        print(f"    @{os.path.abspath(master_script_path)}")


def plot_state_network(
        probs: np.ndarray,
        state_structures: dict,
        save_dir: str,
        protein_name: str,
        lag_time: int = 1,
        stride: int = 1,
        timestep: float = 0.001
):
    """
    Create a network plot showing transitions between non-empty states with representative structures.

    Parameters
    ----------
    probs : np.ndarray
        State probability trajectory with shape [n_frames, n_states] from analyze_vampnet_outputs
    state_structures : dict
        Dictionary mapping state numbers to lists of PDB file paths
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein
    lag_time : int, optional
        Lag time for transition matrix calculation in ns
    stride : int, optional
        Stride used when extracting frames from trajectory
    timestep : float, optional
        Trajectory timestep in ns
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        # Use existing function to calculate transition matrices
        trans_matrix, trans_matrix_no_self = calculate_transition_matrices(
            probs=probs,
            lag_time=lag_time,
            stride=stride,
            timestep=timestep
        )

        # Calculate state populations from probabilities
        states = np.argmax(probs, axis=1)
        unique, counts = np.unique(states, return_counts=True)
        n_states = probs.shape[1]
        avg_state_pops = np.zeros(n_states)
        avg_state_pops[unique] = counts / len(states)

        # Identify non-empty states (those with structures or populations)
        non_empty_states = set()
        for i, pop in enumerate(avg_state_pops):
            if pop > 0 and i in state_structures and len(state_structures[i]) > 0:
                non_empty_states.add(i)

        # If all states are empty, warn and return
        if not non_empty_states:
            print("Warning: No non-empty states found. Cannot create network plot.")
            return

        # Create mapping from old state indices to new contiguous indices
        non_empty_states = sorted(list(non_empty_states))
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(non_empty_states)}

        # Create reduced transition matrix with only non-empty states
        n_active_states = len(non_empty_states)
        reduced_trans_matrix = np.zeros((n_active_states, n_active_states))
        reduced_trans_matrix_no_self = np.zeros((n_active_states, n_active_states))
        reduced_pops = np.zeros(n_active_states)

        # Fill reduced matrices and populations
        for i, old_i in enumerate(non_empty_states):
            reduced_pops[i] = avg_state_pops[old_i]
            for j, old_j in enumerate(non_empty_states):
                reduced_trans_matrix[i, j] = trans_matrix[old_i, old_j]
                reduced_trans_matrix_no_self[i, j] = trans_matrix_no_self[old_i, old_j]

        # Create figure
        fig, ax = plt.subplots(figsize=(24, 24))

        # Setup node positions with larger radius
        angles = np.linspace(0, 2 * np.pi, n_active_states, endpoint=False)
        radius = 2.0  # Increased radius
        pos = {i: (radius * np.cos(angle), radius * np.sin(angle))
               for i, angle in enumerate(angles)}

        # Draw transitions with separate curves for forward/backward
        transition_labels = {}
        for i in range(n_active_states):
            for j in range(n_active_states):
                if i != j and reduced_trans_matrix_no_self[i, j] > 0:
                    try:
                        # Forward transitions (blue, outer curve)
                        if i < j:
                            color = 'blue'
                            rad = -0.3
                        # Backward transitions (red, inner curve)
                        else:
                            color = 'red'
                            rad = -0.3

                        # Draw thicker arrows based on probability
                        arrow = ax.annotate("",
                                            xy=pos[j], xycoords='data',
                                            xytext=pos[i], textcoords='data',
                                            arrowprops=dict(arrowstyle="-|>",
                                                            connectionstyle=f"arc3,rad={rad}",
                                                            color=color,
                                                            lw=4 * reduced_trans_matrix_no_self[i, j] + 2,
                                                            # Thicker arrows
                                                            alpha=0.7,
                                                            mutation_scale=20))  # Controls arrowhead size

                        # Store transition information
                        state_pair = tuple(sorted([i, j]))
                        if state_pair not in transition_labels:
                            transition_labels[state_pair] = []

                        # Use original state numbers for labels
                        orig_i = non_empty_states[i]
                        orig_j = non_empty_states[j]
                        transition_labels[state_pair].append({
                            'text': f'S{orig_i + 1}→S{orig_j + 1}: {reduced_trans_matrix_no_self[i, j]:.2f}',
                            'color': color
                        })
                    except Exception as e:
                        print(f"Warning: Error drawing transition from state {i} to {j}: {str(e)}")

        # Add labels for all transitions
        for (i, j), labels in transition_labels.items():
            try:
                # Calculate middle point
                mid_x = (pos[i][0] + pos[j][0]) / 2
                mid_y = (pos[i][1] + pos[j][1]) / 2

                # Calculate perpendicular offset
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                angle = np.arctan2(dy, dx)
                perp_angle = angle + np.pi / 2

                offset = 0.2
                offset_x = offset * np.cos(perp_angle)
                offset_y = offset * np.sin(perp_angle)

                # Stack labels vertically
                for idx, label in enumerate(labels):
                    vertical_spacing = 0.15  # Adjust this value to control vertical spacing
                    y_offset = vertical_spacing * (idx - (len(labels) - 1) / 2)

                    ax.text(mid_x + offset_x,
                            mid_y + offset_y + y_offset,
                            label['text'],
                            ha='center', va='center',
                            color=label['color'],
                            bbox=dict(facecolor='white', alpha=0.7),
                            fontsize=12)
            except Exception as e:
                print(f"Warning: Error adding transition label: {str(e)}")

        # Draw nodes and add structure images
        for i in range(n_active_states):
            orig_i = non_empty_states[i]

            try:
                # Load and display structure image
                img_path = os.path.join(save_dir,
                                        f"state_{orig_i + 1}/images/{protein_name}_state_{orig_i + 1}_iso.png")
                if os.path.exists(img_path):
                    img = imread(img_path)
                    img_size = 0.8  # Increased image size
                    ax.imshow(img,
                              extent=[pos[i][0] - img_size / 2, pos[i][0] + img_size / 2,
                                      pos[i][1] - img_size / 2, pos[i][1] + img_size / 2])
                else:
                    print(f"Warning: Image file not found: {img_path}")
                    # Draw a placeholder circle
                    circle = plt.Circle(pos[i], 0.4, fill=True, color='lightgray')
                    ax.add_patch(circle)

                    # Add state number to the placeholder
                    ax.text(pos[i][0], pos[i][1], f"S{orig_i + 1}",
                            ha='center', va='center', fontsize=14, fontweight='bold')

                # Add state label and population with larger font
                ax.text(pos[i][0], pos[i][1] - 0.5,
                        f'State {orig_i + 1}\n{reduced_pops[i]:.1%}',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7),
                        fontsize=14)
            except Exception as e:
                print(f"Warning: Error adding state {orig_i + 1} to plot: {str(e)}")

        # Add legend with larger font
        ax.plot([], [], color='blue', label='Forward transitions', linewidth=4)
        ax.plot([], [], color='red', label='Backward transitions', linewidth=4)
        ax.legend(loc='upper right', fontsize=12)

        # Increase plot bounds
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
        plt.axis('equal')
        plt.axis('off')
        plt.title(f'State Transition Network - {protein_name} (Non-empty States Only)', fontsize=16)

        # Save plot with suffix indicating lag time
        suffix = f"lag{lag_time}"
        plot_path = os.path.join(save_dir, f"{protein_name}_state_network_{suffix}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved state network plot to: {plot_path} with {n_active_states} non-empty states.")
        print(f"Original states included: {', '.join([f'S{s + 1}' for s in non_empty_states])}")

        # Return information about the network
        return {
            'plot_path': plot_path,
            'transition_matrix': reduced_trans_matrix,
            'transition_matrix_no_self': reduced_trans_matrix_no_self,
            'state_populations': reduced_pops,
            'non_empty_states': non_empty_states
        }

    except Exception as e:
        print(f"Error creating state network plot: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_state_populations_newrec(probs: List[np.ndarray],
                           save_dir: str,
                           protein_name: str):
    """
    Plot state populations across all trajectories.

    Parameters
    ----------
    probs : List[np.ndarray]
        List of state probability trajectories
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein for file naming
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get number of states from first trajectory
    n_states = probs[0].shape[1]

    # Calculate state populations across all trajectories
    all_probs = np.vstack(probs)
    state_assignments = np.argmax(all_probs, axis=1)
    population_counts = np.bincount(state_assignments, minlength=n_states)
    populations = population_counts / len(state_assignments)

    # Plot populations
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot with tab10 colormap colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_states]
    bars = ax.bar(range(1, n_states + 1), populations, color=colors)

    # Add value labels on top of bars
    for bar, pop in zip(bars, populations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{pop:.3f}', ha='center', va='bottom')

    # Customize plot
    ax.set_xlabel('State')
    ax.set_ylabel('Population')
    ax.set_title(f'State Populations - {protein_name}')
    ax.set_xticks(range(1, n_states + 1))
    ax.set_xticklabels([f'State {i + 1}' for i in range(n_states)])
    ax.set_ylim(0, max(populations) * 1.15)  # Add space for value labels

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save plot
    plot_path = os.path.join(save_dir, f"{protein_name}_state_populations.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save to CSV
    csv_path = os.path.join(save_dir, f"{protein_name}_state_populations.csv")
    pd.DataFrame({
        'State': [f'State {i + 1}' for i in range(n_states)],
        'Population': populations
    }).to_csv(csv_path, index=False)

    print(f"Saved state populations plot to: {plot_path}")
    print(f"Saved state populations to: {csv_path}")

    return populations


def plot_state_evolution_newrec(probs: List[np.ndarray],
                         save_dir: str,
                         protein_name: str,
                         timestep: float = 1.0,
                         time_unit: str = 'ns',
                         max_frames_per_plot: int = 5000):
    """
    Plot state probability evolution over time.

    Parameters
    ----------
    probs : List[np.ndarray]
        List of state probability trajectories
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein for file naming
    timestep : float, optional
        Time between frames in time_unit
    time_unit : str, optional
        Time unit for x-axis
    max_frames_per_plot : int, optional
        Maximum number of frames to show in a single plot
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get number of states from first trajectory
    n_states = probs[0].shape[1]

    # Get colors from tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_states]

    # Plot each trajectory separately
    for traj_idx, prob in enumerate(probs):
        n_frames = prob.shape[0]

        # Split into chunks if too many frames
        n_chunks = max(1, int(np.ceil(n_frames / max_frames_per_plot)))

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * max_frames_per_plot
            end_idx = min((chunk_idx + 1) * max_frames_per_plot, n_frames)

            chunk_prob = prob[start_idx:end_idx]
            chunk_frames = end_idx - start_idx

            # Create time values
            time_values = np.arange(start_idx, end_idx) * timestep

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot each state's probability
            for state_idx in range(n_states):
                ax.plot(time_values, chunk_prob[:, state_idx],
                        label=f'State {state_idx + 1}',
                        color=colors[state_idx], linewidth=2)

            # Customize plot
            ax.set_xlabel(f'Time ({time_unit})')
            ax.set_ylabel('State Probability')

            if n_chunks > 1:
                title = f'State Probability Evolution - {protein_name}\nTrajectory {traj_idx + 1}, Frames {start_idx}-{end_idx - 1}'
            else:
                title = f'State Probability Evolution - {protein_name}\nTrajectory {traj_idx + 1}'

            ax.set_title(title)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add tight layout
            plt.tight_layout()

            # Save plot
            if n_chunks > 1:
                plot_path = os.path.join(save_dir,
                                         f"{protein_name}_traj{traj_idx + 1}_chunk{chunk_idx + 1}_state_evolution.png")
            else:
                plot_path = os.path.join(save_dir, f"{protein_name}_traj{traj_idx + 1}_state_evolution.png")

            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved state evolution plot to: {plot_path}")

    return None
