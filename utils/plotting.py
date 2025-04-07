import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from matplotlib.colors import ListedColormap


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


def calculate_transition_matrices(probs: List[np.ndarray], lag_time: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate transition probability matrices from state probabilities.

    Parameters
    ----------
    probs : List[np.ndarray]
        List of state probability trajectories, each with shape [n_frames, n_states]
    lag_time : int, optional
        Lag time for transition matrix calculation

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (transition_matrix, transition_matrix_without_self_transitions)
    """
    # Extract dimensions
    n_traj = len(probs)
    n_states = probs[0].shape[1]

    # Initialize transition count matrices
    transition_counts = np.zeros((n_states, n_states))

    # Process each trajectory
    for traj in probs:
        n_frames = traj.shape[0]

        # Skip if trajectory is too short for the lag time
        if n_frames <= lag_time:
            print(f"Warning: Trajectory with {n_frames} frames is too short for lag time {lag_time}. Skipping.")
            continue

        # Hard-assign states based on maximum probability
        state_assignments = np.argmax(traj, axis=1)

        # Count transitions with the specified lag time
        for t in range(n_frames - lag_time):
            from_state = state_assignments[t]
            to_state = state_assignments[t + lag_time]
            transition_counts[from_state, to_state] += 1

    # Calculate transition probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)

    # Avoid division by zero
    row_sums[row_sums == 0] = 1

    # Normalize to get transition probabilities
    transition_matrix = transition_counts / row_sums

    # Create a copy for non-self transitions
    transition_matrix_no_self = transition_matrix.copy()

    # Set diagonal to zero for non-self transition matrix
    np.fill_diagonal(transition_matrix_no_self, 0)

    # Re-normalize non-self transition matrix
    row_sums_no_self = transition_matrix_no_self.sum(axis=1, keepdims=True)
    row_sums_no_self[row_sums_no_self == 0] = 1  # Avoid division by zero
    transition_matrix_no_self = transition_matrix_no_self / row_sums_no_self

    return transition_matrix, transition_matrix_no_self


def plot_transition_probabilities(probs: List[np.ndarray],
                                  save_dir: str,
                                  protein_name: str,
                                  lag_time: int = 1,
                                  cmap_name: str = 'YlOrRd',
                                  fig_size: Tuple[int, int] = (10, 8),
                                  font_size: int = 10):
    """
    Plot both standard and non-self transition probability matrices.

    Parameters
    ----------
    probs : List[np.ndarray]
        List of state probability trajectories
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein for file naming
    lag_time : int, optional
        Lag time for transition matrix calculation
    cmap_name : str, optional
        Matplotlib colormap name for the heatmaps
    fig_size : Tuple[int, int], optional
        Figure size in inches
    font_size : int, optional
        Font size for annotations
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Calculate transition matrices
    trans_matrix, trans_matrix_no_self = calculate_transition_matrices(probs, lag_time)

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
        title = f"State Transition Probabilities - Lag {lag_time}"
        if suffix == "no_self":
            title = f"Non-Self {title}"
        ax.set_title(f'{title}\n{protein_name}')

        # Adjust layout
        plt.tight_layout()

        # Save outputs
        plot_path = os.path.join(save_dir, f"{protein_name}_transition_matrix_{suffix}_lag{lag_time}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save to CSV
        csv_path = os.path.join(save_dir, f"{protein_name}_transition_matrix_{suffix}_lag{lag_time}.csv")
        pd.DataFrame(matrix, index=state_labels, columns=state_labels).to_csv(csv_path)

        print(f"Saved {suffix} transition matrix plot to: {plot_path}")
        print(f"Saved {suffix} transition matrix to: {csv_path}")

    return trans_matrix, trans_matrix_no_self


def plot_state_populations(probs: List[np.ndarray],
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


def plot_state_evolution(probs: List[np.ndarray],
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
