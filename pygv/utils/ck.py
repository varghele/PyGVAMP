import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional


def estimate_koopman_op(traj: Union[np.ndarray, List[np.ndarray]], lag: int) -> np.ndarray:
    """
    Estimate the Koopman operator from trajectory data.

    Parameters
    ----------
    traj : Union[np.ndarray, List[np.ndarray]]
        Trajectory data with shape [n_frames, n_states] or list of such arrays
    lag : int
        Lag time for Koopman operator estimation

    Returns
    -------
    np.ndarray
        Estimated Koopman operator with shape [n_states, n_states]
    """
    if isinstance(traj, list):
        # Handle list of trajectories
        n_states = traj[0].shape[1]
        C00 = np.zeros((n_states, n_states))
        C0t = np.zeros((n_states, n_states))

        for t in traj:
            # Skip trajectories that are too short
            if len(t) <= lag:
                continue

            # Calculate correlation matrices
            x_0 = t[:-lag]
            x_t = t[lag:]

            C00 += x_0.T @ x_0
            C0t += x_0.T @ x_t
    else:
        # Handle single trajectory
        n_states = traj.shape[1]
        x_0 = traj[:-lag]
        x_t = traj[lag:]

        C00 = x_0.T @ x_0
        C0t = x_0.T @ x_t

    # Calculate Koopman operator
    # Use pseudoinverse for numerical stability
    koopman = np.linalg.pinv(C00) @ C0t

    return koopman


def get_ck_test(traj: Union[np.ndarray, List[np.ndarray]], steps: int, tau: int) -> List[np.ndarray]:
    """
    Perform Chapman-Kolmogorov test comparing predicted vs estimated dynamics.

    Parameters
    ----------
    traj : Union[np.ndarray, List[np.ndarray]]
        Trajectory data with probabilities (output from VAMPNet classifier)
    steps : int
        Number of prediction steps
    tau : int
        Lag time between steps

    Returns
    -------
    List[np.ndarray]
        [predicted, estimated] arrays of shape (n_states, n_states, steps)
    """
    if isinstance(traj, list):
        n_states = traj[0].shape[1]
    else:
        n_states = traj.shape[1]

    predicted = np.zeros((n_states, n_states, steps))
    estimated = np.zeros((n_states, n_states, steps))

    # Identity matrix for initial condition (t=0)
    predicted[:, :, 0] = np.identity(n_states)
    estimated[:, :, 0] = np.identity(n_states)

    # Calculate base Koopman operator at lag time tau
    base_koopman = estimate_koopman_op(traj, tau)

    # For each initial state
    for i in range(n_states):
        # Create unit vector for this state
        vector = np.zeros(n_states)
        vector[i] = 1.0

        for n in range(1, steps):
            # For predicted: use Chapman-Kolmogorov equation (power of Koopman)
            koop_pred = np.linalg.matrix_power(base_koopman, n)

            # For estimated: directly estimate Koopman at lag time n*tau
            koop_est = estimate_koopman_op(traj, tau * n)

            # Calculate probabilities
            predicted[i, :, n] = vector @ koop_pred
            estimated[i, :, n] = vector @ koop_est

    return [predicted, estimated]


def plot_ck_test(pred: np.ndarray, est: np.ndarray, steps: int, tau: int,
                 save_folder: str, lag_time_unit: str = 'frames',
                 filename: str = 'ck_test.png',
                 title: str = 'Chapman-Kolmogorov Test',
                 figsize: Optional[Tuple[int, int]] = None):
    """
    Plot Chapman-Kolmogorov test results.

    Parameters
    ----------
    pred : np.ndarray
        Predicted dynamics array from get_ck_test
    est : np.ndarray
        Estimated dynamics array from get_ck_test
    steps : int
        Number of prediction steps
    tau : int
        Lag time between steps
    save_folder : str
        Directory to save plot
    lag_time_unit : str, optional
        Unit for lag time (e.g., 'frames', 'ns'), by default 'frames'
    filename : str, optional
        Name of output file, by default 'ck_test.png'
    title : str, optional
        Title for the plot
    figsize : Tuple[int, int], optional
        Custom figure size
    """
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    n_states = pred.shape[0]

    # Calculate default figure size if not provided
    if figsize is None:
        figsize = (2.5 * n_states, 2 * n_states)

    # Create figure
    fig, axes = plt.subplots(n_states, n_states, sharex=True, sharey=True,
                             figsize=figsize, squeeze=False)

    # X-axis values
    x_values = np.arange(0, steps * tau, tau)

    # Plot each transition
    for i in range(n_states):
        for j in range(n_states):
            ax = axes[i, j]

            # Plot predicted (solid blue line)
            ax.plot(x_values, pred[i, j], color='blue', linestyle='-',
                    linewidth=2, label='Predicted')

            # Plot estimated (dashed red line)
            ax.plot(x_values, est[i, j], color='red', linestyle='--',
                    linewidth=2, label='Estimated')

            # Add title for each subplot
            ax.set_title(f'State {i + 1} → {j + 1}', fontsize='small')

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)

    # Set y-limits to ensure we see the full probability range
    for ax in axes.flatten():
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, steps * tau)

    # Set x-ticks to a reasonable number for all plots (not just the bottom row)
    x_ticks = np.round(np.linspace(0, steps * tau, min(4, steps)))
    x_tick_labels = [f"{tick:.1f}" for tick in x_ticks]

    for i in range(n_states):
        for j in range(n_states):
            ax = axes[i, j]
            ax.set_xticks(x_ticks)

            # Only show x labels on the bottom row
            if i == n_states - 1:
                ax.set_xticklabels(x_tick_labels, fontsize=10)
            else:
                ax.set_xticklabels([])

            # Set y-ticks with proper formatting
            y_ticks = [0.0, 0.5, 1.0]
            ax.set_yticks(y_ticks)

            # Only show y labels on the leftmost column
            if j == 0:
                ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=10)
            else:
                ax.set_yticklabels([])

    # Add labels to the outer axes
    for ax in axes[-1, :]:
        ax.set_xlabel(f'Lag time ({lag_time_unit})', fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel('Probability', fontsize=12)

    # Add legend to the top-right subplot
    axes[0, -1].legend(loc='upper right', fontsize='small')

    # Add overall title
    fig.suptitle(title, fontsize=16)

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    # Save figure
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chapman-Kolmogorov test plot saved to: {save_path}")

    return fig, axes


def run_ck_analysis(
        probs: np.ndarray,
        save_dir: str,
        protein_name: str,
        lag_times_ns: list = [1, 5, 10],  # lag times in nanoseconds
        steps: int = 5,
        stride: int = 1,
        timestep: float = 0.001  # trajectory timestep in ns
):
    """
    Run Chapman-Kolmogorov test analysis with multiple lag times in nanoseconds.

    Parameters
    ----------
    probs : np.ndarray
        State probability trajectory with shape [n_frames, n_states]
    save_dir : str
        Directory to save results
    protein_name : str
        Name of the protein for file naming
    lag_times_ns : list, optional
        List of lag times to test in nanoseconds, default [1, 5, 10]
    steps : int, optional
        Number of prediction steps, default 5
    stride : int, optional
        Stride used when extracting frames from trajectory
    timestep : float, optional
        Trajectory timestep in nanoseconds

    Returns
    -------
    dict
        Dictionary of test results for each lag time
    """
    # Create save folder
    ck_folder = os.path.join(save_dir, 'chapman_kolmogorov')
    os.makedirs(ck_folder, exist_ok=True)

    # Effective timestep in ns (accounting for stride)
    effective_timestep = timestep * stride

    print(f"Trajectory timestep: {timestep} ns")
    print(f"Stride: {stride}")
    print(f"Effective timestep: {effective_timestep} ns")

    results = {}

    # Run test for each lag time
    for lag_time_ns in lag_times_ns:
        print(f"\nRunning Chapman-Kolmogorov test with lag time {lag_time_ns} ns...")

        # Convert lag time from ns to frames
        lag_frames = int(round(lag_time_ns / effective_timestep))
        print(f"Lag time {lag_time_ns} ns corresponds to {lag_frames} frames")

        # Get test results using existing function
        predicted, estimated = get_ck_test(probs, steps, lag_frames)

        # Plot results with physical units
        fig, axes = plot_ck_test(
            pred=predicted,
            est=estimated,
            steps=steps,
            tau=lag_time_ns,
            save_folder=ck_folder,
            lag_time_unit='ns',  # Use nanoseconds as unit
            filename=f"{protein_name}_ck_test_lag{lag_time_ns}ns.png",
            title=f"Chapman-Kolmogorov Test - {protein_name}\nLag Time: {lag_time_ns} ns"
        )

        # Store results
        results[lag_time_ns] = {
            'predicted': predicted,
            'estimated': estimated,
            'lag_frames': lag_frames
        }

        # Calculate error metrics
        mse = np.mean((predicted - estimated) ** 2)
        mae = np.mean(np.abs(predicted - estimated))
        print(f"Lag time {lag_time_ns} ns: MSE = {mse:.4f}, MAE = {mae:.4f}")

    # Create comparison plot of MSE values for different lag times
    if len(results) > 1:
        # Calculate MSE values for each lag time
        mse_values = []
        for lag_time_ns in lag_times_ns:
            if lag_time_ns in results:
                mse = np.mean((results[lag_time_ns]['predicted'] - results[lag_time_ns]['estimated']) ** 2)
                mse_values.append(mse)
            else:
                mse_values.append(np.nan)

        # Create figure with higher resolution for professional appearance
        plt.figure(figsize=(12, 6), dpi=100)

        # Create colormap for bars - blue to red gradient depending on MSE value
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(mse_values)))

        # Create bar plot with nicer formatting
        bars = plt.bar(range(len(lag_times_ns)), mse_values,
                       color=colors, edgecolor='black', linewidth=0.5,
                       alpha=0.8)

        # Format y-axis with log scale and grid
        plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Make x-ticks sparser
        if len(lag_times_ns) > 6:
            # Show only a subset of x-ticks for readability when many lag times are present
            step = max(1, len(lag_times_ns) // 5)  # Show approximately 5 ticks
            tick_positions = range(0, len(lag_times_ns), step)
            tick_labels = [f"{lag_times_ns[i]}" for i in tick_positions]
            plt.xticks(tick_positions, tick_labels, fontsize=12)
        else:
            # Show all ticks if there are few enough
            plt.xticks(range(len(lag_times_ns)), [f"{lt}" for lt in lag_times_ns], fontsize=12)

        """# Add value labels on top of bars, formatted to appropriate precision
        for i, (bar, mse) in enumerate(zip(bars, mse_values)):
            if not np.isnan(mse):
                # Format numbers based on their magnitude
                if mse < 0.001:
                    val_text = f"{mse:.2e}"  # Scientific notation for very small values
                elif mse < 0.01:
                    val_text = f"{mse:.4f}"
                else:
                    val_text = f"{mse:.3f}"

                # Position the text above the bar
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.05,  # Position slightly above the bar
                    val_text,
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    rotation=0,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2')
                )"""

        # Add clear axis labels and title
        plt.ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
        plt.xlabel('Lag Time (ns)', fontsize=14, fontweight='bold')
        plt.title(f'Chapman-Kolmogorov Test Error Comparison\n{protein_name}',
                  fontsize=16, fontweight='bold')

        plt.tight_layout()

        # Save comparison plot
        comparison_path = os.path.join(ck_folder, f"{protein_name}_ck_test_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved lag time comparison plot to {comparison_path}")

    return results


# TODO: Delete
def run_ck_analysis_old(probs: List[np.ndarray],
                    save_dir: str,
                    protein_name: str,
                    tau_values: List[int] = [1, 5, 10],
                    steps: int = 5,
                    lag_time_unit: str = 'frames'):
    """
    Run Chapman-Kolmogorov test analysis with multiple lag times.

    Parameters
    ----------
    probs : List[np.ndarray]
        List of state probability trajectories
    save_dir : str
        Directory to save results
    protein_name : str
        Name of the protein for file naming
    tau_values : List[int], optional
        List of lag times to test, by default [1, 5, 10]
    steps : int, optional
        Number of prediction steps, by default 5
    lag_time_unit : str, optional
        Unit for lag time, by default 'frames'

    Returns
    -------
    dict
        Dictionary of test results for each lag time
    """
    # Create save folder
    ck_folder = os.path.join(save_dir, 'chapman_kolmogorov')
    os.makedirs(ck_folder, exist_ok=True)

    results = {}

    # Run test for each lag time
    for tau in tau_values:
        print(f"Running Chapman-Kolmogorov test with lag time {tau} {lag_time_unit}...")

        # Get test results
        predicted, estimated = get_ck_test(probs, steps, tau)

        # Plot results
        fig, axes = plot_ck_test(
            pred=predicted,
            est=estimated,
            steps=steps,
            tau=tau,
            save_folder=ck_folder,
            lag_time_unit=lag_time_unit,
            filename=f"{protein_name}_ck_test_tau{tau}.png",
            title=f"Chapman-Kolmogorov Test - {protein_name}\nLag Time: {tau} {lag_time_unit}"
        )

        # Store results
        results[tau] = {
            'predicted': predicted,
            'estimated': estimated
        }

        # Calculate error metrics
        mse = np.mean((predicted - estimated) ** 2)
        mae = np.mean(np.abs(predicted - estimated))
        print(f"Lag time {tau}: MSE = {mse:.4f}, MAE = {mae:.4f}")

    return results
