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

            # Remove axis labels except for outer plots
            if i < n_states - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    # Set y-limits to ensure we see the full probability range
    for ax in axes.flatten():
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, steps * tau)

    # Set x-ticks to a reasonable number
    for ax in axes[-1, :]:
        ax.set_xticks(np.round(np.linspace(0, steps * tau, min(4, steps))))

    # Add labels to the outer axes
    for ax in axes[-1, :]:
        ax.set_xlabel(f'Lag time ({lag_time_unit})')
    for ax in axes[:, 0]:
        ax.set_ylabel('Probability')

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


def run_ck_analysis(probs: List[np.ndarray],
                    save_folder: str,
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
    save_folder : str
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
    ck_folder = os.path.join(save_folder, 'chapman_kolmogorov')
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
