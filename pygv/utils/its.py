import matplotlib.pyplot as plt
import os
import numpy as np
from pygv.utils.ck import estimate_koopman_op


def get_its(
        probs: np.ndarray,
        lag_times_ns: list,
        stride: int = 1,
        timestep: float = 0.001  # trajectory timestep in ns
):
    """
    Calculate implied timescales from state probability data at multiple lag times.

    Parameters
    ----------
    probs : np.ndarray
        State probability trajectory with shape [n_frames, n_states]
    lag_times_ns : list
        List of lag times in nanoseconds
    stride : int, optional
        Stride used when extracting frames from trajectory
    timestep : float, optional
        Trajectory timestep in nanoseconds

    Returns
    -------
    tuple
        (its_array, lag_times_ns)
        - its_array: Implied timescales array of shape (n_states-1, n_lags)
        - lag_times_ns: The actual lag times used in nanoseconds
    """
    # Get dimensions
    n_states = probs.shape[1]

    # Prepare arrays for storing results
    its_array = np.zeros((n_states - 1, len(lag_times_ns)))

    # Effective timestep (accounting for stride)
    effective_timestep = timestep * stride

    print(f"Calculating ITS for {n_states} states with {len(lag_times_ns)} lag times")
    print(f"Trajectory timestep: {timestep} ns, Stride: {stride}, Effective timestep: {effective_timestep} ns")

    # Calculate ITS for each lag time
    for t, lag_ns in enumerate(lag_times_ns):
        # Convert lag time to frames
        lag_frames = int(round(lag_ns / effective_timestep))
        print(f"Lag time {lag_ns} ns corresponds to {lag_frames} frames")

        # Calculate Koopman operator
        koopman_op = estimate_koopman_op(probs, lag_frames)

        # Get eigenvalues and calculate implied timescales
        k_eigvals, _ = np.linalg.eig(np.real(koopman_op))
        k_eigvals = np.sort(np.abs(k_eigvals))[::-1]  # Sort in descending order

        # Exclude the stationary eigenvector (eigenvalue 1)
        k_eigvals = k_eigvals[1:]

        # Calculate implied timescales: -τ/ln|λ|
        its = -lag_ns / np.log(k_eigvals)

        # Store in results array
        its_array[:, t] = its

    return its_array, lag_times_ns


def plot_its(
        its: np.ndarray,
        lag_times_ns: list,
        save_dir: str,
        protein_name: str,
        ylog: bool = True,
        ylim: tuple = None,
        n_states_to_plot: int = None,
        figsize: tuple = (10, 7)
):
    """
    Plot implied timescales (ITS) as a function of lag time.

    Parameters
    ----------
    its : np.ndarray
        Array of shape (n_states-1, n_lags) containing implied timescales in nanoseconds
    lag_times_ns : list
        Array or list of lag times in nanoseconds
    save_dir : str
        Directory to save the plot
    protein_name : str
        Name of the protein for plot title and filename
    ylog : bool, optional
        Whether to use logarithmic scale for y-axis, default=True
    ylim : tuple, optional
        Y-axis limits (min, max) in nanoseconds, default=None (auto)
    n_states_to_plot : int, optional
        Number of slowest processes to plot, default=None (all)
    figsize : tuple, optional
        Figure size as (width, height) in inches, default=(10, 7)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Convert lag_times to array if needed
    lag_times_ns = np.array(lag_times_ns)

    # Default to plotting all states if not specified
    if n_states_to_plot is None:
        n_states_to_plot = its.shape[0]
    else:
        n_states_to_plot = min(n_states_to_plot, its.shape[0])

    # Create figure
    plt.figure(figsize=figsize)

    # Create colormap for processes
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_states_to_plot))

    # Plot implied timescales
    for i in range(n_states_to_plot):
        process_num = i + 2  # Add 2 since we exclude the stationary process (process 1)
        plt.plot(lag_times_ns, its[i], 'o-', color=colors[i], linewidth=2,
                 label=f"Process {process_num}")

    # Add identity line for reference
    plt.plot(lag_times_ns, lag_times_ns, 'k--', linewidth=1, label='$t_i = \\tau$')

    # Fill region below identity line
    plt.fill_between(lag_times_ns, lag_times_ns, 0.001, alpha=0.1, color='gray')

    # Set axis labels and title
    plt.xlabel('Lag Time (ns)', fontsize=14)
    plt.ylabel('Implied Timescales (ns)', fontsize=14)
    plt.title(f'Implied Timescales - {protein_name}', fontsize=16)

    # Set y-axis limits if provided
    if ylim:
        plt.ylim(ylim)

    # Set y-axis scale
    if ylog:
        plt.yscale('log')
        if not plt.gca().get_yscale() == 'log':
            print("Warning: Failed to set log scale for y-axis")

    # Set x-axis scale to match y-axis for identity line clarity
    #if ylog:
    #    plt.xscale('log')

    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(loc='best', fontsize=12)

    # Ensure proper spacing
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(save_dir, f"{protein_name}_implied_timescales.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved implied timescales plot to: {plot_path}")

    # Clean up
    plt.close()

    return plot_path


def analyze_implied_timescales(
        probs: np.ndarray,
        save_dir: str,
        protein_name: str,
        lag_times_ns: list = [1, 2, 5, 10, 20, 50],
        stride: int = 1,
        timestep: float = 0.001,  # trajectory timestep in ns
        ylog: bool = True,
        n_states_to_plot: int = None
):
    """
    Analyze and plot implied timescales from VAMPNet probability data.

    Parameters
    ----------
    probs : np.ndarray
        State probability trajectory with shape [n_frames, n_states]
    save_dir : str
        Directory to save results
    protein_name : str
        Name of the protein for file naming
    lag_times_ns : list, optional
        List of lag times to analyze in nanoseconds
    stride : int, optional
        Stride used when extracting frames from trajectory
    timestep : float, optional
        Trajectory timestep in nanoseconds
    ylog : bool, optional
        Whether to use logarithmic scale for y-axis, default=True
    n_states_to_plot : int, optional
        Number of slowest processes to plot, default=None (all)

    Returns
    -------
    dict
        Dictionary with implied timescales data and plot path
    """
    # Create its directory if it doesn't exist
    its_dir = os.path.join(save_dir, 'implied_timescales')
    os.makedirs(its_dir, exist_ok=True)

    print(f"Analyzing implied timescales for {protein_name} with lag times: {lag_times_ns} ns")

    try:
        # Calculate implied timescales
        its_array, lag_times = get_its(
            probs=probs,
            lag_times_ns=lag_times_ns,
            stride=stride,
            timestep=timestep
        )

        # Save raw ITS data
        its_file = os.path.join(its_dir, f"{protein_name}_its_data.npz")
        np.savez(its_file, its=its_array, lag_times=lag_times)
        print(f"Saved ITS data to {its_file}")

        # Plot implied timescales
        plot_path = plot_its(
            its=its_array,
            lag_times_ns=lag_times,
            save_dir=its_dir,
            protein_name=protein_name,
            ylog=ylog,
            n_states_to_plot=n_states_to_plot
        )

        # Create summary of results
        n_processes = its_array.shape[0]
        mean_its = np.mean(its_array, axis=1)
        max_its = np.max(its_array, axis=1)

        summary_file = os.path.join(its_dir, f"{protein_name}_its_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Implied Timescales Analysis for {protein_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of processes analyzed: {n_processes}\n")
            f.write(f"Lag times analyzed: {lag_times} ns\n\n")
            f.write("Process Summary:\n")

            for i in range(n_processes):
                process_num = i + 2  # Add 2 since we exclude the stationary process
                f.write(f"Process {process_num}: Mean ITS = {mean_its[i]:.2f} ns, Max ITS = {max_its[i]:.2f} ns\n")

            f.write("\nDetailed ITS for each lag time:\n")
            f.write("\nLag Time (ns) | " + " | ".join([f"Process {i + 2}" for i in range(min(5, n_processes))]) + "\n")
            f.write("-" * (12 + 12 * min(5, n_processes)) + "\n")

            for t, lag in enumerate(lag_times):
                row = f"{lag:11.2f} | "
                row += " | ".join([f"{its_array[i, t]:9.2f}" for i in range(min(5, n_processes))])
                f.write(row + "\n")

        print(f"Saved ITS summary to {summary_file}")

        return {
            'its_array': its_array,
            'lag_times': lag_times,
            'plot_path': plot_path,
            'summary_path': summary_file
        }

    except Exception as e:
        print(f"Error analyzing implied timescales: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
