import matplotlib.pyplot as plt
import numpy as np


def plot_vamp_scores(scores, save_path=None, show_plot=True, smoothing=None, title="VAMPNet Training Performance"):
    """
    Plot the VAMP score curve from training.

    Parameters:
    -----------
    scores : list
        List of VAMP scores across epochs
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot
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

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, ax
