import numpy as np
import torch
from pygv.clustering.nspdk import cluster_md_trajectory_pyg_nspdk
import os

# Example usage
def main():
    """
    Example of using PyTorch Geometric-based NSPDK for MD trajectory clustering
    """
    # Parameters
    trajectory_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/r0/traj0001.xtc')  # Your trajectory file
    topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/prot.pdb')  # Your topology file

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Perform NSPDK clustering
    cluster_labels, kernel_matrix = cluster_md_trajectory_pyg_nspdk(
        trajectory_file=trajectory_file,
        topology_file=topology_file,
        n_clusters=5,
        selection='name CA',  # Or your specific selection
        max_radius=3,
        max_distance=4,
        device='cuda'
    )

    # Save results
    np.save("pyg_nspdk_cluster_labels.npy", cluster_labels)
    np.save("pyg_nspdk_kernel_matrix.npy", kernel_matrix)

    print("PyTorch Geometric NSPDK clustering completed!")

    return cluster_labels, kernel_matrix


if __name__ == "__main__":
    main()