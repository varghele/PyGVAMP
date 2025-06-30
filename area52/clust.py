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
    cluster_labels, kernel_matrix, nspdk = cluster_md_trajectory_pyg_nspdk(
        trajectory_file=trajectory_file,
        topology_file=topology_file,
        n_clusters=8,
        selection='name CA',
        r=2,  # Smaller radius for efficiency
        d=3,  # Smaller max distance
        device='cuda',
        batch_size=8  # Small batch size for memory efficiency
    )


    # Save results
    np.save("pyg_nspdk_cluster_labels.npy", cluster_labels)
    np.save("pyg_nspdk_kernel_matrix.npy", kernel_matrix)

    print("PyTorch Geometric NSPDK clustering completed!")

    return cluster_labels, kernel_matrix


if __name__ == "__main__":
    main()