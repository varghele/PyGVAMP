import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm
import time


def get_nbrs(all_coords, num_neighbors=10):
    '''
    Find k-nearest neighbors for each atom in the trajectory.

    Inputs:
    - all_coords: a trajectory with shape [T, num_atoms, dim]
        T: number of steps
        dim: number of dimensions (3 coordinates)
    - num_neighbors: number of neighbors to find for each atom

    Returns:
    - all_dists: distances to neighbors [n_frames, n_atoms, num_neighbors]
    - all_inds: indices of neighbors [n_frames, n_atoms, num_neighbors]
    '''
    k_nbr = num_neighbors + 1  # +1 to account for self (which will be removed)

    if type(all_coords) == list:
        all_dists = []
        all_inds = []
        for i in range(len(all_coords)):
            dists = []
            inds = []
            tmp_coords = all_coords[i]
            for j in tqdm(range(len(tmp_coords)), desc=f"Processing trajectory {i + 1}/{len(all_coords)}"):
                tree = BallTree(tmp_coords[j], leaf_size=3)
                dist, ind = tree.query(tmp_coords[j], k=k_nbr)
                dists.append(dist[:, 1:])  # Remove self
                inds.append(ind[:, 1:])  # Remove self

            dists = np.array(dists)
            inds = np.array(inds)
            all_dists.append(dists)
            all_inds.append(inds)
    else:
        all_inds = []
        all_dists = []
        dists = []
        inds = []
        for i in tqdm(range(len(all_coords)), desc="Processing frames"):
            tree = BallTree(all_coords[i], leaf_size=3)
            dist, ind = tree.query(all_coords[i], k=k_nbr)
            dists.append(dist[:, 1:])  # Remove self
            inds.append(ind[:, 1:])  # Remove self
        all_dists = np.array(dists)
        all_inds = np.array(inds)

    return all_dists, all_inds


def generate_test_data(n_frames=5, n_atoms=50, noise_level=0.1):
    """Generate synthetic test data for neighbor finding."""
    print(f"Generating {n_frames} frames with {n_atoms} atoms each...")

    # Create a base structure (here a simple cubic lattice)
    n_per_dim = int(np.ceil(np.power(n_atoms, 1 / 3)))
    x = np.linspace(0, 10, n_per_dim)
    y = np.linspace(0, 10, n_per_dim)
    z = np.linspace(0, 10, n_per_dim)

    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z)

    # Flatten and take only what we need
    base_coords = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T[:n_atoms]

    # Create frames with random perturbations
    all_coords = np.zeros((n_frames, n_atoms, 3))

    for i in range(n_frames):
        # Add random noise to positions
        noise = np.random.normal(0, noise_level, size=(n_atoms, 3))
        all_coords[i] = base_coords + noise

    print(f"Created dataset with shape: {all_coords.shape}")
    return all_coords


def test_single_trajectory():
    """Test get_nbrs function with a single trajectory."""
    print("\n=== Testing get_nbrs with a single trajectory ===")

    # Generate test data
    num_neighbors = 10
    coords = generate_test_data(n_frames=3, n_atoms=50)

    # Time the function
    start_time = time.time()
    dists, inds = get_nbrs(coords, num_neighbors=num_neighbors)
    elapsed_time = time.time() - start_time

    # Print output information
    print(f"\nFunction completed in {elapsed_time:.4f} seconds")
    print(f"Input shape: {coords.shape}")
    print(f"Output distances shape: {dists.shape}")
    print(f"Output indices shape: {inds.shape}")

    # Print a sample of the results
    frame_idx = 0
    atom_idx = 0
    print(f"\nSample results for frame {frame_idx}, atom {atom_idx}:")
    print(f"Neighbor indices: {inds[frame_idx, atom_idx]}")
    print(f"Neighbor distances: {dists[frame_idx, atom_idx]}")


def test_multiple_trajectories():
    """Test get_nbrs function with multiple trajectories."""
    print("\n=== Testing get_nbrs with multiple trajectories ===")

    # Generate two trajectories with different motions
    n_frames = 2
    n_atoms = 50
    num_neighbors = 10

    # Trajectory 1: Simple cubic lattice with noise
    traj1 = generate_test_data(n_frames, n_atoms, noise_level=0.1)

    # Trajectory 2: Same atoms but different movement pattern
    traj2 = generate_test_data(n_frames, n_atoms, noise_level=0.2)

    # Put both trajectories in a list
    all_trajs = [traj1, traj2]

    # Find neighbors for both trajectories
    print("Finding neighbors for multiple trajectories...")
    start_time = time.time()
    dists, inds = get_nbrs(all_trajs, num_neighbors=num_neighbors)
    elapsed_time = time.time() - start_time

    # Print output shape
    print(f"\nFunction completed in {elapsed_time:.4f} seconds")
    print(f"Input shapes: {[traj.shape for traj in all_trajs]}")
    print(f"Output distances shapes: {[d.shape for d in dists]}")
    print(f"Output indices shapes: {[i.shape for i in inds]}")

    # Print sample results
    traj_idx = 0
    frame_idx = 0
    atom_idx = 0
    print(f"\nSample results for trajectory {traj_idx}, frame {frame_idx}, atom {atom_idx}:")
    print(f"Neighbor indices: {inds[traj_idx][frame_idx, atom_idx]}")
    print(f"Neighbor distances: {dists[traj_idx][frame_idx, atom_idx]}")


def main():
    """Run tests on the get_nbrs function."""
    print("=== Testing get_nbrs function ===")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Test with a single trajectory
    test_single_trajectory()

    # Test with multiple trajectories
    test_multiple_trajectories()

    print("\n=== Tests completed successfully ===")


if __name__ == "__main__":
    main()
