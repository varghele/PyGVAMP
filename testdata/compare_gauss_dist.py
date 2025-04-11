import torch
import matplotlib.pyplot as plt
import numpy as np


# Implementation 1 - from your VAMPNet dataset
def compute_gaussian_expanded_distances_1(distances, distance_min, distance_max, gaussian_expansion_dim):
    """First implementation"""
    K = gaussian_expansion_dim
    d_range = distance_max - distance_min
    sigma = d_range / K

    # Filter out invalid distances (diagonals set to -1)
    valid_mask = distances >= 0

    # Reshape to prepare for broadcasting
    distances_reshaped = distances.reshape(-1, 1)  # [num_edges, 1]

    # Calculate μ_t values [1, K]
    mu_values = torch.linspace(distance_min, distance_max, K).view(1, -1)

    # Compute expanded features: exp(-(d_ij - μ_t)²/σ²)
    expanded_features = torch.zeros((distances_reshaped.shape[0], K),
                                    device=distances.device,
                                    dtype=torch.float32)

    # Apply computation only to valid distances
    valid_indices = torch.nonzero(valid_mask).squeeze()
    valid_distances = distances_reshaped[valid_indices]

    # Compute Gaussian expansion only for valid distances
    valid_expanded = torch.exp(-((valid_distances - mu_values) ** 2) / (sigma ** 2))

    # Place results back in the output tensor
    expanded_features[valid_indices] = valid_expanded

    return expanded_features


# Implementation 2 - GaussianDistance class
class GaussianDistance:
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax + step, step)
        self.num_features = len(self.filter)
        if var is None:
            var = step
        self.var = var

    def expand(self, distance):
        return torch.exp(-(torch.unsqueeze(distance, -1) - self.filter) ** 2 / self.var ** 2)


def compare_implementations():
    """Compare both Gaussian distance expansion implementations"""
    print("Comparing Gaussian distance expansion implementations...")

    # Test parameters
    distance_min = 0.0
    distance_max = 10.0

    # For implementation 1
    gaussian_expansion_dim = 8

    # For implementation 2
    # Calculate step to get the same number of filters
    step = (distance_max - distance_min) / (gaussian_expansion_dim - 1)
    var1 = (distance_max - distance_min) / gaussian_expansion_dim  # Impl 1 variance

    # Create a GaussianDistance instance
    gd = GaussianDistance(distance_min, distance_max, step, var=var1)

    # Generate test distances
    distances = torch.linspace(0, 10, 100)

    # Add some invalid distances (-1)
    test_distances = torch.cat([distances, torch.tensor([-1.0, -1.0, -1.0])])

    # Process with both implementations
    expanded1 = compute_gaussian_expanded_distances_1(
        test_distances, distance_min, distance_max, gaussian_expansion_dim
    )

    # For impl2, we need to filter out negative values
    valid_mask = test_distances >= 0
    valid_distances = test_distances[valid_mask]
    expanded2_valid = gd.expand(valid_distances)

    # Create full expanded2 including invalid entries
    expanded2 = torch.zeros((test_distances.shape[0], gd.num_features), dtype=torch.float32)
    expanded2[valid_mask] = expanded2_valid

    # Compare shapes
    print(f"Implementation 1 output shape: {expanded1.shape}")
    print(f"Implementation 2 output shape: {expanded2.shape}")

    # Calculate difference
    if expanded1.shape == expanded2.shape:
        difference = (expanded1 - expanded2).abs()
        max_diff = difference.max().item()
        mean_diff = difference.mean().item()
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Check if they're approximately equal (accounting for numerical precision)
        are_equal = max_diff < 1e-5
        print(f"Implementations produce approximately same results: {are_equal}")
    else:
        print("Cannot compare directly due to different shapes")

    # Visualization
    # Plot the filter basis functions for visual comparison
    plt.figure(figsize=(12, 8))

    # Implementation 1
    mu_values = torch.linspace(distance_min, distance_max, gaussian_expansion_dim)
    sigma = (distance_max - distance_min) / gaussian_expansion_dim

    # Plot basis functions
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 10, 1000)
    for i, mu in enumerate(mu_values):
        y = np.exp(-((x - mu.item()) ** 2) / sigma ** 2)
        plt.plot(x, y, label=f"μ={mu.item():.2f}")

    plt.title("Implementation 1 Basis Functions")
    plt.xlabel("Distance")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)

    # Implementation 2
    plt.subplot(2, 1, 2)
    for i, mu in enumerate(gd.filter):
        y = np.exp(-((x - mu.item()) ** 2) / gd.var ** 2)
        plt.plot(x, y, label=f"μ={mu.item():.2f}")

    plt.title("Implementation 2 Basis Functions")
    plt.xlabel("Distance")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gaussian_basis_comparison.png")
    print("Saved visualization to 'gaussian_basis_comparison.png'")

    # Detailed comparison for a few specific distances
    print("\nDetailed comparison for specific distances:")
    sample_distances = [1.0, 3.5, 7.2]
    for dist in sample_distances:
        print(f"\nDistance: {dist}")

        # Implementation 1
        tensor_dist = torch.tensor([dist])
        exp1 = compute_gaussian_expanded_distances_1(
            tensor_dist, distance_min, distance_max, gaussian_expansion_dim
        )

        # Implementation 2
        exp2 = gd.expand(tensor_dist)

        print(f"Implementation 1: {exp1.flatten().tolist()}")
        print(f"Implementation 2: {exp2.flatten().tolist()}")

        diff = (exp1 - exp2).abs()
        print(f"Absolute difference: {diff.flatten().tolist()}")

    return expanded1, expanded2


if __name__ == "__main__":
    expanded1, expanded2 = compare_implementations()
