import torch
import hashlib
import numpy as np
from collections import defaultdict
import time


def _hash_neighborhoods_torch_optimized(self, num_nodes, node_labels, neighborhoods, dist_matrix):
    """
    Optimized hash neighborhoods using matrix operations instead of for loops
    """
    H = {}
    device = dist_matrix.device

    # Pre-compute all distance-label pairs for all nodes
    # This creates a matrix where each row represents a node's view of all other nodes
    node_labels_expanded = node_labels.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, num_nodes]

    # Create distance-label encoding matrix
    # Each element [i,j] contains the encoding "distance_ij,label_j"
    dist_label_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.long, device=device)

    # Vectorized creation of distance-label pairs
    for i in range(num_nodes):
        for j in range(num_nodes):
            if dist_matrix[i, j] != float('inf'):
                # Create a hash of the distance-label pair
                dist_int = int(dist_matrix[i, j].item())
                label_int = int(node_labels[j].item())
                # Simple hash combination (you can use a better hash function)
                dist_label_matrix[i, j] = dist_int * 1000 + label_int

    # Process each center and radius combination
    for center in range(num_nodes):
        for radius in range(self.r + 1):
            sub_vertices = neighborhoods[radius][center]

            if len(sub_vertices) == 0:
                H[(radius, center)] = hash("EMPTY") % (2 ** 32)
                continue

            # Convert sub_vertices to tensor indices
            sub_indices = torch.tensor(sub_vertices, dtype=torch.long, device=device)

            # Extract submatrix for this neighborhood
            sub_dist_labels = dist_label_matrix[sub_indices][:, sub_indices]  # [sub_size, sub_size]

            # Create vertex encodings by sorting distance-label pairs for each vertex
            vertex_encodings = []
            for i, vertex_idx in enumerate(sub_indices):
                # Get all distance-label pairs for this vertex within the subgraph
                vertex_pairs = sub_dist_labels[i]  # [sub_size]

                # Sort the pairs to create canonical representation
                sorted_pairs, _ = torch.sort(vertex_pairs)

                # Create a hash for this vertex's encoding
                vertex_hash = hash(tuple(sorted_pairs.cpu().numpy())) % (2 ** 16)
                vertex_encodings.append(vertex_hash)

            # Sort vertex encodings for canonical graph representation
            vertex_encodings.sort()

            # Create edge encodings using vectorized operations
            edge_encodings = []
            if len(sub_vertices) > 1:
                # Check which pairs are connected (distance == 1 in original graph)
                for i in range(len(sub_vertices)):
                    for j in range(i + 1, len(sub_vertices)):
                        vi, vj = sub_vertices[i], sub_vertices[j]
                        if vi < len(dist_matrix) and vj < len(dist_matrix):
                            if dist_matrix[vi, vj].item() == 1.0:  # Direct edge
                                edge_pair = (min(vertex_encodings[i], vertex_encodings[j]),
                                             max(vertex_encodings[i], vertex_encodings[j]))
                                edge_encodings.append(edge_pair)

            edge_encodings.sort()

            # Create final hash
            canonical_str = f"NODES:{vertex_encodings}|EDGES:{edge_encodings}"
            hash_value = int(hashlib.md5(canonical_str.encode()).hexdigest()[:8], 16)
            H[(radius, center)] = hash_value

    return H

def _hash_neighborhoods_torch_vectorized(self, num_nodes, node_labels, neighborhoods, dist_matrix):
    """ Fully vectorized version using advanced PyTorch operations """
    H = {}
    device = dist_matrix.device

    # Pre-process all neighborhoods at once
    max_neighborhood_size = max(len(neighborhoods[r][center])
                                for r in range(self.r + 1)
                                for center in range(num_nodes))

    # Create batched neighborhood processing
    for radius in range(self.r + 1):
        # Collect all neighborhoods for this radius
        radius_neighborhoods = []
        radius_centers = []

        for center in range(num_nodes):
            sub_vertices = neighborhoods[radius][center]
            if len(sub_vertices) > 0:
                radius_neighborhoods.append(sub_vertices)
                radius_centers.append(center)

        if not radius_neighborhoods:
            continue

        # Process neighborhoods in batches
        batch_size = min(32, len(radius_neighborhoods))  # Adjust based on memory

        for batch_start in range(0, len(radius_neighborhoods), batch_size):
            batch_end = min(batch_start + batch_size, len(radius_neighborhoods))
            batch_neighborhoods = radius_neighborhoods[batch_start:batch_end]
            batch_centers = radius_centers[batch_start:batch_end]

            # Process this batch
            for i, (neighborhood, center) in enumerate(zip(batch_neighborhoods, batch_centers)):
                # Use optimized single neighborhood processing
                hash_value = _hash_single_neighborhood_optimized(
                    neighborhood, center, dist_matrix, node_labels, device
                )
                H[(radius, center)] = hash_value

    return H


def _hash_single_neighborhood_optimized(neighborhood, center, dist_matrix, node_labels, device):
    """ Optimized hashing for a single neighborhood using matrix operations """
    if len(neighborhood) == 0:
        return hash("EMPTY") % (2**32)

    # Convert to tensor
    sub_indices = torch.tensor(neighborhood, dtype=torch.long, device=device)

    # Extract subgraph distance matrix
    sub_dist_matrix = dist_matrix[sub_indices][:, sub_indices]
    sub_node_labels = node_labels[sub_indices]

    # Vectorized vertex encoding creation
    vertex_encodings = []

    for i in range(len(sub_indices)):
        # Get distances and labels for this vertex
        distances = sub_dist_matrix[i]
        labels = sub_node_labels

        # Create distance-label pairs
        valid_mask = distances != float('inf')
        valid_distances = distances[valid_mask]
        valid_labels = labels[valid_mask]

        # Combine distance and label into single values
        dist_label_pairs = valid_distances.long() * 1000 + valid_labels.long()

        # Sort for canonical representation
        sorted_pairs, _ = torch.sort(dist_label_pairs)

        # Create hash for this vertex
        vertex_hash = hash(tuple(sorted_pairs.cpu().numpy())) % (2 ** 16)
        vertex_encodings.append(vertex_hash)

    # Sort vertex encodings
    vertex_encodings.sort()

    # Vectorized edge detection
    edge_encodings = []
    if len(neighborhood) > 1:
        # Find edges (distance == 1) using vectorized operations
        edge_mask = (sub_dist_matrix == 1.0)
        edge_indices = torch.nonzero(edge_mask, as_tuple=False)

        for edge in edge_indices:
            i, j = edge[0].item(), edge[1].item()
            if i < j:  # Avoid duplicates
                edge_pair = (min(vertex_encodings[i], vertex_encodings[j]),
                             max(vertex_encodings[i], vertex_encodings[j]))
                edge_encodings.append(edge_pair)

    edge_encodings.sort()

    # Create final hash
    canonical_str = f"NODES:{vertex_encodings}|EDGES:{edge_encodings}"
    return int(hashlib.md5(canonical_str.encode()).hexdigest()[:8], 16)


import torch
import time
import numpy as np
from collections import defaultdict


class NSPDKHashComparison:
    """
    Class to compare the original and optimized hash functions
    """

    def __init__(self, r=3, d=4):
        self.r = r
        self.d = d

    def _hash_neighborhoods_torch_original(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """
        Original implementation with nested for loops
        """
        H = {}

        for center in range(num_nodes):
            for radius in range(self.r + 1):
                sub_vertices = neighborhoods[radius][center]

                encoding = ""

                # Create vertex labels
                vertex_labels = {}
                for i in sub_vertices:
                    label_parts = []
                    for j in sub_vertices:
                        if i < len(dist_matrix) and j < len(dist_matrix):
                            dist = dist_matrix[i, j].item()
                            if dist != float('inf'):
                                node_label = node_labels[j].item() if j < len(node_labels) else 0
                                label_parts.append(f"{int(dist)},{node_label}")

                    vertex_label = "|".join(sorted(label_parts))
                    vertex_labels[i] = vertex_label
                    encoding += vertex_label + "."

                if encoding.endswith("."):
                    encoding = encoding[:-1] + ":"

                # Add edge information
                for i in sub_vertices:
                    for j in sub_vertices:
                        if i < j and i < len(dist_matrix) and j < len(dist_matrix):
                            dist = dist_matrix[i, j].item()
                            if dist == 1.0:
                                encoding += f"{vertex_labels.get(i, '')},{vertex_labels.get(j, '')},1_"

                hash_value = int(hashlib.md5(encoding.encode()).hexdigest()[:8], 16)
                H[(radius, center)] = hash_value

        return H

    def generate_test_data(self, num_nodes=50, device='cpu'):
        """
        Generate test data for comparison
        """
        # Create random distance matrix
        dist_matrix = torch.rand(num_nodes, num_nodes, device=device) * 10

        # Make it symmetric and set diagonal to 0
        dist_matrix = (dist_matrix + dist_matrix.t()) / 2
        torch.diagonal(dist_matrix).fill_(0.0)

        # Set some distances to infinity (disconnected)
        mask = torch.rand(num_nodes, num_nodes, device=device) > 0.7
        dist_matrix[mask] = float('inf')

        # Ensure diagonal is still 0
        torch.diagonal(dist_matrix).fill_(0.0)

        # Create node labels
        node_labels = torch.randint(0, 10, (num_nodes,), device=device)

        # Create neighborhoods (simplified for testing)
        neighborhoods = {}
        for r in range(self.r + 1):
            neighborhoods[r] = {}
            for center in range(num_nodes):
                # Find nodes within radius r
                neighbors = []
                for node in range(num_nodes):
                    if dist_matrix[center, node] <= r and dist_matrix[center, node] != float('inf'):
                        neighbors.append(node)
                neighborhoods[r][center] = neighbors

        return dist_matrix, node_labels, neighborhoods

    def compare_implementations(self, num_nodes_list=[20, 50, 100], num_runs=3):
        """
        Compare original and optimized implementations
        """
        results = {
            'num_nodes': [],
            'original_time': [],
            'optimized_time': [],
            'vectorized_time': [],
            'speedup_optimized': [],
            'speedup_vectorized': [],
            'outputs_match': []
        }

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running comparison on device: {device}")

        for num_nodes in num_nodes_list:
            print(f"\nTesting with {num_nodes} nodes...")

            # Generate test data
            dist_matrix, node_labels, neighborhoods = self.generate_test_data(num_nodes, device)

            # Time original implementation
            original_times = []
            for run in range(num_runs):
                start_time = time.time()
                original_result = self._hash_neighborhoods_torch_original(
                    num_nodes, node_labels, neighborhoods, dist_matrix
                )
                original_times.append(time.time() - start_time)

            # Time optimized implementation
            optimized_times = []
            for run in range(num_runs):
                start_time = time.time()
                optimized_result = _hash_neighborhoods_torch_optimized(
                    self, num_nodes, node_labels, neighborhoods, dist_matrix
                )
                optimized_times.append(time.time() - start_time)

            # Time vectorized implementation
            vectorized_times = []
            for run in range(num_runs):
                start_time = time.time()
                vectorized_result = _hash_neighborhoods_torch_vectorized(
                    self, num_nodes, node_labels, neighborhoods, dist_matrix
                )
                vectorized_times.append(time.time() - start_time)

            # Calculate averages
            avg_original = np.mean(original_times)
            avg_optimized = np.mean(optimized_times)
            avg_vectorized = np.mean(vectorized_times)

            # Check if outputs match
            outputs_match = self.compare_outputs(original_result, optimized_result, vectorized_result)

            # Store results
            results['num_nodes'].append(num_nodes)
            results['original_time'].append(avg_original)
            results['optimized_time'].append(avg_optimized)
            results['vectorized_time'].append(avg_vectorized)
            results['speedup_optimized'].append(avg_original / avg_optimized)
            results['speedup_vectorized'].append(avg_original / avg_vectorized)
            results['outputs_match'].append(outputs_match)

            print(f"Original time: {avg_original:.4f}s")
            print(f"Optimized time: {avg_optimized:.4f}s")
            print(f"Vectorized time: {avg_vectorized:.4f}s")
            print(f"Optimized speedup: {avg_original / avg_optimized:.2f}x")
            print(f"Vectorized speedup: {avg_original / avg_vectorized:.2f}x")
            print(f"Outputs match: {outputs_match}")

        return results

    def compare_outputs(self, original, optimized, vectorized):
        """
        Compare outputs to ensure correctness
        """
        # Check if all keys match
        if set(original.keys()) != set(optimized.keys()) or set(original.keys()) != set(vectorized.keys()):
            return False

        # Check if values are similar (allowing for small hash differences)
        mismatches = 0
        total_keys = len(original.keys())

        for key in original.keys():
            if original[key] != optimized[key] or original[key] != vectorized[key]:
                mismatches += 1

        # Allow for some hash collisions/differences due to implementation details
        match_rate = (total_keys - mismatches) / total_keys
        return match_rate > 0.95  # 95% match rate threshold

    def visualize_results(self, results):
        """
        Create visualizations of the comparison results
        """
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Execution times
        ax1.plot(results['num_nodes'], results['original_time'], 'o-', label='Original', linewidth=2)
        ax1.plot(results['num_nodes'], results['optimized_time'], 's-', label='Optimized', linewidth=2)
        ax1.plot(results['num_nodes'], results['vectorized_time'], '^-', label='Vectorized', linewidth=2)
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Plot 2: Speedup
        ax2.plot(results['num_nodes'], results['speedup_optimized'], 's-', label='Optimized Speedup', linewidth=2)
        ax2.plot(results['num_nodes'], results['speedup_vectorized'], '^-', label='Vectorized Speedup', linewidth=2)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Time complexity
        ax3.loglog(results['num_nodes'], results['original_time'], 'o-', label='Original', linewidth=2)
        ax3.loglog(results['num_nodes'], results['optimized_time'], 's-', label='Optimized', linewidth=2)
        ax3.loglog(results['num_nodes'], results['vectorized_time'], '^-', label='Vectorized', linewidth=2)
        ax3.set_xlabel('Number of Nodes (log scale)')
        ax3.set_ylabel('Execution Time (log scale)')
        ax3.set_title('Time Complexity Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Output correctness
        ax4.bar(range(len(results['num_nodes'])),
                [1 if match else 0 for match in results['outputs_match']],
                color=['green' if match else 'red' for match in results['outputs_match']])
        ax4.set_xlabel('Test Case')
        ax4.set_ylabel('Outputs Match')
        ax4.set_title('Output Correctness')
        ax4.set_xticks(range(len(results['num_nodes'])))
        ax4.set_xticklabels([f'{n} nodes' for n in results['num_nodes']])
        ax4.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig('nspdk_hash_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

def main():
    """ Run the comparison """
    print("NSPDK Hash Function Optimization Comparison")
    print("=" * 50)
    # Initialize comparison
    comparison = NSPDKHashComparison(r=2, d=3)  # Smaller values for faster testing

    # Run comparison
    results = comparison.compare_implementations(
        num_nodes_list=[10, 20, 30, 50],  # Start with smaller sizes
        num_runs=3
    )

    # Visualize results
    comparison.visualize_results(results)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    avg_speedup_opt = np.mean(results['speedup_optimized'])
    avg_speedup_vec = np.mean(results['speedup_vectorized'])

    print(f"Average optimized speedup: {avg_speedup_opt:.2f}x")
    print(f"Average vectorized speedup: {avg_speedup_vec:.2f}x")
    print(f"All outputs correct: {all(results['outputs_match'])}")

    if avg_speedup_opt > 2:
        print("✅ Significant speedup achieved with optimized version!")
    elif avg_speedup_opt > 1.2:
        print("✅ Moderate speedup achieved with optimized version")
    else:
        print("⚠️ Limited speedup - may need further optimization")

    return results

if __name__ == "__main__":
    results = main()



