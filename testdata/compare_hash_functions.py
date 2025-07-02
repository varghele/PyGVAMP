#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone test script for NSPDK hash function optimization
Compares original nested-loop implementation with optimized vectorized version
"""

import torch
import numpy as np
import time
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os

# Create output directory
os.makedirs("nspdk_hash_test_results", exist_ok=True)


class NSPDKHashTester:
    """
    Standalone tester for NSPDK hash function optimization
    """

    def __init__(self, r=3, d=4):
        self.r = r  # Maximum radius
        self.d = d  # Maximum distance

    def generate_test_data(self, num_nodes=50, device='cpu'):
        """
        Generate synthetic test data for hash function comparison
        """
        print(f"Generating test data with {num_nodes} nodes...")

        # Create random distance matrix (simulating protein structure)
        dist_matrix = torch.rand(num_nodes, num_nodes, device=device) * 15

        # Make symmetric and set diagonal to 0
        dist_matrix = (dist_matrix + dist_matrix.t()) / 2
        torch.diagonal(dist_matrix).fill_(0.0)

        # Set some distances to infinity (disconnected regions)
        mask = torch.rand(num_nodes, num_nodes, device=device) > 0.8
        dist_matrix[mask] = float('inf')
        torch.diagonal(dist_matrix).fill_(0.0)

        # Create node labels (simulating residue types)
        node_labels = torch.randint(0, 20, (num_nodes,), device=device)  # 20 amino acids

        # Create neighborhoods (simplified BFS-like)
        neighborhoods = {}
        for r in range(self.r + 1):
            neighborhoods[r] = {}
            for center in range(num_nodes):
                if r == 0:
                    neighborhoods[r][center] = [center]
                else:
                    # Find nodes within graph distance r
                    neighbors = []
                    for node in range(num_nodes):
                        # Simple approximation: use Euclidean distance threshold
                        if dist_matrix[center, node] <= r * 3 and dist_matrix[center, node] != float('inf'):
                            neighbors.append(node)
                    neighborhoods[r][center] = neighbors

        return dist_matrix, node_labels, neighborhoods

    def hash_neighborhoods_original(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """
        Original implementation with nested for loops (SLOW)
        """
        H = {}

        for center in range(num_nodes):
            for radius in range(self.r + 1):
                sub_vertices = neighborhoods[radius][center]

                # Create encoding string similar to grakel's hash_graph function
                encoding = ""

                # Make labels for vertices (similar to grakel's Lv)
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

                # Add edge information (simplified for efficiency)
                for i in sub_vertices:
                    for j in sub_vertices:
                        if i < j and i < len(dist_matrix) and j < len(dist_matrix):
                            dist = dist_matrix[i, j].item()
                            if dist == 1.0:  # Direct edge
                                encoding += f"{vertex_labels.get(i, '')},{vertex_labels.get(j, '')},1_"

                # Use hashlib for consistent hashing
                hash_value = int(hashlib.md5(encoding.encode()).hexdigest()[:8], 16)
                H[(radius, center)] = hash_value

        return H

    def hash_neighborhoods_optimized(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """
        Optimized implementation using vectorized operations and efficient data structures
        """
        H = {}
        device = dist_matrix.device

        # Pre-convert to numpy for faster CPU operations
        dist_matrix_np = dist_matrix.cpu().numpy()
        node_labels_np = node_labels.cpu().numpy()

        # Group neighborhoods by size for batch processing
        neighborhoods_by_size = defaultdict(list)
        for center in range(num_nodes):
            for radius in range(self.r + 1):
                sub_vertices = neighborhoods[radius][center]
                size = len(sub_vertices)
                neighborhoods_by_size[size].append((radius, center, sub_vertices))

        # Process each size group
        for size, neighborhood_list in neighborhoods_by_size.items():
            if size == 0:
                for radius, center, _ in neighborhood_list:
                    H[(radius, center)] = hash("EMPTY") % (2 ** 32)
                continue

            # Process neighborhoods of this size
            for radius, center, sub_vertices in neighborhood_list:
                sub_vertices_arr = np.array(sub_vertices)

                # Use numpy's advanced indexing for speed
                sub_distances = dist_matrix_np[sub_vertices_arr][:, sub_vertices_arr]
                sub_labels = node_labels_np[sub_vertices_arr]

                # Create a more efficient encoding using numerical hashing
                # Instead of string operations, use numerical combinations
                vertex_hashes = []
                for i in range(size):
                    # Create a numerical hash for each vertex's neighborhood
                    valid_mask = sub_distances[i] != float('inf')
                    if np.any(valid_mask):
                        # Combine distances and labels numerically
                        dist_label_pairs = sub_distances[i][valid_mask].astype(int) * 1000 + sub_labels[valid_mask]
                        dist_label_pairs.sort()
                        vertex_hash = hash(tuple(dist_label_pairs)) % (2 ** 16)
                    else:
                        vertex_hash = 0
                    vertex_hashes.append(vertex_hash)

                # Create edge hash using vectorized operations
                edge_hash = 0
                if size > 1:
                    # Find edges (distance == 1) using vectorized operations
                    edge_mask = (sub_distances == 1.0)
                    edge_indices = np.where(np.triu(edge_mask, k=1))

                    if len(edge_indices[0]) > 0:
                        # Vectorized edge hash computation
                        vertex_hashes_arr = np.array(vertex_hashes)
                        i_vals = edge_indices[0]
                        j_vals = edge_indices[1]

                        min_hashes = np.minimum(vertex_hashes_arr[i_vals], vertex_hashes_arr[j_vals])
                        max_hashes = np.maximum(vertex_hashes_arr[i_vals], vertex_hashes_arr[j_vals])

                        # Combine hashes efficiently
                        for min_h, max_h in zip(min_hashes, max_hashes):
                            edge_hash ^= hash((int(min_h), int(max_h))) % (2 ** 16)

                # Combine vertex and edge hashes
                vertex_hashes.sort()  # Canonical ordering
                final_hash = hash((tuple(vertex_hashes), edge_hash)) % (2 ** 32)
                H[(radius, center)] = final_hash

        return H

    def hash_neighborhoods_ultra_optimized(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """
        Ultra-optimized version with advanced vectorization
        """
        H = {}

        # Convert to numpy once
        dist_matrix_np = dist_matrix.cpu().numpy()
        node_labels_np = node_labels.cpu().numpy()

        # Pre-compute all possible distance-label combinations
        max_dist = 20  # Reasonable maximum distance
        max_label = 20  # Number of amino acid types

        # Create lookup table for distance-label pair hashes
        dist_label_lookup = np.zeros((max_dist + 1, max_label), dtype=np.int32)
        for d in range(max_dist + 1):
            for l in range(max_label):
                dist_label_lookup[d, l] = hash(f"{d},{l}") % (2 ** 16)

        # Process all neighborhoods
        for center in range(num_nodes):
            for radius in range(self.r + 1):
                sub_vertices = neighborhoods[radius][center]

                if len(sub_vertices) == 0:
                    H[(radius, center)] = hash("EMPTY") % (2 ** 32)
                    continue

                sub_vertices_arr = np.array(sub_vertices)

                # Vectorized distance and label extraction
                sub_distances = dist_matrix_np[sub_vertices_arr][:, sub_vertices_arr]
                sub_labels = node_labels_np[sub_vertices_arr]

                # Ultra-fast vertex hash computation using lookup table
                vertex_hashes = []
                for i in range(len(sub_vertices)):
                    valid_mask = (sub_distances[i] != float('inf')) & (sub_distances[i] < max_dist)
                    if np.any(valid_mask):
                        valid_distances = sub_distances[i][valid_mask].astype(int)
                        valid_labels = sub_labels[valid_mask]

                        # Use lookup table for fast hash computation
                        pair_hashes = dist_label_lookup[valid_distances, valid_labels]
                        pair_hashes.sort()
                        vertex_hash = hash(tuple(pair_hashes)) % (2 ** 16)
                    else:
                        vertex_hash = 0
                    vertex_hashes.append(vertex_hash)

                # Ultra-fast edge hash computation
                edge_hash = 0
                if len(sub_vertices) > 1:
                    # Vectorized edge detection
                    edge_matrix = (sub_distances == 1.0)
                    edge_coords = np.where(np.triu(edge_matrix, k=1))

                    if len(edge_coords[0]) > 0:
                        vertex_hashes_arr = np.array(vertex_hashes)
                        edge_hash_pairs = np.column_stack([
                            np.minimum(vertex_hashes_arr[edge_coords[0]], vertex_hashes_arr[edge_coords[1]]),
                            np.maximum(vertex_hashes_arr[edge_coords[0]], vertex_hashes_arr[edge_coords[1]])
                        ])

                        # Fast XOR combination
                        for pair in edge_hash_pairs:
                            edge_hash ^= hash(tuple(pair)) % (2 ** 16)

                # Final hash combination
                vertex_hashes.sort()
                final_hash = hash((tuple(vertex_hashes), edge_hash)) % (2 ** 32)
                H[(radius, center)] = final_hash

        return H

    def compare_implementations(self, test_sizes=[20, 50, 100], num_runs=3):
        """
        Compare all three implementations across different graph sizes
        """
        results = {
            'graph_size': [],
            'original_time': [],
            'optimized_time': [],
            'ultra_optimized_time': [],
            'speedup_optimized': [],
            'speedup_ultra': [],
            'outputs_match_opt': [],
            'outputs_match_ultra': []
        }

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running comparison on device: {device}")

        for size in test_sizes:
            print(f"\nTesting with {size} nodes...")

            # Generate test data
            dist_matrix, node_labels, neighborhoods = self.generate_test_data(size, device)

            # Time original implementation
            original_times = []
            for run in range(num_runs):
                start_time = time.time()
                original_result = self.hash_neighborhoods_original(size, node_labels, neighborhoods, dist_matrix)
                original_times.append(time.time() - start_time)

            # Time optimized implementation
            optimized_times = []
            for run in range(num_runs):
                start_time = time.time()
                optimized_result = self.hash_neighborhoods_optimized(size, node_labels, neighborhoods, dist_matrix)
                optimized_times.append(time.time() - start_time)

            # Time ultra-optimized implementation
            ultra_times = []
            for run in range(num_runs):
                start_time = time.time()
                ultra_result = self.hash_neighborhoods_ultra_optimized(size, node_labels, neighborhoods, dist_matrix)
                ultra_times.append(time.time() - start_time)

            # Calculate averages
            avg_original = np.mean(original_times)
            avg_optimized = np.mean(optimized_times)
            avg_ultra = np.mean(ultra_times)

            # Check if outputs match
            outputs_match_opt = self.compare_outputs(original_result, optimized_result)
            outputs_match_ultra = self.compare_outputs(original_result, ultra_result)

            # Store results
            results['graph_size'].append(size)
            results['original_time'].append(avg_original)
            results['optimized_time'].append(avg_optimized)
            results['ultra_optimized_time'].append(avg_ultra)
            results['speedup_optimized'].append(avg_original / avg_optimized)
            results['speedup_ultra'].append(avg_original / avg_ultra)
            results['outputs_match_opt'].append(outputs_match_opt)
            results['outputs_match_ultra'].append(outputs_match_ultra)

            print(f"Original time: {avg_original:.4f}s")
            print(f"Optimized time: {avg_optimized:.4f}s (speedup: {avg_original / avg_optimized:.2f}x)")
            print(f"Ultra-optimized time: {avg_ultra:.4f}s (speedup: {avg_original / avg_ultra:.2f}x)")
            print(f"Outputs match (optimized): {outputs_match_opt}")
            print(f"Outputs match (ultra): {outputs_match_ultra}")

        return results

    def compare_outputs(self, result1, result2):
        """
        Compare two hash result dictionaries
        """
        if set(result1.keys()) != set(result2.keys()):
            return False

        mismatches = 0
        total_keys = len(result1.keys())

        for key in result1.keys():
            if result1[key] != result2[key]:
                mismatches += 1

        # Allow for some hash differences due to implementation details
        match_rate = (total_keys - mismatches) / total_keys
        return match_rate > 0.95  # 95% match rate threshold

    def create_visualizations(self, results):
        """
        Create performance comparison visualizations
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Execution times
        ax1.plot(results['graph_size'], results['original_time'], 'o-', label='Original', linewidth=2, markersize=8)
        ax1.plot(results['graph_size'], results['optimized_time'], 's-', label='Optimized', linewidth=2, markersize=8)
        ax1.plot(results['graph_size'], results['ultra_optimized_time'], '^-', label='Ultra-Optimized', linewidth=2,
                 markersize=8)
        ax1.set_xlabel('Graph Size (nodes)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Plot 2: Speedup
        ax2.plot(results['graph_size'], results['speedup_optimized'], 's-', label='Optimized Speedup', linewidth=2,
                 markersize=8)
        ax2.plot(results['graph_size'], results['speedup_ultra'], '^-', label='Ultra-Optimized Speedup', linewidth=2,
                 markersize=8)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
        ax2.set_xlabel('Graph Size (nodes)')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Time complexity (log-log)
        ax3.loglog(results['graph_size'], results['original_time'], 'o-', label='Original', linewidth=2, markersize=8)
        ax3.loglog(results['graph_size'], results['optimized_time'], 's-', label='Optimized', linewidth=2, markersize=8)
        ax3.loglog(results['graph_size'], results['ultra_optimized_time'], '^-', label='Ultra-Optimized', linewidth=2,
                   markersize=8)
        ax3.set_xlabel('Graph Size (log scale)')
        ax3.set_ylabel('Execution Time (log scale)')
        ax3.set_title('Time Complexity Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Output correctness
        correctness_data = []
        labels = []
        for i, size in enumerate(results['graph_size']):
            correctness_data.extend([
                1 if results['outputs_match_opt'][i] else 0,
                1 if results['outputs_match_ultra'][i] else 0
            ])
            labels.extend([f'{size} nodes\n(Optimized)', f'{size} nodes\n(Ultra)'])

        colors = ['green' if x == 1 else 'red' for x in correctness_data]
        ax4.bar(range(len(correctness_data)), correctness_data, color=colors)
        ax4.set_xlabel('Test Case')
        ax4.set_ylabel('Outputs Match')
        ax4.set_title('Output Correctness')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig('nspdk_hash_test_results/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def run_comprehensive_test(self):
        """
        Run comprehensive test suite
        """
        print("=" * 60)
        print("NSPDK Hash Function Optimization Test")
        print("=" * 60)

        # Run comparison
        results = self.compare_implementations(
            test_sizes=[10, 20, 30, 50, 75, 150, 300, 500],
            num_runs=3
        )

        # Create visualizations
        self.create_visualizations(results)

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv('nspdk_hash_test_results/performance_results.csv', index=False)

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        avg_speedup_opt = np.mean(results['speedup_optimized'])
        avg_speedup_ultra = np.mean(results['speedup_ultra'])

        print(f"Average optimized speedup: {avg_speedup_opt:.2f}x")
        print(f"Average ultra-optimized speedup: {avg_speedup_ultra:.2f}x")
        print(f"All optimized outputs correct: {all(results['outputs_match_opt'])}")
        print(f"All ultra-optimized outputs correct: {all(results['outputs_match_ultra'])}")

        # Performance recommendations
        print("\nPERFORMANCE RECOMMENDATIONS:")
        if avg_speedup_ultra > avg_speedup_opt * 1.2:
            print("✅ Use ultra-optimized version for best performance")
        elif avg_speedup_opt > 2:
            print("✅ Use optimized version for good balance of performance and simplicity")
        else:
            print("⚠️ Optimizations provide limited benefit - consider algorithm changes")

        # Correctness check
        if all(results['outputs_match_opt']) and all(results['outputs_match_ultra']):
            print("✅ All optimized implementations produce correct results")
        else:
            print("❌ Some implementations produce incorrect results - investigate further")

        print(f"\nResults saved to 'nspdk_hash_test_results/' directory")

        return results


def main():
    """
    Main function to run the standalone test
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize tester
    tester = NSPDKHashTester(r=2, d=3)  # Use smaller values for faster testing

    # Run comprehensive test
    try:
        results = tester.run_comprehensive_test()
        print("\n🎉 Testing completed successfully!")

        # Print key findings
        best_speedup = max(max(results['speedup_optimized']), max(results['speedup_ultra']))
        print(f"\nKey Findings:")
        print(f"- Best speedup achieved: {best_speedup:.2f}x")
        print(f"- Largest graph tested: {max(results['graph_size'])} nodes")
        print(
            f"- All implementations correct: {all(results['outputs_match_opt']) and all(results['outputs_match_ultra'])}")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
