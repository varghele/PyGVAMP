#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick test script to benchmark GraKeL's NSPDK kernel performance
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os

# Import grakel
try:
    from grakel import Graph
    from grakel.kernels import NeighborhoodSubgraphPairwiseDistance

    print("✅ Successfully imported grakel")
except ImportError:
    print("❌ Could not import grakel. Please install it with:")
    print("pip install grakel")
    exit(1)


def generate_synthetic_molecular_graphs(n_graphs=100, min_nodes=10, max_nodes=50):
    """
    Generate synthetic molecular-like graphs for testing

    Parameters
    ----------
    n_graphs : int
        Number of graphs to generate
    min_nodes : int
        Minimum number of nodes per graph
    max_nodes : int
        Maximum number of nodes per graph

    Returns
    -------
    graphs : list
        List of grakel Graph objects
    """
    print(f"Generating {n_graphs} synthetic molecular graphs...")

    graphs = []
    np.random.seed(42)  # For reproducibility

    # Define some "atom types" for node labels
    atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br']

    for i in range(n_graphs):
        # Random number of nodes
        n_nodes = np.random.randint(min_nodes, max_nodes + 1)

        # Create a connected graph (molecular-like)
        # Start with a random tree, then add some cycles
        edges = []

        # Create a spanning tree first (ensures connectivity)
        for node in range(1, n_nodes):
            parent = np.random.randint(0, node)
            edges.append((parent, node))

        # Add some additional edges to create cycles (like rings in molecules)
        n_extra_edges = np.random.randint(0, min(n_nodes // 3, 5))
        for _ in range(n_extra_edges):
            u = np.random.randint(0, n_nodes)
            v = np.random.randint(0, n_nodes)
            if u != v and (u, v) not in edges and (v, u) not in edges:
                edges.append((u, v))

        # Create adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for u, v in edges:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1  # Undirected graph

        # Create node labels (atom types)
        node_labels = {}
        for node in range(n_nodes):
            node_labels[node] = np.random.choice(atom_types)

        # Create edge labels (bond types)
        edge_labels = {}
        for u, v in edges:
            bond_type = np.random.choice(['single', 'double', 'triple'], p=[0.7, 0.25, 0.05])
            edge_labels[(u, v)] = bond_type
            edge_labels[(v, u)] = bond_type  # Symmetric

        # Create grakel Graph object
        graph = Graph(adj_matrix, node_labels, edge_labels)
        graphs.append(graph)

    print(f"Generated {len(graphs)} graphs with {min_nodes}-{max_nodes} nodes each")
    return graphs


def benchmark_nspdk_kernel(graphs, r_values=[1, 2, 3], d_values=[2, 3, 4, 5]):
    """
    Benchmark NSPDK kernel with different parameter settings

    Parameters
    ----------
    graphs : list
        List of grakel Graph objects
    r_values : list
        List of radius values to test
    d_values : list
        List of distance values to test

    Returns
    -------
    results : dict
        Dictionary containing timing results
    """
    print("Benchmarking NSPDK kernel with different parameters...")

    results = {
        'r': [],
        'd': [],
        'n_graphs': [],
        'fit_time': [],
        'transform_time': [],
        'total_time': [],
        'kernel_matrix_shape': [],
        'memory_usage_mb': []
    }

    n_graphs = len(graphs)

    for r in r_values:
        for d in d_values:
            print(f"\nTesting NSPDK with r={r}, d={d}")

            # Initialize NSPDK kernel
            nspdk = NeighborhoodSubgraphPairwiseDistance(r=r, d=d, normalize=True)

            # Measure fit time
            start_time = time.time()
            nspdk.fit(graphs)
            fit_time = time.time() - start_time

            # Measure transform time (compute kernel matrix)
            start_time = time.time()
            kernel_matrix = nspdk.fit_transform(graphs)
            transform_time = time.time() - start_time

            total_time = fit_time + transform_time

            # Estimate memory usage (rough approximation)
            memory_usage_mb = kernel_matrix.nbytes / (1024 * 1024)

            # Store results
            results['r'].append(r)
            results['d'].append(d)
            results['n_graphs'].append(n_graphs)
            results['fit_time'].append(fit_time)
            results['transform_time'].append(transform_time)
            results['total_time'].append(total_time)
            results['kernel_matrix_shape'].append(kernel_matrix.shape)
            results['memory_usage_mb'].append(memory_usage_mb)

            print(f"  Fit time: {fit_time:.4f}s")
            print(f"  Transform time: {transform_time:.4f}s")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Kernel matrix shape: {kernel_matrix.shape}")
            print(f"  Memory usage: {memory_usage_mb:.2f} MB")

            # Basic validation
            assert kernel_matrix.shape == (n_graphs, n_graphs), "Kernel matrix has wrong shape"
            assert np.allclose(kernel_matrix, kernel_matrix.T), "Kernel matrix is not symmetric"
            assert np.all(np.diag(kernel_matrix) >= 0), "Diagonal elements should be non-negative"

    return results


def benchmark_scalability(base_graphs, graph_counts=[50, 100, 200, 500]):
    """
    Test how NSPDK performance scales with number of graphs

    Parameters
    ----------
    base_graphs : list
        Base set of graphs to sample from
    graph_counts : list
        Different numbers of graphs to test

    Returns
    -------
    scalability_results : dict
        Dictionary containing scalability results
    """
    print("\nTesting scalability with different numbers of graphs...")

    scalability_results = {
        'n_graphs': [],
        'total_time': [],
        'time_per_graph': [],
        'time_per_comparison': []
    }

    # Use fixed parameters for scalability test
    r, d = 2, 3

    for n_graphs in graph_counts:
        if n_graphs > len(base_graphs):
            # Repeat graphs if we need more
            graphs = (base_graphs * ((n_graphs // len(base_graphs)) + 1))[:n_graphs]
        else:
            graphs = base_graphs[:n_graphs]

        print(f"\nTesting with {n_graphs} graphs...")

        # Initialize and run NSPDK
        nspdk = NeighborhoodSubgraphPairwiseDistance(r=r, d=d, normalize=True)

        start_time = time.time()
        kernel_matrix = nspdk.fit_transform(graphs)
        total_time = time.time() - start_time

        time_per_graph = total_time / n_graphs
        time_per_comparison = total_time / (n_graphs * n_graphs)

        scalability_results['n_graphs'].append(n_graphs)
        scalability_results['total_time'].append(total_time)
        scalability_results['time_per_graph'].append(time_per_graph)
        scalability_results['time_per_comparison'].append(time_per_comparison)

        print(f"  Total time: {total_time:.4f}s")
        print(f"  Time per graph: {time_per_graph:.6f}s")
        print(f"  Time per comparison: {time_per_comparison:.8f}s")

    return scalability_results


def create_visualizations(results, scalability_results, output_dir="grakel_nspdk_results"):
    """
    Create visualizations of the benchmarking results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # 1. Parameter comparison heatmap
    plt.figure(figsize=(12, 8))

    # Create pivot table for heatmap
    pivot_table = df.pivot(index='r', columns='d', values='total_time')

    plt.subplot(2, 2, 1)
    plt.imshow(pivot_table.values, cmap='viridis', aspect='auto')
    plt.colorbar(label='Total Time (seconds)')
    plt.title('NSPDK Performance: Total Time')
    plt.xlabel('Distance (d)')
    plt.ylabel('Radius (r)')
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)

    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            plt.text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                     ha='center', va='center', color='white', fontweight='bold')

    # 2. Memory usage heatmap
    memory_pivot = df.pivot(index='r', columns='d', values='memory_usage_mb')

    plt.subplot(2, 2, 2)
    plt.imshow(memory_pivot.values, cmap='plasma', aspect='auto')
    plt.colorbar(label='Memory Usage (MB)')
    plt.title('NSPDK Memory Usage')
    plt.xlabel('Distance (d)')
    plt.ylabel('Radius (r)')
    plt.xticks(range(len(memory_pivot.columns)), memory_pivot.columns)
    plt.yticks(range(len(memory_pivot.index)), memory_pivot.index)

    # Add text annotations
    for i in range(len(memory_pivot.index)):
        for j in range(len(memory_pivot.columns)):
            plt.text(j, i, f'{memory_pivot.iloc[i, j]:.1f}',
                     ha='center', va='center', color='white', fontweight='bold')

    # 3. Scalability plot
    plt.subplot(2, 2, 3)
    scal_df = pd.DataFrame(scalability_results)
    plt.loglog(scal_df['n_graphs'], scal_df['total_time'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Graphs')
    plt.ylabel('Total Time (seconds)')
    plt.title('Scalability: Total Time vs Number of Graphs')
    plt.grid(True, alpha=0.3)

    # Add theoretical O(n²) line for comparison
    x_theory = np.array(scal_df['n_graphs'])
    y_theory = scal_df['total_time'].iloc[0] * (x_theory / x_theory[0]) ** 2
    plt.loglog(x_theory, y_theory, '--', alpha=0.7, label='O(n²) theoretical')
    plt.legend()

    # 4. Time per comparison
    plt.subplot(2, 2, 4)
    plt.semilogx(scal_df['n_graphs'], scal_df['time_per_comparison'] * 1000, 'o-',
                 linewidth=2, markersize=8, color='red')
    plt.xlabel('Number of Graphs')
    plt.ylabel('Time per Comparison (ms)')
    plt.title('Efficiency: Time per Graph Comparison')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/nspdk_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save detailed results
    df.to_csv(f"{output_dir}/nspdk_parameter_results.csv", index=False)
    pd.DataFrame(scalability_results).to_csv(f"{output_dir}/nspdk_scalability_results.csv", index=False)

    print(f"\nResults saved to {output_dir}/")


def print_performance_summary(results, scalability_results):
    """
    Print a summary of the performance results
    """
    df = pd.DataFrame(results)
    scal_df = pd.DataFrame(scalability_results)

    print("\n" + "=" * 60)
    print("GRAKEL NSPDK PERFORMANCE SUMMARY")
    print("=" * 60)

    # Parameter analysis
    fastest_config = df.loc[df['total_time'].idxmin()]
    slowest_config = df.loc[df['total_time'].idxmax()]

    print(f"\nParameter Analysis:")
    print(f"  Fastest configuration: r={fastest_config['r']}, d={fastest_config['d']}")
    print(f"    Time: {fastest_config['total_time']:.4f}s")
    print(f"    Memory: {fastest_config['memory_usage_mb']:.2f} MB")

    print(f"  Slowest configuration: r={slowest_config['r']}, d={slowest_config['d']}")
    print(f"    Time: {slowest_config['total_time']:.4f}s")
    print(f"    Memory: {slowest_config['memory_usage_mb']:.2f} MB")

    print(f"  Speedup factor: {slowest_config['total_time'] / fastest_config['total_time']:.2f}x")

    # Scalability analysis
    print(f"\nScalability Analysis:")
    print(f"  Tested with {min(scal_df['n_graphs'])}-{max(scal_df['n_graphs'])} graphs")
    print(f"  Time per graph comparison: {scal_df['time_per_comparison'].mean() * 1000:.4f} ms (average)")
    print(f"  Best time per comparison: {scal_df['time_per_comparison'].min() * 1000:.4f} ms")
    print(f"  Worst time per comparison: {scal_df['time_per_comparison'].max() * 1000:.4f} ms")

    # Performance recommendations
    print(f"\nRecommendations:")
    if fastest_config['r'] <= 2 and fastest_config['d'] <= 3:
        print("  ✅ Small parameter values (r≤2, d≤3) provide good performance")
    else:
        print("  ⚠️ Consider using smaller parameter values for better performance")

    if scal_df['time_per_comparison'].iloc[-1] < scal_df['time_per_comparison'].iloc[0] * 2:
        print("  ✅ Good scalability - time per comparison remains stable")
    else:
        print("  ⚠️ Performance degrades with larger datasets")

    # Comparison with typical molecular datasets
    avg_time_per_comparison = scal_df['time_per_comparison'].mean()
    print(f"\nDataset Size Estimates (based on {avg_time_per_comparison * 1000:.4f} ms per comparison):")
    for dataset_size in [1000, 5000, 10000, 50000]:
        total_comparisons = dataset_size * dataset_size
        estimated_time = total_comparisons * avg_time_per_comparison
        print(f"  {dataset_size:5d} molecules: ~{estimated_time / 60:.1f} minutes")


def main():
    """
    Main function to run the GraKeL NSPDK performance test
    """
    print("GraKeL NSPDK Performance Test")
    print("=" * 40)

    # Generate test data
    graphs = generate_synthetic_molecular_graphs(n_graphs=2000, min_nodes=200, max_nodes=400)

    # Test different parameter combinations
    print("\n1. Testing different parameter combinations...")
    results = benchmark_nspdk_kernel(
        graphs[:100],  # Use subset for parameter testing
        r_values=[1, 2, 3],
        d_values=[2, 3, 4, 5]
    )

    # Test scalability
    print("\n2. Testing scalability...")
    scalability_results = benchmark_scalability(
        graphs,
        graph_counts=[50, 100, 150, 200]
    )

    # Create visualizations
    print("\n3. Creating visualizations...")
    create_visualizations(results, scalability_results)

    # Print summary
    print_performance_summary(results, scalability_results)

    print("\n🎉 GraKeL NSPDK performance test completed!")

    return results, scalability_results


if __name__ == "__main__":
    try:
        results, scalability_results = main()
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()
