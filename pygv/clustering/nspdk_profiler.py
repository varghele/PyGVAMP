import torch
import torch.nn.functional as F
import numpy as np
import mdtraj as md
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from tqdm import tqdm
import hashlib
import os
import time
import cProfile
import pstats
import io
from memory_profiler import profile, memory_usage
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse, k_hop_subgraph


class ProfiledPyGNSPDK:
    """
    PyTorch Geometric NSPDK with comprehensive profiling capabilities
    """

    def __init__(self, r=3, d=4, device='cuda', batch_size=32):
        self.r = r
        self.d = d
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # Profiling data
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        self.operation_counts = defaultdict(int)

    def time_function(self, func_name):
        """Decorator to time function execution"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()

                # Memory before
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                else:
                    mem_before = psutil.Process().memory_info().rss

                result = func(*args, **kwargs)

                # Memory after
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_after = torch.cuda.memory_allocated()
                else:
                    mem_after = psutil.Process().memory_info().rss

                end_time = time.time()

                # Store timing and memory data
                execution_time = end_time - start_time
                memory_used = mem_after - mem_before

                self.timing_data[func_name].append(execution_time)
                self.memory_data[func_name].append(memory_used)
                self.operation_counts[func_name] += 1

                return result

            return wrapper

        return decorator

    @time_function('trajectory_to_pyg_dataset')
    def trajectory_to_pyg_dataset(self, trajectory, selection='name CA'):
        """Convert MD trajectory to PyTorch Geometric dataset with profiling"""
        print(f"Converting {trajectory.n_frames} frames to PyG dataset...")

        atom_indices = trajectory.topology.select(selection)
        n_atoms = len(atom_indices)

        atom_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]

        class TrajectoryDataset(torch.utils.data.Dataset):
            def __init__(self, trajectory, atom_indices, atom_pairs, device, parent_profiler):
                self.trajectory = trajectory
                self.atom_indices = atom_indices
                self.atom_pairs = atom_pairs
                self.device = device
                self.n_atoms = len(atom_indices)
                self.profiler = parent_profiler

                # Pre-compute node labels once
                self.node_labels = torch.zeros(self.n_atoms, dtype=torch.long)
                for i, atom_idx in enumerate(atom_indices):
                    atom = trajectory.topology.atom(atom_idx)
                    label_str = f"{atom.residue.name}_{atom.residue.resSeq}"
                    label_hash = hash(label_str) % 1000
                    self.node_labels[i] = label_hash

            def __len__(self):
                return self.trajectory.n_frames

            def __getitem__(self, idx):
                start_time = time.time()

                # Extract single frame
                frame = self.trajectory[idx]

                # Compute distances for this frame only
                distances = md.compute_distances(frame, self.atom_pairs)[0] * 10

                # Create edges based on distance cutoff
                edge_indices = []
                edge_attrs = []

                for pair_idx, (i, j) in enumerate(self.atom_pairs):
                    if distances[pair_idx] < 12.0:
                        edge_indices.extend([[i, j], [j, i]])
                        edge_attrs.extend([distances[pair_idx], distances[pair_idx]])

                if len(edge_indices) > 0:
                    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0,), dtype=torch.float)

                # Track timing for dataset item creation
                item_time = time.time() - start_time
                self.profiler.timing_data['dataset_getitem'].append(item_time)

                return Data(
                    x=self.node_labels.clone().unsqueeze(1).float(),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    node_labels=self.node_labels.clone(),
                    num_nodes=self.n_atoms,
                    frame_idx=idx
                )

        dataset = TrajectoryDataset(trajectory, atom_indices, atom_pairs, self.device, self)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return dataloader

    @time_function('compute_shortest_paths_batch')
    def compute_shortest_paths_batch(self, batch):
        """Compute shortest paths for a batch of graphs with profiling"""
        shortest_paths = []

        for i in range(batch.num_graphs):
            start_time = time.time()

            mask = batch.batch == i
            num_nodes = mask.sum().item()

            if num_nodes == 0:
                continue

            # Get edges for this graph
            edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
            graph_edges = batch.edge_index[:, edge_mask]

            # Remap node indices
            node_mapping = torch.zeros(batch.num_nodes, dtype=torch.long, device=self.device)
            node_mapping[mask] = torch.arange(num_nodes, device=self.device)
            graph_edges = node_mapping[graph_edges]

            # Floyd-Warshall algorithm
            dist_matrix = torch.full((num_nodes, num_nodes), float('inf'), device=self.device)
            torch.diagonal(dist_matrix).fill_(0.0)

            if graph_edges.size(1) > 0:
                dist_matrix[graph_edges[0], graph_edges[1]] = 1.0

            # This is the bottleneck for large graphs
            for k in range(num_nodes):
                dist_ik = dist_matrix[:, k].unsqueeze(1)
                dist_kj = dist_matrix[k, :].unsqueeze(0)
                dist_matrix = torch.min(dist_matrix, dist_ik + dist_kj)

            shortest_paths.append(dist_matrix)

            # Track per-graph timing
            graph_time = time.time() - start_time
            self.timing_data['floyd_warshall_single_graph'].append(graph_time)

        return shortest_paths

    @time_function('extract_neighborhoods_batch')
    def extract_neighborhoods_batch(self, batch, shortest_paths):
        """Extract neighborhood subgraphs with detailed profiling"""
        all_features = defaultdict(dict)

        for graph_idx in range(batch.num_graphs):
            if graph_idx >= len(shortest_paths):
                continue

            start_time = time.time()

            mask = batch.batch == graph_idx
            num_nodes = mask.sum().item()
            node_labels = batch.node_labels[mask]
            dist_matrix = shortest_paths[graph_idx]

            # Extract neighborhoods for each radius
            neighborhoods = {}
            neighborhood_time = time.time()

            for r in range(self.r + 1):
                neighborhoods[r] = {}
                for center in range(num_nodes):
                    if r == 0:
                        neighborhoods[r][center] = [center]
                    else:
                        neighbors = torch.where(dist_matrix[center] <= r)[0].tolist()
                        neighborhoods[r][center] = sorted(neighbors)

            self.timing_data['neighborhood_extraction'].append(time.time() - neighborhood_time)

            # Hash neighborhoods
            hash_time = time.time()
            H = self._hash_neighborhoods_torch(num_nodes, node_labels, neighborhoods, dist_matrix)
            self.timing_data['neighborhood_hashing'].append(time.time() - hash_time)

            # Extract features for each distance level
            feature_time = time.time()
            for d_level in range(self.d + 1):
                for i in range(num_nodes):
                    for j in range(i, num_nodes):
                        distance = dist_matrix[i, j].item()

                        if distance == d_level and distance != float('inf'):
                            for r in range(self.r + 1):
                                hash_i = H.get((r, i), 0)
                                hash_j = H.get((r, j), 0)

                                if hash_i <= hash_j:
                                    feature_key = (hash_i, hash_j)
                                else:
                                    feature_key = (hash_j, hash_i)

                                level_key = (r, d_level)
                                if level_key not in all_features:
                                    all_features[level_key] = defaultdict(int)

                                all_features[level_key][(graph_idx, feature_key)] += 1

            self.timing_data['feature_extraction'].append(time.time() - feature_time)
            self.timing_data['single_graph_processing'].append(time.time() - start_time)

        return all_features

    @time_function('hash_neighborhoods_torch')
    def _hash_neighborhoods_torch(self, num_nodes, node_labels, neighborhoods, dist_matrix):
        """Hash neighborhoods with profiling"""
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

    def profile_memory_usage(self, func, *args, **kwargs):
        """Profile memory usage of a function"""

        def wrapper():
            return func(*args, **kwargs)

        mem_usage = memory_usage(wrapper, interval=0.1)
        return max(mem_usage) - min(mem_usage), mem_usage

    def benchmark_scalability(self, trajectory_file, topology_file,
                              frame_counts=[100, 500, 1000, 2000],
                              selection='name CA'):
        """Benchmark scalability with different numbers of frames"""
        print("Benchmarking scalability...")

        scalability_results = {
            'frame_counts': [],
            'total_times': [],
            'memory_usage': [],
            'frames_per_second': []
        }

        for frame_count in frame_counts:
            print(f"\nTesting with {frame_count} frames...")

            # Load limited trajectory
            traj = md.load(trajectory_file, top=topology_file)
            if frame_count < traj.n_frames:
                traj = traj[:frame_count]

            # Reset profiling data
            self.timing_data.clear()
            self.memory_data.clear()

            start_time = time.time()

            # Run the pipeline
            dataloader = self.trajectory_to_pyg_dataset(traj, selection=selection)
            kernel_matrix = self.fit_transform(dataloader)

            total_time = time.time() - start_time

            # Calculate metrics
            frames_per_second = frame_count / total_time
            peak_memory = max([max(values) if values else 0 for values in self.memory_data.values()])

            scalability_results['frame_counts'].append(frame_count)
            scalability_results['total_times'].append(total_time)
            scalability_results['memory_usage'].append(peak_memory)
            scalability_results['frames_per_second'].append(frames_per_second)

            print(f"Total time: {total_time:.2f}s")
            print(f"Frames per second: {frames_per_second:.2f}")
            print(f"Peak memory: {peak_memory / 1e6:.2f} MB")

        return scalability_results

    def create_profiling_report(self, output_dir="profiling_results"):
        """Create comprehensive profiling report"""
        os.makedirs(output_dir, exist_ok=True)

        # Timing analysis
        timing_df = pd.DataFrame([
            {
                'Function': func_name,
                'Total_Time': sum(times),
                'Mean_Time': np.mean(times),
                'Std_Time': np.std(times),
                'Min_Time': min(times),
                'Max_Time': max(times),
                'Call_Count': len(times)
            }
            for func_name, times in self.timing_data.items()
        ])

        timing_df = timing_df.sort_values('Total_Time', ascending=False)
        timing_df.to_csv(f"{output_dir}/timing_analysis.csv", index=False)

        # Memory analysis
        memory_df = pd.DataFrame([
            {
                'Function': func_name,
                'Total_Memory': sum(memories),
                'Mean_Memory': np.mean(memories),
                'Peak_Memory': max(memories) if memories else 0,
                'Call_Count': len(memories)
            }
            for func_name, memories in self.memory_data.items()
        ])

        memory_df = memory_df.sort_values('Peak_Memory', ascending=False)
        memory_df.to_csv(f"{output_dir}/memory_analysis.csv", index=False)

        # Create visualizations
        self._create_profiling_plots(timing_df, memory_df, output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("PROFILING SUMMARY")
        print("=" * 60)
        print("\nTop 5 Time Consumers:")
        for _, row in timing_df.head().iterrows():
            print(f"  {row['Function']}: {row['Total_Time']:.3f}s ({row['Call_Count']} calls)")

        print("\nTop 5 Memory Consumers:")
        for _, row in memory_df.head().iterrows():
            print(f"  {row['Function']}: {row['Peak_Memory'] / 1e6:.1f} MB peak")

        return timing_df, memory_df

    def _create_profiling_plots(self, timing_df, memory_df, output_dir):
        """Create profiling visualization plots"""

        # Timing breakdown pie chart
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        top_functions = timing_df.head(8)
        plt.pie(top_functions['Total_Time'], labels=top_functions['Function'], autopct='%1.1f%%')
        plt.title('Time Distribution by Function')

        # Memory usage bar chart
        plt.subplot(2, 2, 2)
        top_memory = memory_df.head(8)
        plt.bar(range(len(top_memory)), top_memory['Peak_Memory'] / 1e6)
        plt.xticks(range(len(top_memory)), top_memory['Function'], rotation=45, ha='right')
        plt.ylabel('Peak Memory (MB)')
        plt.title('Peak Memory Usage by Function')

        # Function call frequency
        plt.subplot(2, 2, 3)
        plt.bar(range(len(timing_df)), timing_df['Call_Count'])
        plt.xticks(range(len(timing_df)), timing_df['Function'], rotation=45, ha='right')
        plt.ylabel('Number of Calls')
        plt.title('Function Call Frequency')

        # Time per call
        plt.subplot(2, 2, 4)
        plt.bar(range(len(timing_df)), timing_df['Mean_Time'])
        plt.xticks(range(len(timing_df)), timing_df['Function'], rotation=45, ha='right')
        plt.ylabel('Mean Time per Call (s)')
        plt.title('Average Time per Function Call')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/profiling_overview.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Detailed timing plot
        plt.figure(figsize=(15, 6))

        # Box plot of timing distributions
        timing_data_for_plot = [times for times in self.timing_data.values() if times]
        function_names = [name for name, times in self.timing_data.items() if times]

        plt.boxplot(timing_data_for_plot, labels=function_names)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Execution Time (s)')
        plt.title('Distribution of Execution Times by Function')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_detailed_profiler(self, func, *args, **kwargs):
        """Run cProfile on a specific function"""
        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()

        # Create string buffer to capture profiler output
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()

        return result, s.getvalue()

    # Include the rest of your NSPDK implementation here
    # (fit, transform, fit_transform methods with @time_function decorators)

    @time_function('fit')
    def fit(self, dataloader):
        """Fit method with profiling"""
        # Your existing fit implementation with profiling decorators
        pass

    @time_function('fit_transform')
    def fit_transform(self, dataloader):
        """Fit transform method with profiling"""
        # Your existing fit_transform implementation
        pass


def run_comprehensive_benchmark(trajectory_file, topology_file):
    """Run comprehensive benchmarking suite"""

    print("Starting comprehensive NSPDK benchmarking...")

    # Initialize profiled NSPDK
    nspdk = ProfiledPyGNSPDK(r=2, d=3, device='cuda', batch_size=16)

    # 1. Scalability benchmark
    print("\n1. Running scalability benchmark...")
    scalability_results = nspdk.benchmark_scalability(
        trajectory_file, topology_file,
        frame_counts=[50, 100, 200, 500],
        selection='name CA'
    )

    # 2. Detailed profiling on medium dataset
    print("\n2. Running detailed profiling...")
    traj = md.load(trajectory_file, top=topology_file)[:200]  # Use 200 frames

    dataloader = nspdk.trajectory_to_pyg_dataset(traj, selection='name CA')

    # Profile the main computation
    result, profile_output = nspdk.run_detailed_profiler(
        nspdk.fit_transform, dataloader
    )

    # 3. Create profiling report
    print("\n3. Creating profiling report...")
    timing_df, memory_df = nspdk.create_profiling_report()

    # 4. Save detailed cProfile output
    with open("profiling_results/detailed_profile.txt", "w") as f:
        f.write(profile_output)

    # 5. Identify bottlenecks and recommendations
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS & RECOMMENDATIONS")
    print("=" * 60)

    # Identify the biggest bottlenecks
    biggest_time_consumer = timing_df.iloc[0]
    print(f"\nBiggest time bottleneck: {biggest_time_consumer['Function']}")
    print(f"  - Takes {biggest_time_consumer['Total_Time']:.2f}s total")
    print(f"  - Called {biggest_time_consumer['Call_Count']} times")
    print(f"  - Average {biggest_time_consumer['Mean_Time']:.4f}s per call")

    biggest_memory_consumer = memory_df.iloc[0]
    print(f"\nBiggest memory bottleneck: {biggest_memory_consumer['Function']}")
    print(f"  - Peak usage: {biggest_memory_consumer['Peak_Memory'] / 1e6:.1f} MB")

    # Provide optimization recommendations
    print("\nOPTIMIZATION RECOMMENDATIONS:")

    if 'floyd_warshall' in biggest_time_consumer['Function']:
        print("  1. Floyd-Warshall is O(n³) - consider using sparse shortest paths")
        print("  2. Implement early termination for max distance")
        print("  3. Use batched sparse operations")

    if 'neighborhood_hashing' in timing_df['Function'].values:
        print("  4. Consider caching hash computations")
        print("  5. Use more efficient string operations")

    if biggest_memory_consumer['Peak_Memory'] > 1e9:  # > 1GB
        print("  6. Reduce batch size to manage memory")
        print("  7. Implement gradient checkpointing")
        print("  8. Use memory mapping for large datasets")

    print("\nDetailed profiling results saved to 'profiling_results/' directory")

    return {
        'scalability': scalability_results,
        'timing_analysis': timing_df,
        'memory_analysis': memory_df,
        'detailed_profile': profile_output
    }


# Example usage
if __name__ == "__main__":
    # Run the comprehensive benchmark
    trajectory_file = "your_trajectory.xtc"
    topology_file = "your_topology.pdb"

    try:
        results = run_comprehensive_benchmark(trajectory_file, topology_file)
        print("\nBenchmarking completed successfully!")
    except Exception as e:
        print(f"Error during benchmarking: {str(e)}")
        import traceback

        traceback.print_exc()
