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
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
from scipy.sparse import csr_matrix

import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse, k_hop_subgraph


def time_function(func_name):
    """Decorator factory to time function execution"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()

            # Memory before (simplified)
            mem_before = psutil.Process().memory_info().rss if hasattr(psutil.Process(), 'memory_info') else 0

            result = func(self, *args, **kwargs)

            # Memory after (simplified)
            mem_after = psutil.Process().memory_info().rss if hasattr(psutil.Process(), 'memory_info') else 0

            end_time = time.time()

            # Store timing data
            execution_time = end_time - start_time
            self.timing_data[func_name].append(execution_time)
            self.operation_counts[func_name] += 1

            return result

        return wrapper

    return decorator


class ProfiledPyGNSPDK:
    """
    PyTorch Geometric NSPDK with simplified profiling capabilities
    """

    def __init__(self, r=3, d=4, device='cuda', batch_size=32):
        self.r = r
        self.d = d
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # Profiling data
        self.timing_data = defaultdict(list)
        self.operation_counts = defaultdict(int)

        # Storage for fitted data (matching grakel's approach)
        self._fit_keys = {}
        self._ngx = 0
        self.X = {}
        self._X_level_norm_factor = {}

    @time_function('trajectory_to_pyg_dataset')
    def trajectory_to_pyg_dataset(self, trajectory, selection='name CA'):
        """Convert MD trajectory to PyTorch Geometric dataset with DataLoader"""
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
        """Compute shortest paths for a batch of graphs"""
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

            # Floyd-Warshall algorithm - THIS IS THE MAIN BOTTLENECK
            dist_matrix = torch.full((num_nodes, num_nodes), float('inf'), device=self.device)
            torch.diagonal(dist_matrix).fill_(0.0)

            if graph_edges.size(1) > 0:
                dist_matrix[graph_edges[0], graph_edges[1]] = 1.0

            # O(n^3) operation - major bottleneck for large graphs
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

    @time_function('fit')
    def fit(self, dataloader):
        """Fit the NSPDK kernel on trajectory data"""
        print("Fitting NSPDK kernel...")

        all_features = defaultdict(dict)
        all_keys = defaultdict(dict)
        graph_count = 0

        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = batch.to(self.device)

            # Compute shortest paths for this batch
            shortest_paths = self.compute_shortest_paths_batch(batch)

            # Extract features for this batch
            batch_features = self.extract_neighborhoods_batch(batch, shortest_paths)

            # Merge features with global feature dictionary
            for level_key, features in batch_features.items():
                for (local_graph_idx, feature_key), count in features.items():
                    global_graph_idx = graph_count + local_graph_idx

                    # Index feature keys
                    keys = all_keys[level_key]
                    if feature_key not in keys:
                        keys[feature_key] = len(keys)

                    feature_idx = keys[feature_key]
                    all_features[level_key][(global_graph_idx, feature_idx)] = count

            graph_count += batch.num_graphs

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Convert to sparse matrices (matching grakel's format)
        self.X = {}
        for level_key, features in all_features.items():
            if len(features) > 0:
                indices, data = zip(*features.items())
                rows, cols = zip(*indices)

                self.X[level_key] = csr_matrix(
                    (data, (rows, cols)),
                    shape=(graph_count, len(all_keys[level_key])),
                    dtype=np.int64
                )

        self._fit_keys = all_keys
        self._ngx = graph_count

        print(f"Fitted NSPDK on {graph_count} graphs with {len(self.X)} feature levels")
        return self

    @time_function('compute_kernel_matrix')
    def compute_kernel_matrix(self):
        """Compute the kernel matrix from fitted features"""
        print("Computing kernel matrix...")

        # Compute normalization factors
        self._X_level_norm_factor = {
            key: np.array(M.power(2).sum(-1)).flatten()
            for key, M in self.X.items()
        }

        # Compute kernel matrix
        S = np.zeros((self._ngx, self._ngx))

        for level_key, M in self.X.items():
            # Compute dot product kernel for this level
            K = M.dot(M.T).toarray()
            K_diag = K.diagonal()

            # Normalize
            with np.errstate(divide='ignore', invalid='ignore'):
                Q = K / np.sqrt(np.outer(K_diag, K_diag))
                Q = np.nan_to_num(Q, nan=1.0)

            S += Q

        # Average over all levels
        if len(self.X) > 0:
            S /= len(self.X)

        return S

    @time_function('fit_transform')
    def fit_transform(self, dataloader):
        """Fit and transform in one step"""
        self.fit(dataloader)
        return self.compute_kernel_matrix()

    def create_profiling_report(self, output_dir="profiling_results"):
        """Create simplified profiling report"""
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

        # Create visualizations
        self._create_profiling_plots(timing_df, output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("PROFILING SUMMARY")
        print("=" * 60)
        print("\nTop 5 Time Consumers:")
        for _, row in timing_df.head().iterrows():
            print(f"  {row['Function']}: {row['Total_Time']:.3f}s ({row['Call_Count']} calls)")

        return timing_df

    def _create_profiling_plots(self, timing_df, output_dir):
        """Create profiling visualization plots"""

        # Timing breakdown pie chart
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        top_functions = timing_df.head(8)
        plt.pie(top_functions['Total_Time'], labels=top_functions['Function'], autopct='%1.1f%%')
        plt.title('Time Distribution by Function')

        # Function call frequency
        plt.subplot(2, 2, 2)
        plt.bar(range(len(timing_df)), timing_df['Call_Count'])
        plt.xticks(range(len(timing_df)), timing_df['Function'], rotation=45, ha='right')
        plt.ylabel('Number of Calls')
        plt.title('Function Call Frequency')

        # Time per call
        plt.subplot(2, 2, 3)
        plt.bar(range(len(timing_df)), timing_df['Mean_Time'])
        plt.xticks(range(len(timing_df)), timing_df['Function'], rotation=45, ha='right')
        plt.ylabel('Mean Time per Call (s)')
        plt.title('Average Time per Function Call')

        # Total time comparison
        plt.subplot(2, 2, 4)
        plt.bar(range(len(timing_df)), timing_df['Total_Time'])
        plt.xticks(range(len(timing_df)), timing_df['Function'], rotation=45, ha='right')
        plt.ylabel('Total Time (s)')
        plt.title('Total Time by Function')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/profiling_overview.png", dpi=300, bbox_inches='tight')
        plt.close()

    def benchmark_scalability(self, trajectory_file, topology_file,
                              frame_counts=[100, 500, 1000],
                              selection='name CA'):
        """Benchmark scalability with different numbers of frames"""
        print("Benchmarking scalability...")

        scalability_results = {
            'frame_counts': [],
            'total_times': [],
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

            start_time = time.time()

            # Run the pipeline
            dataloader = self.trajectory_to_pyg_dataset(traj, selection=selection)
            kernel_matrix = self.fit_transform(dataloader)

            total_time = time.time() - start_time

            # Calculate metrics
            frames_per_second = frame_count / total_time

            scalability_results['frame_counts'].append(frame_count)
            scalability_results['total_times'].append(total_time)
            scalability_results['frames_per_second'].append(frames_per_second)

            print(f"Total time: {total_time:.2f}s")
            print(f"Frames per second: {frames_per_second:.2f}")

        return scalability_results


def run_simplified_benchmark(trajectory_file, topology_file):
    """Run simplified benchmarking suite"""

    print("Starting simplified NSPDK benchmarking...")

    # Initialize profiled NSPDK
    nspdk = ProfiledPyGNSPDK(r=2, d=3, device='cuda', batch_size=16)

    # 1. Scalability benchmark
    print("\n1. Running scalability benchmark...")
    scalability_results = nspdk.benchmark_scalability(
        trajectory_file, topology_file,
        frame_counts=[50, 100, 200],
        selection='name CA'
    )

    # 2. Detailed profiling on medium dataset
    print("\n2. Running detailed profiling...")
    traj = md.load(trajectory_file, top=topology_file)[:200]  # Use 200 frames

    dataloader = nspdk.trajectory_to_pyg_dataset(traj, selection='name CA')

    # Profile the main computation
    start_time = time.time()
    result = nspdk.fit_transform(dataloader)
    total_time = time.time() - start_time

    # 3. Create profiling report
    print("\n3. Creating profiling report...")
    timing_df = nspdk.create_profiling_report()

    # 4. Identify bottlenecks and recommendations
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS & RECOMMENDATIONS")
    print("=" * 60)

    # Identify the biggest bottlenecks
    biggest_time_consumer = timing_df.iloc[0]
    print(f"\nBiggest time bottleneck: {biggest_time_consumer['Function']}")
    print(f"  - Takes {biggest_time_consumer['Total_Time']:.2f}s total")
    print(f"  - Called {biggest_time_consumer['Call_Count']} times")
    print(f"  - Average {biggest_time_consumer['Mean_Time']:.4f}s per call")

    # Provide optimization recommendations
    print("\nOPTIMIZATION RECOMMENDATIONS:")

    if 'floyd_warshall' in biggest_time_consumer['Function']:
        print("  1. Floyd-Warshall is O(n³) - consider using sparse shortest paths")
        print("  2. Implement early termination for max distance")
        print("  3. Use batched sparse operations")
        print("  4. Consider approximate distance methods for large graphs")

    if 'neighborhood_hashing' in timing_df['Function'].values:
        print("  5. Consider caching hash computations")
        print("  6. Use more efficient string operations")
        print("  7. Implement incremental hashing")

    if 'dataset_getitem' in timing_df['Function'].values:
        print("  8. Pre-compute distances for all frames")
        print("  9. Use memory mapping for large trajectories")
        print("  10. Implement parallel data loading")

    print("\nDetailed profiling results saved to 'profiling_results/' directory")

    return {
        'scalability': scalability_results,
        'timing_analysis': timing_df,
        'total_time': total_time
    }


# Example usage
if __name__ == "__main__":
    # Run the simplified benchmark
    #trajectory_file = "your_trajectory.xtc"
    #topology_file = "your_topology.pdb"

    # Parameters
    trajectory_file = os.path.expanduser(
        '~/PycharmProjects/DDVAMP/datasets/ATR/r0/traj0001.xtc')  # Your trajectory file
    topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/prot.pdb')  # Your topology file

    try:
        results = run_simplified_benchmark(trajectory_file, topology_file)
        print("\nBenchmarking completed successfully!")

        # Print key findings
        timing_df = results['timing_analysis']
        print(f"\nKey Findings:")
        print(f"- Total processing time: {results['total_time']:.2f} seconds")
        print(f"- Main bottleneck: {timing_df.iloc[0]['Function']}")
        print(
            f"- Bottleneck accounts for {timing_df.iloc[0]['Total_Time'] / results['total_time'] * 100:.1f}% of total time")

    except Exception as e:
        print(f"Error during benchmarking: {str(e)}")
        import traceback

        traceback.print_exc()
