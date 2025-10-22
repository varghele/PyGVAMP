import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.manifold import TSNE
from torch.backends.mkl import verbose
from torch_geometric.data import Data
from sklearn.metrics import silhouette_score
import time


# from area52.create_dataset import topology_file
from pygv.dataset.vampnet_dataset_with_AA import VAMPNetDataset
from pygv.clustering.graph2vec import Graph2Vec  # Replace with actual import
import os
import glob


def test_trajectory_graph2vec_birch(trajectory_path, topology_file, max_trajectories=10):
    """Test Graph2Vec on real trajectory data with BIRCH clustering and partial_fit."""

    # Find trajectory files
    trajectory_files = glob.glob(os.path.join(trajectory_path, "*.xtc"))[:max_trajectories]

    if not trajectory_files:
        print(f"No .xtc files found in {trajectory_path}")
        return

    print(f"Found {len(trajectory_files)} trajectory files")

    # Create VAMPNet dataset
    print("Loading trajectory data...")
    dataset = VAMPNetDataset(
        trajectory_files=trajectory_files,
        topology_file=topology_file,
        lag_time=10,  # 1 ns lag time
        n_neighbors=20,
        node_embedding_dim=16,
        gaussian_expansion_dim=8,
        selection="name CA",
        stride=10,  # Take every 10th frame for speed
        chunk_size=100,
        cache_dir="./cache",
        use_cache=False
    )

    # Get individual frames dataset (no time-lagged pairs)
    frames_dataset = dataset.get_AA_frames(return_pairs=False)

    print(f"Dataset contains {len(frames_dataset)} frames")
    print(f"Each graph has {dataset.n_atoms} atoms")

    # Train Graph2Vec
    print("Training Graph2Vec...")
    model = Graph2Vec(
        embedding_dim=512,
        max_degree=3,
        epochs=10,
        batch_size=1024,
        min_count=5,
        negative_samples=50,
        learning_rate=0.025,
    )

    model.fit(frames_dataset, len(frames_dataset))
    embeddings = model.get_embeddings().numpy()

    # BIRCH clustering with partial_fit
    print("BIRCH clustering with partial_fit...")
    start_time = time.time()

    # Initialize BIRCH clusterer
    birch = Birch(
        n_clusters=None,  # Let BIRCH determine clusters automatically
        threshold=0.5,  # Distance threshold for merging subclusters
        branching_factor=50,  # Maximum number of subclusters in each node
        compute_labels=True,  # Compute cluster labels
        copy=True
    )

    # Use partial_fit for incremental learning
    batch_size = 1000  # Process embeddings in batches
    n_batches = (len(embeddings) + batch_size - 1) // batch_size

    print(f"Processing {len(embeddings)} embeddings in {n_batches} batches of size {batch_size}")

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(embeddings))
        batch_embeddings = embeddings[start_idx:end_idx]

        print(f"Processing batch {i + 1}/{n_batches} (samples {start_idx}:{end_idx})")

        # Use partial_fit for incremental learning
        birch.partial_fit(batch_embeddings)

    # Get final cluster labels for all data
    print("Getting final cluster labels...")
    cluster_labels = birch.predict(embeddings)

    # Get number of clusters
    n_clusters = len(set(cluster_labels))
    end_time = time.time()

    print(f"BIRCH clustering completed in {end_time - start_time:.2f} seconds")
    print(f"Found {n_clusters} clusters")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")

    # Calculate silhouette score if we have more than 1 cluster
    if n_clusters > 1:
        # For large datasets, sample for silhouette score calculation
        if len(embeddings) > 5000:
            sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_labels = cluster_labels[sample_indices]
            sil_score = silhouette_score(sample_embeddings, sample_labels)
            print(f"Silhouette score (sampled): {sil_score:.3f}")
        else:
            sil_score = silhouette_score(embeddings, cluster_labels)
            print(f"Silhouette score: {sil_score:.3f}")

    # Reduce to 2D for visualization
    print("Creating 2D visualization...")
    perplexity = min(30, len(embeddings) - 1)

    # For large datasets, sample for t-SNE
    if len(embeddings) > 5000:
        sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
        sample_embeddings = embeddings[sample_indices]
        sample_labels = cluster_labels[sample_indices]
        sample_frame_indices = sample_indices
        print(f"Using {len(sample_embeddings)} samples for t-SNE visualization")
    else:
        sample_embeddings = embeddings
        sample_labels = cluster_labels
        sample_frame_indices = np.arange(len(embeddings))

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(sample_embeddings)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot 1: Clusters
    plt.subplot(1, 3, 1)
    scatter1 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=sample_labels, cmap='tab20', alpha=0.7, s=20)
    plt.title(f'BIRCH Trajectory Clusters (n={n_clusters})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter1, label='Cluster')

    # Plot 2: Time evolution
    plt.subplot(1, 3, 2)
    scatter2 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=sample_frame_indices, cmap='viridis', alpha=0.7, s=20)
    plt.title('Trajectory Time Evolution')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter2, label='Frame Index')

    # Plot 3: Trajectory path (if enough points)
    plt.subplot(1, 3, 3)
    if len(embeddings_2d) > 10:
        # Sort by frame index for proper trajectory path
        sorted_indices = np.argsort(sample_frame_indices)
        sorted_embeddings_2d = embeddings_2d[sorted_indices]

        # Plot trajectory as connected path
        plt.plot(sorted_embeddings_2d[:, 0], sorted_embeddings_2d[:, 1], 'b-', alpha=0.3, linewidth=0.5)
        plt.scatter(sorted_embeddings_2d[0, 0], sorted_embeddings_2d[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(sorted_embeddings_2d[-1, 0], sorted_embeddings_2d[-1, 1], c='red', s=100, marker='s', label='End')
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=sample_frame_indices, cmap='viridis', s=20)

    plt.title('Trajectory Path (BIRCH)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    plt.tight_layout()
    plt.show()

    # Print cluster summary
    print(f"\nBIRCH Clustering Results:")
    print(f"Number of frames: {len(embeddings)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Threshold used: {birch.threshold}")
    print(f"Branching factor: {birch.branching_factor}")

    # Analyze cluster characteristics
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster size distribution:")
    for label, count in zip(unique_labels, counts):
        cluster_frames = np.where(cluster_labels == label)[0]
        frame_range = f"{cluster_frames.min()}-{cluster_frames.max()}" if len(cluster_frames) > 1 else str(
            cluster_frames[0])
        percentage = (count / len(cluster_labels)) * 100
        print(f"Cluster {label}: {count} frames ({percentage:.1f}%), frame range: {frame_range}")

    # BIRCH-specific information
    print(f"\nBIRCH Model Information:")
    print(f"Number of subclusters: {birch.n_features_in_}")
    if hasattr(birch, 'subcluster_centers_'):
        print(f"Subcluster centers shape: {birch.subcluster_centers_.shape}")

    return embeddings, cluster_labels, birch


def compare_birch_parameters(embeddings, parameter_sets):
    """Compare different BIRCH parameter configurations."""
    print("\nComparing BIRCH parameter configurations...")

    results = []

    for i, params in enumerate(parameter_sets):
        print(f"\nTesting parameter set {i + 1}: {params}")

        start_time = time.time()

        # Create BIRCH with current parameters
        birch = Birch(**params)

        # Use partial_fit for incremental learning
        batch_size = 1000
        n_batches = (len(embeddings) + batch_size - 1) // batch_size

        for j in range(n_batches):
            start_idx = j * batch_size
            end_idx = min((j + 1) * batch_size, len(embeddings))
            batch_embeddings = embeddings[start_idx:end_idx]
            birch.partial_fit(batch_embeddings)

        # Get cluster labels
        cluster_labels = birch.predict(embeddings)
        n_clusters = len(set(cluster_labels))

        end_time = time.time()

        # Calculate silhouette score (sample if too large)
        if len(embeddings) > 5000:
            sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
            sil_score = silhouette_score(embeddings[sample_indices], cluster_labels[sample_indices])
        else:
            sil_score = silhouette_score(embeddings, cluster_labels) if n_clusters > 1 else -1

        results.append({
            'params': params,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'time': end_time - start_time,
            'cluster_sizes': np.bincount(cluster_labels)
        })

        print(f"  Clusters: {n_clusters}, Silhouette: {sil_score:.3f}, Time: {end_time - start_time:.2f}s")

    return results


if __name__ == "__main__":
    # Update these paths for your data
    #trajectory_path = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/r1')
    #topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    trajectory_path = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/r1')
    topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/topol.pdb')

    # Test with trajectory data
    try:
        embeddings, clusters, birch_model = test_trajectory_graph2vec_birch(
            trajectory_path=trajectory_path,
            topology_file=topology_file,
            max_trajectories=100000  # Limit for testing
        )

        print("\n✅ SUCCESS: Graph2Vec trajectory analysis with BIRCH completed!")

        # Optional: Compare different BIRCH parameter configurations
        print("\n" + "=" * 60)
        print("PARAMETER COMPARISON")
        print("=" * 60)

        parameter_sets = [
            {'threshold': 0.3, 'branching_factor': 50, 'n_clusters': None},
            {'threshold': 0.5, 'branching_factor': 50, 'n_clusters': None},
            {'threshold': 0.7, 'branching_factor': 50, 'n_clusters': None},
            {'threshold': 0.5, 'branching_factor': 30, 'n_clusters': None},
            {'threshold': 0.5, 'branching_factor': 100, 'n_clusters': None},
        ]

        comparison_results = compare_birch_parameters(embeddings, parameter_sets)

        # Find best configuration based on silhouette score
        best_config = max(comparison_results, key=lambda x: x['silhouette_score'])
        print(f"\nBest configuration: {best_config['params']}")
        print(f"Best silhouette score: {best_config['silhouette_score']:.3f}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("Make sure you have:")
        print("1. Trajectory files (.xtc) in the specified directory")
        print("2. A topology file (.pdb, .gro, etc.)")
        print("3. MDTraj installed: pip install mdtraj")
