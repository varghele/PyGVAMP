import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, HDBSCAN, Birch
from sklearn.manifold import TSNE
from torch.backends.mkl import verbose
from torch_geometric.data import Data
from sklearn.metrics import silhouette_score
import time

from pygv.dataset.vampnet_dataset import VAMPNetDataset  # Use use_amino_acid_encoding=True for AA features
from pygv.clustering.graph2vec import Graph2Vec  # Replace with actual import
import os
import glob


def test_trajectory_graph2vec_minibatch(trajectory_path, topology_file, max_trajectories=10):
    """Test Graph2Vec on real trajectory data with MiniBatchKMeans clustering and partial_fit."""

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
        stride=1,  # Take every 10th frame for speed
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

    print(f"Embeddings shape: {embeddings.shape}")

    # MiniBatchKMeans clustering with partial_fit
    print("MiniBatchKMeans clustering with partial_fit...")
    start_time = time.time()

    # Initialize MiniBatchKMeans
    n_clusters = min(8, len(embeddings) // 100)  # Reasonable number of clusters
    minibatch_kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1000,  # Process 1000 samples at a time
        max_iter=100,
        random_state=42,
        reassignment_ratio=0.01,  # Controls how often cluster centers are reassigned
        max_no_improvement=20,  # Stop if no improvement for 10 iterations
        verbose=1  # Show progress
    )

    # Use partial_fit for incremental learning
    batch_size = 1000
    n_batches = (len(embeddings) + batch_size - 1) // batch_size

    print(f"Processing {len(embeddings)} embeddings in {n_batches} batches of size {batch_size}")

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(embeddings))
        batch_embeddings = embeddings[start_idx:end_idx]

        print(f"Processing batch {i + 1}/{n_batches} (samples {start_idx}:{end_idx})")

        # Use partial_fit for incremental learning
        minibatch_kmeans.partial_fit(batch_embeddings)

    # Get final cluster labels for all data
    print("Getting final cluster labels...")
    cluster_labels = minibatch_kmeans.predict(embeddings)

    end_time = time.time()

    print(f"MiniBatchKMeans clustering completed in {end_time - start_time:.2f} seconds")
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

    perplexity = min(30, len(sample_embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(sample_embeddings)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot 1: Clusters
    plt.subplot(1, 3, 1)
    scatter1 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=sample_labels, cmap='tab20', alpha=0.7, s=20)
    plt.title(f'MiniBatchKMeans Clusters (n={n_clusters})')
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

    # Plot 3: Trajectory path
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

    plt.title('Trajectory Path (MiniBatchKMeans)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    plt.tight_layout()
    plt.show()

    # Print cluster summary
    print(f"\nMiniBatchKMeans Clustering Results:")
    print(f"Number of frames: {len(embeddings)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Batch size used: {minibatch_kmeans.batch_size}")

    # Analyze cluster characteristics
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster size distribution:")
    for label, count in zip(unique_labels, counts):
        cluster_frames = np.where(cluster_labels == label)[0]
        frame_range = f"{cluster_frames.min()}-{cluster_frames.max()}" if len(cluster_frames) > 1 else str(
            cluster_frames[0])
        percentage = (count / len(cluster_labels)) * 100
        print(f"Cluster {label}: {count} frames ({percentage:.1f}%), frame range: {frame_range}")

    return embeddings, cluster_labels, minibatch_kmeans


def test_birch_hdbscan_pipeline(trajectory_path, topology_file, max_trajectories=10):
    """Test BIRCH as preprocessing for HDBSCAN on high-dimensional embeddings."""

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
        lag_time=10,
        n_neighbors=20,
        node_embedding_dim=16,
        gaussian_expansion_dim=8,
        selection="name CA",
        stride=10,
        chunk_size=100,
        cache_dir="./cache",
        use_cache=False
    )

    # Get individual frames dataset
    frames_dataset = dataset.get_AA_frames(return_pairs=False)

    print(f"Dataset contains {len(frames_dataset)} frames")

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

    print(f"Original embeddings shape: {embeddings.shape}")

    # Step 1: Use BIRCH for dimensionality reduction/preprocessing
    print("Step 1: BIRCH preprocessing for dimensionality reduction...")
    start_time = time.time()

    # Use BIRCH to create subclusters (not final clusters)
    birch_preprocessor = Birch(
        n_clusters=None,  # Don't create final clusters yet
        threshold=0.01,  # Lower threshold for more subclusters
        branching_factor=50,
        compute_labels=False  # We only want the subcluster centers
    )

    # Fit BIRCH incrementally
    batch_size = 1000
    n_batches = (len(embeddings) + batch_size - 1) // batch_size

    print(f"Processing {len(embeddings)} embeddings in {n_batches} batches for BIRCH preprocessing")

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(embeddings))
        batch_embeddings = embeddings[start_idx:end_idx]

        print(f"BIRCH batch {i + 1}/{n_batches} (samples {start_idx}:{end_idx})")
        birch_preprocessor.partial_fit(batch_embeddings)

    # Get subcluster centers as reduced representation
    subcluster_centers = birch_preprocessor.subcluster_centers_
    print(f"BIRCH created {len(subcluster_centers)} subclusters")
    print(f"Reduced from {embeddings.shape} to {subcluster_centers.shape}")

    birch_time = time.time() - start_time
    print(f"BIRCH preprocessing completed in {birch_time:.2f} seconds")

    # Step 2: Apply HDBSCAN to the reduced representation
    print("Step 2: HDBSCAN clustering on BIRCH subclusters...")
    hdbscan_start = time.time()

    hdbscan = HDBSCAN(
        min_cluster_size=max(3, len(subcluster_centers) // 20),
        min_samples=2,
        metric='cosine',
        cluster_selection_epsilon=0.05,
        n_jobs=-1
    )

    # Cluster the subcluster centers
    subcluster_labels = hdbscan.fit_predict(subcluster_centers)
    n_clusters = len(set(subcluster_labels)) - (1 if -1 in subcluster_labels else 0)

    hdbscan_time = time.time() - hdbscan_start
    total_time = birch_time + hdbscan_time

    print(f"HDBSCAN clustering completed in {hdbscan_time:.2f} seconds")
    print(f"Total BIRCH+HDBSCAN time: {total_time:.2f} seconds")
    print(f"Found {n_clusters} clusters in subclusters")

    # Step 3: Assign original points to clusters via BIRCH
    print("Step 3: Assigning original points to clusters...")

    # Get subcluster assignments for original points
    original_subcluster_assignments = birch_preprocessor.predict(embeddings)

    # Map original points to final clusters through subclusters
    final_labels = np.full(len(embeddings), -1)  # Initialize with noise label

    for i, subcluster_id in enumerate(original_subcluster_assignments):
        if subcluster_id < len(subcluster_labels):
            final_labels[i] = subcluster_labels[subcluster_id]

    print(f"Final clustering: {len(set(final_labels)) - (1 if -1 in final_labels else 0)} clusters")
    print(
        f"Noise points: {list(final_labels).count(-1)} ({100 * list(final_labels).count(-1) / len(final_labels):.1f}%)")

    # Calculate silhouette score on a sample
    if n_clusters > 1:
        sample_size = min(5000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        sample_labels = final_labels[sample_indices]

        # Filter out noise points for silhouette score
        non_noise_mask = sample_labels != -1
        if np.sum(non_noise_mask) > 1 and len(set(sample_labels[non_noise_mask])) > 1:
            sil_score = silhouette_score(sample_embeddings[non_noise_mask], sample_labels[non_noise_mask])
            print(f"Silhouette score: {sil_score:.3f}")

    # Visualization
    print("Creating visualization...")
    sample_size = min(5000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    sample_labels = final_labels[sample_indices]

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(sample_embeddings)

    plt.figure(figsize=(12, 4))

    # Plot 1: Final clusters
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=sample_labels, cmap='tab20', alpha=0.7, s=20)
    plt.title(f'BIRCH+HDBSCAN Clusters (n={n_clusters})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter1, label='Cluster')

    # Plot 2: Subcluster centers
    plt.subplot(1, 2, 2)
    if len(subcluster_centers) <= 1000:  # Only plot if not too many
        subcluster_2d = tsne.fit_transform(subcluster_centers)
        scatter2 = plt.scatter(subcluster_2d[:, 0], subcluster_2d[:, 1],
                               c=subcluster_labels, cmap='tab20', alpha=0.8, s=50, marker='s')
        plt.title(f'BIRCH Subclusters (n={len(subcluster_centers)})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter2, label='Subcluster')
    else:
        plt.text(0.5, 0.5, f'Too many subclusters\nto visualize\n({len(subcluster_centers)})',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('BIRCH Subclusters (too many to plot)')

    plt.tight_layout()
    plt.show()

    return embeddings, final_labels, birch_preprocessor, hdbscan


if __name__ == "__main__":
    # Update these paths for your data
    # trajectory_path = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/r1')
    # topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    trajectory_path = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/r1')
    topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/topol.pdb')

    print("=" * 60)
    print("TESTING MINIBATCHKMEANS WITH PARTIAL_FIT")
    print("=" * 60)

    try:
        embeddings1, clusters1, model1 = test_trajectory_graph2vec_minibatch(
            trajectory_path=trajectory_path,
            topology_file=topology_file,
            max_trajectories=100000
        )

        print("\n✅ SUCCESS: MiniBatchKMeans trajectory analysis completed!")

    except Exception as e:
        print(f"❌ ERROR in MiniBatchKMeans: {str(e)}")

    print("\n" + "=" * 60)
    print("TESTING BIRCH + HDBSCAN PIPELINE")
    print("=" * 60)

    try:
        embeddings2, clusters2, birch_model, hdbscan_model = test_birch_hdbscan_pipeline(
            trajectory_path=trajectory_path,
            topology_file=topology_file,
            max_trajectories=100000
        )

        print("\n✅ SUCCESS: BIRCH+HDBSCAN trajectory analysis completed!")

    except Exception as e:
        print(f"❌ ERROR in BIRCH+HDBSCAN: {str(e)}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("MiniBatchKMeans:")
    print("  ✅ Excellent for high-dimensional data")
    print("  ✅ Memory efficient with partial_fit")
    print("  ✅ Fast and scalable")
    print("  ❌ Requires pre-specifying number of clusters")

    print("\nBIRCH + HDBSCAN:")
    print("  ✅ Automatic cluster detection")
    print("  ✅ Handles noise points")
    print("  ✅ Good for varying cluster densities")
    print("  ❌ More complex pipeline")
    print("  ❌ BIRCH preprocessing adds overhead")
