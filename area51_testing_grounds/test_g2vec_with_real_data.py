import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, OPTICS, HDBSCAN
from sklearn.manifold import TSNE
from torch.backends.mkl import verbose
from torch_geometric.data import Data
from sklearn.metrics import silhouette_score
import time

from torch_geometric.graphgym import compute_loss

#from area52.create_dataset import topology_file
from pygv.dataset.vampnet_dataset_with_AA import VAMPNetDataset
from pygv.clustering.graph2vec import Graph2Vec  # Replace with actual import
import os
import glob


def test_trajectory_graph2vec(trajectory_path, topology_file, max_trajectories=10):
    """Test Graph2Vec on real trajectory data with clustering and visualization."""

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
        n_neighbors=10,
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
        embedding_dim=4098,
        max_degree=2,
        epochs=100,
        batch_size=1024,
        min_count=5,
        negative_samples=50,
        learning_rate=0.025,
    )

    model.fit(frames_dataset, len(frames_dataset))
    embeddings = model.get_embeddings().numpy()

    # Cluster embeddings
    print("Clustering trajectories...")

    # Cluster embeddings with timing
    print("Clustering trajectories...")
    start_time = time.time()

    # Use verbose output and timing
    hdbscan = HDBSCAN(
        min_cluster_size=10,  # Minimum points to form a cluster
        min_samples=5,  # Conservative noise threshold
        metric='cosine',  # Good for high-dimensional embeddings
        cluster_selection_epsilon=0.01,  # Helps merge close clusters
        cluster_selection_method='eom',  # Excess of Mass (default, usually best)
        n_jobs=-2,  # Use all available cores
    )

    print("HDBSCAN clustering started...")
    cluster_labels = hdbscan.fit_predict(embeddings)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    end_time = time.time()

    print(f"HDBSCAN clustering completed in {end_time - start_time:.2f} seconds")
    print(f"Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
    print(
        f"Noise points: {list(cluster_labels).count(-1)} ({100 * list(cluster_labels).count(-1) / len(cluster_labels):.1f}%)")

    # Calculate silhouette score
    if n_clusters > 1:
        sil_score = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette score: {sil_score:.3f}")

    # Reduce to 2D for visualization
    print("Creating 2D visualization...")
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create time-based coloring (frame index)
    frame_indices = np.arange(len(embeddings))

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot 1: Clusters
    plt.subplot(1, 3, 1)
    scatter1 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=cluster_labels, cmap='tab10', alpha=0.7, s=20)
    plt.title(f'Trajectory Clusters (n={n_clusters})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter1, label='Cluster')

    # Plot 2: Time evolution
    plt.subplot(1, 3, 2)
    scatter2 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=frame_indices, cmap='viridis', alpha=0.7, s=20)
    plt.title('Trajectory Time Evolution')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter2, label='Frame Index')

    # Plot 3: Trajectory path (if enough points)
    plt.subplot(1, 3, 3)
    if len(embeddings_2d) > 10:
        # Plot trajectory as connected path
        plt.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'b-', alpha=0.3, linewidth=0.5)
        plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], c='red', s=100, marker='s', label='End')
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=frame_indices, cmap='viridis', s=20)

    plt.title('Trajectory Path')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    plt.tight_layout()
    plt.show()

    # Print cluster summary
    print(f"\nClustering Results:")
    print(f"Number of frames: {len(embeddings)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        cluster_frames = np.where(cluster_labels == i)[0]
        frame_range = f"{cluster_frames.min()}-{cluster_frames.max()}" if len(cluster_frames) > 1 else str(
            cluster_frames[0])
        print(f"Cluster {i}: {cluster_size} frames, frame range: {frame_range}")

    return embeddings, cluster_labels


if __name__ == "__main__":
    # Update these paths for your data
    #trajectory_path = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/r1')
    #topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/topol.pdb')
    trajectory_path = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/r1')
    topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    #trajectory_path = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/r0/')
    #topology_file = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/prot.pdb')

    # Test with trajectory data
    try:
        embeddings, clusters = test_trajectory_graph2vec(
            trajectory_path=trajectory_path,
            topology_file=topology_file,
            max_trajectories=100000  # Limit for testing
        )

        print("\n✅ SUCCESS: Graph2Vec trajectory analysis completed!")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("Make sure you have:")
        print("1. Trajectory files (.xtc) in the specified directory")
        print("2. A topology file (.pdb, .gro, etc.)")
        print("3. MDTraj installed: pip install mdtraj")