"""
State Discovery Module for Unsupervised Molecular State Detection

This module provides unsupervised state discovery using Graph2Vec embeddings
and clustering analysis. It helps determine the optimal number of states
for VAMPNet training by analyzing the natural clustering structure in
molecular dynamics trajectory data.

Clustering is performed on both raw Graph2Vec embeddings and UMAP-reduced
embeddings, and the result with the better silhouette score is recommended.
"""

import os
import json
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from pygv.clustering.graph2vec import Graph2Vec


class StateDiscovery:
    """
    Unsupervised state discovery using Graph2Vec + clustering.

    This class trains a Graph2Vec model on molecular graphs to learn
    fixed-dimensional embeddings, then performs clustering analysis
    to recommend the optimal number of states.

    Clustering runs on both the raw high-dimensional embeddings and
    UMAP-reduced embeddings. The result with the better silhouette
    score is used for the final recommendation.

    Parameters
    ----------
    embedding_dim : int, default=64
        Dimension of Graph2Vec embeddings
    max_degree : int, default=2
        Maximum WL iteration depth for Graph2Vec
    g2v_epochs : int, default=50
        Number of training epochs for Graph2Vec
    g2v_min_count : int, default=5
        Minimum subgraph frequency to be included in Graph2Vec vocabulary
    umap_dim : int, default=2
        Dimensionality for UMAP reduction used in clustering
    min_k : int, default=2
        Minimum number of clusters to test
    max_k : int, default=15
        Maximum number of clusters to test
    random_state : int, default=42
        Random seed for reproducibility
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_degree: int = 2,
        g2v_epochs: int = 50,
        g2v_min_count: int = 5,
        umap_dim: int = 2,
        min_k: int = 2,
        max_k: int = 15,
        random_state: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.max_degree = max_degree
        self.g2v_epochs = g2v_epochs
        self.g2v_min_count = g2v_min_count
        self.umap_dim = umap_dim
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state

        # Results storage
        self.embeddings = None
        self.umap_embeddings = None
        self.inertias = {}
        self.silhouette_scores = {}
        self.cluster_labels = {}
        self.umap_inertias = {}
        self.umap_silhouette_scores = {}
        self.umap_cluster_labels = {}
        self.best_k = None
        self.elbow_k = None
        self.umap_best_k = None
        self.umap_elbow_k = None
        self.chosen_source = None  # 'raw' or 'umap'
        self.g2v_model = None

    def fit(self, frames_dataset) -> 'StateDiscovery':
        """
        Run Graph2Vec and clustering analysis.

        Parameters
        ----------
        frames_dataset : Dataset
            PyTorch dataset returning individual frame graphs

        Returns
        -------
        StateDiscovery
            Self for method chaining
        """
        print("\n=== STATE DISCOVERY ===")

        # Step 1: Train Graph2Vec
        print("\n--- Training Graph2Vec ---")
        num_graphs = len(frames_dataset)
        print(f"Training on {num_graphs} graphs")

        self.g2v_model = Graph2Vec(
            embedding_dim=self.embedding_dim,
            max_degree=self.max_degree,
            epochs=self.g2v_epochs,
            min_count=self.g2v_min_count,
            batch_size=64,
            num_workers=4,
        )
        self.g2v_model.fit(frames_dataset, num_graphs)

        # Step 2: Get embeddings
        print("\n--- Extracting embeddings ---")
        self.embeddings = self.g2v_model.get_embeddings().numpy()
        print(f"Embeddings shape: {self.embeddings.shape}")

        # Step 3: Clustering on raw embeddings
        print("\n--- Clustering on raw embeddings ---")
        self._run_clustering_analysis(
            self.embeddings,
            self.silhouette_scores,
            self.inertias,
            self.cluster_labels,
            desc="Raw embeddings",
        )
        raw_best_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        raw_best_score = self.silhouette_scores[raw_best_k]
        self.elbow_k = self._find_elbow(self.inertias)
        print(f"Raw: best k={raw_best_k} (silhouette={raw_best_score:.3f}), "
              f"elbow k={self.elbow_k}")

        # Step 4: UMAP reduction + clustering
        umap_best_k = None
        umap_best_score = -1.0
        try:
            import umap
            print(f"\n--- UMAP reduction (dim={self.umap_dim}) + clustering ---")
            reducer = umap.UMAP(
                n_components=self.umap_dim,
                random_state=self.random_state,
                n_neighbors=min(15, len(self.embeddings) - 1),
                min_dist=0.1,
            )
            self.umap_embeddings = reducer.fit_transform(self.embeddings)
            print(f"UMAP embeddings shape: {self.umap_embeddings.shape}")

            self._run_clustering_analysis(
                self.umap_embeddings,
                self.umap_silhouette_scores,
                self.umap_inertias,
                self.umap_cluster_labels,
                desc="UMAP embeddings",
            )
            umap_best_k = max(self.umap_silhouette_scores, key=self.umap_silhouette_scores.get)
            umap_best_score = self.umap_silhouette_scores[umap_best_k]
            self.umap_elbow_k = self._find_elbow(self.umap_inertias)
            print(f"UMAP: best k={umap_best_k} (silhouette={umap_best_score:.3f}), "
                  f"elbow k={self.umap_elbow_k}")
        except ImportError:
            print("\n--- UMAP not installed, skipping UMAP clustering ---")
            print("Install umap-learn for dual clustering: pip install umap-learn")

        # Step 5: Pick the better result
        self.umap_best_k = umap_best_k
        if umap_best_score > raw_best_score:
            self.best_k = umap_best_k
            self.chosen_source = 'umap'
            # Use UMAP cluster labels as the primary labels
            self.cluster_labels = self.umap_cluster_labels
            self.silhouette_scores = self.umap_silhouette_scores
            self.inertias = self.umap_inertias
            self.elbow_k = self.umap_elbow_k
            print(f"\n>> UMAP clustering selected (silhouette {umap_best_score:.3f} > {raw_best_score:.3f})")
        else:
            self.best_k = raw_best_k
            self.chosen_source = 'raw'
            print(f"\n>> Raw embedding clustering selected (silhouette {raw_best_score:.3f} >= {umap_best_score:.3f})")

        print(f"Recommended n_states: {self.best_k}")

        return self

    def _run_clustering_analysis(self, data, silhouette_dict, inertia_dict, labels_dict, desc=""):
        """Run KMeans for different k values and calculate metrics."""
        k_range = range(self.min_k, self.max_k + 1)

        for k in tqdm(k_range, desc=f"Testing cluster sizes ({desc})"):
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )
            labels = kmeans.fit_predict(data)

            # Store results
            labels_dict[k] = labels
            inertia_dict[k] = kmeans.inertia_

            # Calculate silhouette score (only valid for k >= 2)
            if k >= 2:
                silhouette_dict[k] = silhouette_score(data, labels)

    def _find_elbow(self, inertia_dict) -> int:
        """Find elbow point using second derivative method."""
        k_values = sorted(inertia_dict.keys())
        inertia_values = [inertia_dict[k] for k in k_values]

        if len(k_values) < 3:
            return k_values[0]

        # Calculate second derivative
        first_deriv = np.diff(inertia_values)
        second_deriv = np.diff(first_deriv)

        # Find the point with maximum curvature (most negative second derivative)
        # Add 1 to offset since we lost two points due to double diff
        elbow_idx = np.argmax(second_deriv) + 1

        return k_values[elbow_idx]

    def get_embeddings(self) -> np.ndarray:
        """Return graph embeddings."""
        if self.embeddings is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.embeddings

    def get_recommended_n_states(self) -> int:
        """Return recommended number of states based on silhouette score."""
        if self.best_k is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_k

    def get_cluster_labels(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Return cluster labels for specific k.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters. If None, uses recommended k.

        Returns
        -------
        np.ndarray
            Cluster labels for each frame
        """
        if n_clusters is None:
            n_clusters = self.best_k

        if n_clusters not in self.cluster_labels:
            raise ValueError(f"Clustering for k={n_clusters} not available. "
                           f"Available: {list(self.cluster_labels.keys())}")

        return self.cluster_labels[n_clusters]

    def plot_results(self, save_dir: str):
        """
        Generate all visualization plots.

        Parameters
        ----------
        save_dir : str
            Directory to save plots and data
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n--- Generating Plots ---")
        print(f"Saving to: {save_dir}")

        # Save embeddings
        np.save(os.path.join(save_dir, 'embeddings.npy'), self.embeddings)
        if self.umap_embeddings is not None:
            np.save(os.path.join(save_dir, 'umap_embeddings.npy'), self.umap_embeddings)

        # Save cluster labels for recommended k
        np.save(
            os.path.join(save_dir, 'cluster_labels.npy'),
            self.cluster_labels[self.best_k]
        )

        # Generate plots
        self._plot_elbow(save_dir)
        self._plot_silhouette(save_dir)
        self._plot_cluster_sizes(save_dir)
        self._plot_embeddings_2d(save_dir)
        self._plot_temporal_evolution(save_dir)

        # Save summary JSON
        self._save_summary(save_dir)

        print("All plots generated successfully")

    def _plot_elbow(self, save_dir: str):
        """Generate elbow plot (inertia vs k)."""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        k_values = sorted(self.inertias.keys())
        inertia_values = [self.inertias[k] for k in k_values]

        # Plot inertia
        ax1.plot(k_values, inertia_values, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')

        # Mark elbow point
        ax1.axvline(x=self.elbow_k, color='r', linestyle='--', linewidth=2,
                   label=f'Elbow at k={self.elbow_k}')

        # Add second derivative on secondary axis
        if len(k_values) >= 3:
            ax2 = ax1.twinx()
            first_deriv = np.diff(inertia_values)
            second_deriv = np.diff(first_deriv)
            # Align with k_values (offset by 1 due to diff)
            ax2.plot(k_values[1:-1], second_deriv, 'g--', alpha=0.5,
                    label='Second derivative')
            ax2.set_ylabel('Second Derivative', color='g', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='g')

        source_label = f" (clustering on {self.chosen_source} embeddings)" if self.chosen_source else ""
        ax1.legend(loc='upper right')
        ax1.set_title(f'Elbow Method for Optimal k{source_label}', fontsize=14)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'elbow_plot.png'), dpi=150)
        plt.close()
        print("  - elbow_plot.png")

    def _plot_silhouette(self, save_dir: str):
        """Generate silhouette score plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        k_values = sorted(self.silhouette_scores.keys())
        scores = [self.silhouette_scores[k] for k in k_values]

        # Plot silhouette scores
        ax.plot(k_values, scores, 'b-o', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)

        # Mark best k
        ax.axvline(x=self.best_k, color='r', linestyle='--', linewidth=2,
                  label=f'Best k={self.best_k} (score={self.silhouette_scores[self.best_k]:.3f})')

        # Highlight best point
        ax.scatter([self.best_k], [self.silhouette_scores[self.best_k]],
                  color='r', s=200, zorder=5, marker='*')

        source_label = f" (clustering on {self.chosen_source} embeddings)" if self.chosen_source else ""
        ax.legend(loc='best')
        ax.set_title(f'Silhouette Score vs Number of Clusters{source_label}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        y_min = min(scores) - 0.05
        y_max = max(scores) + 0.05
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'silhouette_plot.png'), dpi=150)
        plt.close()
        print("  - silhouette_plot.png")

    def _plot_cluster_sizes(self, save_dir: str):
        """Generate cluster size distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = self.cluster_labels[self.best_k]
        unique, counts = np.unique(labels, return_counts=True)

        # Sort by size (largest first)
        sorted_idx = np.argsort(counts)[::-1]
        sorted_labels = unique[sorted_idx]
        sorted_counts = counts[sorted_idx]
        percentages = 100 * sorted_counts / len(labels)

        # Create bar chart
        bars = ax.bar(range(len(sorted_labels)), sorted_counts,
                     color=plt.cm.tab10(np.arange(len(sorted_labels))))

        # Add percentage labels on bars
        for i, (count, pct) in enumerate(zip(sorted_counts, percentages)):
            ax.text(i, count + len(labels)*0.01, f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Number of Frames', fontsize=12)
        ax.set_title(f'Cluster Size Distribution (k={self.best_k})', fontsize=14)
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels([f'State {l}' for l in sorted_labels])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cluster_sizes.png'), dpi=150)
        plt.close()
        print("  - cluster_sizes.png")

    def _plot_embeddings_2d(self, save_dir: str):
        """Generate 2D embedding visualization using t-SNE and optionally UMAP."""
        labels = self.cluster_labels[self.best_k]

        # t-SNE
        print("  Computing t-SNE projection...")
        tsne = TSNE(
            n_components=2,
            random_state=self.random_state,
            perplexity=min(30, len(self.embeddings) - 1),
            max_iter=1000,
        )
        embeddings_2d = tsne.fit_transform(self.embeddings)

        self._create_embedding_scatter(
            embeddings_2d, labels,
            os.path.join(save_dir, 'tsne_embeddings.png'),
            title='t-SNE Visualization of Graph Embeddings'
        )
        print("  - tsne_embeddings.png")

        # UMAP visualization (always 2D for plotting, separate from clustering UMAP)
        try:
            import umap
            print("  Computing UMAP projection (2D for visualization)...")
            reducer = umap.UMAP(
                n_components=2,
                random_state=self.random_state,
                n_neighbors=min(15, len(self.embeddings) - 1),
                min_dist=0.1,
            )
            embeddings_umap_2d = reducer.fit_transform(self.embeddings)

            self._create_embedding_scatter(
                embeddings_umap_2d, labels,
                os.path.join(save_dir, 'umap_embeddings.png'),
                title='UMAP Visualization of Graph Embeddings'
            )
            print("  - umap_embeddings.png")
        except ImportError:
            print("  - umap_embeddings.png (skipped - umap-learn not installed)")

    def _plot_temporal_evolution(self, save_dir: str):
        """Generate temporal evolution plots showing trajectory through embedding space."""
        labels = self.cluster_labels[self.best_k]
        n_frames = len(self.embeddings)
        frame_indices = np.arange(n_frames)

        # Compute t-SNE (reuse if already computed, but we need to recompute here)
        print("  Computing t-SNE for temporal visualization...")
        tsne = TSNE(
            n_components=2,
            random_state=self.random_state,
            perplexity=min(30, n_frames - 1),
            max_iter=1000,
        )
        embeddings_2d = tsne.fit_transform(self.embeddings)

        # Plot 1: 2D scatter colored by time
        self._plot_temporal_2d(embeddings_2d, frame_indices, save_dir)

        # Plot 2: 3D scatter with frame index as z-axis
        self._plot_temporal_3d(embeddings_2d, frame_indices, labels, save_dir)

        # Plot 3: State evolution over time
        self._plot_state_timeline(labels, frame_indices, save_dir)

    def _plot_temporal_2d(self, embeddings_2d: np.ndarray, frame_indices: np.ndarray, save_dir: str):
        """Create 2D scatter plot colored by frame index (time)."""
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=frame_indices,
            cmap='viridis',
            alpha=0.6,
            s=20,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frame Index (Time)', fontsize=12)

        # Add trajectory line (connecting consecutive frames)
        ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1],
                'k-', alpha=0.1, linewidth=0.5, zorder=1)

        # Mark start and end points
        ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1],
                  c='green', s=200, marker='^', edgecolors='black',
                  linewidths=2, label='Start', zorder=5)
        ax.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1],
                  c='red', s=200, marker='s', edgecolors='black',
                  linewidths=2, label='End', zorder=5)

        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('Temporal Evolution in Embedding Space\n(colored by frame index)', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tsne_temporal.png'), dpi=150)
        plt.close()
        print("  - tsne_temporal.png")

    def _plot_temporal_3d(self, embeddings_2d: np.ndarray, frame_indices: np.ndarray,
                          labels: np.ndarray, save_dir: str):
        """Create 3D scatter plot with frame index as z-axis."""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Color by cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.arange(len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                frame_indices[mask],
                c=[colors[i]],
                label=f'State {label}',
                alpha=0.6,
                s=20,
            )

        # Add trajectory line
        ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], frame_indices,
                'k-', alpha=0.15, linewidth=0.5)

        ax.set_xlabel('t-SNE Component 1', fontsize=10)
        ax.set_ylabel('t-SNE Component 2', fontsize=10)
        ax.set_zlabel('Frame Index (Time)', fontsize=10)
        ax.set_title('3D Temporal Evolution\n(z-axis = frame index)', fontsize=14)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tsne_temporal_3d.png'), dpi=150)
        plt.close()
        print("  - tsne_temporal_3d.png")

    def _plot_state_timeline(self, labels: np.ndarray, frame_indices: np.ndarray, save_dir: str):
        """Plot state assignments over time as a timeline."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.arange(len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        # Top plot: scatter plot of states over time
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(frame_indices[mask], [label] * np.sum(mask),
                       c=[colors[i]], s=10, alpha=0.7, label=f'State {label}')

        ax1.set_xlabel('Frame Index', fontsize=12)
        ax1.set_ylabel('State', fontsize=12)
        ax1.set_title('State Assignments Over Time', fontsize=14)
        ax1.set_yticks(unique_labels)
        ax1.set_yticklabels([f'State {l}' for l in unique_labels])
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='x')

        # Bottom plot: state as colored bar
        # Create color array for each frame
        frame_colors = np.array([color_map[l] for l in labels])

        # Plot as a colored bar using imshow
        ax2.imshow([labels], aspect='auto', cmap=plt.cm.tab10,
                   extent=[0, len(labels), 0, 1], vmin=0, vmax=len(unique_labels)-1)
        ax2.set_xlabel('Frame Index', fontsize=12)
        ax2.set_ylabel('')
        ax2.set_yticks([])
        ax2.set_title('State Timeline (color bar)', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'state_timeline.png'), dpi=150)
        plt.close()
        print("  - state_timeline.png")

    def _create_embedding_scatter(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        title: str
    ):
        """Create a 2D scatter plot of embeddings colored by cluster."""
        fig, ax = plt.subplots(figsize=(12, 10))

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.arange(len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            count = np.sum(mask)
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f'State {label} (n={count})',
                alpha=0.6,
                s=20,
            )

        # Calculate and plot centroids
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centroid = embeddings_2d[mask].mean(axis=0)
            ax.scatter(
                centroid[0], centroid[1],
                c=[colors[i]],
                marker='X',
                s=200,
                edgecolors='black',
                linewidths=2,
            )

        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        source_label = f", source={self.chosen_source}" if self.chosen_source else ""
        ax.set_title(f'{title}\n(k={self.best_k}, silhouette={self.silhouette_scores[self.best_k]:.3f}{source_label})',
                    fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _save_summary(self, save_dir: str):
        """Save summary JSON with all metrics and recommendations."""
        # Calculate cluster sizes for recommended k
        labels = self.cluster_labels[self.best_k]
        unique, counts = np.unique(labels, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        cluster_sizes = (counts[sorted_idx] / len(labels)).tolist()

        summary = {
            'recommended_n_states': int(self.best_k),
            'selection_method': 'best_silhouette_across_raw_and_umap',
            'chosen_source': self.chosen_source,
            'silhouette_scores': {str(k): float(v) for k, v in self.silhouette_scores.items()},
            'inertias': {str(k): float(v) for k, v in self.inertias.items()},
            'best_silhouette_score': float(self.silhouette_scores[self.best_k]),
            'best_silhouette_k': int(self.best_k),
            'elbow_k': int(self.elbow_k),
            'cluster_sizes': cluster_sizes,
            'graph2vec_config': {
                'embedding_dim': self.embedding_dim,
                'max_degree': self.max_degree,
                'epochs': self.g2v_epochs,
                'min_count': self.g2v_min_count,
                'umap_dim': self.umap_dim,
            },
            'n_frames': len(self.embeddings),
            'embedding_shape': list(self.embeddings.shape),
        }

        with open(os.path.join(save_dir, 'discovery_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("  - discovery_summary.json")
