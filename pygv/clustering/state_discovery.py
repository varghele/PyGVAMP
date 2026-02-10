"""
State Discovery Module for Unsupervised Molecular State Detection

This module provides unsupervised state discovery using Graph2Vec embeddings
and clustering analysis. It helps determine the optimal number of states
for VAMPNet training by analyzing the natural clustering structure in
molecular dynamics trajectory data.

Clustering is swept across multiple embedding sources (raw, UMAP at
configurable dimensions, t-SNE at 2D and 3D). For each source, four
metrics are evaluated:
- Silhouette score (KMeans)
- Elbow method (KMeans inertia)
- BIC (Gaussian Mixture Model)
- AIC (Gaussian Mixture Model)

The best embedding source is selected by silhouette score. The recommended
number of states is the **maximum** k across all four metrics from the
winning source, erring on the side of more states (since states can be
merged post-training but cannot be split).
"""

import os
import json
import warnings
import numpy as np
from typing import Optional, List
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from pygv.clustering.graph2vec import Graph2Vec


class StateDiscovery:
    """
    Unsupervised state discovery using Graph2Vec + clustering.

    This class trains a Graph2Vec model on molecular graphs to learn
    fixed-dimensional embeddings, then sweeps clustering across multiple
    embedding sources to recommend the optimal number of states.

    Clustering sources swept:
    - Raw high-dimensional Graph2Vec embeddings (always)
    - UMAP-reduced embeddings at each dimension in ``umap_dims``
    - t-SNE-reduced embeddings at dimensions 2 and 3 (automatic)

    For each source, KMeans and GMM are fitted for every k in
    ``[min_k, max_k]``. The best source is chosen by silhouette score.
    The recommended k is ``max(silhouette_k, elbow_k, bic_k, aic_k)``
    from the winning source — biasing toward more states.

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
    umap_dims : list of int, optional
        UMAP dimensionalities to sweep for clustering (default ``[2]``).
    min_k : int, default=2
        Minimum number of clusters to test
    max_k : int, default=15
        Maximum number of clusters to test
    random_state : int, default=42
        Random seed for reproducibility
    """

    # Use diagonal covariance for GMM when dimensionality exceeds this
    _GMM_FULL_COV_THRESHOLD = 50

    def __init__(
        self,
        embedding_dim: int = 64,
        max_degree: int = 2,
        g2v_epochs: int = 50,
        g2v_min_count: int = 5,
        umap_dims: Optional[List[int]] = None,
        min_k: int = 2,
        max_k: int = 15,
        random_state: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.max_degree = max_degree
        self.g2v_epochs = g2v_epochs
        self.g2v_min_count = g2v_min_count
        self.umap_dims = umap_dims if umap_dims is not None else [2]
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state

        # Results storage
        self.embeddings = None
        self.g2v_model = None

        # Per-source sweep results:
        #   source_name -> {silhouette_scores, inertias, bic_scores,
        #                   aic_scores, cluster_labels, silhouette_k,
        #                   elbow_k, bic_k, aic_k, best_score, data}
        self.sweep_results = {}

        # Primary results (set to the winning source after sweep)
        self.silhouette_scores = {}
        self.inertias = {}
        self.bic_scores = {}
        self.aic_scores = {}
        self.cluster_labels = {}
        self.best_k = None
        self.elbow_k = None
        self.chosen_source = None
        self.metric_recommendations = {}

    def fit(self, frames_dataset) -> 'StateDiscovery':
        """
        Run Graph2Vec and clustering sweep.

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

        # Step 3: Clustering sweep
        print("\n--- Clustering sweep ---")

        # Always cluster raw embeddings
        self._cluster_and_evaluate(self.embeddings, 'raw')

        # UMAP sweep
        for dim in self.umap_dims:
            try:
                import umap
                print(f"  UMAP reduction (dim={dim})...")
                reducer = umap.UMAP(
                    n_components=dim,
                    random_state=self.random_state,
                    n_neighbors=min(15, len(self.embeddings) - 1),
                    min_dist=0.1,
                )
                reduced = reducer.fit_transform(self.embeddings)
                self._cluster_and_evaluate(reduced, f'umap_{dim}')
            except ImportError:
                print("  UMAP not installed, skipping UMAP sweep.")
                print("  Install umap-learn: pip install umap-learn")
                break

        # t-SNE sweep (always 2D and 3D)
        for dim in (2, 3):
            print(f"  t-SNE reduction (dim={dim})...")
            tsne = TSNE(
                n_components=dim,
                random_state=self.random_state,
                perplexity=min(30, len(self.embeddings) - 1),
                max_iter=1000,
            )
            reduced = tsne.fit_transform(self.embeddings)
            self._cluster_and_evaluate(reduced, f'tsne_{dim}')

        # Pick winner and recommend k
        self._select_best_source()

        return self

    def _cluster_and_evaluate(self, data, source_name):
        """Run KMeans + GMM clustering on *data* and store results."""
        result = {
            'silhouette_scores': {},
            'inertias': {},
            'bic_scores': {},
            'aic_scores': {},
            'cluster_labels': {},
            'data': data,
        }

        self._run_clustering_analysis(
            data,
            result['silhouette_scores'],
            result['inertias'],
            result['bic_scores'],
            result['aic_scores'],
            result['cluster_labels'],
            desc=source_name,
        )

        # Best k per metric
        if result['silhouette_scores']:
            result['silhouette_k'] = max(
                result['silhouette_scores'],
                key=result['silhouette_scores'].get,
            )
            result['best_score'] = result['silhouette_scores'][
                result['silhouette_k']]
        else:
            result['silhouette_k'] = self.min_k
            result['best_score'] = -1.0

        result['elbow_k'] = self._find_elbow(result['inertias'])
        result['bic_k'] = min(result['bic_scores'],
                              key=result['bic_scores'].get)
        result['aic_k'] = min(result['aic_scores'],
                              key=result['aic_scores'].get)

        # For source comparison, best_k is the silhouette-optimal k
        result['best_k'] = result['silhouette_k']

        self.sweep_results[source_name] = result
        print(f"    {source_name}: silhouette_k={result['silhouette_k']} "
              f"(score={result['best_score']:.3f}), "
              f"elbow_k={result['elbow_k']}, "
              f"bic_k={result['bic_k']}, aic_k={result['aic_k']}")

    def _select_best_source(self):
        """Select the best source by silhouette, recommend k via max across metrics."""
        # Pick source with best silhouette score
        best_source = max(
            self.sweep_results,
            key=lambda s: self.sweep_results[s]['best_score'],
        )

        winner = self.sweep_results[best_source]

        # k recommendation: max across all metrics (err toward more states)
        self.metric_recommendations = {
            'silhouette': winner['silhouette_k'],
            'elbow': winner['elbow_k'],
            'bic': winner['bic_k'],
            'aic': winner['aic_k'],
        }
        recommended_k = max(self.metric_recommendations.values())

        # Set primary results to the winner
        self.silhouette_scores = winner['silhouette_scores']
        self.inertias = winner['inertias']
        self.bic_scores = winner['bic_scores']
        self.aic_scores = winner['aic_scores']
        self.cluster_labels = winner['cluster_labels']
        self.best_k = recommended_k
        self.elbow_k = winner['elbow_k']
        self.chosen_source = best_source

        # Print sweep summary
        print(f"\n--- Sweep Summary ---")
        for source_name, result in self.sweep_results.items():
            marker = "  << BEST SOURCE" if source_name == best_source else ""
            print(f"  {source_name}: silhouette_k={result['silhouette_k']} "
                  f"({result['best_score']:.3f}), "
                  f"elbow={result['elbow_k']}, "
                  f"bic={result['bic_k']}, "
                  f"aic={result['aic_k']}{marker}")

        print(f"\n--- Metric Recommendations (source={best_source}) ---")
        for metric, k in self.metric_recommendations.items():
            print(f"  {metric}: k={k}")
        print(f"\n  >> Recommended n_states: {self.best_k} "
              f"(max across metrics)")

    def _run_clustering_analysis(self, data, silhouette_dict, inertia_dict,
                                 bic_dict, aic_dict, labels_dict, desc=""):
        """Run KMeans and GMM for different k values."""
        k_range = range(self.min_k, self.max_k + 1)
        n_features = data.shape[1]
        cov_type = ('diag' if n_features > self._GMM_FULL_COV_THRESHOLD
                    else 'full')

        for k in tqdm(k_range, desc=f"KMeans + GMM ({desc})"):
            # KMeans
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )
            labels = kmeans.fit_predict(data)

            labels_dict[k] = labels
            inertia_dict[k] = kmeans.inertia_

            if k >= 2:
                silhouette_dict[k] = silhouette_score(data, labels)

            # GMM for BIC / AIC
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=self.random_state,
                    n_init=3,
                    max_iter=200,
                )
                gmm.fit(data)
            bic_dict[k] = gmm.bic(data)
            aic_dict[k] = gmm.aic(data)

    def _find_elbow(self, inertia_dict) -> int:
        """Find elbow point using second derivative method."""
        k_values = sorted(inertia_dict.keys())
        inertia_values = [inertia_dict[k] for k in k_values]

        if len(k_values) < 3:
            return k_values[0]

        first_deriv = np.diff(inertia_values)
        second_deriv = np.diff(first_deriv)

        elbow_idx = np.argmax(second_deriv) + 1
        return k_values[elbow_idx]

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_embeddings(self) -> np.ndarray:
        """Return raw Graph2Vec embeddings."""
        if self.embeddings is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.embeddings

    def get_recommended_n_states(self) -> int:
        """Return recommended number of states (max across metrics)."""
        if self.best_k is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_k

    def get_cluster_labels(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Return cluster labels for specific k (from the winning source).

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters. If None, uses recommended k.
        """
        if n_clusters is None:
            n_clusters = self.best_k

        if n_clusters not in self.cluster_labels:
            raise ValueError(f"Clustering for k={n_clusters} not available. "
                           f"Available: {list(self.cluster_labels.keys())}")

        return self.cluster_labels[n_clusters]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

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

        # Save raw embeddings
        np.save(os.path.join(save_dir, 'embeddings.npy'), self.embeddings)

        # Save reduced embeddings from sweep
        for source_name, result in self.sweep_results.items():
            if source_name != 'raw':
                np.save(
                    os.path.join(save_dir, f'{source_name}_embeddings.npy'),
                    result['data'],
                )

        # Save cluster labels for recommended k
        np.save(
            os.path.join(save_dir, 'cluster_labels.npy'),
            self.cluster_labels[self.best_k],
        )

        # Generate plots
        self._plot_elbow(save_dir)
        self._plot_silhouette(save_dir)
        self._plot_bic_aic(save_dir)
        self._plot_metric_recommendations(save_dir)
        self._plot_cluster_sizes(save_dir)
        self._plot_sweep_comparison(save_dir)
        self._plot_embeddings_2d(save_dir)
        self._plot_temporal_evolution(save_dir)

        # Save summary JSON
        self._save_summary(save_dir)

        print("All plots generated successfully")

    def _plot_elbow(self, save_dir: str):
        """Generate elbow plot (inertia vs k) for the winning source."""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        k_values = sorted(self.inertias.keys())
        inertia_values = [self.inertias[k] for k in k_values]

        ax1.plot(k_values, inertia_values, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', color='b',
                        fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')

        ax1.axvline(x=self.elbow_k, color='r', linestyle='--', linewidth=2,
                     label=f'Elbow at k={self.elbow_k}')

        if len(k_values) >= 3:
            ax2 = ax1.twinx()
            first_deriv = np.diff(inertia_values)
            second_deriv = np.diff(first_deriv)
            ax2.plot(k_values[1:-1], second_deriv, 'g--', alpha=0.5,
                     label='Second derivative')
            ax2.set_ylabel('Second Derivative', color='g', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='g')

        ax1.legend(loc='upper right')
        ax1.set_title(
            f'Elbow Method for Optimal k (source={self.chosen_source})',
            fontsize=14,
        )
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'elbow_plot.png'), dpi=150)
        plt.close()
        print("  - elbow_plot.png")

    def _plot_silhouette(self, save_dir: str):
        """Generate silhouette score plot for the winning source."""
        fig, ax = plt.subplots(figsize=(10, 6))

        k_values = sorted(self.silhouette_scores.keys())
        scores = [self.silhouette_scores[k] for k in k_values]

        ax.plot(k_values, scores, 'b-o', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)

        sil_k = self.metric_recommendations['silhouette']
        ax.axvline(x=sil_k, color='r', linestyle='--', linewidth=2,
                    label=f'Best k={sil_k} '
                          f'(score={self.silhouette_scores[sil_k]:.3f})')
        ax.scatter([sil_k], [self.silhouette_scores[sil_k]],
                   color='r', s=200, zorder=5, marker='*')

        ax.legend(loc='best')
        ax.set_title(
            f'Silhouette Score vs k (source={self.chosen_source})',
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)

        y_min = min(scores) - 0.05
        y_max = max(scores) + 0.05
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'silhouette_plot.png'), dpi=150)
        plt.close()
        print("  - silhouette_plot.png")

    def _plot_bic_aic(self, save_dir: str):
        """Generate BIC and AIC vs k plot for the winning source."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        k_values = sorted(self.bic_scores.keys())
        bic_values = [self.bic_scores[k] for k in k_values]
        aic_values = [self.aic_scores[k] for k in k_values]

        bic_k = self.metric_recommendations['bic']
        aic_k = self.metric_recommendations['aic']

        # BIC
        ax1.plot(k_values, bic_values, 'b-o', linewidth=2, markersize=8)
        ax1.axvline(x=bic_k, color='r', linestyle='--', linewidth=2,
                     label=f'Best k={bic_k}')
        ax1.scatter([bic_k], [self.bic_scores[bic_k]],
                    color='r', s=200, zorder=5, marker='*')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('BIC (lower is better)', fontsize=12)
        ax1.set_title(f'Bayesian Information Criterion\n'
                      f'(source={self.chosen_source})', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # AIC
        ax2.plot(k_values, aic_values, 'g-o', linewidth=2, markersize=8)
        ax2.axvline(x=aic_k, color='r', linestyle='--', linewidth=2,
                     label=f'Best k={aic_k}')
        ax2.scatter([aic_k], [self.aic_scores[aic_k]],
                    color='r', s=200, zorder=5, marker='*')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('AIC (lower is better)', fontsize=12)
        ax2.set_title(f'Akaike Information Criterion\n'
                      f'(source={self.chosen_source})', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bic_aic_plot.png'), dpi=150)
        plt.close()
        print("  - bic_aic_plot.png")

    def _plot_metric_recommendations(self, save_dir: str):
        """Bar chart showing the recommended k from each metric."""
        metrics = list(self.metric_recommendations.keys())
        k_values = [self.metric_recommendations[m] for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))

        bar_colors = ['#2196F3'] * len(metrics)
        bars = ax.bar(range(len(metrics)), k_values, color=bar_colors)

        # Add value labels on bars
        for i, k in enumerate(k_values):
            ax.text(i, k + 0.1, str(k), ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

        # Horizontal line at the recommended k
        ax.axhline(y=self.best_k, color='#4CAF50', linestyle='--',
                   linewidth=3,
                   label=f'Recommended: k={self.best_k} (max)')

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.upper() for m in metrics], fontsize=12)
        ax.set_ylabel('Recommended k', fontsize=12)
        ax.set_title(
            f'Per-Metric Recommendations (source={self.chosen_source})\n'
            f'Final recommendation = max across metrics (err toward more states)',
            fontsize=13,
        )
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(k_values) + 1.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metric_recommendations.png'),
                    dpi=150)
        plt.close()
        print("  - metric_recommendations.png")

    def _plot_cluster_sizes(self, save_dir: str):
        """Generate cluster size distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = self.cluster_labels[self.best_k]
        unique, counts = np.unique(labels, return_counts=True)

        sorted_idx = np.argsort(counts)[::-1]
        sorted_labels = unique[sorted_idx]
        sorted_counts = counts[sorted_idx]
        percentages = 100 * sorted_counts / len(labels)

        ax.bar(range(len(sorted_labels)), sorted_counts,
               color=plt.cm.tab10(np.arange(len(sorted_labels))))

        for i, (count, pct) in enumerate(zip(sorted_counts, percentages)):
            ax.text(i, count + len(labels) * 0.01, f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Number of Frames', fontsize=12)
        ax.set_title(f'Cluster Size Distribution (k={self.best_k})',
                     fontsize=14)
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels([f'State {l}' for l in sorted_labels])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cluster_sizes.png'), dpi=150)
        plt.close()
        print("  - cluster_sizes.png")

    def _plot_sweep_comparison(self, save_dir: str):
        """Bar chart comparing the best silhouette score of every sweep source."""
        sources = list(self.sweep_results.keys())
        best_scores = [self.sweep_results[s]['best_score'] for s in sources]
        best_ks = [self.sweep_results[s]['silhouette_k'] for s in sources]

        fig, ax = plt.subplots(figsize=(max(8, len(sources) * 1.5), 6))

        bar_colors = ['#4CAF50' if s == self.chosen_source else '#2196F3'
                      for s in sources]
        ax.bar(range(len(sources)), best_scores, color=bar_colors)

        for i, (score, k) in enumerate(zip(best_scores, best_ks)):
            ax.text(i, score + 0.005, f'k={k}\n{score:.3f}',
                    ha='center', va='bottom', fontsize=9)

        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.set_ylabel('Best Silhouette Score', fontsize=12)
        ax.set_title('Clustering Source Comparison (green = selected)',
                     fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sweep_comparison.png'), dpi=150)
        plt.close()
        print("  - sweep_comparison.png")

    def _get_2d_projection(self, method):
        """
        Return a 2D projection, reusing sweep data when available.

        Parameters
        ----------
        method : str
            ``'tsne'`` or ``'umap'``

        Returns
        -------
        np.ndarray or None
            2D projection of the raw embeddings, or None if unavailable.
        """
        key = f'{method}_2'
        if key in self.sweep_results:
            data = self.sweep_results[key]['data']
            if data.shape[1] == 2:
                return data

        if method == 'tsne':
            tsne = TSNE(
                n_components=2,
                random_state=self.random_state,
                perplexity=min(30, len(self.embeddings) - 1),
                max_iter=1000,
            )
            return tsne.fit_transform(self.embeddings)

        if method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=self.random_state,
                    n_neighbors=min(15, len(self.embeddings) - 1),
                    min_dist=0.1,
                )
                return reducer.fit_transform(self.embeddings)
            except ImportError:
                return None

        return None

    def _plot_embeddings_2d(self, save_dir: str):
        """Generate 2D embedding visualizations using t-SNE and UMAP."""
        labels = self.cluster_labels[self.best_k]

        # t-SNE 2D
        print("  Computing t-SNE 2D projection...")
        tsne_2d = self._get_2d_projection('tsne')
        self._create_embedding_scatter(
            tsne_2d, labels,
            os.path.join(save_dir, 'tsne_embeddings.png'),
            title='t-SNE Visualization of Graph Embeddings',
        )
        print("  - tsne_embeddings.png")

        # UMAP 2D
        print("  Computing UMAP 2D projection...")
        umap_2d = self._get_2d_projection('umap')
        if umap_2d is not None:
            self._create_embedding_scatter(
                umap_2d, labels,
                os.path.join(save_dir, 'umap_embeddings.png'),
                title='UMAP Visualization of Graph Embeddings',
            )
            print("  - umap_embeddings.png")
        else:
            print("  - umap_embeddings.png (skipped - umap-learn not installed)")

    def _plot_temporal_evolution(self, save_dir: str):
        """Generate temporal evolution plots."""
        labels = self.cluster_labels[self.best_k]
        n_frames = len(self.embeddings)
        frame_indices = np.arange(n_frames)

        print("  Computing t-SNE 2D for temporal visualization...")
        tsne_2d = self._get_2d_projection('tsne')

        self._plot_temporal_2d(tsne_2d, frame_indices, save_dir)
        self._plot_temporal_3d(tsne_2d, frame_indices, labels, save_dir,
                               method_name='t-SNE', filename='tsne_temporal_3d.png')
        self._plot_state_timeline(labels, frame_indices, save_dir)

        # UMAP 3D temporal plot
        print("  Computing UMAP 2D for temporal visualization...")
        umap_2d = self._get_2d_projection('umap')
        if umap_2d is not None:
            self._plot_temporal_3d(umap_2d, frame_indices, labels, save_dir,
                                   method_name='UMAP', filename='umap_temporal_3d.png')
        else:
            print("  - umap_temporal_3d.png (skipped - umap-learn not installed)")

    def _plot_temporal_2d(self, embeddings_2d: np.ndarray,
                          frame_indices: np.ndarray, save_dir: str):
        """Create 2D scatter plot colored by frame index (time)."""
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=frame_indices, cmap='viridis', alpha=0.6, s=20,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frame Index (Time)', fontsize=12)

        ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1],
                'k-', alpha=0.1, linewidth=0.5, zorder=1)

        ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1],
                   c='green', s=200, marker='^', edgecolors='black',
                   linewidths=2, label='Start', zorder=5)
        ax.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1],
                   c='red', s=200, marker='s', edgecolors='black',
                   linewidths=2, label='End', zorder=5)

        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('Temporal Evolution in Embedding Space\n'
                     '(colored by frame index)', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tsne_temporal.png'), dpi=150)
        plt.close()
        print("  - tsne_temporal.png")

    def _plot_temporal_3d(self, embeddings_2d: np.ndarray,
                          frame_indices: np.ndarray, labels: np.ndarray,
                          save_dir: str, method_name: str = 't-SNE',
                          filename: str = 'tsne_temporal_3d.png'):
        """Create 3D scatter plot with frame index as z-axis."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.arange(len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                frame_indices[mask],
                c=[colors[i]], label=f'State {label}', alpha=0.6, s=20,
            )

        ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], frame_indices,
                'k-', alpha=0.15, linewidth=0.5)

        ax.set_xlabel(f'{method_name} Component 1', fontsize=10)
        ax.set_ylabel(f'{method_name} Component 2', fontsize=10)
        ax.set_zlabel('Frame Index (Time)', fontsize=10)
        ax.set_title(f'3D Temporal Evolution ({method_name})\n'
                     f'(z-axis = frame index)', fontsize=14)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        print(f"  - {filename}")

    def _plot_state_timeline(self, labels: np.ndarray,
                             frame_indices: np.ndarray, save_dir: str):
        """Plot state assignments over time as a timeline."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                       height_ratios=[2, 1])

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.arange(len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(frame_indices[mask], [label] * np.sum(mask),
                        c=[colors[i]], s=10, alpha=0.7,
                        label=f'State {label}')

        ax1.set_xlabel('Frame Index', fontsize=12)
        ax1.set_ylabel('State', fontsize=12)
        ax1.set_title('State Assignments Over Time', fontsize=14)
        ax1.set_yticks(unique_labels)
        ax1.set_yticklabels([f'State {l}' for l in unique_labels])
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='x')

        ax2.imshow([labels], aspect='auto', cmap=plt.cm.tab10,
                   extent=[0, len(labels), 0, 1],
                   vmin=0, vmax=len(unique_labels) - 1)
        ax2.set_xlabel('Frame Index', fontsize=12)
        ax2.set_ylabel('')
        ax2.set_yticks([])
        ax2.set_title('State Timeline (color bar)', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'state_timeline.png'), dpi=150)
        plt.close()
        print("  - state_timeline.png")

    def _create_embedding_scatter(self, embeddings_2d: np.ndarray,
                                  labels: np.ndarray, save_path: str,
                                  title: str):
        """Create a 2D scatter plot of embeddings colored by cluster."""
        fig, ax = plt.subplots(figsize=(12, 10))

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.arange(len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            count = np.sum(mask)
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[colors[i]], label=f'State {label} (n={count})',
                alpha=0.6, s=20,
            )

        for i, label in enumerate(unique_labels):
            mask = labels == label
            centroid = embeddings_2d[mask].mean(axis=0)
            ax.scatter(
                centroid[0], centroid[1], c=[colors[i]],
                marker='X', s=200, edgecolors='black', linewidths=2,
            )

        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(
            f'{title}\n(k={self.best_k}, '
            f'source={self.chosen_source})',
            fontsize=14,
        )
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _save_summary(self, save_dir: str):
        """Save summary JSON with all metrics and recommendations."""
        labels = self.cluster_labels[self.best_k]
        unique, counts = np.unique(labels, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        cluster_sizes = (counts[sorted_idx] / len(labels)).tolist()

        # Per-source sweep details (without numpy arrays)
        sweep_details = {}
        for source_name, result in self.sweep_results.items():
            sweep_details[source_name] = {
                'silhouette_k': int(result['silhouette_k']),
                'best_silhouette': float(result['best_score']),
                'elbow_k': int(result['elbow_k']),
                'bic_k': int(result['bic_k']),
                'aic_k': int(result['aic_k']),
                'silhouette_scores': {
                    str(k): float(v)
                    for k, v in result['silhouette_scores'].items()
                },
                'bic_scores': {
                    str(k): float(v)
                    for k, v in result['bic_scores'].items()
                },
                'aic_scores': {
                    str(k): float(v)
                    for k, v in result['aic_scores'].items()
                },
            }

        summary = {
            'recommended_n_states': int(self.best_k),
            'selection_method': 'max_across_metrics',
            'chosen_source': self.chosen_source,
            'metric_recommendations': {
                k: int(v) for k, v in self.metric_recommendations.items()
            },
            'elbow_k': int(self.elbow_k),
            'cluster_sizes': cluster_sizes,
            'sweep_sources': list(self.sweep_results.keys()),
            'sweep_details': sweep_details,
            'graph2vec_config': {
                'embedding_dim': self.embedding_dim,
                'max_degree': self.max_degree,
                'epochs': self.g2v_epochs,
                'min_count': self.g2v_min_count,
                'umap_dims': self.umap_dims,
            },
            'n_frames': len(self.embeddings),
            'embedding_shape': list(self.embeddings.shape),
        }

        with open(os.path.join(save_dir, 'discovery_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("  - discovery_summary.json")
