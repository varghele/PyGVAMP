"""
Unit tests for the interactive HTML report generation module.

Tests cover:
- reduce_embeddings_to_2d: shape, UMAP/t-SNE paths, edge cases
- aggregate_edge_attention_to_residue: shape, None handling, correctness
- generate_interactive_report: pygviz missing guard, orchestration
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def seed():
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_embeddings_high_dim():
    """50 frames, 16-dim embeddings."""
    return np.random.randn(50, 16).astype(np.float32)


@pytest.fixture
def sample_embeddings_2d():
    """50 frames, already 2D."""
    return np.random.randn(50, 2).astype(np.float32)


@pytest.fixture
def sample_probs():
    """50 frames, 3 states — rows sum to 1."""
    raw = np.random.rand(50, 3).astype(np.float32)
    return raw / raw.sum(axis=1, keepdims=True)


@pytest.fixture
def sample_edge_data():
    """Consistent per-frame edge attentions and indices for 10 nodes."""
    n_frames = 50
    n_nodes = 10
    n_edges = 20

    edge_attentions = []
    edge_indices = []
    for _ in range(n_frames):
        sources = np.random.randint(0, n_nodes, size=n_edges)
        targets = np.random.randint(0, n_nodes, size=n_edges)
        idx = np.stack([sources, targets], axis=0)
        att = np.random.rand(n_edges).astype(np.float32)
        edge_attentions.append(att)
        edge_indices.append(idx)

    return edge_attentions, edge_indices, n_nodes


# ============================================================================
# Tests for reduce_embeddings_to_2d
# ============================================================================

class TestReduceEmbeddingsTo2d:

    def test_output_shape(self, sample_embeddings_high_dim, seed):
        from pygv.utils.interactive_report import reduce_embeddings_to_2d
        result = reduce_embeddings_to_2d(sample_embeddings_high_dim, method='tsne')
        assert result.shape == (50, 2)

    def test_passthrough_2d(self, sample_embeddings_2d):
        from pygv.utils.interactive_report import reduce_embeddings_to_2d
        result = reduce_embeddings_to_2d(sample_embeddings_2d)
        np.testing.assert_array_equal(result, sample_embeddings_2d)

    def test_tsne_fallback_when_umap_missing(self, sample_embeddings_high_dim, seed):
        from pygv.utils.interactive_report import reduce_embeddings_to_2d
        with patch.dict('sys.modules', {'umap': None}):
            result = reduce_embeddings_to_2d(sample_embeddings_high_dim, method='umap')
            assert result.shape == (50, 2)

    def test_tsne_explicit(self, sample_embeddings_high_dim, seed):
        from pygv.utils.interactive_report import reduce_embeddings_to_2d
        result = reduce_embeddings_to_2d(sample_embeddings_high_dim, method='tsne')
        assert result.shape == (50, 2)

    def test_invalid_input_1d(self):
        from pygv.utils.interactive_report import reduce_embeddings_to_2d
        with pytest.raises(ValueError, match="Expected 2D"):
            reduce_embeddings_to_2d(np.array([1, 2, 3]))

    def test_small_dataset(self, seed):
        """5 samples — perplexity must adapt."""
        from pygv.utils.interactive_report import reduce_embeddings_to_2d
        small = np.random.randn(5, 8).astype(np.float32)
        result = reduce_embeddings_to_2d(small, method='tsne')
        assert result.shape == (5, 2)

    def test_no_nans_in_output(self, sample_embeddings_high_dim, seed):
        from pygv.utils.interactive_report import reduce_embeddings_to_2d
        result = reduce_embeddings_to_2d(sample_embeddings_high_dim, method='tsne')
        assert not np.any(np.isnan(result))


# ============================================================================
# Tests for aggregate_edge_attention_to_residue
# ============================================================================

class TestAggregateEdgeAttentionToResidue:

    def test_output_shape(self, sample_edge_data):
        from pygv.utils.interactive_report import aggregate_edge_attention_to_residue
        att, idx, n_nodes = sample_edge_data
        result = aggregate_edge_attention_to_residue(att, idx, n_nodes)
        assert result.shape == (len(att), n_nodes)

    def test_none_entries_produce_zeros(self):
        from pygv.utils.interactive_report import aggregate_edge_attention_to_residue
        att = [None, None, None]
        idx = [None, None, None]
        result = aggregate_edge_attention_to_residue(att, idx, n_nodes=5)
        assert result.shape == (3, 5)
        np.testing.assert_array_equal(result, 0.0)

    def test_mixed_none_and_valid(self):
        from pygv.utils.interactive_report import aggregate_edge_attention_to_residue
        edge_idx = np.array([[0, 1], [1, 0]])  # 2 edges: 0->1, 1->0
        edge_att = np.array([0.8, 0.2], dtype=np.float32)
        att = [None, edge_att, None]
        idx = [None, edge_idx, None]
        result = aggregate_edge_attention_to_residue(att, idx, n_nodes=3)
        assert result.shape == (3, 3)
        # Frame 0 and 2: all zeros
        np.testing.assert_array_equal(result[0], 0.0)
        np.testing.assert_array_equal(result[2], 0.0)
        # Frame 1: node 1 receives edge 0->1 with att 0.8, node 0 receives 1->0 with att 0.2
        assert result[1, 0] == pytest.approx(0.2, abs=1e-5)
        assert result[1, 1] == pytest.approx(0.8, abs=1e-5)
        assert result[1, 2] == pytest.approx(0.0)

    def test_multiple_edges_to_same_target_averaged(self):
        from pygv.utils.interactive_report import aggregate_edge_attention_to_residue
        # 3 edges all pointing to node 0
        edge_idx = np.array([[1, 2, 3], [0, 0, 0]])
        edge_att = np.array([0.3, 0.6, 0.9], dtype=np.float32)
        result = aggregate_edge_attention_to_residue([edge_att], [edge_idx], n_nodes=4)
        expected_mean = (0.3 + 0.6 + 0.9) / 3.0
        assert result[0, 0] == pytest.approx(expected_mean, abs=1e-5)

    def test_single_frame_single_edge(self):
        from pygv.utils.interactive_report import aggregate_edge_attention_to_residue
        edge_idx = np.array([[0], [1]])
        edge_att = np.array([0.5], dtype=np.float32)
        result = aggregate_edge_attention_to_residue([edge_att], [edge_idx], n_nodes=2)
        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.5, abs=1e-5)


# ============================================================================
# Tests for generate_interactive_report
# ============================================================================

class TestGenerateInteractiveReport:

    def test_returns_none_when_pygviz_missing(self, sample_probs, sample_embeddings_2d, sample_edge_data, tmp_path):
        from pygv.utils.interactive_report import generate_interactive_report
        att, idx, n_nodes = sample_edge_data
        with patch.dict('sys.modules', {'pygviz': None, 'pygviz.md_visualizer': None}):
            result = generate_interactive_report(
                probs=sample_probs,
                embeddings=sample_embeddings_2d,
                edge_attentions=att,
                edge_indices=idx,
                topology_file="dummy.pdb",
                save_dir=str(tmp_path),
                n_nodes=n_nodes,
            )
            assert result is None

    def test_calls_visualizer_and_returns_path(self, sample_probs, sample_embeddings_2d, sample_edge_data, tmp_path):
        from pygv.utils.interactive_report import generate_interactive_report

        att, idx, n_nodes = sample_edge_data

        mock_viz_instance = MagicMock()
        mock_viz_class = MagicMock(return_value=mock_viz_instance)

        with patch('pygv.utils.interactive_report.reduce_embeddings_to_2d', return_value=sample_embeddings_2d):
            with patch('pygviz.md_visualizer.MDTrajectoryVisualizer', mock_viz_class):
                result = generate_interactive_report(
                    probs=sample_probs,
                    embeddings=sample_embeddings_2d,
                    edge_attentions=att,
                    edge_indices=idx,
                    topology_file="dummy.pdb",
                    save_dir=str(tmp_path),
                    protein_name="test_protein",
                    n_nodes=n_nodes,
                )

        assert result is not None
        assert result.endswith("test_protein_interactive_report.html")
        mock_viz_instance.add_timescale.assert_called_once()
        mock_viz_instance.set_protein_structure.assert_called_once()
        mock_viz_instance.generate.assert_called_once()

    def test_infers_n_nodes_from_edge_indices(self, sample_probs, sample_embeddings_2d, sample_edge_data, tmp_path):
        from pygv.utils.interactive_report import generate_interactive_report

        att, idx, _ = sample_edge_data

        mock_viz_instance = MagicMock()
        mock_viz_class = MagicMock(return_value=mock_viz_instance)

        with patch('pygv.utils.interactive_report.reduce_embeddings_to_2d', return_value=sample_embeddings_2d):
            with patch('pygviz.md_visualizer.MDTrajectoryVisualizer', mock_viz_class):
                result = generate_interactive_report(
                    probs=sample_probs,
                    embeddings=sample_embeddings_2d,
                    edge_attentions=att,
                    edge_indices=idx,
                    topology_file="dummy.pdb",
                    save_dir=str(tmp_path),
                    n_nodes=None,  # should be inferred
                )

        assert result is not None

    def test_returns_none_when_no_edge_indices_and_no_n_nodes(self, sample_probs, sample_embeddings_2d, tmp_path):
        from pygv.utils.interactive_report import generate_interactive_report

        all_none_att = [None] * 50
        all_none_idx = [None] * 50

        result = generate_interactive_report(
            probs=sample_probs,
            embeddings=sample_embeddings_2d,
            edge_attentions=all_none_att,
            edge_indices=all_none_idx,
            topology_file="dummy.pdb",
            save_dir=str(tmp_path),
            n_nodes=None,
        )
        assert result is None
