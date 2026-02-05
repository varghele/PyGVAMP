"""
Unit tests for the analysis pipeline.

Tests cover:
- State assignment from probabilities
- Transition matrix calculation
- Attention map computation
- analyze_vampnet_outputs function
- extract_residue_indices_from_selection function
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device('cpu')


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_probabilities():
    """Create sample state probabilities."""
    # 100 frames, 3 states
    probs = np.random.dirichlet([1, 1, 1], size=100)
    return probs


@pytest.fixture
def sample_probabilities_with_clear_states():
    """Create probabilities with clear state assignments."""
    n_frames = 100
    n_states = 3
    probs = np.zeros((n_frames, n_states))

    # First 40 frames in state 0
    probs[:40, 0] = 0.9
    probs[:40, 1] = 0.05
    probs[:40, 2] = 0.05

    # Next 30 frames in state 1
    probs[40:70, 0] = 0.05
    probs[40:70, 1] = 0.9
    probs[40:70, 2] = 0.05

    # Last 30 frames in state 2
    probs[70:, 0] = 0.05
    probs[70:, 1] = 0.05
    probs[70:, 2] = 0.9

    return probs


@pytest.fixture
def sample_edge_attentions():
    """Create sample edge attention values."""
    n_frames = 50
    n_atoms = 10
    n_neighbors = 5
    n_edges = n_atoms * n_neighbors

    attentions = []
    for _ in range(n_frames):
        # Random attention values
        att = np.random.rand(n_edges).astype(np.float32)
        attentions.append(att)

    return attentions


@pytest.fixture
def sample_edge_indices():
    """Create sample edge indices matching attentions."""
    n_frames = 50
    n_atoms = 10
    n_neighbors = 5

    indices = []
    for _ in range(n_frames):
        # Create k-NN style edge indices
        src = []
        dst = []
        for i in range(n_atoms):
            for j in range(n_neighbors):
                src.append(i)
                # Connect to different atoms (not self)
                dst.append((i + j + 1) % n_atoms)

        edge_index = np.array([src, dst], dtype=np.int64)
        indices.append(edge_index)

    return indices


def create_synthetic_graph(num_nodes=10, node_dim=16, edge_dim=16):
    """Create a synthetic PyG graph."""
    # Node features
    x = torch.randn(num_nodes, node_dim)

    # Create edges (k-NN style)
    n_neighbors = min(5, num_nodes - 1)
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(n_neighbors):
            src.append(i)
            dst.append((i + j + 1) % num_nodes)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.randn(len(src), edge_dim)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class SyntheticFramesDataset(Dataset):
    """Synthetic dataset returning single graphs (not pairs)."""

    def __init__(self, n_samples=50, num_nodes=10, node_dim=16, edge_dim=16):
        self.n_samples = n_samples
        self.graphs = []
        for _ in range(n_samples):
            self.graphs.append(create_synthetic_graph(num_nodes, node_dim, edge_dim))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graphs[idx]


@pytest.fixture
def synthetic_frames_dataset():
    """Create synthetic frames dataset."""
    return SyntheticFramesDataset(n_samples=50, num_nodes=10, node_dim=16, edge_dim=16)


@pytest.fixture
def mock_model(device):
    """Create a mock VAMPNet model."""
    model = Mock()
    model.eval = Mock(return_value=None)
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=iter([torch.randn(10, 10)]))

    n_states = 3
    embedding_dim = 32

    def mock_forward(batch, return_features=False, apply_classifier=True):
        if hasattr(batch, 'batch'):
            batch_size = batch.batch.max().item() + 1
        else:
            batch_size = 1

        probs = torch.softmax(torch.randn(batch_size, n_states), dim=1)
        embeddings = torch.randn(batch_size, embedding_dim)

        if return_features:
            return probs, embeddings
        return probs

    def mock_get_attention(batch, device=None):
        if hasattr(batch, 'batch'):
            n_edges = batch.edge_index.shape[1]
        else:
            n_edges = batch.edge_index.shape[1]

        features = torch.randn(10, embedding_dim)
        attentions = [torch.rand(n_edges)]
        return features, attentions

    model.__call__ = mock_forward
    model.side_effect = mock_forward
    model.get_attention = mock_get_attention

    return model


@pytest.fixture
def mock_topology():
    """Create mock MDTraj topology."""
    mock_top = Mock()

    # Create mock atoms with residues
    atoms = []
    residue_names = ['ALA', 'GLY', 'VAL', 'LEU', 'ILE',
                     'PRO', 'PHE', 'TYR', 'TRP', 'SER']

    for i in range(10):
        atom = Mock()
        atom.index = i
        atom.residue = Mock()
        atom.residue.index = i
        atom.residue.resSeq = i + 1
        atom.residue.name = residue_names[i]
        atoms.append(atom)

    mock_top.atom = lambda i: atoms[i]
    mock_top.atoms = atoms
    mock_top.n_atoms = 10
    mock_top.n_residues = 10
    mock_top.select = Mock(return_value=np.arange(10))

    return mock_top


# ============================================================================
# Test Classes
# ============================================================================

class TestStateAssignment:
    """Tests for state assignment from probabilities."""

    def test_argmax_gives_state_assignments(self, sample_probabilities):
        """State assignments are argmax of probabilities."""
        states = np.argmax(sample_probabilities, axis=1)

        assert states.shape == (100,)
        assert np.all(states >= 0)
        assert np.all(states < 3)

    def test_state_assignments_match_highest_probability(self, sample_probabilities_with_clear_states):
        """Clear state assignments match highest probability."""
        probs = sample_probabilities_with_clear_states
        states = np.argmax(probs, axis=1)

        # First 40 should be state 0
        assert np.all(states[:40] == 0)
        # Next 30 should be state 1
        assert np.all(states[40:70] == 1)
        # Last 30 should be state 2
        assert np.all(states[70:] == 2)

    def test_state_values_in_valid_range(self, sample_probabilities):
        """State assignments are in [0, n_states-1]."""
        n_states = sample_probabilities.shape[1]
        states = np.argmax(sample_probabilities, axis=1)

        assert states.min() >= 0
        assert states.max() < n_states

    def test_unique_states_count(self, sample_probabilities_with_clear_states):
        """All states are represented in clear assignments."""
        probs = sample_probabilities_with_clear_states
        states = np.argmax(probs, axis=1)
        unique_states = np.unique(states)

        assert len(unique_states) == 3


class TestTransitionMatrix:
    """Tests for transition matrix calculation."""

    def test_transition_matrix_shape(self, sample_probabilities):
        """Transition matrix has shape (n_states, n_states)."""
        from pygv.utils.analysis import calculate_transition_matrices

        n_states = sample_probabilities.shape[1]
        T, T_no_self = calculate_transition_matrices(
            sample_probabilities,
            lag_time=0.001,
            stride=1,
            timestep=0.001
        )

        assert T.shape == (n_states, n_states)
        assert T_no_self.shape == (n_states, n_states)

    def test_transition_matrix_rows_sum_to_one(self, sample_probabilities_with_clear_states):
        """Transition matrix rows sum to 1."""
        from pygv.utils.analysis import calculate_transition_matrices

        T, _ = calculate_transition_matrices(
            sample_probabilities_with_clear_states,
            lag_time=0.001,
            stride=1,
            timestep=0.001
        )

        row_sums = T.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_values_in_zero_one(self, sample_probabilities):
        """Transition matrix values are in [0, 1]."""
        from pygv.utils.analysis import calculate_transition_matrices

        T, T_no_self = calculate_transition_matrices(
            sample_probabilities,
            lag_time=0.001,
            stride=1,
            timestep=0.001
        )

        assert np.all(T >= 0)
        assert np.all(T <= 1)
        assert np.all(T_no_self >= 0)
        assert np.all(T_no_self <= 1)

    def test_transition_matrix_no_self_has_zero_diagonal(self, sample_probabilities_with_clear_states):
        """Transition matrix without self-transitions has zero diagonal."""
        from pygv.utils.analysis import calculate_transition_matrices

        _, T_no_self = calculate_transition_matrices(
            sample_probabilities_with_clear_states,
            lag_time=0.001,
            stride=1,
            timestep=0.001
        )

        diagonal = np.diag(T_no_self)
        assert np.allclose(diagonal, 0.0)

    def test_short_trajectory_returns_identity(self):
        """Short trajectory (< lag_frames) returns identity matrix."""
        from pygv.utils.analysis import calculate_transition_matrices

        # 5 frames, but lag requires 10 frames
        probs = np.random.dirichlet([1, 1, 1], size=5)

        T, T_no_self = calculate_transition_matrices(
            probs,
            lag_time=0.01,  # 10 ns lag
            stride=1,
            timestep=0.001  # 1 ns timestep -> needs 10 frames
        )

        # Should return identity matrix for T
        assert np.allclose(T, np.eye(3))

    def test_lag_frames_calculation(self):
        """Lag frames are calculated correctly from lag_time and timestep."""
        from pygv.utils.analysis import calculate_transition_matrices

        probs = np.random.dirichlet([1, 1, 1], size=100)

        # lag_time=0.005 ns, timestep=0.001 ns, stride=1 -> lag_frames=5
        T, _ = calculate_transition_matrices(
            probs,
            lag_time=0.005,
            stride=1,
            timestep=0.001
        )

        # Should complete without error
        assert T.shape == (3, 3)


class TestStateEdgeAttentionMaps:
    """Tests for state-specific edge attention map calculation."""

    def test_attention_maps_shape(self, sample_probabilities, sample_edge_attentions, sample_edge_indices):
        """Attention maps have shape (n_states, n_atoms, n_atoms)."""
        from pygv.utils.analysis import calculate_state_edge_attention_maps

        # Use subset matching
        probs = sample_probabilities[:50]

        with tempfile.TemporaryDirectory() as tmpdir:
            maps, populations = calculate_state_edge_attention_maps(
                edge_attentions=sample_edge_attentions,
                edge_indices=sample_edge_indices,
                probs=probs,
                save_dir=tmpdir,
                protein_name="test"
            )

        n_states = probs.shape[1]
        n_atoms = 10

        assert maps.shape == (n_states, n_atoms, n_atoms)

    def test_state_populations_sum_to_one(self, sample_probabilities, sample_edge_attentions, sample_edge_indices):
        """State populations sum to 1."""
        from pygv.utils.analysis import calculate_state_edge_attention_maps

        probs = sample_probabilities[:50]

        with tempfile.TemporaryDirectory() as tmpdir:
            _, populations = calculate_state_edge_attention_maps(
                edge_attentions=sample_edge_attentions,
                edge_indices=sample_edge_indices,
                probs=probs,
                save_dir=tmpdir,
                protein_name="test"
            )

        assert np.isclose(populations.sum(), 1.0)

    def test_state_populations_non_negative(self, sample_probabilities, sample_edge_attentions, sample_edge_indices):
        """State populations are non-negative."""
        from pygv.utils.analysis import calculate_state_edge_attention_maps

        probs = sample_probabilities[:50]

        with tempfile.TemporaryDirectory() as tmpdir:
            _, populations = calculate_state_edge_attention_maps(
                edge_attentions=sample_edge_attentions,
                edge_indices=sample_edge_indices,
                probs=probs,
                save_dir=tmpdir,
                protein_name="test"
            )

        assert np.all(populations >= 0)

    def test_attention_maps_non_negative(self, sample_probabilities, sample_edge_attentions, sample_edge_indices):
        """Attention maps have non-negative values (averaged from non-negative attentions)."""
        from pygv.utils.analysis import calculate_state_edge_attention_maps

        probs = sample_probabilities[:50]

        with tempfile.TemporaryDirectory() as tmpdir:
            maps, _ = calculate_state_edge_attention_maps(
                edge_attentions=sample_edge_attentions,
                edge_indices=sample_edge_indices,
                probs=probs,
                save_dir=tmpdir,
                protein_name="test"
            )

        assert np.all(maps >= 0)

    def test_saves_files_when_save_dir_provided(self, sample_probabilities, sample_edge_attentions, sample_edge_indices):
        """Files are saved when save_dir is provided."""
        from pygv.utils.analysis import calculate_state_edge_attention_maps

        probs = sample_probabilities[:50]

        with tempfile.TemporaryDirectory() as tmpdir:
            calculate_state_edge_attention_maps(
                edge_attentions=sample_edge_attentions,
                edge_indices=sample_edge_indices,
                probs=probs,
                save_dir=tmpdir,
                protein_name="test"
            )

            # Check files were created
            assert os.path.exists(os.path.join(tmpdir, "test_state_attention_maps.npy"))
            assert os.path.exists(os.path.join(tmpdir, "test_state_populations.npy"))
            assert os.path.exists(os.path.join(tmpdir, "test_state_counts.txt"))


class TestExtractResidueIndices:
    """Tests for extract_residue_indices_from_selection."""

    def test_returns_tuple(self, mock_topology):
        """Function returns tuple of indices and names."""
        from pygv.utils.analysis import extract_residue_indices_from_selection

        result = extract_residue_indices_from_selection(
            selection_string="name CA",
            topology=mock_topology
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_indices_are_list(self, mock_topology):
        """Residue indices are returned as list."""
        from pygv.utils.analysis import extract_residue_indices_from_selection

        indices, names = extract_residue_indices_from_selection(
            selection_string="name CA",
            topology=mock_topology
        )

        assert isinstance(indices, list)
        assert len(indices) > 0

    def test_names_are_list(self, mock_topology):
        """Residue names are returned as list."""
        from pygv.utils.analysis import extract_residue_indices_from_selection

        indices, names = extract_residue_indices_from_selection(
            selection_string="name CA",
            topology=mock_topology
        )

        assert isinstance(names, list)
        assert len(names) > 0

    def test_indices_and_names_same_length(self, mock_topology):
        """Indices and names lists have same length."""
        from pygv.utils.analysis import extract_residue_indices_from_selection

        indices, names = extract_residue_indices_from_selection(
            selection_string="name CA",
            topology=mock_topology
        )

        assert len(indices) == len(names)

    def test_names_contain_residue_info(self, mock_topology):
        """Residue names contain name and number."""
        from pygv.utils.analysis import extract_residue_indices_from_selection

        indices, names = extract_residue_indices_from_selection(
            selection_string="name CA",
            topology=mock_topology
        )

        # Names should be like "ALA1", "GLY2", etc.
        for name in names:
            assert len(name) > 1
            # Should have letters and numbers
            assert any(c.isalpha() for c in name)
            assert any(c.isdigit() for c in name)

    def test_empty_selection_raises_error(self, mock_topology):
        """Empty selection raises ValueError."""
        from pygv.utils.analysis import extract_residue_indices_from_selection

        # Mock empty selection
        mock_topology.select.return_value = np.array([])

        with pytest.raises(ValueError, match="returned no atoms"):
            extract_residue_indices_from_selection(
                selection_string="resid 9999",
                topology=mock_topology
            )


class TestAnalyzeVAMPNetOutputs:
    """Tests for the main analyze_vampnet_outputs function."""

    def test_returns_tuple_of_four(self, mock_model, synthetic_frames_dataset, device):
        """Function returns tuple of 4 elements."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_probs_shape(self, mock_model, synthetic_frames_dataset, device):
        """Probabilities have correct shape."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            probs, _, _, _ = analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

        n_samples = len(synthetic_frames_dataset)
        assert probs.shape[0] == n_samples
        assert probs.shape[1] == 3  # n_states from mock

    def test_embeddings_shape(self, mock_model, synthetic_frames_dataset, device):
        """Embeddings have correct shape."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            _, embeddings, _, _ = analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

        n_samples = len(synthetic_frames_dataset)
        assert embeddings.shape[0] == n_samples
        assert embeddings.shape[1] == 32  # embedding_dim from mock

    def test_returns_numpy_by_default(self, mock_model, synthetic_frames_dataset, device):
        """Returns numpy arrays by default."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            probs, embeddings, _, _ = analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

        assert isinstance(probs, np.ndarray)
        assert isinstance(embeddings, np.ndarray)

    def test_returns_tensors_when_requested(self, mock_model, synthetic_frames_dataset, device):
        """Returns tensors when return_tensors=True."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            probs, embeddings, _, _ = analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=True
            )

        assert isinstance(probs, torch.Tensor)
        assert isinstance(embeddings, torch.Tensor)

    def test_creates_output_files(self, mock_model, synthetic_frames_dataset, device):
        """Creates expected output files."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

            # Check main files
            assert os.path.exists(os.path.join(tmpdir, 'transformed_traj.npz'))
            assert os.path.exists(os.path.join(tmpdir, 'embeddings.npz'))
            assert os.path.exists(os.path.join(tmpdir, 'metadata.txt'))

            # Check attention directories
            assert os.path.isdir(os.path.join(tmpdir, 'edge_attentions'))
            assert os.path.isdir(os.path.join(tmpdir, 'edge_indices'))

    def test_edge_attentions_is_list(self, mock_model, synthetic_frames_dataset, device):
        """Edge attentions is a list."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, attentions, _ = analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

        assert isinstance(attentions, list)

    def test_edge_indices_is_list(self, mock_model, synthetic_frames_dataset, device):
        """Edge indices is a list."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, indices = analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

        assert isinstance(indices, list)

    def test_probs_are_valid_distribution(self, mock_model, synthetic_frames_dataset, device):
        """Probabilities are valid probability distributions."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            probs, _, _, _ = analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

        # All values should be non-negative
        assert np.all(probs >= 0)
        # Rows should sum to 1
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)


class TestProbabilityProperties:
    """Tests for probability output properties."""

    def test_probabilities_non_negative(self, sample_probabilities):
        """All probability values are non-negative."""
        assert np.all(sample_probabilities >= 0)

    def test_probabilities_at_most_one(self, sample_probabilities):
        """All probability values are at most 1."""
        assert np.all(sample_probabilities <= 1)

    def test_probabilities_sum_to_one(self, sample_probabilities):
        """Probability rows sum to 1."""
        row_sums = sample_probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_clear_state_has_high_probability(self, sample_probabilities_with_clear_states):
        """Dominant state has high probability."""
        probs = sample_probabilities_with_clear_states
        max_probs = probs.max(axis=1)

        # All max probabilities should be high (0.9)
        assert np.all(max_probs >= 0.9)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_frame_probabilities(self):
        """Handle single frame probabilities."""
        probs = np.array([[0.3, 0.5, 0.2]])
        states = np.argmax(probs, axis=1)

        assert states.shape == (1,)
        assert states[0] == 1

    def test_two_state_system(self):
        """Handle two-state system."""
        from pygv.utils.analysis import calculate_transition_matrices

        probs = np.random.dirichlet([1, 1], size=100)

        T, T_no_self = calculate_transition_matrices(
            probs,
            lag_time=0.001,
            stride=1,
            timestep=0.001
        )

        assert T.shape == (2, 2)
        assert np.allclose(T.sum(axis=1), 1.0, atol=1e-6)

    def test_many_states_system(self):
        """Handle system with many states."""
        from pygv.utils.analysis import calculate_transition_matrices

        n_states = 10
        probs = np.random.dirichlet([1]*n_states, size=100)

        T, T_no_self = calculate_transition_matrices(
            probs,
            lag_time=0.001,
            stride=1,
            timestep=0.001
        )

        assert T.shape == (n_states, n_states)
        assert np.allclose(T.sum(axis=1), 1.0, atol=1e-6)

    def test_missing_attention_handled(self, sample_probabilities):
        """Handle missing attention values."""
        from pygv.utils.analysis import calculate_state_edge_attention_maps

        probs = sample_probabilities[:10]

        # Mix of valid and None attentions
        attentions = [np.random.rand(50) for _ in range(5)] + [None] * 5
        indices = [np.array([[0,1,2,3,4]*10, [1,2,3,4,0]*10]) for _ in range(5)] + [None] * 5

        with tempfile.TemporaryDirectory() as tmpdir:
            maps, populations = calculate_state_edge_attention_maps(
                edge_attentions=attentions,
                edge_indices=indices,
                probs=probs,
                save_dir=tmpdir,
                protein_name="test"
            )

        # Should complete without error
        assert maps is not None
        assert populations is not None


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_small_probabilities(self):
        """Handle very small probability values."""
        probs = np.array([
            [1e-10, 1 - 2e-10, 1e-10],
            [0.33, 0.34, 0.33]
        ])

        states = np.argmax(probs, axis=1)
        assert states[0] == 1

    def test_equal_probabilities(self):
        """Handle equal probabilities."""
        probs = np.array([
            [1/3, 1/3, 1/3],
            [0.5, 0.5, 0.0]
        ])

        # argmax returns first max index
        states = np.argmax(probs, axis=1)
        assert states.shape == (2,)

    def test_deterministic_results(self, seed):
        """Results are deterministic with fixed seed."""
        np.random.seed(42)
        probs1 = np.random.dirichlet([1, 1, 1], size=100)

        np.random.seed(42)
        probs2 = np.random.dirichlet([1, 1, 1], size=100)

        assert np.array_equal(probs1, probs2)


class TestMetadataOutput:
    """Tests for metadata output."""

    def test_metadata_file_contains_info(self, mock_model, synthetic_frames_dataset, device):
        """Metadata file contains expected information."""
        from pygv.utils.analysis import analyze_vampnet_outputs

        loader = DataLoader(synthetic_frames_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyze_vampnet_outputs(
                model=mock_model,
                data_loader=loader,
                save_folder=tmpdir,
                batch_size=8,
                device=device,
                return_tensors=False
            )

            # Read metadata
            with open(os.path.join(tmpdir, 'metadata.txt'), 'r') as f:
                content = f.read()

            assert 'Number of frames' in content
            assert 'Number of states' in content
            assert 'Embedding dimension' in content


class TestStateCountsOutput:
    """Tests for state counts output."""

    def test_state_counts_file_format(self, sample_probabilities, sample_edge_attentions, sample_edge_indices):
        """State counts file has correct format."""
        from pygv.utils.analysis import calculate_state_edge_attention_maps

        probs = sample_probabilities[:50]

        with tempfile.TemporaryDirectory() as tmpdir:
            calculate_state_edge_attention_maps(
                edge_attentions=sample_edge_attentions,
                edge_indices=sample_edge_indices,
                probs=probs,
                save_dir=tmpdir,
                protein_name="test"
            )

            # Read counts file
            with open(os.path.join(tmpdir, 'test_state_counts.txt'), 'r') as f:
                lines = f.readlines()

            # First line should be header
            assert 'State' in lines[0]
            assert 'Count' in lines[0]
            assert 'Population' in lines[0]

            # Should have n_states + 1 lines (header + data)
            assert len(lines) == probs.shape[1] + 1
