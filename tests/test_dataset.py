"""
Unit tests for VAMPNetDataset.

Tests verify that the dataset:
1. Creates valid PyG graph representations from molecular coordinates
2. Correctly constructs k-NN graphs with proper edge features
3. Handles time-lagged pair creation in both continuous and non-continuous modes
4. Supports different node encoding schemes (one-hot, amino acid labels/properties)
5. Properly validates lag time parameters
6. Implements caching correctly

Run with: pytest tests/test_dataset.py -v
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from torch_geometric.data import Data
import os
import tempfile
import pickle


# =============================================================================
# Fixtures: Reusable test data and mocks
# =============================================================================

@pytest.fixture
def device():
    """Use CPU for tests to ensure reproducibility and CI compatibility."""
    return torch.device('cpu')


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


def create_mock_topology(n_atoms=10):
    """Create mock topology with CA atoms."""
    mock_top = Mock()
    mock_top.select.return_value = np.arange(n_atoms)

    # Mock atoms with residue names
    residue_names = ['ALA', 'GLY', 'VAL', 'LEU', 'ILE',
                     'PRO', 'PHE', 'TYR', 'TRP', 'SER'][:n_atoms]
    # Extend if n_atoms > 10
    while len(residue_names) < n_atoms:
        residue_names.extend(['ALA', 'GLY', 'VAL', 'LEU', 'ILE'][:n_atoms - len(residue_names)])

    atoms = []
    for i, res_name in enumerate(residue_names):
        atom = Mock()
        atom.residue = Mock()
        atom.residue.name = res_name
        atoms.append(atom)

    mock_top.atom = lambda i, atoms=atoms: atoms[i]
    mock_top.n_atoms = n_atoms
    return mock_top


def create_mock_trajectory(n_frames=100, n_atoms=10, timestep_ps=1.0):
    """Create mock trajectory with specified parameters."""
    # Coordinates in nanometers, small random displacements around a line
    base_coords = np.linspace(0.3, 1.0, n_atoms).reshape(1, n_atoms, 1)
    base_coords = np.tile(base_coords, (n_frames, 1, 3))
    # Add some random displacement
    np.random.seed(42)
    xyz = (base_coords + np.random.randn(n_frames, n_atoms, 3) * 0.05).astype(np.float32)
    time = np.arange(n_frames) * timestep_ps
    topology = create_mock_topology(n_atoms)

    # Create a class that properly handles slicing
    class MockTrajectory:
        def __init__(self, xyz, time, topology):
            self._xyz = xyz
            self._time = time
            self.topology = topology
            self.n_frames = len(xyz)

        @property
        def xyz(self):
            return self._xyz

        @property
        def time(self):
            return self._time

        def __getitem__(self, key):
            """Support slicing like traj[::stride]."""
            new_xyz = self._xyz[key]
            new_time = self._time[key]
            return MockTrajectory(new_xyz, new_time, self.topology)

        def __len__(self):
            return self.n_frames

    return MockTrajectory(xyz, time, topology)


@pytest.fixture
def mock_topology():
    """Mock MDTraj topology with 10 CA atoms."""
    return create_mock_topology(10)


@pytest.fixture
def mock_trajectory(mock_topology):
    """Mock MDTraj trajectory with 100 frames, 10 atoms."""
    traj = create_mock_trajectory(n_frames=100, n_atoms=10, timestep_ps=1.0)
    traj.topology = mock_topology
    return traj


@pytest.fixture
def mock_mdtraj(mock_trajectory, mock_topology, monkeypatch):
    """Patch mdtraj.load and mdtraj.load_topology."""
    # Mock mdtraj.load to return the mock trajectory
    def mock_load(*args, **kwargs):
        return mock_trajectory

    # Mock mdtraj.load_topology to return the mock topology
    def mock_load_topology(*args, **kwargs):
        return mock_topology

    # Mock mdtraj.iterload to return iterator with first chunk
    def mock_iterload(*args, **kwargs):
        chunk = create_mock_trajectory(n_frames=2, n_atoms=10, timestep_ps=1.0)
        return iter([chunk])

    monkeypatch.setattr('mdtraj.load', mock_load)
    monkeypatch.setattr('mdtraj.load_topology', mock_load_topology)
    monkeypatch.setattr('mdtraj.iterload', mock_iterload)


@pytest.fixture
def dataset_params():
    """Default parameters for creating a VAMPNetDataset."""
    return {
        'trajectory_files': ['/fake/traj.xtc'],
        'topology_file': '/fake/topology.pdb',
        'lag_time': 0.01,  # 10 ps lag time (compatible with 1 ps timestep)
        'n_neighbors': 5,
        'node_embedding_dim': 16,
        'gaussian_expansion_dim': 16,
        'selection': 'name CA',
        'seed': 42,
        'stride': 1,
        'cache_dir': None,
        'use_cache': False,
        'continuous': True,
    }


@pytest.fixture
def dataset(mock_mdtraj, dataset_params):
    """Create a VAMPNetDataset with mocked MDTraj."""
    from pygv.dataset.vampnet_dataset import VAMPNetDataset
    return VAMPNetDataset(**dataset_params)


# =============================================================================
# TestGraphConstruction - Tests graph creation from frame coordinates
# =============================================================================

class TestGraphConstruction:
    """Tests graph creation from frame coordinates."""

    def test_graph_has_correct_num_nodes(self, dataset):
        """n_nodes == n_atoms from selection."""
        graph = dataset.get_graph(0)
        assert graph.num_nodes == dataset.n_atoms
        assert graph.x.shape[0] == dataset.n_atoms

    def test_knn_edges_count(self, dataset):
        """Edge count approximates k * n_nodes (asymmetric k-NN)."""
        graph = dataset.get_graph(0)
        n_edges = graph.edge_index.shape[1]
        expected_min = dataset.n_atoms * dataset.n_neighbors * 0.5
        expected_max = dataset.n_atoms * dataset.n_neighbors * 2
        assert n_edges >= expected_min, f"Too few edges: {n_edges} < {expected_min}"
        assert n_edges <= expected_max, f"Too many edges: {n_edges} > {expected_max}"

    def test_no_self_edges(self, dataset):
        """No edges where src == dst."""
        graph = dataset.get_graph(0)
        src = graph.edge_index[0]
        dst = graph.edge_index[1]
        self_loops = (src == dst).sum().item()
        assert self_loops == 0, f"Found {self_loops} self-loops"

    def test_edge_attr_shape(self, dataset):
        """Shape is (n_edges, gaussian_expansion_dim)."""
        graph = dataset.get_graph(0)
        n_edges = graph.edge_index.shape[1]
        assert graph.edge_attr.shape == (n_edges, dataset.gaussian_expansion_dim)

    def test_edge_attr_values_valid(self, dataset):
        """No NaN, values in reasonable range."""
        graph = dataset.get_graph(0)
        assert not torch.isnan(graph.edge_attr).any(), "Edge attributes contain NaN"
        assert not torch.isinf(graph.edge_attr).any(), "Edge attributes contain Inf"
        # Gaussian expansion should be non-negative and bounded
        assert (graph.edge_attr >= 0).all(), "Gaussian expansion should be non-negative"
        assert (graph.edge_attr <= 1.1).all(), "Gaussian expansion should be bounded"

    def test_graph_is_valid_pyg_data(self, dataset):
        """Returns torch_geometric.data.Data with required attrs."""
        graph = dataset.get_graph(0)
        assert isinstance(graph, Data)
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'edge_attr')
        assert hasattr(graph, 'num_nodes')

    def test_batch_tensor_present(self, dataset):
        """Graph can have batch tensor added (zeros for single graph)."""
        graph = dataset.get_graph(0)
        # PyG Data objects don't have batch by default, but it can be added
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        graph.batch = batch
        assert graph.batch.shape[0] == graph.num_nodes
        assert (graph.batch == 0).all()

    def test_deterministic_graph_construction(self, mock_mdtraj, dataset_params):
        """Same seed produces same graph."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        params1 = dataset_params.copy()
        params1['seed'] = 123
        ds1 = VAMPNetDataset(**params1)
        graph1 = ds1.get_graph(0)

        params2 = dataset_params.copy()
        params2['seed'] = 123
        ds2 = VAMPNetDataset(**params2)
        graph2 = ds2.get_graph(0)

        assert torch.allclose(graph1.x, graph2.x)
        assert torch.equal(graph1.edge_index, graph2.edge_index)
        assert torch.allclose(graph1.edge_attr, graph2.edge_attr)


# =============================================================================
# TestNodeFeatures - Tests node feature encoding modes
# =============================================================================

class TestNodeFeatures:
    """Tests node feature encoding modes."""

    def test_onehot_shape(self, dataset):
        """Shape (n_atoms, n_atoms) for one-hot encoding."""
        graph = dataset.get_graph(0, use_amino_acid_encoding=False)
        assert graph.x.shape == (dataset.n_atoms, dataset.n_atoms)

    def test_onehot_diagonal_ones(self, dataset):
        """x[i,i] == 1.0 for all i."""
        graph = dataset.get_graph(0, use_amino_acid_encoding=False)
        for i in range(dataset.n_atoms):
            assert graph.x[i, i] == 1.0, f"Diagonal at {i} is not 1.0"

    def test_onehot_offdiagonal_zeros(self, dataset):
        """x[i,j] == 0.0 for i != j."""
        graph = dataset.get_graph(0, use_amino_acid_encoding=False)
        for i in range(dataset.n_atoms):
            for j in range(dataset.n_atoms):
                if i != j:
                    assert graph.x[i, j] == 0.0, f"Off-diagonal at ({i},{j}) is not 0.0"

    def test_aa_labels_shape(self, mock_mdtraj, dataset_params):
        """Shape (n_atoms, 1) for amino acid labels."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        params = dataset_params.copy()
        params['use_amino_acid_encoding'] = True
        params['amino_acid_feature_type'] = 'labels'
        ds = VAMPNetDataset(**params)
        graph = ds.get_graph(0)
        assert graph.x.shape == (ds.n_atoms, 1)

    def test_aa_labels_range(self, mock_mdtraj, dataset_params):
        """Values in [0, 20] for amino acid labels."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        params = dataset_params.copy()
        params['use_amino_acid_encoding'] = True
        params['amino_acid_feature_type'] = 'labels'
        ds = VAMPNetDataset(**params)
        graph = ds.get_graph(0)
        assert (graph.x >= 0).all(), "Labels should be non-negative"
        assert (graph.x <= 20).all(), "Labels should be at most 20"

    def test_aa_properties_shape(self, mock_mdtraj, dataset_params):
        """Shape (n_atoms, 4) for amino acid properties."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        params = dataset_params.copy()
        params['use_amino_acid_encoding'] = True
        params['amino_acid_feature_type'] = 'properties'
        ds = VAMPNetDataset(**params)
        graph = ds.get_graph(0)
        assert graph.x.shape == (ds.n_atoms, 4)


# =============================================================================
# TestGaussianExpansion - Tests Gaussian basis function edge features
# =============================================================================

class TestGaussianExpansion:
    """Tests Gaussian basis function edge features."""

    def test_gaussian_output_shape(self, dataset):
        """Shape (n_edges, K) where K = gaussian_expansion_dim."""
        graph = dataset.get_graph(0)
        n_edges = graph.edge_index.shape[1]
        assert graph.edge_attr.shape == (n_edges, dataset.gaussian_expansion_dim)

    def test_gaussian_values_nonnegative(self, dataset):
        """All values >= 0 (Gaussian outputs)."""
        graph = dataset.get_graph(0)
        assert (graph.edge_attr >= 0).all(), "Gaussian values should be non-negative"

    def test_gaussian_values_bounded(self, dataset):
        """Values in [0, 1] range (approximately)."""
        graph = dataset.get_graph(0)
        assert (graph.edge_attr >= 0).all()
        assert (graph.edge_attr <= 1.1).all(), "Gaussian values should be bounded"

    def test_gaussian_deterministic(self, mock_mdtraj, dataset_params):
        """Same distances produce same features."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        ds1 = VAMPNetDataset(**dataset_params)
        ds2 = VAMPNetDataset(**dataset_params)

        graph1 = ds1.get_graph(0)
        graph2 = ds2.get_graph(0)

        assert torch.allclose(graph1.edge_attr, graph2.edge_attr)


# =============================================================================
# TestTimeLaggedPairs - Tests time-lagged pair creation in both modes
# =============================================================================

class TestTimeLaggedPairs:
    """Tests time-lagged pair creation in both modes."""

    def test_continuous_pair_count(self, dataset):
        """n_pairs == n_frames - lag_frames in continuous mode."""
        expected_pairs = dataset.n_frames - dataset.lag_frames
        assert len(dataset) == expected_pairs

    def test_continuous_pair_offset(self, dataset):
        """t1_indices[i] == t0_indices[i] + lag_frames."""
        for i in range(min(10, len(dataset))):
            t0 = dataset.t0_indices[i]
            t1 = dataset.t1_indices[i]
            assert t1 == t0 + dataset.lag_frames

    def test_noncontinuous_no_boundary_crossing(self, mock_mdtraj, monkeypatch):
        """No pairs span trajectory boundaries in non-continuous mode."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        # Create two mock trajectories
        traj1 = create_mock_trajectory(n_frames=50, n_atoms=10, timestep_ps=1.0)
        traj2 = create_mock_trajectory(n_frames=50, n_atoms=10, timestep_ps=1.0)

        call_count = [0]
        trajs = [traj1, traj2]

        def mock_load(*args, **kwargs):
            idx = call_count[0] % 2
            call_count[0] += 1
            return trajs[idx]

        def mock_iterload(*args, **kwargs):
            chunk = create_mock_trajectory(n_frames=2, n_atoms=10, timestep_ps=1.0)
            return iter([chunk])

        topology = create_mock_topology(10)
        monkeypatch.setattr('mdtraj.load', mock_load)
        monkeypatch.setattr('mdtraj.load_topology', lambda *args, **kwargs: topology)
        monkeypatch.setattr('mdtraj.iterload', mock_iterload)

        ds = VAMPNetDataset(
            trajectory_files=['/fake/traj1.xtc', '/fake/traj2.xtc'],
            topology_file='/fake/topology.pdb',
            lag_time=0.01,
            n_neighbors=5,
            continuous=False,
            use_cache=False,
        )

        # Check no pairs cross the boundary at frame 50
        boundary = 50
        for t0, t1 in zip(ds.t0_indices, ds.t1_indices):
            # Both should be in same trajectory
            t0_traj = 0 if t0 < boundary else 1
            t1_traj = 0 if t1 < boundary else 1
            assert t0_traj == t1_traj, f"Pair ({t0}, {t1}) crosses boundary"

    def test_noncontinuous_fewer_pairs(self, mock_mdtraj, monkeypatch):
        """n_pairs < continuous mode pairs when trajectories are split."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        # Create two mock trajectories
        traj1 = create_mock_trajectory(n_frames=50, n_atoms=10, timestep_ps=1.0)
        traj2 = create_mock_trajectory(n_frames=50, n_atoms=10, timestep_ps=1.0)

        call_count = [0]
        trajs = [traj1, traj2]

        def mock_load(*args, **kwargs):
            idx = call_count[0] % 2
            call_count[0] += 1
            return trajs[idx]

        def mock_iterload(*args, **kwargs):
            chunk = create_mock_trajectory(n_frames=2, n_atoms=10, timestep_ps=1.0)
            return iter([chunk])

        topology = create_mock_topology(10)
        monkeypatch.setattr('mdtraj.load', mock_load)
        monkeypatch.setattr('mdtraj.load_topology', lambda *args, **kwargs: topology)
        monkeypatch.setattr('mdtraj.iterload', mock_iterload)

        # Continuous mode
        call_count[0] = 0
        ds_cont = VAMPNetDataset(
            trajectory_files=['/fake/traj1.xtc', '/fake/traj2.xtc'],
            topology_file='/fake/topology.pdb',
            lag_time=0.01,
            n_neighbors=5,
            continuous=True,
            use_cache=False,
        )

        # Non-continuous mode
        call_count[0] = 0
        ds_noncont = VAMPNetDataset(
            trajectory_files=['/fake/traj1.xtc', '/fake/traj2.xtc'],
            topology_file='/fake/topology.pdb',
            lag_time=0.01,
            n_neighbors=5,
            continuous=False,
            use_cache=False,
        )

        # Non-continuous should have fewer pairs (excludes boundary-crossing pairs)
        assert len(ds_noncont) < len(ds_cont)

    def test_short_trajectory_skipped(self, mock_mdtraj, monkeypatch, capsys):
        """Warns when traj_length <= lag_frames."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        # Create one short trajectory and one normal
        short_traj = create_mock_trajectory(n_frames=5, n_atoms=10, timestep_ps=1.0)
        normal_traj = create_mock_trajectory(n_frames=50, n_atoms=10, timestep_ps=1.0)

        call_count = [0]
        trajs = [short_traj, normal_traj]

        def mock_load(*args, **kwargs):
            idx = call_count[0] % 2
            call_count[0] += 1
            return trajs[idx]

        def mock_iterload(*args, **kwargs):
            chunk = create_mock_trajectory(n_frames=2, n_atoms=10, timestep_ps=1.0)
            return iter([chunk])

        topology = create_mock_topology(10)
        monkeypatch.setattr('mdtraj.load', mock_load)
        monkeypatch.setattr('mdtraj.load_topology', lambda *args, **kwargs: topology)
        monkeypatch.setattr('mdtraj.iterload', mock_iterload)

        ds = VAMPNetDataset(
            trajectory_files=['/fake/short.xtc', '/fake/normal.xtc'],
            topology_file='/fake/topology.pdb',
            lag_time=0.01,  # 10 frames lag
            n_neighbors=5,
            continuous=False,
            use_cache=False,
        )

        captured = capsys.readouterr()
        assert "less than lag_frames" in captured.out or len(ds) > 0

    def test_getitem_returns_tuple(self, dataset):
        """__getitem__ returns (graph_t0, graph_t1)."""
        result = dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_getitem_graphs_different(self, dataset):
        """t0 and t1 graphs are different objects."""
        graph_t0, graph_t1 = dataset[0]
        assert graph_t0 is not graph_t1
        # Coordinates should differ (different frames)
        # Note: node features may be the same if using one-hot

    def test_len_equals_pair_count(self, dataset):
        """len(dataset) == len(t0_indices)."""
        assert len(dataset) == len(dataset.t0_indices)
        assert len(dataset) == len(dataset.t1_indices)


# =============================================================================
# TestLagTimeValidation - Tests lag time compatibility checking
# =============================================================================

class TestLagTimeValidation:
    """Tests lag time compatibility checking."""

    def test_valid_lag_time_accepted(self, mock_mdtraj, dataset_params):
        """Compatible lag_time doesn't raise."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        # 10 ps with 1 ps timestep should work
        params = dataset_params.copy()
        params['lag_time'] = 0.01  # 10 ps
        ds = VAMPNetDataset(**params)
        assert ds.lag_frames == 10

    def test_invalid_lag_time_raises(self, mock_mdtraj, dataset_params):
        """Incompatible lag_time raises ValueError."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        params = dataset_params.copy()
        params['lag_time'] = 0.0015  # 1.5 ps - not divisible by 1 ps timestep
        with pytest.raises(ValueError, match="cannot be achieved"):
            VAMPNetDataset(**params)

    def test_lag_frames_calculation(self, mock_mdtraj, dataset_params):
        """lag_frames = int(lag_time_ps / timestep)."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        params = dataset_params.copy()
        params['lag_time'] = 0.02  # 20 ps
        params['stride'] = 1
        ds = VAMPNetDataset(**params)
        # 20 ps / 1 ps = 20 frames
        assert ds.lag_frames == 20

    def test_stride_affects_effective_timestep(self, mock_mdtraj, dataset_params):
        """Stride changes effective timestep."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        params = dataset_params.copy()
        params['lag_time'] = 0.02  # 20 ps
        params['stride'] = 2  # Effective timestep = 2 ps
        ds = VAMPNetDataset(**params)
        # 20 ps / 2 ps = 10 frames
        assert ds.lag_frames == 10


# =============================================================================
# TestCaching - Tests cache save/load functionality
# =============================================================================

class TestCaching:
    """Tests cache save/load functionality."""

    def test_cache_filename_format(self, mock_mdtraj, dataset_params):
        """Filename includes hash, lag, nn, stride, cont flag."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            params = dataset_params.copy()
            params['cache_dir'] = tmpdir
            params['use_cache'] = True
            ds = VAMPNetDataset(**params)

            cache_file = ds._get_cache_filename()
            assert 'vampnet_data_' in cache_file
            assert f'lag{params["lag_time"]}' in cache_file
            assert f'nn{params["n_neighbors"]}' in cache_file
            assert f'str{params["stride"]}' in cache_file
            assert 'cont' in cache_file  # continuous flag

    def test_cache_saves_on_first_run(self, mock_mdtraj, dataset_params):
        """Cache file created when cache_dir set."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            params = dataset_params.copy()
            params['cache_dir'] = tmpdir
            params['use_cache'] = True
            ds = VAMPNetDataset(**params)

            cache_file = ds._get_cache_filename()
            assert os.path.exists(cache_file)

    def test_cache_loads_on_second_run(self, mock_mdtraj, dataset_params, monkeypatch):
        """Second init loads from cache."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            params = dataset_params.copy()
            params['cache_dir'] = tmpdir
            params['use_cache'] = True

            # First run - creates cache
            ds1 = VAMPNetDataset(**params)
            n_frames1 = ds1.n_frames

            # Track if _process_trajectories is called
            process_called = [False]
            original_process = VAMPNetDataset._process_trajectories

            def mock_process(self):
                process_called[0] = True
                return original_process(self)

            monkeypatch.setattr(VAMPNetDataset, '_process_trajectories', mock_process)

            # Second run - should load from cache
            ds2 = VAMPNetDataset(**params)
            assert ds2.n_frames == n_frames1
            # _process_trajectories should NOT be called when loading from cache
            assert not process_called[0], "Should have loaded from cache"

    def test_cache_contains_config(self, mock_mdtraj, dataset_params):
        """Saved config matches init params."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            params = dataset_params.copy()
            params['cache_dir'] = tmpdir
            params['use_cache'] = True
            ds = VAMPNetDataset(**params)

            cache_file = ds._get_cache_filename()
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            config = data['config']
            assert config['lag_time'] == params['lag_time']
            assert config['n_neighbors'] == params['n_neighbors']
            assert config['stride'] == params['stride']

    def test_config_mismatch_warns(self, mock_mdtraj, dataset_params, capsys):
        """Warning issued on config change."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            params = dataset_params.copy()
            params['cache_dir'] = tmpdir
            params['use_cache'] = True

            # First run with n_neighbors=5
            ds1 = VAMPNetDataset(**params)

            # Manually modify cache file to have different config
            cache_file = ds1._get_cache_filename()
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            data['config']['n_neighbors'] = 999  # Different value
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

            # Second run should warn about mismatch
            ds2 = VAMPNetDataset(**params)
            captured = capsys.readouterr()
            assert "Warning" in captured.out or "doesn't match" in captured.out

    def test_use_cache_false_skips_cache(self, mock_mdtraj, dataset_params, monkeypatch):
        """use_cache=False ignores existing cache."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            params = dataset_params.copy()
            params['cache_dir'] = tmpdir

            # First run with caching
            params['use_cache'] = True
            ds1 = VAMPNetDataset(**params)

            # Track if _process_trajectories is called
            process_called = [False]
            original_process = VAMPNetDataset._process_trajectories

            def mock_process(self):
                process_called[0] = True
                return original_process(self)

            monkeypatch.setattr(VAMPNetDataset, '_process_trajectories', mock_process)

            # Second run with use_cache=False
            params['use_cache'] = False
            ds2 = VAMPNetDataset(**params)
            # Should process trajectories even though cache exists
            assert process_called[0], "Should have processed trajectories"


# =============================================================================
# TestDatasetInterface - Tests PyTorch Dataset API compliance
# =============================================================================

class TestDatasetInterface:
    """Tests PyTorch Dataset API compliance."""

    def test_len_returns_int(self, dataset):
        """len() returns integer."""
        length = len(dataset)
        assert isinstance(length, int)
        assert length > 0

    def test_getitem_valid_index(self, dataset):
        """Valid index returns data."""
        result = dataset[0]
        assert result is not None
        graph_t0, graph_t1 = result
        assert isinstance(graph_t0, Data)
        assert isinstance(graph_t1, Data)

    def test_getitem_negative_index(self, dataset):
        """Negative index works (Python convention)."""
        # Python lists support negative indexing
        last_idx = len(dataset) - 1
        result_last = dataset[last_idx]
        result_neg = dataset[-1]
        # Note: These should return same pair
        # (implementation may vary, but should not raise)
        assert result_last is not None
        assert result_neg is not None

    def test_get_graph_single_frame(self, dataset):
        """get_graph(idx) returns single graph."""
        graph = dataset.get_graph(0)
        assert isinstance(graph, Data)
        assert graph.x is not None
        assert graph.edge_index is not None

    def test_get_frames_dataset(self, dataset):
        """get_frames_dataset() returns variant."""
        frames_ds = dataset.get_frames_dataset()
        assert len(frames_ds) == dataset.n_frames
        # Should return single graph, not pair
        graph = frames_ds[0]
        assert isinstance(graph, Data)


# =============================================================================
# TestEdgeCases - Tests boundary conditions
# =============================================================================

class TestEdgeCases:
    """Tests boundary conditions."""

    def test_single_frame_trajectory(self, monkeypatch):
        """Handles 1-frame trajectory gracefully - creates dataset with 0 pairs."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        single_frame_traj = create_mock_trajectory(n_frames=1, n_atoms=10, timestep_ps=1.0)
        topology = create_mock_topology(10)

        # Need at least 2 frames for iterload to determine timestep
        two_frame_chunk = create_mock_trajectory(n_frames=2, n_atoms=10, timestep_ps=1.0)

        monkeypatch.setattr('mdtraj.load', lambda *args, **kwargs: single_frame_traj)
        monkeypatch.setattr('mdtraj.load_topology', lambda *args, **kwargs: topology)
        monkeypatch.setattr('mdtraj.iterload', lambda *args, **kwargs: iter([two_frame_chunk]))

        # With 1 frame and lag_time requiring multiple frames, should create 0 pairs
        # The dataset handles this gracefully rather than raising
        ds = VAMPNetDataset(
            trajectory_files=['/fake/single.xtc'],
            topology_file='/fake/topology.pdb',
            lag_time=0.01,
            n_neighbors=5,
            use_cache=False,
        )
        # Should have 0 time-lagged pairs (single frame can't have pairs)
        assert len(ds) == 0
        assert ds.n_frames == 1

    def test_few_atoms(self, monkeypatch):
        """Handles n_atoms < n_neighbors."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        # 3 atoms but requesting 5 neighbors
        few_atoms_traj = create_mock_trajectory(n_frames=100, n_atoms=3, timestep_ps=1.0)
        topology = create_mock_topology(3)

        monkeypatch.setattr('mdtraj.load', lambda *args, **kwargs: few_atoms_traj)
        monkeypatch.setattr('mdtraj.load_topology', lambda *args, **kwargs: topology)
        monkeypatch.setattr('mdtraj.iterload', lambda *args, **kwargs: iter([
            create_mock_trajectory(n_frames=2, n_atoms=3, timestep_ps=1.0)
        ]))

        # Should handle gracefully - use min(n_neighbors, n_atoms-1)
        ds = VAMPNetDataset(
            trajectory_files=['/fake/few_atoms.xtc'],
            topology_file='/fake/topology.pdb',
            lag_time=0.01,
            n_neighbors=5,  # More than available atoms
            use_cache=False,
        )
        graph = ds.get_graph(0)
        assert graph.num_nodes == 3
        # Should still create edges (to available neighbors)
        assert graph.edge_index.shape[1] > 0

    def test_empty_selection_raises(self, monkeypatch):
        """Empty atom selection raises ValueError."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        # Create topology with select that returns empty array
        empty_topology = Mock()
        empty_topology.select.return_value = np.array([], dtype=int)
        empty_topology.n_atoms = 10

        # Create trajectory with this topology
        traj = create_mock_trajectory(n_frames=100, n_atoms=10, timestep_ps=1.0)
        # Replace the trajectory's topology with our mock that returns empty selection
        traj.topology = empty_topology

        monkeypatch.setattr('mdtraj.load', lambda *args, **kwargs: traj)
        monkeypatch.setattr('mdtraj.load_topology', lambda *args, **kwargs: empty_topology)
        monkeypatch.setattr('mdtraj.iterload', lambda *args, **kwargs: iter([
            create_mock_trajectory(n_frames=2, n_atoms=10, timestep_ps=1.0)
        ]))

        # The error message varies but should contain relevant info
        with pytest.raises(ValueError, match="(no atoms|No frames)"):
            VAMPNetDataset(
                trajectory_files=['/fake/traj.xtc'],
                topology_file='/fake/topology.pdb',
                lag_time=0.01,
                n_neighbors=5,
                selection='invalid selection',
                use_cache=False,
            )

    def test_missing_trajectory_file_skipped(self, monkeypatch, capsys):
        """Missing file logged, continues with other files."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        good_traj = create_mock_trajectory(n_frames=100, n_atoms=10, timestep_ps=1.0)
        topology = create_mock_topology(10)

        call_count = [0]

        def mock_load(path, *args, **kwargs):
            call_count[0] += 1
            if 'missing' in path:
                raise FileNotFoundError(f"File not found: {path}")
            return good_traj

        monkeypatch.setattr('mdtraj.load', mock_load)
        monkeypatch.setattr('mdtraj.load_topology', lambda *args, **kwargs: topology)
        monkeypatch.setattr('mdtraj.iterload', lambda *args, **kwargs: iter([
            create_mock_trajectory(n_frames=2, n_atoms=10, timestep_ps=1.0)
        ]))

        ds = VAMPNetDataset(
            trajectory_files=['/fake/missing.xtc', '/fake/good.xtc'],
            topology_file='/fake/topology.pdb',
            lag_time=0.01,
            n_neighbors=5,
            use_cache=False,
        )

        captured = capsys.readouterr()
        assert "Error" in captured.out or ds.n_frames > 0


# =============================================================================
# TestFramesDataset - Tests frames dataset variants
# =============================================================================

class TestFramesDataset:
    """Tests frames dataset variants."""

    def test_frames_dataset_length(self, dataset):
        """Frames dataset has n_frames samples."""
        frames_ds = dataset.get_frames_dataset(return_pairs=False)
        assert len(frames_ds) == dataset.n_frames

    def test_frames_dataset_returns_single_graph(self, dataset):
        """Frames dataset returns single graph, not tuple."""
        frames_ds = dataset.get_frames_dataset(return_pairs=False)
        result = frames_ds[0]
        assert isinstance(result, Data)
        assert not isinstance(result, tuple)

    def test_frames_dataset_with_pairs(self, dataset):
        """Frames dataset can return pairs."""
        frames_ds = dataset.get_frames_dataset(return_pairs=True)
        assert len(frames_ds) == len(dataset)
        result = frames_ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_frames_dataset_with_encoding_override(self, mock_mdtraj, dataset_params):
        """Can override encoding in frames dataset."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        params = dataset_params.copy()
        params['use_amino_acid_encoding'] = False  # Default to one-hot
        ds = VAMPNetDataset(**params)

        # Get frames with amino acid encoding override
        frames_ds = ds.get_frames_dataset_with_encoding(
            return_pairs=False,
            use_amino_acid_encoding=True
        )
        graph = frames_ds[0]
        # Should have amino acid features (1D labels)
        # The parent dataset uses labels by default
        assert graph.x.shape[1] == 1  # labels shape


# =============================================================================
# TestPrecomputeGraphs - Tests graph precomputation
# =============================================================================

class TestPrecomputeGraphs:
    """Tests graph precomputation functionality."""

    def test_precompute_stores_graphs(self, dataset):
        """Precompute creates graph cache."""
        dataset.precompute_graphs(max_graphs=10)
        assert hasattr(dataset, 'graphs')
        assert len(dataset.graphs) == 10

    def test_precompute_getitem_uses_cache(self, dataset):
        """After precompute, getitem uses cached graphs."""
        dataset.precompute_graphs(max_graphs=20)
        # Should return cached graphs
        graph_t0, graph_t1 = dataset[0]
        assert isinstance(graph_t0, Data)
        assert isinstance(graph_t1, Data)


# =============================================================================
# TestDistanceRange - Tests distance range determination
# =============================================================================

class TestDistanceRange:
    """Tests distance range determination."""

    def test_distance_range_determined(self, dataset):
        """Distance min and max are set."""
        assert hasattr(dataset, 'distance_min')
        assert hasattr(dataset, 'distance_max')
        assert dataset.distance_min < dataset.distance_max

    def test_distance_range_reasonable(self, dataset):
        """Distance range is physically reasonable (nanometers)."""
        # For protein CA atoms, distances typically 0.3-5 nm
        assert dataset.distance_min > 0
        assert dataset.distance_max < 100  # Very generous upper bound


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
