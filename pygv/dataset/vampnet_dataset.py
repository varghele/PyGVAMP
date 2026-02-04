"""
VAMPNet Dataset - Unified implementation with optional amino acid encoding.

This module provides a unified VAMPNetDataset class that can create graph representations
from MD trajectories with either one-hot node encoding or amino acid property encoding.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import mdtraj as md
from typing import List, Optional, Literal
import pickle
from tqdm import tqdm

from pygv.utils.features import get_amino_acid_features, get_amino_acid_labels


class VAMPNetDataset(Dataset):
    """
    VAMPNet dataset for creating graph representations from MD trajectories.

    This dataset converts molecular dynamics trajectories into time-lagged pairs of
    graphs suitable for training VAMPNet models. Each molecular structure is represented
    as a k-nearest neighbor graph with Gaussian-expanded edge features.

    Parameters
    ----------
    trajectory_files : List[str]
        List of MD trajectory files (.xtc, .dcd, etc.)
    topology_file : str
        Topology file for the trajectories (.pdb)
    lag_time : float, default=1
        Time lag between pairs (in nanoseconds)
    n_neighbors : int, default=10
        Number of nearest neighbors for graph construction
    node_embedding_dim : int, default=16
        Dimension of node embeddings (used for positional encoding)
    gaussian_expansion_dim : int, default=16
        Dimension of Gaussian expansion for edge features
    distance_min : float, optional
        Minimum distance for Gaussian expansion (auto-determined if None)
    distance_max : float, optional
        Maximum distance for Gaussian expansion (auto-determined if None)
    selection : str, default="name CA"
        MDTraj selection string for atoms to include
    seed : int, default=42
        Random seed for reproducibility
    stride : int, default=1
        Process every 'stride' frames from trajectories
    chunk_size : int, default=1000
        Number of frames to process at once (memory management)
    cache_dir : str, optional
        Directory to save/load cached data
    use_cache : bool, default=True
        Whether to use cached data if available
    use_amino_acid_encoding : bool, default=False
        If True, use amino acid property-based node features.
        If False, use one-hot encoding for node features.
    amino_acid_feature_type : str, default="labels"
        Type of amino acid features to use when use_amino_acid_encoding=True.
        Options: "labels" (integer labels 0-20) or "properties" (4D property vector)
    continuous : bool, default=True
        If True, treat all trajectory files as one continuous trajectory (original behavior).
        If False, treat each trajectory file as independent - time-lagged pairs will not
        cross trajectory boundaries. Use False when trajectories are from independent
        simulations (e.g., multiple replicas, different starting conditions).
    """

    def __init__(
            self,
            trajectory_files: List[str],
            topology_file: str,
            lag_time: float = 1,
            n_neighbors: int = 10,
            node_embedding_dim: int = 16,
            gaussian_expansion_dim: int = 16,
            distance_min: Optional[float] = None,
            distance_max: Optional[float] = None,
            selection: str = "name CA",
            seed: int = 42,
            stride: int = 1,
            chunk_size: int = 1000,
            cache_dir: Optional[str] = None,
            use_cache: bool = True,
            use_amino_acid_encoding: bool = False,
            amino_acid_feature_type: Literal["labels", "properties"] = "labels",
            continuous: bool = True,
    ):
        super(VAMPNetDataset, self).__init__()

        self.trajectory_files = trajectory_files
        self.topology_file = topology_file
        self.lag_time = lag_time
        self.n_neighbors = n_neighbors
        self.node_embedding_dim = node_embedding_dim
        self.gaussian_expansion_dim = gaussian_expansion_dim
        self.selection = selection
        self.stride = stride
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        self.use_amino_acid_encoding = use_amino_acid_encoding
        self.amino_acid_feature_type = amino_acid_feature_type
        self.continuous = continuous

        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load topology (needed for amino acid encoding and atom selection)
        self.topology = md.load_topology(self.topology_file)

        # Load from cache or process trajectories
        cache_loaded = False
        if use_cache and self.cache_dir:
            cache_loaded = self._load_from_cache()

        if not cache_loaded:
            # Check if lag time is compatible with trajectory timestep and stride
            self._infer_timestep()

            # Process trajectories and create graphs
            self._process_trajectories()

            # Create trainable node embeddings
            self.node_embeddings = torch.nn.Parameter(
                torch.zeros(self.n_atoms, self.node_embedding_dim)
            )

            # Initialize using position encoding (better than random for large graphs)
            self._initialize_node_embeddings()

            # Determine min and max distances for Gaussian expansion
            if distance_min is None or distance_max is None:
                self._determine_distance_range()
            else:
                self.distance_min = distance_min
                self.distance_max = distance_max

            # Create time-lagged pairs for VAMPNet
            self._create_time_lagged_pairs()

            # Save processed data to cache
            if self.cache_dir:
                self._save_to_cache()

    def _process_trajectories(self):
        """Process trajectory files and select atoms with tqdm progress.

        Also tracks trajectory boundaries for non-continuous trajectory handling.
        """
        print(f"Processing {len(self.trajectory_files)} trajectory files...")

        self.frames = []
        self.atom_indices = None
        self.trajectory_boundaries = [0]  # Start indices of each trajectory

        for traj_file in tqdm(self.trajectory_files, desc="Trajectory files", unit="file"):
            try:
                traj = md.load(traj_file, top=self.topology_file)

                if self.atom_indices is None:
                    self.atom_indices = traj.topology.select(self.selection)
                    if len(self.atom_indices) == 0:
                        raise ValueError(f"Selection '{self.selection}' returned no atoms")
                    print(f"Selected {len(self.atom_indices)} atoms with selection: '{self.selection}'")

                traj = traj[::self.stride]
                coords = traj.xyz[:, self.atom_indices, :]
                self.frames.extend(coords)

                # Track where this trajectory ends (= where next one starts)
                self.trajectory_boundaries.append(len(self.frames))

            except Exception as e:
                print(f"Error processing {traj_file}: {str(e)}")
                continue

        if not self.frames:
            raise ValueError("No frames were loaded from trajectories")

        self.frames = np.array(self.frames)
        self.n_frames = len(self.frames)
        self.n_atoms = len(self.atom_indices)

        # Report trajectory info
        n_trajectories = len(self.trajectory_boundaries) - 1
        if n_trajectories > 1:
            frames_per_traj = [self.trajectory_boundaries[i+1] - self.trajectory_boundaries[i]
                              for i in range(n_trajectories)]
            print(f"Loaded {n_trajectories} trajectories with frames: {frames_per_traj}")
            if not self.continuous:
                print(f"Non-continuous mode: pairs will not cross trajectory boundaries")

        print(f"Total frames: {self.n_frames}, Atoms per frame: {self.n_atoms}")

    def _determine_distance_range(self):
        """Determine minimum and maximum distances from data samples."""
        print("Calculating distance range from data samples...")

        sample_size = self.n_frames
        indices = np.random.choice(self.n_frames, sample_size, replace=False)

        min_distances = []
        max_distances = []

        for idx in tqdm(indices, desc="Computing distance range", unit="frame"):
            coords = self.frames[idx]
            distances = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
            np.fill_diagonal(distances, -1)
            valid_distances = distances[distances > 0]

            if len(valid_distances) > 0:
                min_distances.append(np.min(valid_distances))
                max_distances.append(np.max(valid_distances))

        if not min_distances or not max_distances:
            raise ValueError("No valid distances found across samples")

        self.distance_min = float(np.min(min_distances))
        self.distance_max = float(np.max(max_distances))

        print(f"Distance range: {self.distance_min:.2f} to {self.distance_max:.2f} Å")

    def _create_time_lagged_pairs(self):
        """Create time-lagged pairs of frame indices.

        When continuous=True (default), pairs can span across trajectory boundaries.
        When continuous=False, pairs that would cross trajectory boundaries are excluded.
        """
        if self.continuous:
            # Original behavior: treat all frames as one continuous trajectory
            self.t0_indices = list(range(self.n_frames - self.lag_frames))
            self.t1_indices = list(range(self.lag_frames, self.n_frames))
        else:
            # Non-continuous: only create pairs within the same trajectory
            self.t0_indices = []
            self.t1_indices = []

            n_trajectories = len(self.trajectory_boundaries) - 1
            for traj_idx in range(n_trajectories):
                traj_start = self.trajectory_boundaries[traj_idx]
                traj_end = self.trajectory_boundaries[traj_idx + 1]
                traj_length = traj_end - traj_start

                # Only create pairs if trajectory is long enough for the lag
                if traj_length > self.lag_frames:
                    # Valid t0 indices: from traj_start to (traj_end - lag_frames - 1)
                    for t0 in range(traj_start, traj_end - self.lag_frames):
                        t1 = t0 + self.lag_frames
                        self.t0_indices.append(t0)
                        self.t1_indices.append(t1)
                else:
                    print(f"Warning: Trajectory {traj_idx} has {traj_length} frames, "
                          f"which is less than lag_frames={self.lag_frames}. Skipping.")

        n_pairs = len(self.t0_indices)
        print(f"Created {n_pairs} time-lagged pairs with lag time {self.lag_time} and "
              f"{self.lag_frames} lag frames. 1 lag frame == {self.lag_time / self.lag_frames} ns")

        if not self.continuous:
            # Report how many pairs were excluded
            max_possible = self.n_frames - self.lag_frames
            excluded = max_possible - n_pairs
            if excluded > 0:
                print(f"Non-continuous mode excluded {excluded} cross-boundary pairs")

    def _compute_gaussian_expanded_distances(self, distances):
        """
        Compute Gaussian expanded distances.

        Formula: e_t(i,j) = exp(-(d_ij - μ_t)²/σ²)
        where μ_t = dmin + t * (dmax - dmin)/K and σ = (dmax - dmin)/K
        """
        K = self.gaussian_expansion_dim
        d_range = self.distance_max - self.distance_min
        sigma = d_range / K

        valid_mask = distances >= 0
        distances_reshaped = distances.reshape(-1, 1)
        mu_values = torch.linspace(self.distance_min, self.distance_max, K).view(1, -1)

        expanded_features = torch.zeros((distances_reshaped.shape[0], K),
                                        device=distances.device,
                                        dtype=torch.float32)

        valid_indices = torch.nonzero(valid_mask).squeeze()
        valid_distances = distances_reshaped[valid_indices]
        valid_expanded = torch.exp(-((valid_distances - mu_values) ** 2) / (sigma ** 2))
        expanded_features[valid_indices] = valid_expanded

        return expanded_features

    def _initialize_node_embeddings(self):
        """Initialize node embeddings with position encoding for better gradient flow."""
        import math

        embeddings = torch.zeros(self.n_atoms, self.node_embedding_dim)
        position = torch.arange(0, self.n_atoms).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.node_embedding_dim, 2) *
                             -(math.log(10000.0) / self.node_embedding_dim))

        embeddings[:, 0::2] = torch.sin(position * div_term)

        if self.node_embedding_dim > 1:
            if div_term.size(0) < self.node_embedding_dim // 2 + self.node_embedding_dim % 2:
                remaining = self.node_embedding_dim // 2 + self.node_embedding_dim % 2 - div_term.size(0)
                extension = torch.exp(torch.arange(div_term.size(0), div_term.size(0) + remaining) *
                                      -(math.log(10000.0) / self.node_embedding_dim))
                div_term = torch.cat([div_term, extension])

            embeddings[:, 1::2] = torch.cos(position * div_term[:self.node_embedding_dim // 2])

        self.node_embeddings = embeddings

    def _create_node_features(self, use_amino_acid_encoding: bool = None):
        """
        Create node features based on encoding type.

        Parameters
        ----------
        use_amino_acid_encoding : bool, optional
            Override the instance setting. If None, uses self.use_amino_acid_encoding.

        Returns
        -------
        torch.Tensor
            Node feature tensor
        """
        if use_amino_acid_encoding is None:
            use_amino_acid_encoding = self.use_amino_acid_encoding

        if use_amino_acid_encoding:
            # Amino acid-based node features
            if self.amino_acid_feature_type == "properties":
                # 4D property vector: [hydrophobic, polar, charged, aromatic]
                node_attr = torch.zeros(self.n_atoms, 4)
                for i, atom_idx in enumerate(self.atom_indices):
                    residue = self.topology.atom(atom_idx).residue
                    properties = get_amino_acid_features(residue.name)
                    node_attr[i] = torch.tensor(properties, dtype=torch.float32)
            else:
                # Integer labels (0-20)
                node_attr = torch.zeros(self.n_atoms, 1)
                for i, atom_idx in enumerate(self.atom_indices):
                    residue = self.topology.atom(atom_idx).residue
                    label = get_amino_acid_labels(residue.name)
                    node_attr[i] = torch.tensor(label, dtype=torch.float32)
        else:
            # One-hot encoding (n_atoms × n_atoms identity matrix)
            node_attr = torch.zeros(self.n_atoms, self.n_atoms)
            for i in range(self.n_atoms):
                node_attr[i, i] = 1.0

        return node_attr

    def _create_graph_from_frame(self, frame_idx: int, use_amino_acid_encoding: bool = None):
        """
        Create a graph representation for a single frame.

        Parameters
        ----------
        frame_idx : int
            Index of the frame to process
        use_amino_acid_encoding : bool, optional
            Override the instance setting for amino acid encoding.
            If None, uses self.use_amino_acid_encoding.

        Returns
        -------
        torch_geometric.data.Data
            Graph representation of the frame
        """
        if use_amino_acid_encoding is None:
            use_amino_acid_encoding = self.use_amino_acid_encoding

        # Get coordinates for the frame
        coords = torch.tensor(self.frames[frame_idx], dtype=torch.float32)

        # Calculate pairwise distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=2))

        # Create mask to identify self-connections
        diag_mask = torch.eye(self.n_atoms, dtype=torch.bool, device=distances.device)
        distances[diag_mask] = -1.0
        valid_mask = ~diag_mask

        # Find k-nearest neighbors for each node
        nn_indices = []
        for i in range(self.n_atoms):
            node_distances = distances[i]
            valid_distances = node_distances[valid_mask[i]]
            valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
            _, top_k_indices = torch.topk(valid_distances, min(self.n_neighbors, len(valid_distances)), largest=False)
            node_nn_indices = valid_indices[top_k_indices]
            nn_indices.append(node_nn_indices)

        nn_indices = torch.stack(nn_indices)

        # Collect directional edges (asymmetric k-NN)
        edge_set = set()
        for i in range(self.n_atoms):
            for j in nn_indices[i]:
                edge_set.add((i, j.item()))

        # Create edge list (flipped for message passing direction)
        directional_edges = [(target, source) for source, target in edge_set]

        source_indices = torch.tensor([edge[0] for edge in directional_edges], device=distances.device)
        target_indices = torch.tensor([edge[1] for edge in directional_edges], device=distances.device)

        # Create edge_index tensor
        edge_index = torch.stack([source_indices, target_indices], dim=0)

        # Get edge distances and compute Gaussian expansion
        edge_distances = distances[source_indices, target_indices]
        edge_attr = self._compute_gaussian_expanded_distances(edge_distances)

        # Create node features
        node_attr = self._create_node_features(use_amino_acid_encoding)

        # Create PyG Data object
        graph = Data(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.n_atoms
        )

        return graph

    def __len__(self):
        """Return the number of time-lagged pairs."""
        return len(self.t0_indices)

    def __getitem__(self, idx):
        """Get a time-lagged pair of graphs for the given index."""
        t0_idx = self.t0_indices[idx]
        t1_idx = self.t1_indices[idx]

        graph_t0 = self._create_graph_from_frame(t0_idx)
        graph_t1 = self._create_graph_from_frame(t1_idx)

        return graph_t0, graph_t1

    def get_graph(self, idx: int, use_amino_acid_encoding: bool = None):
        """
        Get graph for a specific frame (without time-lagging).

        Parameters
        ----------
        idx : int
            Frame index
        use_amino_acid_encoding : bool, optional
            Override encoding setting for this call

        Returns
        -------
        torch_geometric.data.Data
            Graph for the specified frame
        """
        return self._create_graph_from_frame(idx, use_amino_acid_encoding)

    def _get_cache_filename(self):
        """Generate a unique cache filename based on dataset parameters."""
        import hashlib
        traj_hash = hashlib.md5(str(sorted(self.trajectory_files)).encode()).hexdigest()[:8]
        cont_flag = "cont" if self.continuous else "noncont"
        cache_name = f"vampnet_data_{traj_hash}_lag{self.lag_time}_nn{self.n_neighbors}_str{self.stride}_{cont_flag}.pkl"
        return os.path.join(self.cache_dir, cache_name)

    def _save_to_cache(self):
        """Save processed data to cache file."""
        if not self.cache_dir:
            print("No cache directory specified. Skipping cache save.")
            return False

        cache_file = self._get_cache_filename()
        print(f"Saving dataset to cache: {cache_file}")

        data = {
            'frames': self.frames,
            'atom_indices': self.atom_indices,
            'distance_min': self.distance_min,
            'distance_max': self.distance_max,
            't0_indices': self.t0_indices,
            't1_indices': self.t1_indices,
            'n_frames': self.n_frames,
            'n_atoms': self.n_atoms,
            'trajectory_boundaries': self.trajectory_boundaries,
            'config': {
                'lag_time': self.lag_time,
                'n_neighbors': self.n_neighbors,
                'node_embedding_dim': self.node_embedding_dim,
                'gaussian_expansion_dim': self.gaussian_expansion_dim,
                'selection': self.selection,
                'stride': self.stride,
                'trajectory_files': self.trajectory_files,
                'topology_file': self.topology_file,
                'use_amino_acid_encoding': self.use_amino_acid_encoding,
                'amino_acid_feature_type': self.amino_acid_feature_type,
                'continuous': self.continuous,
            }
        }

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Dataset successfully cached to {cache_file}")
            return True
        except Exception as e:
            print(f"Error saving dataset to cache: {str(e)}")
            return False

    def _load_from_cache(self):
        """Load processed data from cache if available."""
        if not self.cache_dir:
            return False

        cache_file = self._get_cache_filename()

        if not os.path.exists(cache_file):
            print(f"No cache file found at {cache_file}")
            return False

        print(f"Loading dataset from cache: {cache_file}")

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            self.frames = data['frames']
            self.atom_indices = data['atom_indices']
            self.distance_min = data['distance_min']
            self.distance_max = data['distance_max']
            self.t0_indices = data['t0_indices']
            self.t1_indices = data['t1_indices']
            self.n_frames = data['n_frames']
            self.n_atoms = data['n_atoms']
            self.trajectory_boundaries = data.get('trajectory_boundaries', [0, self.n_frames])

            cached_config = data['config']
            if (cached_config['lag_time'] != self.lag_time or
                    cached_config['n_neighbors'] != self.n_neighbors or
                    cached_config['gaussian_expansion_dim'] != self.gaussian_expansion_dim or
                    cached_config['selection'] != self.selection or
                    cached_config['stride'] != self.stride or
                    cached_config.get('continuous', True) != self.continuous):
                print("Warning: Current configuration doesn't match cached configuration.")
                print("Using cached data anyway, but consider regenerating if needed.")

            print(f"Successfully loaded dataset from cache: {self.n_frames} frames, {self.n_atoms} atoms")
            return True
        except Exception as e:
            print(f"Error loading dataset from cache: {str(e)}")
            return False

    def _infer_timestep(self) -> float:
        """
        Infer the trajectory timestep in picoseconds and check lag time compatibility.

        Returns
        -------
        float
            Timestep in picoseconds

        Raises
        ------
        ValueError
            If timestep can't be determined or lag time is incompatible
        """
        if not self.trajectory_files:
            raise ValueError("No trajectory files provided")

        try:
            traj_iterator = md.iterload(self.trajectory_files[0], top=self.topology_file, chunk=2)
            first_chunk = next(traj_iterator)

            if len(first_chunk.time) < 2:
                raise ValueError("Trajectory must have at least 2 frames to infer timestep")

            timestep = first_chunk.time[1] - first_chunk.time[0]
        except Exception as e:
            print(f"Warning: Could not determine timestep from trajectory time data: {str(e)}")
            timestep = None

        lag_time_ps = self.lag_time * 1000.0
        effective_timestep = timestep * self.stride

        if lag_time_ps % effective_timestep != 0:
            closest_achievable = round(lag_time_ps / effective_timestep) * effective_timestep
            raise ValueError(
                f"Requested lag time of {self.lag_time} ns ({lag_time_ps} ps) cannot be achieved "
                f"with timestep of {timestep} ps and stride of {self.stride}. "
                f"The effective timestep is {effective_timestep} ps. "
                f"Consider using a lag time of {closest_achievable / 1000.0:.3f} ns instead, "
                f"or adjust the stride to {int(lag_time_ps / timestep)}."
            )

        self.lag_frames = int(lag_time_ps / effective_timestep)

        print(f"Trajectory timestep: {timestep:.3f} ps")
        print(f"Effective timestep with stride {self.stride}: {effective_timestep:.3f} ps")
        print(f"Lag time: {self.lag_time:.3f} ns ({lag_time_ps} ps, {self.lag_frames} frames)")

        return timestep

    @classmethod
    def from_cache(cls, cache_file: str, node_embedding_dim: int = 16):
        """
        Create a dataset directly from a cache file.

        Parameters
        ----------
        cache_file : str
            Path to the cache file
        node_embedding_dim : int, default=16
            Node embedding dimension

        Returns
        -------
        VAMPNetDataset
            Dataset loaded from cache
        """
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        print(f"Creating dataset from cache file: {cache_file}")

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            instance = cls.__new__(cls)
            super(VAMPNetDataset, instance).__init__()

            instance.frames = data['frames']
            instance.atom_indices = data['atom_indices']
            instance.distance_min = data['distance_min']
            instance.distance_max = data['distance_max']
            instance.t0_indices = data['t0_indices']
            instance.t1_indices = data['t1_indices']
            instance.n_frames = data['n_frames']
            instance.n_atoms = data['n_atoms']
            instance.trajectory_boundaries = data.get('trajectory_boundaries', [0, data['n_frames']])

            config = data['config']
            instance.lag_time = config['lag_time']
            instance.n_neighbors = config['n_neighbors']
            instance.gaussian_expansion_dim = config['gaussian_expansion_dim']
            instance.selection = config['selection']
            instance.stride = config['stride']
            instance.trajectory_files = config['trajectory_files']
            instance.topology_file = config['topology_file']
            instance.use_amino_acid_encoding = config.get('use_amino_acid_encoding', False)
            instance.amino_acid_feature_type = config.get('amino_acid_feature_type', 'labels')
            instance.continuous = config.get('continuous', True)

            instance.node_embedding_dim = node_embedding_dim
            instance.cache_dir = os.path.dirname(cache_file)

            # Load topology for amino acid encoding
            instance.topology = md.load_topology(instance.topology_file)

            print(f"Successfully created dataset from cache: {instance.n_frames} frames, {instance.n_atoms} atoms")
            return instance
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from cache: {str(e)}")

    def precompute_graphs(self, max_graphs: int = None):
        """
        Precompute all graphs to speed up data loading.

        Parameters
        ----------
        max_graphs : int, optional
            Maximum number of graphs to precompute (None for all)
        """
        print("Precomputing graphs...")

        num_graphs = self.n_frames
        if max_graphs is not None and max_graphs < num_graphs:
            num_graphs = max_graphs
            print(f"Precomputing {num_graphs} graphs (limited by max_graphs)")
        else:
            print(f"Precomputing all {num_graphs} graphs")

        self.graphs = []

        for idx in tqdm(range(num_graphs), desc="Creating graphs", unit="graph"):
            graph = self._create_graph_from_frame(idx)
            self.graphs.append(graph)

        print(f"Precomputed {len(self.graphs)} graphs")

        self._original_getitem = self.__getitem__

        def new_getitem(idx):
            t0_idx = self.t0_indices[idx]
            t1_idx = self.t1_indices[idx]

            if t0_idx < len(self.graphs) and t1_idx < len(self.graphs):
                return self.graphs[t0_idx], self.graphs[t1_idx]
            else:
                return self._create_graph_from_frame(t0_idx), self._create_graph_from_frame(t1_idx)

        self.__getitem__ = new_getitem.__get__(self)

    def get_frames_dataset(self, return_pairs: bool = False):
        """
        Create a dataset that returns individual frames instead of time-lagged pairs.

        Uses the instance's use_amino_acid_encoding setting.

        Parameters
        ----------
        return_pairs : bool, default=False
            If True, return time-lagged pairs
            If False, return individual frames

        Returns
        -------
        VAMPNetFramesDataset
            Dataset returning individual frames
        """
        parent = self

        class VAMPNetFramesDataset(torch.utils.data.Dataset):
            def __init__(self, parent_dataset, return_pairs=False):
                self.parent = parent_dataset
                self.return_pairs = return_pairs
                self.n_samples = len(parent_dataset.t0_indices) if return_pairs else parent_dataset.n_frames

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                if self.return_pairs:
                    return self.parent.__getitem__(idx)
                else:
                    return self.parent._create_graph_from_frame(idx)

        return VAMPNetFramesDataset(parent, return_pairs=return_pairs)

    def get_frames_dataset_with_encoding(self, return_pairs: bool = False, use_amino_acid_encoding: bool = False):
        """
        Create a dataset that returns individual frames with explicit encoding control.

        Parameters
        ----------
        return_pairs : bool, default=False
            If True, return time-lagged pairs
            If False, return individual frames
        use_amino_acid_encoding : bool, default=False
            Override the encoding type for this dataset

        Returns
        -------
        VAMPNetFramesDataset
            Dataset returning individual frames with specified encoding
        """
        parent = self

        class VAMPNetFramesDataset(torch.utils.data.Dataset):
            def __init__(self, parent_dataset, return_pairs, use_aa_encoding):
                self.parent = parent_dataset
                self.return_pairs = return_pairs
                self.use_aa_encoding = use_aa_encoding
                self.n_samples = len(parent_dataset.t0_indices) if return_pairs else parent_dataset.n_frames

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                if self.return_pairs:
                    t0_idx = self.parent.t0_indices[idx]
                    t1_idx = self.parent.t1_indices[idx]
                    graph_t0 = self.parent._create_graph_from_frame(t0_idx, use_amino_acid_encoding=self.use_aa_encoding)
                    graph_t1 = self.parent._create_graph_from_frame(t1_idx, use_amino_acid_encoding=self.use_aa_encoding)
                    return graph_t0, graph_t1
                else:
                    return self.parent._create_graph_from_frame(idx, use_amino_acid_encoding=self.use_aa_encoding)

        return VAMPNetFramesDataset(parent, return_pairs, use_amino_acid_encoding)

    # Backward compatibility alias
    def get_AA_frames(self, return_pairs: bool = False):
        """
        Create a dataset with amino acid encoding (backward compatibility).

        This is equivalent to get_frames_dataset_with_encoding(return_pairs, use_amino_acid_encoding=True).

        Parameters
        ----------
        return_pairs : bool, default=False
            If True, return time-lagged pairs

        Returns
        -------
        VAMPNetFramesDataset
            Dataset with amino acid encoded node features
        """
        return self.get_frames_dataset_with_encoding(return_pairs=return_pairs, use_amino_acid_encoding=True)
