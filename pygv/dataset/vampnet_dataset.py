import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import mdtraj as md
from typing import List, Tuple, Optional, Union
import pickle
from tqdm import tqdm


class VAMPNetDataset(Dataset):
    def __init__(
            self,
            trajectory_files: List[str],
            topology_file: str,
            lag_time: float = 1,
            n_neighbors: int = 10,  # M nearest neighbors
            node_embedding_dim: int = 16,
            gaussian_expansion_dim: int = 16,  # K in the paper
            distance_min: Optional[float] = None,
            distance_max: Optional[float] = None,
            selection: str = "name CA",  # Select C-alpha atoms by default
            seed: int = 42,
            stride: int = 1,  # Take every 'stride' frame
            chunk_size: int = 1000,  # Process trajectories in chunks to save memory
            cache_dir: Optional[str] = None,  # Directory to save/load cached data
            use_cache: bool = True,  # Whether to use cached data if available
    ):
        """
        Create a VAMPNet dataset from MD trajectories, representing each structure as a graph.

        Args:
            trajectory_files: List of MD trajectory files
            topology_file: Topology file for the trajectories
            lag_time: Time lag between pairs (in ns)
            n_neighbors: Number of nearest neighbors for graph construction (M in the paper)
            node_embedding_dim: Dimension of random node features
            gaussian_expansion_dim: Dimension of Gaussian expansion (K in the paper)
            distance_min: Minimum distance for Gaussian expansion (if None, will be determined from data)
            distance_max: Maximum distance for Gaussian expansion (if None, will be determined from data)
            selection: MDTraj selection string for atoms to include
            seed: Random seed for reproducibility
            stride: Process every 'stride' frames from trajectories
            chunk_size: Number of frames to process at once (to avoid memory issues)
            cache_dir: Directory to save/load cached data
            use_cache: Whether to use cached data if available
        """
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

        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load from cache or process trajectories
        cache_loaded = False
        if use_cache and self.cache_dir:
            cache_loaded = self._load_from_cache()

        if not cache_loaded:
            # Check if lag time is compatible with trajectory timestep and stride
            self._infer_timestep()

            # Process trajectories and create graphs
            self._process_trajectories()

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
        """Process trajectory files and select atoms with tqdm progress"""
        print(f"Processing {len(self.trajectory_files)} trajectory files...")

        # Initialize an empty list for storing selected frames
        self.frames = []
        self.atom_indices = None

        # Process each trajectory file with progress bar
        for traj_file in tqdm(self.trajectory_files, desc="Trajectory files", unit="file"):
            try:
                # Load the entire trajectory first
                traj = md.load(traj_file, top=self.topology_file)

                # Select atoms (only need to do this once)
                if self.atom_indices is None:
                    self.atom_indices = traj.topology.select(self.selection)
                    if len(self.atom_indices) == 0:
                        raise ValueError(f"Selection '{self.selection}' returned no atoms")
                    print(f"Selected {len(self.atom_indices)} atoms with selection: '{self.selection}'")

                # Apply stride to frames
                traj = traj[::self.stride]

                # Extract coordinates for selected atoms with progress bar
                coords = traj.xyz[:, self.atom_indices, :]
                self.frames.extend(coords)

                #print(f"Processed {len(traj)} frames from {traj_file}")

            except Exception as e:
                print(f"Error processing {traj_file}: {str(e)}")
                continue

        if not self.frames:
            raise ValueError("No frames were loaded from trajectories")

        self.frames = np.array(self.frames)
        self.n_frames = len(self.frames)
        self.n_atoms = len(self.atom_indices)

        print(f"Total frames: {self.n_frames}, Atoms per frame: {self.n_atoms}")

    def _determine_distance_range(self):
        """Determine minimum and maximum distances from data samples with progress bar"""
        print("Calculating distance range from data samples...")

        # Take a reasonable subset of frames to calculate distances
        sample_size = self.n_frames #min(100, self.n_frames)
        indices = np.random.choice(self.n_frames, sample_size, replace=False)

        min_distances = []
        max_distances = []

        # Use tqdm progress bar
        # for idx in tqdm(indices, desc="Computing distance range", unit="frame"):
        for idx in indices:
            # Calculate pairwise distances for selected atoms in this frame
            coords = self.frames[idx]

            # Compute all pairwise distances
            distances = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))

            # Exclude self-distances (diagonal)
            np.fill_diagonal(distances, -1)

            # Convert to mask to filter out -1 values
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
        """Create time-lagged pairs of frame indices"""
        self.t0_indices = list(range(self.n_frames - self.lag_frames))
        self.t1_indices = list(range(self.lag_frames, self.n_frames))

        print(f"Created {len(self.t0_indices)} time-lagged pairs with lag time {self.lag_time} and "
              f"{self.lag_frames} lag frames. 1 lag frame == {self.lag_time/self.lag_frames} ns")

    def _compute_gaussian_expanded_distances(self, distances):
        """
        Compute Gaussian expanded distances according to the formula:
        e_t(i,j) = exp(-(d_ij - μ_t)²/σ²)

        where:
        - d_ij is the distance between atoms i and j
        - μ_t = dmin + t * (dmax - dmin)/K
        - σ = (dmax - dmin)/K
        - t = 0, 1, ..., K-1
        - K is the Gaussian expansion dimension
        """
        K = self.gaussian_expansion_dim
        d_range = self.distance_max - self.distance_min
        sigma = d_range / K

        # Filter out invalid distances (diagonals set to -1)
        valid_mask = distances >= 0

        # Reshape to prepare for broadcasting
        distances_reshaped = distances.reshape(-1, 1)  # [num_edges, 1]

        # Calculate μ_t values [1, K]
        mu_values = torch.linspace(self.distance_min, self.distance_max, K).view(1, -1)

        # Compute expanded features: exp(-(d_ij - μ_t)²/σ²)
        expanded_features = torch.zeros((distances_reshaped.shape[0], K),
                                        device=distances.device,
                                        dtype=torch.float32)

        # Apply computation only to valid distances
        valid_indices = torch.nonzero(valid_mask).squeeze()
        valid_distances = distances_reshaped[valid_indices]

        # Compute Gaussian expansion only for valid distances
        valid_expanded = torch.exp(-((valid_distances - mu_values) ** 2) / (sigma ** 2))

        # Place results back in the output tensor
        expanded_features[valid_indices] = valid_expanded

        return expanded_features

    def _create_graph_from_frame_old(self, frame_idx):
        """
        Create a graph representation for a single frame.

        Args:
            frame_idx: Index of the frame to process

        Returns:
            torch_geometric.data.Data: Graph representation of the frame
        """
        # Get coordinates for the frame
        coords = torch.tensor(self.frames[frame_idx], dtype=torch.float32)

        # Calculate pairwise distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [n_atoms, n_atoms, 3]
        distances = torch.sqrt((diff ** 2).sum(dim=2))  # [n_atoms, n_atoms]

        # Find M nearest neighbors for each atom
        # Set self-distances to -1 to exclude them
        distances.fill_diagonal_(float(-1))

        # Get top-k smallest distances for each atom
        _, nn_indices = torch.topk(distances, self.n_neighbors, dim=1, largest=False)

        # Create edge indices
        source_indices = torch.arange(self.n_atoms).repeat_interleave(self.n_neighbors)
        target_indices = nn_indices.reshape(-1)
        edge_index = torch.stack([source_indices, target_indices], dim=0)

        # Get edge distances
        edge_distances = distances[source_indices, target_indices]

        # Compute Gaussian expanded edge features
        edge_attr = self._compute_gaussian_expanded_distances(edge_distances)

        # Generate random node features
        node_attr = torch.randn(self.n_atoms, self.node_embedding_dim)

        # Create PyG Data object
        graph = Data(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.n_atoms
        )

        return graph

    def _create_graph_from_frame(self, frame_idx):
        """
        Create a graph representation for a single frame.

        Args:
            frame_idx: Index of the frame to process

        Returns:
            torch_geometric.data.Data: Graph representation of the frame
        """
        # Get coordinates for the frame
        coords = torch.tensor(self.frames[frame_idx], dtype=torch.float32)

        # Calculate pairwise distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [n_atoms, n_atoms, 3]
        distances = torch.sqrt((diff ** 2).sum(dim=2))  # [n_atoms, n_atoms]

        # Create a mask to identify self-connections (diagonal elements)
        diag_mask = torch.eye(self.n_atoms, dtype=torch.bool, device=distances.device)

        # Set self-distances to -1 (keep your original approach)
        distances[diag_mask] = -1.0

        # Create a mask for valid distances (excluding self-connections)
        valid_mask = ~diag_mask

        # For each node, get indices of the k-nearest neighbors (excluding self)
        nn_indices = []
        for i in range(self.n_atoms):
            # Get distances from node i to all other nodes
            node_distances = distances[i]
            # Mask out the self-connection
            valid_distances = node_distances[valid_mask[i]]
            # Get indices of valid nodes (excluding self)
            valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
            # Get top-k nearest neighbors
            _, top_k_indices = torch.topk(valid_distances, min(self.n_neighbors, len(valid_distances)), largest=False)
            # Map back to original indices
            node_nn_indices = valid_indices[top_k_indices]
            # Add to list
            nn_indices.append(node_nn_indices)

        # Stack indices for all nodes
        nn_indices = torch.stack(nn_indices)

        # Create a set to track all edges for ensuring bidirectionality
        edge_set = set()

        # First, collect all the original directional edges
        for i in range(self.n_atoms):
            for j in nn_indices[i]:
                edge_set.add((i, j.item()))

        # Create a list of all bidirectional edges
        bidirectional_edges = []
        for source, target in edge_set:
            bidirectional_edges.append((source, target))
            # Add the reverse edge if it doesn't already exist
            if (target, source) not in edge_set:
                bidirectional_edges.append((target, source))

        # Convert to tensors for source and target indices
        source_indices = torch.tensor([edge[0] for edge in bidirectional_edges], device=distances.device)
        target_indices = torch.tensor([edge[1] for edge in bidirectional_edges], device=distances.device)

        # Create the edge_index tensor
        edge_index = torch.stack([source_indices, target_indices], dim=0)

        # Quick bidirectionality check
        edge_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        assert all((target, source) in edge_pairs for source, target in edge_pairs), "Graph is not bidirectional"

        # Get edge distances (non-negative real distances)
        edge_distances = distances[source_indices, target_indices]

        # Compute Gaussian expanded edge features
        edge_attr = self._compute_gaussian_expanded_distances(edge_distances)

        # Generate random node features
        #node_attr = torch.randn(self.n_atoms, self.node_embedding_dim)
        node_attr = torch.nn.Embedding(num_embeddings=self.n_atoms, embedding_dim=self.node_embedding_dim)

        # Create PyG Data object
        graph = Data(
            x=node_attr.weight.data,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.n_atoms
        )

        return graph

    def __len__(self):
        """Return the number of time-lagged pairs"""
        return len(self.t0_indices)

    def __getitem__(self, idx):
        """Get a time-lagged pair of graphs for the given index"""
        t0_idx = self.t0_indices[idx]
        t1_idx = self.t1_indices[idx]

        graph_t0 = self._create_graph_from_frame(t0_idx)
        graph_t1 = self._create_graph_from_frame(t1_idx)

        return graph_t0, graph_t1

    def get_graph(self, idx):
        """Get graph for a specific frame (without time-lagging)"""
        return self._create_graph_from_frame(idx)

    def _get_cache_filename(self):
        """Generate a unique cache filename based on dataset parameters"""
        # Create a hash from the trajectory files to ensure uniqueness
        import hashlib
        traj_hash = hashlib.md5(str(sorted(self.trajectory_files)).encode()).hexdigest()[:8]

        # Format parameters into the filename
        cache_name = f"vampnet_data_{traj_hash}_lag{self.lag_time}_nn{self.n_neighbors}_str{self.stride}.pkl"
        return os.path.join(self.cache_dir, cache_name)

    def _save_to_cache(self):
        """Save processed data to cache file"""
        if not self.cache_dir:
            print("No cache directory specified. Skipping cache save.")
            return False

        cache_file = self._get_cache_filename()
        print(f"Saving dataset to cache: {cache_file}")

        # Prepare data to save
        data = {
            'frames': self.frames,
            'atom_indices': self.atom_indices,
            'distance_min': self.distance_min,
            'distance_max': self.distance_max,
            't0_indices': self.t0_indices,
            't1_indices': self.t1_indices,
            'n_frames': self.n_frames,
            'n_atoms': self.n_atoms,
            # Save configuration parameters as well for reference
            'config': {
                'lag_time': self.lag_time,
                'n_neighbors': self.n_neighbors,
                'node_embedding_dim': self.node_embedding_dim,
                'gaussian_expansion_dim': self.gaussian_expansion_dim,
                'selection': self.selection,
                'stride': self.stride,
                'trajectory_files': self.trajectory_files,
                'topology_file': self.topology_file
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
        """Load processed data from cache if available"""
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

            # Load data
            self.frames = data['frames']
            self.atom_indices = data['atom_indices']
            self.distance_min = data['distance_min']
            self.distance_max = data['distance_max']
            self.t0_indices = data['t0_indices']
            self.t1_indices = data['t1_indices']
            self.n_frames = data['n_frames']
            self.n_atoms = data['n_atoms']

            # Verify configuration compatibility
            cached_config = data['config']
            if (cached_config['lag_time'] != self.lag_time or
                    cached_config['n_neighbors'] != self.n_neighbors or
                    cached_config['gaussian_expansion_dim'] != self.gaussian_expansion_dim or
                    cached_config['selection'] != self.selection or
                    cached_config['stride'] != self.stride):
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

        This method determines the timestep of the trajectory and verifies that
        the requested lag time (in ns) can be achieved with the current stride.

        Returns:
        -------
        float
            Timestep in picoseconds

        Raises:
        -------
        ValueError
            If the timestep can't be determined or if the lag time can't be achieved
            with current stride.
        """
        if not self.trajectory_files:
            raise ValueError("No trajectory files provided")

        # Try direct iterload approach (most reliable)
        try:
            # Load first few frames using iterload
            traj_iterator = md.iterload(self.trajectory_files[0], top=self.topology_file, chunk=2)
            first_chunk = next(traj_iterator)

            if len(first_chunk.time) < 2:
                raise ValueError("Trajectory must have at least 2 frames to infer timestep")

            # Calculate timestep from time differences
            timestep = first_chunk.time[1] - first_chunk.time[0]
        except Exception as e:
            # If time information is not available, try alternative methods
            print(f"Warning: Could not determine timestep from trajectory time data: {str(e)}")
            timestep = None

        # Convert lag_time from ns to ps for comparison
        lag_time_ps = self.lag_time * 1000.0

        # Calculate effective timestep with stride
        effective_timestep = timestep * self.stride

        # Check if lag time is achievable with the effective timestep
        if lag_time_ps % effective_timestep != 0:
            closest_achievable = round(lag_time_ps / effective_timestep) * effective_timestep
            raise ValueError(
                f"Requested lag time of {self.lag_time} ns ({lag_time_ps} ps) cannot be achieved "
                f"with timestep of {timestep} ps and stride of {self.stride}. "
                f"The effective timestep is {effective_timestep} ps. "
                f"Consider using a lag time of {closest_achievable / 1000.0:.3f} ns instead, "
                f"or adjust the stride to {int(lag_time_ps / timestep)}."
            )

        # Calculate lag time in frames
        self.lag_frames = int(lag_time_ps / effective_timestep)

        print(f"Trajectory timestep: {timestep:.3f} ps")
        print(f"Effective timestep with stride {self.stride}: {effective_timestep:.3f} ps")
        print(f"Lag time: {self.lag_time:.3f} ns ({lag_time_ps} ps, {self.lag_frames} frames)")

        return timestep

    @classmethod
    def from_cache(cls, cache_file, node_embedding_dim=16):
        """Create a dataset directly from a cache file"""
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        print(f"Creating dataset from cache file: {cache_file}")

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            # Create a new instance of the class
            instance = cls.__new__(cls)
            super(VAMPNetDataset, instance).__init__()

            # Load data from cache
            instance.frames = data['frames']
            instance.atom_indices = data['atom_indices']
            instance.distance_min = data['distance_min']
            instance.distance_max = data['distance_max']
            instance.t0_indices = data['t0_indices']
            instance.t1_indices = data['t1_indices']
            instance.n_frames = data['n_frames']
            instance.n_atoms = data['n_atoms']

            # Set configuration from cached data
            config = data['config']
            instance.lag_time = config['lag_time']
            instance.n_neighbors = config['n_neighbors']
            instance.gaussian_expansion_dim = config['gaussian_expansion_dim']
            instance.selection = config['selection']
            instance.stride = config['stride']
            instance.trajectory_files = config['trajectory_files']
            instance.topology_file = config['topology_file']

            # Set node embedding dimension (can be different from cached)
            instance.node_embedding_dim = node_embedding_dim

            # Set cache related attributes
            instance.cache_dir = os.path.dirname(cache_file)

            print(f"Successfully created dataset from cache: {instance.n_frames} frames, {instance.n_atoms} atoms")
            return instance
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from cache: {str(e)}")

    def precompute_graphs(self, max_graphs=None):
        """
        Precompute all graphs to speed up data loading.

        Args:
            max_graphs: Maximum number of graphs to precompute (None for all)
        """
        print("Precomputing graphs...")

        # Determine number of graphs to compute
        num_graphs = self.n_frames
        if max_graphs is not None and max_graphs < num_graphs:
            num_graphs = max_graphs
            print(f"Precomputing {num_graphs} graphs (limited by max_graphs)")
        else:
            print(f"Precomputing all {num_graphs} graphs")

        self.graphs = []

        # Use tqdm for progress tracking
        for idx in tqdm(range(num_graphs), desc="Creating graphs", unit="graph"):
            graph = self._create_graph_from_frame(idx)
            self.graphs.append(graph)

        print(f"Precomputed {len(self.graphs)} graphs")

        # Update getitem to use precomputed graphs if available
        self._original_getitem = self.__getitem__

        def new_getitem(idx):
            t0_idx = self.t0_indices[idx]
            t1_idx = self.t1_indices[idx]

            # Use precomputed graphs if available, otherwise compute on-the-fly
            if t0_idx < len(self.graphs) and t1_idx < len(self.graphs):
                return self.graphs[t0_idx], self.graphs[t1_idx]
            else:
                return self._create_graph_from_frame(t0_idx), self._create_graph_from_frame(t1_idx)

        self.__getitem__ = new_getitem.__get__(self)  # Bind method to instance
