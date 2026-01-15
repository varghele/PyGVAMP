"""Data processing and validation utilities for MD trajectory visualization."""

import numpy as np
import json
from typing import Optional, Dict, List, Tuple, Union
import warnings


class DataProcessor:
    """Handles data validation, normalization, and conversion for visualization."""

    @staticmethod
    def validate_embeddings(embeddings: np.ndarray, expected_dims: int = 2) -> np.ndarray:
        """
        Validate embedding array shape and values.

        Parameters
        ----------
        embeddings : np.ndarray
            Array of shape (n_frames, n_dimensions)
        expected_dims : int, default=2
            Expected number of dimensions (typically 2 for 2D embeddings)

        Returns
        -------
        np.ndarray
            Validated embeddings array

        Raises
        ------
        ValueError
            If embeddings have invalid shape or contain NaN/Inf values
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")

        if embeddings.shape[1] != expected_dims:
            raise ValueError(f"Embeddings must have {expected_dims} dimensions, "
                           f"got {embeddings.shape[1]}")

        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain NaN or Inf values")

        return embeddings.astype(np.float32)

    @staticmethod
    def validate_frame_indices(frame_indices: np.ndarray, n_embeddings: int) -> np.ndarray:
        """
        Validate frame indices array.

        Parameters
        ----------
        frame_indices : np.ndarray
            Array of frame indices
        n_embeddings : int
            Number of embedding points (must match length of frame_indices)

        Returns
        -------
        np.ndarray
            Validated frame indices

        Raises
        ------
        ValueError
            If frame indices are invalid
        """
        if not isinstance(frame_indices, np.ndarray):
            frame_indices = np.array(frame_indices)

        if frame_indices.ndim != 1:
            raise ValueError(f"Frame indices must be 1D array, got shape {frame_indices.shape}")

        if len(frame_indices) != n_embeddings:
            raise ValueError(f"Frame indices length ({len(frame_indices)}) must match "
                           f"number of embeddings ({n_embeddings})")

        if np.any(frame_indices < 0):
            raise ValueError("Frame indices cannot be negative")

        return frame_indices.astype(np.int32)

    @staticmethod
    def validate_state_assignments(states: np.ndarray, n_embeddings: int) -> np.ndarray:
        """
        Validate state assignment array.

        Parameters
        ----------
        states : np.ndarray
            Array of state assignments
        n_embeddings : int
            Number of embedding points

        Returns
        -------
        np.ndarray
            Validated state assignments

        Raises
        ------
        ValueError
            If state assignments are invalid
        """
        if not isinstance(states, np.ndarray):
            states = np.array(states)

        if states.ndim != 1:
            raise ValueError(f"State assignments must be 1D array, got shape {states.shape}")

        if len(states) != n_embeddings:
            raise ValueError(f"State assignments length ({len(states)}) must match "
                           f"number of embeddings ({n_embeddings})")

        if np.any(states < 0):
            raise ValueError("State assignments cannot be negative")

        return states.astype(np.int32)

    @staticmethod
    def validate_transition_matrix(trans_matrix: np.ndarray) -> np.ndarray:
        """
        Validate transition matrix.

        Parameters
        ----------
        trans_matrix : np.ndarray
            Transition matrix of shape (n_states, n_states)

        Returns
        -------
        np.ndarray
            Validated transition matrix

        Raises
        ------
        ValueError
            If transition matrix is invalid
        """
        if not isinstance(trans_matrix, np.ndarray):
            trans_matrix = np.array(trans_matrix)

        if trans_matrix.ndim != 2:
            raise ValueError(f"Transition matrix must be 2D array, got shape {trans_matrix.shape}")

        if trans_matrix.shape[0] != trans_matrix.shape[1]:
            raise ValueError(f"Transition matrix must be square, got shape {trans_matrix.shape}")

        if np.any(trans_matrix < 0) or np.any(trans_matrix > 1):
            warnings.warn("Transition matrix contains values outside [0, 1] range")

        # Check if rows sum to 1 (within tolerance)
        row_sums = np.sum(trans_matrix, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            warnings.warn("Transition matrix rows do not sum to 1")

        return trans_matrix.astype(np.float32)

    @staticmethod
    def validate_attention_values(attention: np.ndarray, n_embeddings: int) -> np.ndarray:
        """
        Validate attention values array.

        Parameters
        ----------
        attention : np.ndarray
            Array of shape (n_frames, n_residues)
        n_embeddings : int
            Number of embedding points

        Returns
        -------
        np.ndarray
            Validated attention values

        Raises
        ------
        ValueError
            If attention values are invalid
        """
        if not isinstance(attention, np.ndarray):
            attention = np.array(attention)

        if attention.ndim != 2:
            raise ValueError(f"Attention values must be 2D array, got shape {attention.shape}")

        if attention.shape[0] != n_embeddings:
            raise ValueError(f"Attention values first dimension ({attention.shape[0]}) "
                           f"must match number of embeddings ({n_embeddings})")

        if np.any(np.isnan(attention)) or np.any(np.isinf(attention)):
            raise ValueError("Attention values contain NaN or Inf values")

        return attention.astype(np.float32)

    @staticmethod
    def normalize_attention(attention_values: np.ndarray,
                          method: str = 'minmax') -> np.ndarray:
        """
        Normalize attention values to 0-1 range.

        Parameters
        ----------
        attention_values : np.ndarray
            Raw attention values
        method : str, default='minmax'
            Normalization method: 'minmax' or 'zscore'

        Returns
        -------
        np.ndarray
            Normalized attention values in [0, 1] range
        """
        if method == 'minmax':
            # Min-max normalization
            min_val = np.min(attention_values)
            max_val = np.max(attention_values)

            if max_val - min_val < 1e-10:
                # All values are the same
                return np.ones_like(attention_values) * 0.5

            normalized = (attention_values - min_val) / (max_val - min_val)

        elif method == 'zscore':
            # Z-score normalization then sigmoid
            mean = np.mean(attention_values)
            std = np.std(attention_values)

            if std < 1e-10:
                # All values are the same
                return np.ones_like(attention_values) * 0.5

            z_scores = (attention_values - mean) / std
            # Apply sigmoid to map to [0, 1]
            normalized = 1 / (1 + np.exp(-z_scores))

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized.astype(np.float32)

    @staticmethod
    def extract_frames_from_trajectory(
        traj_path: str,
        topology_path: str,
        frame_indices: np.ndarray
    ) -> List[str]:
        """
        Extract specific frames from trajectory and convert to PDB strings.

        Parameters
        ----------
        traj_path : str
            Path to trajectory file (e.g., .xtc, .dcd)
        topology_path : str
            Path to topology file (e.g., .pdb, .gro)
        frame_indices : np.ndarray
            Indices of frames to extract

        Returns
        -------
        List[str]
            List of PDB strings, one per frame

        Raises
        ------
        ImportError
            If MDTraj is not installed
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("MDTraj is required for trajectory processing. "
                            "Install with: pip install mdtraj")

        # Load trajectory
        traj = md.load(traj_path, top=topology_path)

        # Extract frames
        pdb_strings = []
        for idx in frame_indices:
            if idx >= traj.n_frames:
                raise ValueError(f"Frame index {idx} out of range (trajectory has {traj.n_frames} frames)")

            frame = traj[idx]
            # Convert to PDB string
            from io import StringIO
            pdb_io = StringIO()
            frame.save_pdb(pdb_io)
            pdb_strings.append(pdb_io.getvalue())

        return pdb_strings

    @staticmethod
    def load_pdb_file(pdb_path: str) -> str:
        """
        Load PDB file and return as string.

        Parameters
        ----------
        pdb_path : str
            Path to PDB file or PDB ID (e.g., '1UBQ')

        Returns
        -------
        str
            PDB file contents
        """
        # Check if it's a PDB ID (4 characters)
        if len(pdb_path) == 4 and pdb_path.isalnum():
            # Fetch from RCSB
            try:
                import urllib.request
                url = f"https://files.rcsb.org/download/{pdb_path.upper()}.pdb"
                with urllib.request.urlopen(url) as response:
                    return response.read().decode('utf-8')
            except Exception as e:
                raise ValueError(f"Failed to fetch PDB {pdb_path} from RCSB: {e}")
        else:
            # Load from file
            with open(pdb_path, 'r') as f:
                return f.read()

    @staticmethod
    def prepare_json_data(timescales_data: List[Dict]) -> str:
        """
        Convert numpy arrays to JSON-serializable format.

        Parameters
        ----------
        timescales_data : List[Dict]
            List of dictionaries containing timescale data

        Returns
        -------
        str
            JSON string with all data
        """
        def convert_to_serializable(obj):
            """Convert numpy arrays and other types to JSON-serializable format."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_data = convert_to_serializable(timescales_data)
        return json.dumps(serializable_data, indent=2)

    @staticmethod
    def compute_embedding_bounds(embeddings_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute bounds across all embeddings for consistent scaling.

        Parameters
        ----------
        embeddings_list : List[np.ndarray]
            List of embedding arrays

        Returns
        -------
        Dict[str, float]
            Dictionary with 'min_x', 'max_x', 'min_y', 'max_y'
        """
        all_embeddings = np.vstack(embeddings_list)

        return {
            'min_x': float(np.min(all_embeddings[:, 0])),
            'max_x': float(np.max(all_embeddings[:, 0])),
            'min_y': float(np.min(all_embeddings[:, 1])),
            'max_y': float(np.max(all_embeddings[:, 1]))
        }

    @staticmethod
    def get_residue_count_from_pdb(pdb_string: str) -> int:
        """
        Get number of residues from PDB string.

        Parameters
        ----------
        pdb_string : str
            PDB file contents

        Returns
        -------
        int
            Number of residues
        """
        residue_ids = set()
        for line in pdb_string.split('\n'):
            if line.startswith('ATOM'):
                # Extract residue number (columns 23-26 in PDB format)
                try:
                    res_id = int(line[22:26].strip())
                    residue_ids.add(res_id)
                except (ValueError, IndexError):
                    continue

        return len(residue_ids)
