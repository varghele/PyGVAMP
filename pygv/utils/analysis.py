import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import pickle
from glob import glob
import mdtraj as md

# Helper function to move batch to device
def to_device(batch, device):
    x_t0, x_t1 = batch
    return (x_t0.to(device), x_t1.to(device))


def extract_residue_indices_from_selection(selection_string, topology):
    """
    Extract the actual residue indices and names from an MDTraj selection string.

    Parameters
    ----------
    selection_string : str
        MDTraj selection string (e.g. "residue 126 to 146 or residue 221 to 259 and name CA")
    topology : mdtraj.Topology
        The trajectory topology containing residue information

    Returns
    -------
    tuple
        - residue_indices: List of residue indices that match the selection
        - residue_names: List of residue names with indices (e.g. "ALA126")
    """
    # Get the atom indices from the selection
    atom_indices = topology.select(selection_string)

    if len(atom_indices) == 0:
        raise ValueError(f"Selection '{selection_string}' returned no atoms")

    # Get the residue indices and names for these atoms, avoiding duplicates
    residue_indices = []
    residue_names = []
    seen_residues = set()

    for atom_idx in atom_indices:
        atom = topology.atom(atom_idx)
        residue = atom.residue

        # Skip duplicates (multiple atoms from the same residue)
        if residue.index in seen_residues:
            continue

        seen_residues.add(residue.index)

        # Get residue number (resSeq) and name
        residue_indices.append(residue.resSeq)
        residue_name = f"{residue.name}{residue.resSeq}"
        residue_names.append(residue_name)

    return residue_indices, residue_names


def analyze_vampnet_outputs(
        model,
        data_loader: DataLoader,
        save_folder: str,
        batch_size: int = 32,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        return_tensors: bool = False
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Analyze VAMPNet outputs for PyG graph data, extracting embeddings, attention, and state probabilities.

    Parameters
    ----------
    model : VAMPNet
        Trained VAMPNet model
    data_loader : DataLoader
        PyG DataLoader containing trajectory data as graphs
    save_folder : str
        Path to save the analysis results
    batch_size : int, optional
        Size of batches for processing, default=32
    device : torch.device, optional
        Device to run analysis on (cuda or cpu)
    return_tensors : bool, optional
        Whether to return torch tensors instead of numpy arrays

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list, list]
        Tuple containing (probs, embeddings, edge_attentions, edge_indices)
        - probs: State probabilities [n_frames, n_states]
        - embeddings: Node embeddings [n_frames, embedding_dim]
        - edge_attentions: List of edge attention values, one per frame
        - edge_indices: List of edge indices, one per frame
    """
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Create output directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Get total number of samples for pre-allocation
    total_samples = len(data_loader.dataset)

    # Get dimensions from a single batch
    sample_batch = next(iter(data_loader))

    # Get output dimensionality by running a sample through the model
    with torch.no_grad():
        # Move sample batch to device
        sample_batch = sample_batch.to(device)

        # Get dimensions for state probabilities and embeddings
        sample_probs, sample_embeddings = model(sample_batch,
                                                return_features=True,
                                                apply_classifier=True)

    # Create tensors to store results for this trajectory
    num_classes = sample_probs.size(1)
    embedding_dim = sample_embeddings.size(1)

    probs = torch.zeros((total_samples, num_classes), device=device)
    embeddings = torch.zeros((total_samples, embedding_dim), device=device)

    # Store edge indices and edge attentions for each frame
    edge_indices = []
    edge_attentions = []

    # Process each batch
    n_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing trajectory")):
            # Move batch to device
            batch = batch.to(device)

            # Get batch size
            if hasattr(batch, 'batch'):
                batch_size_current = batch.batch.max().item() + 1
                # Get unique batch IDs to identify each graph in the batch
                batch_ids = torch.unique(batch.batch).cpu().numpy()
            else:
                batch_size_current = 1
                batch_ids = [0]

            # Get state probabilities and embeddings
            batch_probs, batch_embeddings = model(batch,
                                                  return_features=True,
                                                  apply_classifier=True)

            # Store results
            probs[n_processed:n_processed + batch_size_current] = batch_probs
            embeddings[n_processed:n_processed + batch_size_current] = batch_embeddings

            # Get edge attention using get_attention
            _, batch_attentions = model.get_attention(batch, device=device)

            # Extract only the last layer's attention (most important)
            if batch_attentions is not None:
                if isinstance(batch_attentions, list) and len(batch_attentions) > 0:
                    last_layer_attention = batch_attentions[-1]

                    # Convert to CPU if needed
                    if not return_tensors:
                        last_layer_attention = last_layer_attention.cpu().numpy()
                else:
                    # Single attention output
                    last_layer_attention = batch_attentions
                    if not return_tensors:
                        last_layer_attention = last_layer_attention.cpu().numpy()

                # Process edge indices and attentions for each individual graph in the batch
                if hasattr(batch, 'batch'):
                    batch_tensor = batch.batch.cpu()
                    edge_index_tensor = batch.edge_index.cpu()

                    # For each graph in the batch
                    for i, batch_id in enumerate(batch_ids):
                        # Get node indices for this graph
                        node_mask = batch_tensor == batch_id
                        node_indices = torch.nonzero(node_mask).squeeze()

                        # Map global node indices to local (graph-specific) indices
                        node_mapping = torch.zeros(node_mask.size(0), dtype=torch.long)
                        node_mapping[node_indices] = torch.arange(node_indices.size(0))

                        # Find edges for this graph
                        edge_mask = torch.isin(edge_index_tensor[0], node_indices)
                        graph_edges = edge_index_tensor[:, edge_mask]

                        # Map to local node indices
                        local_edges = node_mapping[graph_edges]

                        # Add edge indices to list
                        edge_indices.append(local_edges.numpy())

                        # Extract attention values for this graph's edges
                        graph_attention = last_layer_attention[edge_mask]
                        edge_attentions.append(graph_attention)
                else:
                    # Single graph case
                    edge_indices.append(batch.edge_index.cpu().numpy())
                    edge_attentions.append(last_layer_attention)
            else:
                # No attention available
                for i in range(batch_size_current):
                    edge_indices.append(None)
                    edge_attentions.append(None)

            # Update processed count
            n_processed += batch_size_current

    # Convert results to numpy arrays if needed
    if not return_tensors:
        probs = probs.cpu().numpy()
        embeddings = embeddings.cpu().numpy()

    # Save results as NPZ files
    np.savez(os.path.join(save_folder, 'transformed_traj.npz'), probs)
    np.savez(os.path.join(save_folder, 'embeddings.npz'), embeddings)

    # Save edge attention values as NPY files
    attention_dir = os.path.join(save_folder, 'edge_attentions')
    os.makedirs(attention_dir, exist_ok=True)

    for i, att in enumerate(edge_attentions):
        if att is not None:
            np.save(os.path.join(attention_dir, f'attention_{i:05d}.npy'), att)

    # Save edge indices as NPY files
    indices_dir = os.path.join(save_folder, 'edge_indices')
    os.makedirs(indices_dir, exist_ok=True)

    for i, idx in enumerate(edge_indices):
        if idx is not None:
            np.save(os.path.join(indices_dir, f'edge_indices_{i:05d}.npy'), idx)

    print(f"Analysis complete. Results saved to {save_folder}")
    print(f"State probabilities shape: {probs.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of edge attention arrays: {len(edge_attentions)}")
    print(f"Number of edge index arrays: {len(edge_indices)}")
    print(f"Number of samples processed: {n_processed}")

    # Save metadata about the arrays for easier loading
    with open(os.path.join(save_folder, 'metadata.txt'), 'w') as f:
        f.write(f"Number of frames: {n_processed}\n")
        f.write(f"Number of states: {num_classes}\n")
        f.write(f"Embedding dimension: {embedding_dim}\n")
        f.write(f"Number of attention arrays: {len(edge_attentions)}\n")
        f.write(f"Number of edge index arrays: {len(edge_indices)}\n")

    return probs, embeddings, edge_attentions, edge_indices


def calculate_transition_matrices(probs: np.ndarray, lag_time: float = 1, stride: int = 1, timestep: float = 0.001) -> \
tuple[np.ndarray, np.ndarray]:
    """
    Calculate transition probability matrices from state probabilities.

    Parameters
    ----------
    probs : np.ndarray
        State probability trajectory with shape [n_frames, n_states]
    lag_time : float, optional
        Lag time for transition matrix calculation in ns
    stride : int, optional
        Stride used when extracting frames from trajectory
    timestep : float, optional
        Trajectory timestep in ns

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing (transition_matrix, transition_matrix_without_self_transitions)
    """
    # Extract dimensions
    n_frames = probs.shape[0]
    n_states = probs.shape[1]

    # Calculate lag in frames
    effective_timestep = timestep * stride  # Time between consecutive frames in the dataset
    lag_frames = int(round(lag_time / effective_timestep))

    print(f"Calculating transition matrix with lag time {lag_time} ns")
    print(f"Effective timestep: {effective_timestep} ns")
    print(f"Lag frames: {lag_frames}")

    # Initialize transition count matrix
    transition_counts = np.zeros((n_states, n_states))

    # Skip if trajectory is too short for the lag time
    if n_frames <= lag_frames:
        print(f"Warning: Trajectory with {n_frames} frames is too short for lag time {lag_time} ns "
              f"({lag_frames} frames). No transitions counted.")
        # Return empty matrices
        return np.eye(n_states), np.zeros((n_states, n_states))

    # Hard-assign states based on maximum probability
    state_assignments = np.argmax(probs, axis=1)

    # Count transitions with the specified lag time
    for t in range(n_frames - lag_frames):
        from_state = state_assignments[t]
        to_state = state_assignments[t + lag_frames]
        transition_counts[from_state, to_state] += 1

    # Calculate transition probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)

    # Avoid division by zero
    row_sums[row_sums == 0] = 1

    # Normalize to get transition probabilities
    transition_matrix = transition_counts / row_sums

    # Create a copy for non-self transitions
    transition_matrix_no_self = transition_matrix.copy()

    # Set diagonal to zero for non-self transition matrix
    np.fill_diagonal(transition_matrix_no_self, 0)

    # Re-normalize non-self transition matrix
    row_sums_no_self = transition_matrix_no_self.sum(axis=1, keepdims=True)
    row_sums_no_self[row_sums_no_self == 0] = 1  # Avoid division by zero
    transition_matrix_no_self = transition_matrix_no_self / row_sums_no_self

    print(f"Transition counts:\n{transition_counts}")
    print(f"Transition matrix shape: {transition_matrix.shape}")

    return transition_matrix, transition_matrix_no_self


def calculate_state_edge_attention_maps(
        edge_attentions: list,
        edge_indices: list,
        probs: np.ndarray,
        save_dir: str = None,
        protein_name: str = "protein"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate attention maps for each state from edge attention values.

    Parameters
    ----------
    edge_attentions : list
        List of edge attention values for each frame [n_frames][n_edges]
    edge_indices : list
        List of edge indices for each frame [n_frames][2, n_edges]
    probs : np.ndarray
        State probability trajectory with shape [n_frames, n_states]
    save_dir : str, optional
        Directory to save attention maps and state populations
    protein_name : str, optional
        Name of the protein for file naming

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        state_attention_maps: Average attention maps for each state [n_states, n_atoms, n_atoms]
        state_populations: Population of each state [n_states]
    """
    # Infer the number of states from probabilities
    num_classes = probs.shape[1]

    # Infer the number of atoms from edge indices
    num_atoms = 0
    for edges in edge_indices:
        if edges is not None:
            max_idx = np.max(edges)
            num_atoms = max(num_atoms, max_idx + 1)
            break

    if num_atoms == 0:
        raise ValueError("Could not determine the number of atoms from edge indices")

    # Determine state assignments from probabilities
    state_assignments = np.argmax(probs, axis=1)

    # Calculate state populations
    unique, counts = np.unique(state_assignments, return_counts=True)
    state_populations = np.zeros(num_classes)
    state_populations[unique] = counts
    state_populations = state_populations / np.sum(state_populations)

    # Initialize state attention maps with zeros
    state_attention_maps = np.zeros((num_classes, num_atoms, num_atoms))

    # Initialize counters to track how many times each edge appears in each state
    edge_counts = np.zeros((num_classes, num_atoms, num_atoms))

    # Process each frame with progress bar
    print(f"Calculating state attention maps for {num_classes} states...")
    for frame_idx in tqdm(range(min(len(edge_attentions), len(edge_indices), len(state_assignments)))):
        # Skip frames with missing data
        if frame_idx >= len(edge_attentions) or edge_attentions[frame_idx] is None or \
                frame_idx >= len(edge_indices) or edge_indices[frame_idx] is None:
            continue

        # Get frame data
        attention = edge_attentions[frame_idx]
        edges = edge_indices[frame_idx]
        state = state_assignments[frame_idx]

        # Check dimensions match
        if len(attention) != edges.shape[1]:
            continue

        # Process each edge in the frame
        for i in range(edges.shape[1]):
            source = int(edges[0, i])
            target = int(edges[1, i])

            # Ensure indices are within bounds
            if 0 <= source < num_atoms and 0 <= target < num_atoms:
                # Add attention value to the corresponding state map
                state_attention_maps[state, source, target] += attention[i]
                # Increment edge count
                edge_counts[state, source, target] += 1

    # Average the attention values by dividing by the counts (avoiding division by zero)
    mask = edge_counts > 0
    state_attention_maps[mask] /= edge_counts[mask]

    # Print state populations
    print(f"\nState populations:")
    for i in range(num_classes):
        print(f"State {i + 1}: {state_populations[i]:.2%}")

    # Save results if save_dir is provided
    if save_dir:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save state attention maps
        attention_map_path = os.path.join(save_dir, f"{protein_name}_state_attention_maps.npy")
        np.save(attention_map_path, state_attention_maps)
        print(f"Saved state attention maps to: {attention_map_path}")

        # Save state populations
        populations_path = os.path.join(save_dir, f"{protein_name}_state_populations.npy")
        np.save(populations_path, state_populations)
        print(f"Saved state populations to: {populations_path}")

        # Save state counts as text file for easy reference
        counts_path = os.path.join(save_dir, f"{protein_name}_state_counts.txt")
        with open(counts_path, "w") as f:
            f.write("State\tCount\tPopulation\n")
            for i in range(num_classes):
                count = int(state_populations[i] * len(state_assignments))
                f.write(f"{i + 1}\t{count}\t{state_populations[i]:.6f}\n")
        print(f"Saved state counts to: {counts_path}")

    return state_attention_maps, state_populations


def generate_state_structures(
        traj_folder: str,
        topology_file: str,
        probs: np.ndarray,
        save_dir: str,
        protein_name: str,
        stride: int = 10,
        n_structures: int = 10,
        prob_threshold: float = 0.7
) -> dict:
    """
    Generate multiple representative PDB structures for each conformational state.

    Parameters
    ----------
    traj_folder : str
        Path to the folder containing trajectory files
    topology_file : str
        Path to the topology file
    probs : np.ndarray
        State probabilities for each frame with shape [n_frames, n_states]
    save_dir : str
        Directory to save the output PDB files
    protein_name : str
        Name of the protein for file naming
    stride : int, optional
        Load every nth frame to reduce memory usage (default: 10)
    n_structures : int, optional
        Number of structures to generate per state (default: 10)
    prob_threshold : float, optional
        Probability threshold for accepting frames as representative of a state (default: 0.7)

    Returns
    -------
    dict
        Dictionary mapping state numbers to lists of PDB file paths,
        sorted by similarity to average structure
    """
    # Make sure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get trajectory files based on available files
    dcd_pattern = os.path.join(traj_folder, "*.dcd")
    xtc_pattern = os.path.join(traj_folder, "*.xtc")
    traditional_pattern = os.path.join(traj_folder, "r?", "traj*")

    # Check for different trajectory formats
    traj_files = []
    for pattern in [dcd_pattern, xtc_pattern, traditional_pattern]:
        files = sorted(glob(pattern))
        if files:
            traj_files.extend(files)
            print(f"Found {len(files)} trajectory files matching {pattern}")

    if not traj_files:
        raise ValueError(f"No trajectory files found in {traj_folder}")

    print(f"Total trajectory files found: {len(traj_files)}")

    # Extract number of states from probability array
    n_states = probs.shape[1]
    n_frames_total = probs.shape[0]

    print(f"Found probabilities for {n_frames_total} frames with {n_states} states")

    # Create a frame index mapping to track original indices after stride
    frame_indices = np.arange(0, n_frames_total, stride)
    strided_probs = probs[frame_indices]

    print(f"Processing trajectories with stride {stride}...")
    print(f"Original frames: {n_frames_total}, Strided frames: {len(strided_probs)}")

    # Load trajectories with stride
    trajs = []
    frame_counts = []

    # Track which original frames correspond to which loaded frames
    frame_mapping = []
    current_idx = 0

    print("Loading trajectories...")
    for traj_file in tqdm(traj_files, desc="Loading trajectories"):
        try:
            traj = md.load(traj_file, top=topology_file, stride=stride)
            trajs.append(traj)
            n_frames = len(traj)
            frame_counts.append(n_frames)

            # Add the frame indices for this trajectory chunk
            frame_mapping.extend(range(current_idx, current_idx + n_frames * stride, stride))
            current_idx += n_frames * stride
        except Exception as e:
            print(f"Warning: Could not load {traj_file}: {str(e)}")

    if not trajs:
        raise ValueError("Failed to load any trajectories")

    print("Combining trajectories...")
    combined_traj = md.join(trajs) if len(trajs) > 1 else trajs[0]
    print(f"Combined trajectory has {len(combined_traj)} frames")

    # Verify we have enough probability data
    if len(strided_probs) < len(combined_traj):
        print(
            f"Warning: Fewer probability entries ({len(strided_probs)}) than trajectory frames ({len(combined_traj)})")
        # Truncate the trajectory to match
        combined_traj = combined_traj[:len(strided_probs)]
    elif len(strided_probs) > len(combined_traj):
        print(f"Warning: More probability entries ({len(strided_probs)}) than trajectory frames ({len(combined_traj)})")
        # Truncate the probabilities to match
        strided_probs = strided_probs[:len(combined_traj)]

    # Get most likely state for each frame (just for comparison with probability-based approach)
    state_assignments = np.argmax(strided_probs, axis=1)
    state_counts = np.bincount(state_assignments, minlength=n_states)
    state_percentages = state_counts / len(state_assignments) * 100

    print("\nState assignment statistics (from most probable state):")
    for state_idx in range(n_states):
        print(f"State {state_idx + 1}: {state_counts[state_idx]} frames ({state_percentages[state_idx]:.2f}%)")

    # Dictionary to store generated structure paths
    state_structures = {}

    # Create summary file
    summary_path = os.path.join(save_dir, f"{protein_name}_state_structures_summary.txt")
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"State structure analysis for {protein_name}\n")
        summary_file.write(f"Total states: {n_states}\n\n")

        for state_idx in range(n_states):
            summary_file.write(f"State {state_idx + 1}:\n")
            summary_file.write(
                f"  - Frame count (max probability): {state_counts[state_idx]} ({state_percentages[state_idx]:.2f}%)\n")

    print("\nProcessing states...")
    for state_idx in range(n_states):
        state_num = state_idx + 1  # Convert to 1-indexed for output
        state_structures[state_idx] = []  # Initialize list for this state

        # Create state-specific directory
        state_dir = os.path.join(save_dir, f"state_{state_num}")
        os.makedirs(state_dir, exist_ok=True)

        # Get probabilities for this state
        state_probs = strided_probs[:, state_idx]

        # Find frames where probability exceeds threshold
        high_prob_indices = np.where(state_probs >= prob_threshold)[0]

        if len(high_prob_indices) == 0:
            print(f"Warning: No frames found for state {state_num} with probability >= {prob_threshold}")
            # Fall back to top N frames by probability
            high_prob_indices = np.argsort(state_probs)[-n_structures:]
            print(f"Using top {len(high_prob_indices)} frames by probability instead")
        elif len(high_prob_indices) < n_structures:
            print(f"Warning: Only {len(high_prob_indices)} frames found for state {state_num} "
                  f"with probability >= {prob_threshold} (requested {n_structures})")
        else:
            print(f"Found {len(high_prob_indices)} frames for state {state_num} with probability >= {prob_threshold}")
            # If we have more frames than needed, select the highest probability ones
            if len(high_prob_indices) > n_structures:
                # Sort by probability (highest first) and take top n_structures
                sorted_indices = sorted(high_prob_indices, key=lambda i: state_probs[i], reverse=True)
                high_prob_indices = sorted_indices[:n_structures]

        # Extract frames for this state
        state_frames = combined_traj[high_prob_indices]
        state_frame_probs = state_probs[high_prob_indices]

        # Append information to summary
        with open(summary_path, 'a') as summary_file:
            summary_file.write(f"  - High probability frames (>= {prob_threshold}): {len(high_prob_indices)}\n")

        if len(high_prob_indices) == 0:
            with open(summary_path, 'a') as summary_file:
                summary_file.write("  - No structures generated\n\n")
            continue

        try:
            # First align all structures to the first frame
            aligned_traj = state_frames.superpose(state_frames[0])

            # Calculate average coordinates across all frames
            average_xyz = np.mean(aligned_traj.xyz, axis=0)

            # Create a new mdtraj trajectory for the average structure
            average_structure = md.Trajectory(
                xyz=average_xyz,
                topology=state_frames.topology
            )

            # Save the average structure
            avg_file = os.path.join(state_dir, f"{protein_name}_state_{state_num}_average.pdb")
            average_structure.save_pdb(avg_file)
            state_structures[state_idx].append(avg_file)

            # Append to summary
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"  - Average structure: {os.path.basename(avg_file)}\n")

            # Calculate RMSD to average structure for all frames
            rmsd_to_avg = np.sqrt(np.mean(np.sum((aligned_traj.xyz - average_xyz) ** 2, axis=2), axis=1))

            # Get indices sorted by RMSD (closest to average first)
            sorted_indices = np.argsort(rmsd_to_avg)

            # Save representative structures
            with open(summary_path, 'a') as summary_file:
                summary_file.write("  - Representative structures:\n")

            for rank, idx in enumerate(sorted_indices):
                frame_idx = high_prob_indices[idx]
                rmsd = rmsd_to_avg[idx]
                prob = state_frame_probs[idx]

                # Get original frame index
                original_idx = frame_mapping[frame_idx] if frame_idx < len(frame_mapping) else -1

                # Output filename with rank, frame index, RMSD, and probability
                output_file = os.path.join(
                    state_dir,
                    f"{protein_name}_state_{state_num}_rank_{rank + 1}_frame_{original_idx}_rmsd_{rmsd:.3f}_prob_{prob:.3f}.pdb"
                )

                # Save the frame as PDB
                combined_traj[frame_idx].save_pdb(output_file)
                state_structures[state_idx].append(output_file)

                # Add to summary
                with open(summary_path, 'a') as summary_file:
                    summary_file.write(f"    - {os.path.basename(output_file)}\n")

            with open(summary_path, 'a') as summary_file:
                summary_file.write("\n")

            print(f"State {state_num}: Saved average structure and {len(sorted_indices)} representative structures")

        except Exception as e:
            print(f"Error processing state {state_num}: {str(e)}")
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"  - Error: {str(e)}\n\n")

    print(f"Summary saved to {summary_path}")
    return state_structures


def save_attention_colored_structures(
        state_structures: dict,
        state_attention_maps: np.ndarray,
        save_dir: str,
        protein_name: str,
        residue_indices: list = None,
        residue_names: list = None
) -> dict:
    """
    Save new versions of existing state structures with attention values as B-factors
    and corresponding PyMOL visualization scripts.

    Parameters
    ----------
    state_structures : dict
        Dictionary mapping state numbers to lists of PDB file paths
    state_attention_maps : np.ndarray
        Attention maps for each state [n_states, n_atoms, n_atoms]
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein for file naming
    residue_indices : list, optional
        List of residue indices that correspond to atoms in attention maps
    residue_names : list, optional
        List of residue names with numbers (e.g., "ALA126")

    Returns
    -------
    dict
        Dictionary mapping state numbers to lists of attention-colored PDB file paths
    """
    # Define a simple scaling function to replace sklearn's scale
    def scale(x):
        """Scale array to range [0,1]"""
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        # Handle case where all values are the same
        return np.zeros_like(x)

    # Calculate scaled attention scores for each state
    state_residue_attention = {}
    for state in range(len(state_attention_maps)):
        # If residue indices are provided, use them to map attention to residues
        if residue_indices is not None:
            # Map attention to residues if needed
            scores = scale(state_attention_maps[state].sum(axis=0))
        else:
            # Otherwise use atom-level attention directly
            scores = scale(state_attention_maps[state].sum(axis=0))

        scores = scores * 90  # Scale to 0-90 range
        state_residue_attention[state] = scores

    attention_structures = {}

    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Process each state
    for state_num, structures in state_structures.items():
        attention_structures[state_num] = []
        state_dir = os.path.join(save_dir, f"state_{state_num + 1}_attention")
        os.makedirs(state_dir, exist_ok=True)

        print(f"Processing state {state_num + 1} structures...")

        # Process each structure in the state
        for pdb_file in structures:
            try:
                # Load structure
                traj = md.load(pdb_file)

                # Create attention PDB filename
                base_name = os.path.basename(pdb_file)
                new_name = base_name.replace('.pdb', '_attention.pdb')
                output_file = os.path.join(state_dir, new_name)

                # Get attention values for this state
                attention_values = state_residue_attention[state_num]

                # Create a mapping from PDB residue numbers to attention values
                attention_map = {}
                if residue_indices is not None:
                    # Map between selected residue indices and attention values
                    for i, res_idx in enumerate(residue_indices):
                        if i < len(attention_values):
                            attention_map[res_idx] = attention_values[i]

                # Set B-factors based on residue indices
                b_factors = []
                for atom in traj.topology.atoms:
                    res_idx = atom.residue.resSeq

                    if residue_indices is not None:
                        # Use residue mapping
                        attention_value = attention_map.get(res_idx, 0.0)
                        b_factors.append(attention_value)
                    else:
                        # Use atom indices directly
                        atom_idx = atom.index
                        if atom_idx < len(attention_values):
                            b_factors.append(attention_values[atom_idx])
                        else:
                            b_factors.append(0.0)

                # Save PDB with attention B-factors
                traj.save_pdb(output_file, bfactors=b_factors)

                # Create corresponding PyMOL script
                script_name = new_name.replace('.pdb', '_view.pml')
                script_path = os.path.join(state_dir, script_name)

                with open(script_path, 'w') as f:
                    f.write(f"load {new_name}\n")
                    f.write("bg_color white\n")
                    f.write("show cartoon\n")
                    f.write("hide lines\n")
                    f.write("spectrum b, blue_white_red\n")
                    f.write("set ray_shadows, 0\n")
                    f.write("set ray_opaque_background, off\n")
                    f.write("set cartoon_fancy_helices, 1\n")

                    # If we have specific residue indices, add labels
                    # TODO: Only re-enable if you want labels next to high attention residues
                    #if residue_names is not None:
                    #    # Add labels for high-attention residues (top quartile)
                    #    if len(attention_values) > 0:
                    #        high_threshold = np.percentile(attention_values, 75)
                    #        for i, res_idx in enumerate(residue_indices):
                    #            if i < len(attention_values) and attention_values[i] >= high_threshold:
                    #                res_name = residue_names[i]
                    #                f.write(f"label chain A and resi {res_idx}, '{res_name}'\n")

                    f.write("zoom\n")
                    f.write("ray 1200, 1200\n")

                attention_structures[state_num].append(output_file)
                print(f"Saved attention-colored structure to {output_file}")
                print(f"Saved PyMOL script to {script_path}")

            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")

    return attention_structures


def save_attention_colored_structures_old(
        state_structures: dict,
        state_attention_maps: np.ndarray,
        save_dir: str,
        protein_name: str
) -> dict:
    """
    Save new versions of existing state structures with attention values as B-factors
    and corresponding PyMOL visualization scripts.

    Parameters
    ----------
    state_structures : dict
        Dictionary mapping state numbers to lists of PDB file paths
    state_attention_maps : np.ndarray
        Attention maps for each state [n_states, n_atoms, n_atoms]
    save_dir : str
        Directory to save the output files
    protein_name : str
        Name of the protein for file naming

    Returns
    -------
    dict
        Dictionary mapping state numbers to lists of attention-colored PDB file paths
    """
    # Define a simple scaling function to replace sklearn's scale
    def scale(x):
        """Scale array to range [0,1]"""
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        # Handle case where all values are the same
        return np.zeros_like(x)

    # Calculate scaled attention scores for each state
    state_residue_attention = {}
    for state in range(len(state_attention_maps)):
        scores = scale(state_attention_maps[state].sum(axis=0))
        scores = scores * 90  # Scale to 0-90 range
        state_residue_attention[state] = scores

    attention_structures = {}

    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Process each state
    for state_num, structures in state_structures.items():
        attention_structures[state_num] = []
        state_dir = os.path.join(save_dir, f"state_{state_num + 1}_attention")
        os.makedirs(state_dir, exist_ok=True)

        print(f"Processing state {state_num + 1} structures...")

        # Process each structure in the state
        for pdb_file in structures:
            try:
                # Load structure
                traj = md.load(pdb_file)

                # Create attention PDB filename
                base_name = os.path.basename(pdb_file)
                new_name = base_name.replace('.pdb', '_attention.pdb')
                output_file = os.path.join(state_dir, new_name)

                # Get attention values for this state
                attention_values = state_residue_attention[state_num]

                # Check if dimensions match
                if len(attention_values) != traj.topology.n_residues:
                    print(f"Warning: Attention map has {len(attention_values)} values, but structure "
                          f"{base_name} has {traj.topology.n_residues} residues.")

                    # If not enough values, pad with zeros
                    if len(attention_values) < traj.topology.n_residues:
                        padding = np.zeros(traj.topology.n_residues - len(attention_values))
                        attention_values = np.concatenate([attention_values, padding])
                    else:
                        # If too many values, truncate
                        attention_values = attention_values[:traj.topology.n_residues]

                # Set B-factors based on residue indices
                b_factors = []
                for atom in traj.topology.atoms:
                    res_idx = atom.residue.index
                    if res_idx < len(attention_values):
                        attention_value = attention_values[res_idx]
                        b_factors.append(attention_value)
                    else:
                        # Fallback for atoms in residues beyond our attention map
                        b_factors.append(0.0)

                # Check that we have b-factors for all atoms
                if len(b_factors) != traj.n_atoms:
                    print(f"Warning: Generated {len(b_factors)} B-factors for {traj.n_atoms} atoms "
                          f"in structure {base_name}")
                    # Pad with zeros if needed
                    if len(b_factors) < traj.n_atoms:
                        b_factors.extend([0.0] * (traj.n_atoms - len(b_factors)))
                    else:
                        # Truncate if too many
                        b_factors = b_factors[:traj.n_atoms]

                # Save PDB with attention B-factors
                traj_with_bfactors = traj
                traj_with_bfactors.save_pdb(output_file, bfactors=b_factors)

                # Create corresponding PyMOL script
                script_name = new_name.replace('.pdb', '_view.pml')
                script_path = os.path.join(state_dir, script_name)

                with open(script_path, 'w') as f:
                    f.write(f"load {new_name}\n")
                    f.write("bg_color white\n")
                    f.write("show cartoon\n")
                    f.write("hide lines\n")
                    f.write("spectrum b, blue_white_red\n")
                    f.write("set ray_shadows, 0\n")
                    f.write("set ray_opaque_background, off\n")
                    f.write("set cartoon_fancy_helices, 1\n")
                    f.write("zoom\n")
                    f.write("ray 1200, 1200\n")

                attention_structures[state_num].append(output_file)
                print(f"Saved attention-colored structure to {output_file}")
                print(f"Saved PyMOL script to {script_path}")

            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")

    return attention_structures


def calculate_state_attention_maps_old(attentions: np.ndarray,
                                   neighbor_indices: np.ndarray,
                                   state_assignments: np.ndarray,
                                   num_classes: int,
                                   num_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate attention maps for each state from node attention values.

    Parameters
    ----------
    attentions : np.ndarray
        Attention values for each frame [n_frames, n_atoms]
    neighbor_indices : np.ndarray
        Neighbor indices for each frame [n_frames, n_atoms, n_neighbors]
    state_assignments : np.ndarray
        State assignments for each frame [n_frames]
    num_classes : int
        Number of states
    num_atoms : int
        Number of atoms in the system

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        state_attention_maps: Average attention maps for each state
        state_populations: Population of each state
    """
    # Calculate state populations
    unique, counts = np.unique(state_assignments, return_counts=True)
    state_populations = np.zeros(num_classes)
    state_populations[unique] = counts
    state_populations = state_populations / np.sum(state_populations)

    # Initialize state attention maps
    state_attention_maps = np.zeros((num_classes, num_atoms))

    # Process each state
    for state in range(num_classes):
        # Create mask for frames in this state
        state_mask = state_assignments == state

        if np.any(state_mask):
            # Get attention for frames in this state and average
            state_attention = attentions[state_mask]

            # Average over frames
            if len(state_attention) > 0:
                state_attention_maps[state] = np.mean(state_attention, axis=0)

    return state_attention_maps, state_populations





