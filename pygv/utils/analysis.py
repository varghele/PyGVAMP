import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import pickle

# Helper function to move batch to device
def to_device(batch, device):
    x_t0, x_t1 = batch
    return (x_t0.to(device), x_t1.to(device))


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
    import numpy as np

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





