import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


def analyze_vampnet_outputs(
        model,
        data_loader: Union[DataLoader, List[DataLoader]],
        save_folder: str,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        return_tensors: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Analyze VAMPNet outputs for PyG graph data, extracting embeddings, attention, and state probabilities.

    Parameters
    ----------
    model : VAMPNet
        Trained VAMPNet model
    data_loader : DataLoader or List[DataLoader]
        PyG DataLoader(s) containing trajectory data as graphs
        If a list is provided, each loader is treated as a separate trajectory
    save_folder : str
        Path to save the analysis results
    batch_size : int, optional
        Size of batches for processing, default=32
    device : str, optional
        Device to run analysis on ('cuda' or 'cpu')
    return_tensors : bool, optional
        Whether to return torch tensors instead of numpy arrays

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        Tuple containing (probs, embeddings, attentions)
    """
    # Set model to evaluation mode and move to device
    model.eval()
    model = model.to(device)

    # Create output directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Convert single loader to list for consistent processing
    if not isinstance(data_loader, list):
        data_loader = [data_loader]

    # Initialize lists for storing results
    probs_list = []
    embeddings_list = []
    attentions_list = []

    # Process each trajectory (dataloader)
    for traj_idx, loader in enumerate(data_loader):
        print(f"Processing trajectory {traj_idx + 1}/{len(data_loader)}...")

        # Get total number of samples for pre-allocation
        total_samples = len(loader.dataset)

        # Get dimensions from a single batch
        sample_batch = next(iter(loader))

        # Get output dimensionality by running a sample through the model
        with torch.no_grad():
            # Get dimensions for state probabilities and embeddings
            sample_probs, sample_embeddings = model(sample_batch.to(device),
                                                    return_features=True,
                                                    apply_classifier=True)

            # Try to get attention from model's encoder
            sample_output = model.encoder(
                sample_batch.x.to(device),
                sample_batch.edge_index.to(device),
                sample_batch.edge_attr.to(device) if hasattr(sample_batch, 'edge_attr') else None,
                sample_batch.batch.to(device) if hasattr(sample_batch, 'batch') else None
            )

            # Check if the output is a tuple with attention
            if isinstance(sample_output, tuple) and len(sample_output) > 1:
                _, (_, sample_attentions) = sample_output
                has_attention = len(sample_attentions) > 0
            else:
                has_attention = False

        # Create numpy arrays to store results for this trajectory
        num_classes = sample_probs.size(1)
        embedding_dim = sample_embeddings.size(1)

        probs = torch.zeros((total_samples, num_classes), device=device)
        embeddings = torch.zeros((total_samples, embedding_dim), device=device)

        # Initialize attention tensor if available
        if has_attention:
            # Get sample dimensions
            if isinstance(sample_attentions, list) and len(sample_attentions) > 0:
                last_attention = sample_attentions[-1]  # Use last layer's attention
                if last_attention is not None:
                    # Determine attention shape from the model
                    attention_shape = list(last_attention.shape)
                    attentions = torch.zeros((total_samples, *attention_shape[1:]), device=device)
                else:
                    has_attention = False

        # Reset dataloader iterator
        loader_iter = iter(loader)

        # Process each batch
        n_processed = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Trajectory {traj_idx + 1}")):
                # Move batch to device
                batch = batch.to(device)

                # Get batch size
                if hasattr(batch, 'batch'):
                    batch_size_current = batch.batch.max().item() + 1
                else:
                    batch_size_current = 1

                # Get state probabilities and embeddings
                batch_probs, batch_embeddings = model(batch,
                                                      return_features=True,
                                                      apply_classifier=True)

                # Store results
                probs[n_processed:n_processed + batch_size_current] = batch_probs
                embeddings[n_processed:n_processed + batch_size_current] = batch_embeddings

                # Get attention if available
                if has_attention:
                    # Get encoder output with attention
                    _, (_, batch_attentions) = model.encoder(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                        batch.batch if hasattr(batch, 'batch') else None
                    )

                    # Store attention from the last layer
                    if len(batch_attentions) > 0:
                        last_attention = batch_attentions[-1]
                        if last_attention is not None:
                            # Reshape attention based on batch assignment
                            for i in range(batch_size_current):
                                # Get node mask for this graph
                                if hasattr(batch, 'batch'):
                                    node_mask = batch.batch == i
                                    # Get number of nodes in this graph
                                    n_nodes = node_mask.sum().item()

                                    # Extract attention for this graph's nodes
                                    if n_nodes > 0:
                                        graph_attention = last_attention[node_mask]
                                        attentions[n_processed + i] = graph_attention

                # Update processed count
                n_processed += batch_size_current

        # Convert to numpy and append to result lists
        if return_tensors:
            probs_list.append(probs.cpu())
            embeddings_list.append(embeddings.cpu())
            if has_attention:
                attentions_list.append(attentions.cpu())
            else:
                attentions_list.append(None)
        else:
            probs_list.append(probs.cpu().numpy())
            embeddings_list.append(embeddings.cpu().numpy())
            if has_attention:
                attentions_list.append(attentions.cpu().numpy())
            else:
                attentions_list.append(None)

    # Save results as NPZ files
    np.savez(os.path.join(save_folder, 'transformed_traj.npz'), *probs_list)
    np.savez(os.path.join(save_folder, 'embeddings.npz'), *embeddings_list)

    # Only save attention if available
    if all(attn is not None for attn in attentions_list):
        np.savez(os.path.join(save_folder, 'attention.npz'), *attentions_list)

    # Save first trajectory results separately
    if len(probs_list) > 0:
        np.savez(os.path.join(save_folder, 'transformed_0_traj.npz'), probs_list[0])
        np.savez(os.path.join(save_folder, 'embeddings_0.npz'), embeddings_list[0])
        if attentions_list[0] is not None:
            np.savez(os.path.join(save_folder, 'attention_0.npz'), attentions_list[0])

    print(f"Analysis complete. Results saved to {save_folder}")
    print(f"State probabilities shape: {probs_list[0].shape}")
    print(f"Embeddings shape: {embeddings_list[0].shape}")
    if attentions_list[0] is not None:
        print(f"Attention shape: {attentions_list[0].shape}")
    print(f"Number of trajectories: {len(probs_list)}")

    return probs_list, embeddings_list, attentions_list
