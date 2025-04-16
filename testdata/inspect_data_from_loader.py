import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj


def inspect_pygvamp_dataloader(dataloader, num_batches=5, save_plots=True):
    """
    Comprehensive inspection of data coming from a PyGVAMP dataloader.

    Parameters:
    -----------
    dataloader : torch.utils.data.DataLoader
        The dataloader to inspect
    num_batches : int, default=5
        Number of batches to inspect
    save_plots : bool, default=True
        Whether to save plots to disk
    """
    print(f"{'=' * 20} PyGVAMP DataLoader Inspection {'=' * 20}")

    # Basic dataloader information
    print(f"Dataloader type: {type(dataloader)}")
    print(f"Dataset type: {type(dataloader.dataset)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    # Sample batches
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        batches.append(batch)

    if not batches:
        print("No batches found in dataloader!")
        return

    # Inspect first batch in detail
    first_batch = batches[0]
    print("\n" + "=" * 60)
    print(f"First batch type: {type(first_batch)}")

    # PyG batch inspection
    if isinstance(first_batch, Batch):
        inspect_pyg_batch(first_batch, 0)
    else:
        print(f"Batch is not a PyG Batch, but: {type(first_batch)}")
        # Try to treat as a tuple or other structure
        if isinstance(first_batch, tuple):
            print(f"Batch is a tuple with {len(first_batch)} elements")
            for i, element in enumerate(first_batch):
                print(f"Element {i} type: {type(element)}")
                if isinstance(element, Batch):
                    inspect_pyg_batch(element, i)
                else:
                    inspect_generic_tensor(element, f"Tuple element {i}")

    # Check consistency across batches
    print("\n" + "=" * 60)
    print("Checking consistency across batches...")

    for i, batch in enumerate(batches):
        if i == 0:
            continue  # Skip first batch, already inspected

        print(f"\nBatch {i}:")
        if isinstance(batch, Batch):
            check_batch_consistency(batch, first_batch, i)
        elif isinstance(batch, tuple) and isinstance(first_batch, tuple):
            print(f"  Tuple length: {len(batch)} (first batch: {len(first_batch)})")

            for j, element in enumerate(batch):
                if j >= len(first_batch):
                    print(f"  Element {j} exists in batch {i} but not in first batch")
                    continue

                if isinstance(element, Batch) and isinstance(first_batch[j], Batch):
                    check_batch_consistency(element, first_batch[j], i, tuple_idx=j)
                else:
                    check_tensor_consistency(element, first_batch[j], f"Tuple element {j}")

    # VAMP-specific checks
    print("\n" + "=" * 60)
    print("Running VAMP-specific checks...")

    # Check if the batch structure matches expected VAMP structure
    # (i.e., contains both instantaneous and time-lagged data)
    check_vamp_batch_structure(batches)

    # Check correlation between instantaneous and time-lagged data
    check_vamp_correlation(batches)

    print("\n" + "=" * 60)
    print("DataLoader inspection complete!")


def inspect_pyg_batch(batch, batch_idx=0, prefix=""):
    """Inspect a PyG batch object"""
    print(f"{prefix}PyG Batch {batch_idx} contents:")

    # Print available attributes
    attrs = [attr for attr in dir(batch) if not attr.startswith('_') and not callable(getattr(batch, attr))]
    print(f"{prefix}  Available attributes: {attrs}")

    # Inspect common PyG attributes
    for attr_name in ['x', 'edge_index', 'edge_attr', 'pos', 'y', 'batch']:
        if hasattr(batch, attr_name) and getattr(batch, attr_name) is not None:
            attr = getattr(batch, attr_name)
            print(f"{prefix}  {attr_name}: shape={attr.shape}, dtype={attr.dtype}, device={attr.device}")

            # Value statistics for numerical tensors
            if attr.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
                print(f"{prefix}    values: min={attr.min().item():.4f}, max={attr.max().item():.4f}, "
                      f"mean={attr.mean().item():.4f}, std={attr.std().item():.4f}")

                # Check for NaN or Inf
                nan_count = torch.isnan(attr).sum().item()
                inf_count = torch.isinf(attr).sum().item()
                if nan_count > 0 or inf_count > 0:
                    print(f"{prefix}    WARNING: Contains {nan_count} NaN and {inf_count} Inf values!")

            # For edge_index, check connectivity
            if attr_name == 'edge_index' and hasattr(batch, 'x'):
                num_nodes = batch.x.size(0)
                num_edges = batch.edge_index.size(1)
                print(f"{prefix}    {num_edges} edges connecting {num_nodes} nodes")

                # Check for isolated nodes
                connected_nodes = torch.unique(batch.edge_index).size(0)
                if connected_nodes < num_nodes:
                    print(f"{prefix}    WARNING: Only {connected_nodes}/{num_nodes} nodes are connected!")

            # For batch index, check number of graphs
            if attr_name == 'batch':
                num_graphs = batch.batch.max().item() + 1
                print(f"{prefix}    Contains {num_graphs} graphs")

    # Graph-level analysis
    if hasattr(batch, 'edge_index') and hasattr(batch, 'x') and hasattr(batch, 'batch'):
        # Check adjacency matrix for each graph
        try:
            adj = to_dense_adj(batch.edge_index, batch=batch.batch)
            print(f"{prefix}  Adjacency tensor shape: {adj.shape}")
            print(f"{prefix}  Connectivity stats: {adj.sum(dim=(1, 2)).tolist()}")
        except Exception as e:
            print(f"{prefix}  Error computing adjacency matrix: {str(e)}")


def inspect_generic_tensor(tensor, name):
    """Inspect a generic tensor"""
    if not isinstance(tensor, torch.Tensor):
        print(f"{name} is not a tensor, but {type(tensor)}")
        return

    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

    if tensor.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
        print(f"  values: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
              f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")

        # Check for NaN or Inf
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"  WARNING: Contains {nan_count} NaN and {inf_count} Inf values!")


def check_batch_consistency(batch, first_batch, batch_idx, tuple_idx=None):
    """Check consistency between a batch and the first batch"""
    prefix = f"  Tuple element {tuple_idx}: " if tuple_idx is not None else "  "

    # Check for same attributes
    first_attrs = [attr for attr in dir(first_batch) if
                   not attr.startswith('_') and not callable(getattr(first_batch, attr))]
    attrs = [attr for attr in dir(batch) if not attr.startswith('_') and not callable(getattr(batch, attr))]

    if set(attrs) != set(first_attrs):
        print(f"{prefix}WARNING: Attributes differ from first batch!")
        print(f"{prefix}  First batch: {first_attrs}")
        print(f"{prefix}  Batch {batch_idx}: {attrs}")

    # Check tensor shapes
    for attr_name in ['x', 'edge_index', 'edge_attr', 'pos', 'y', 'batch']:
        if hasattr(first_batch, attr_name) and hasattr(batch, attr_name):
            first_attr = getattr(first_batch, attr_name)
            curr_attr = getattr(batch, attr_name)

            if first_attr is None and curr_attr is not None:
                print(f"{prefix}WARNING: {attr_name} is None in first batch but not in batch {batch_idx}")
                continue

            if first_attr is not None and curr_attr is None:
                print(f"{prefix}WARNING: {attr_name} is not None in first batch but None in batch {batch_idx}")
                continue

            if first_attr is None and curr_attr is None:
                continue

            if first_attr.shape != curr_attr.shape:
                print(f"{prefix}WARNING: {attr_name} shape differs! "
                      f"First: {first_attr.shape}, Batch {batch_idx}: {curr_attr.shape}")


def check_tensor_consistency(tensor, first_tensor, name):
    """Check consistency between a tensor and the first tensor"""
    if not isinstance(tensor, torch.Tensor) or not isinstance(first_tensor, torch.Tensor):
        print(f"  {name}: Type differs! First: {type(first_tensor)}, Current: {type(tensor)}")
        return

    if tensor.shape != first_tensor.shape:
        print(f"  {name}: Shape differs! First: {first_tensor.shape}, Current: {tensor.shape}")

    if tensor.dtype != first_tensor.dtype:
        print(f"  {name}: Dtype differs! First: {first_tensor.dtype}, Current: {tensor.dtype}")


def check_vamp_batch_structure(batches):
    """Check if the batch structure is appropriate for VAMP score calculation"""
    first_batch = batches[0]

    # VAMP expects either:
    # 1. A single batch with data split in half (data, data_lagged)
    # 2. A tuple/batch with separate data and time-lagged data

    if isinstance(first_batch, Batch):
        # Case 1: Data should be splittable into two halves
        if hasattr(first_batch, 'x'):
            n_samples = first_batch.x.size(0)
            if n_samples % 2 != 0:
                print("WARNING: Number of samples is not even, cannot split into instantaneous and time-lagged!")
                return

            # Check if first half and second half are different
            x_first_half = first_batch.x[:n_samples // 2]
            x_second_half = first_batch.x[n_samples // 2:]

            diff = (x_first_half - x_second_half).abs().mean().item()
            print(f"Difference between first and second half of x: {diff:.6f}")

            if diff < 1e-8:
                print("WARNING: First and second half of data are almost identical!")
                print("This suggests the data is not properly time-lagged.")

            # Compute correlation between halves
            try:
                corr = compute_correlation(x_first_half, x_second_half)
                print(f"Correlation between first and second half: {corr:.4f}")
            except Exception as e:
                print(f"Error computing correlation: {str(e)}")

    elif isinstance(first_batch, tuple) and len(first_batch) == 2:
        # Case 2: Tuple with data and time-lagged data
        data, data_lagged = first_batch

        if isinstance(data, torch.Tensor) and isinstance(data_lagged, torch.Tensor):
            if data.shape != data_lagged.shape:
                print(f"WARNING: Data shape {data.shape} doesn't match lagged data shape {data_lagged.shape}!")

            diff = (data - data_lagged).abs().mean().item()
            print(f"Difference between data and lagged data: {diff:.6f}")

            if diff < 1e-8:
                print("WARNING: Data and lagged data are almost identical!")
                print("This suggests the data is not properly time-lagged.")

            # Compute correlation
            try:
                corr = compute_correlation(data, data_lagged)
                print(f"Correlation between data and lagged data: {corr:.4f}")
            except Exception as e:
                print(f"Error computing correlation: {str(e)}")
    else:
        print("WARNING: Batch structure doesn't match expected VAMP format!")
        print(f"Expected a Batch or a tuple of (data, data_lagged), got {type(first_batch)}")


def check_vamp_correlation(batches):
    """Check correlation between instantaneous and time-lagged data across multiple batches"""
    all_data = []
    all_lagged = []

    for batch in batches:
        if isinstance(batch, Batch) and hasattr(batch, 'x'):
            n_samples = batch.x.size(0)
            if n_samples % 2 == 0:
                all_data.append(batch.x[:n_samples // 2])
                all_lagged.append(batch.x[n_samples // 2:])
        elif isinstance(batch, tuple) and len(batch) == 2:
            data, data_lagged = batch
            if isinstance(data, torch.Tensor) and isinstance(data_lagged, torch.Tensor):
                all_data.append(data)
                all_lagged.append(data_lagged)

    if all_data and all_lagged:
        try:
            # Concatenate data across batches
            data_tensor = torch.cat(all_data, dim=0)
            lagged_tensor = torch.cat(all_lagged, dim=0)

            # Compute overall correlation
            corr = compute_correlation(data_tensor, lagged_tensor)
            print(f"\nOverall correlation across all batches: {corr:.4f}")

            # Create visualization
            plot_data_distribution(data_tensor, lagged_tensor)

        except Exception as e:
            print(f"Error in correlation analysis: {str(e)}")
    else:
        print("Could not extract enough data for correlation analysis")


def compute_correlation(x, y):
    """Compute correlation between two tensors"""
    if x.ndim > 2:
        x = x.reshape(x.size(0), -1)
    if y.ndim > 2:
        y = y.reshape(y.size(0), -1)

    # Compute means
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)

    # Center the data
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Compute correlation
    numer = torch.sum(x_centered * y_centered)
    denom = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))

    return (numer / (denom + 1e-8)).item()


def plot_data_distribution(data, data_lagged, save_path="vamp_data_analysis.png"):
    """Create visualization of data distribution"""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Flatten data if needed
    if data.ndim > 2:
        data_flat = data.reshape(data.size(0), -1)
        lagged_flat = data_lagged.reshape(data_lagged.size(0), -1)
    else:
        data_flat = data
        lagged_flat = data_lagged

    # 1. PCA projection to 2D
    try:
        from sklearn.decomposition import PCA

        # Convert to numpy
        data_np = data_flat.detach().cpu().numpy()
        lagged_np = lagged_flat.detach().cpu().numpy()

        # Apply PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_np)
        lagged_2d = pca.transform(lagged_np)

        # Plot
        axes[0, 0].scatter(data_2d[:100, 0], data_2d[:100, 1], label="Data", alpha=0.7)
        axes[0, 0].scatter(lagged_2d[:100, 0], lagged_2d[:100, 1], label="Lagged", alpha=0.7)
        axes[0, 0].set_title("PCA 2D Projection (first 100 samples)")
        axes[0, 0].legend()

        # Plot lines connecting corresponding points
        for i in range(min(50, data_2d.shape[0])):
            axes[0, 0].plot([data_2d[i, 0], lagged_2d[i, 0]],
                            [data_2d[i, 1], lagged_2d[i, 1]], 'k-', alpha=0.1)

    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f"PCA Error: {str(e)}", ha='center', va='center')

    # 2. Feature distributions
    try:
        # Plot histograms of the first feature
        axes[0, 1].hist(data_flat[:, 0].detach().cpu().numpy(), bins=30, alpha=0.7, label="Data")
        axes[0, 1].hist(lagged_flat[:, 0].detach().cpu().numpy(), bins=30, alpha=0.7, label="Lagged")
        axes[0, 1].set_title("Distribution of First Feature")
        axes[0, 1].legend()
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f"Histogram Error: {str(e)}", ha='center', va='center')

    # 3. Feature-wise correlation
    try:
        n_features = min(20, data_flat.size(1))
        corrs = []

        for i in range(n_features):
            corr = compute_correlation(data_flat[:, i:i + 1], lagged_flat[:, i:i + 1])
            corrs.append(corr)

        axes[1, 0].bar(range(len(corrs)), corrs)
        axes[1, 0].set_xlabel("Feature Index")
        axes[1, 0].set_ylabel("Correlation")
        axes[1, 0].set_title(f"Feature-wise Correlation (first {n_features} features)")
        axes[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f"Correlation Error: {str(e)}", ha='center', va='center')

    # 4. Mean absolute difference
    try:
        n_features = min(20, data_flat.size(1))
        diffs = []

        for i in range(n_features):
            diff = (data_flat[:, i] - lagged_flat[:, i]).abs().mean().item()
            diffs.append(diff)

        axes[1, 1].bar(range(len(diffs)), diffs)
        axes[1, 1].set_xlabel("Feature Index")
        axes[1, 1].set_ylabel("Mean Abs Difference")
        axes[1, 1].set_title(f"Feature-wise Mean Abs Difference (first {n_features})")
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f"Difference Error: {str(e)}", ha='center', va='center')

    # Finalize and save
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    from torch_geometric.loader import DataLoader
    from your_module import YourDataset  # Replace with actual import

    # Create dataset and dataloader
    dataset = YourDataset(...)  # Replace with your dataset initialization
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Inspect dataloader
    inspect_pygvamp_dataloader(dataloader)
