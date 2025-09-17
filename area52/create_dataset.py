from pygv.dataset.vampnet_dataset import VAMPNetDataset
import os
import glob
import torch

def find_xtc_files(base_path):
    """
    Find all .xtc files within the given directory and its subdirectories

    Args:
        base_path: Base directory to search

    Returns:
        List of absolute paths to .xtc files
    """
    # Ensure base_path exists
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    # Use recursive glob to find all .xtc files
    xtc_files = glob.glob(os.path.join(base_path, '**', '*.xtc'), recursive=True)

    # Sort the files for consistency
    xtc_files.sort()

    print(f"Found {len(xtc_files)} .xtc files in {base_path} and subdirectories")

    # Print the first few files for verification
    if xtc_files:
        print("Sample files:")
        for file in xtc_files[:5]:  # Show the first 5 files
            print(f"  - {file}")
        if len(xtc_files) > 5:
            print(f"  ... and {len(xtc_files) - 5} more")

    return xtc_files


# Specify the base directory path
base_path = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/"
#base_path = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ATR/"
# First, let's find all the .xtc files
xtc_files = find_xtc_files(base_path)
#print(xtc_files)

# Assuming you have a topology file in the same directory or nearby
# You might need to adjust this path
topology_file = os.path.join(base_path, "topol.pdb")  # Adjust as needed
#topology_file = os.path.join(base_path, "prot.pdb")  # Adjust as needed

# Initialize the dataset
dataset = VAMPNetDataset(
    trajectory_files=xtc_files,
    topology_file=topology_file,
    lag_time=20,  # Lag time in nanoseconds
    n_neighbors=20,  # Number of neighbors for graph construction
    node_embedding_dim=16,
    gaussian_expansion_dim=8,
    selection="name CA",  # Select only C-alpha atoms
    stride=40,  # Take every 2nd frame to reduce dataset size
    cache_dir="testdata",
    use_cache=False
)

# Then use the dataset as usual
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Get the first batch by using iter and next
dataiter = iter(loader)
batch = next(dataiter)

# Unpack the batch - it contains two time-lagged graph batches
batch_t0, batch_t1 = batch

# Print and compare basic information about both batches
print("=== BATCH T0 (Current Time) ===")
print(f"Batch size: {batch_t0.num_graphs}")
print(f"Total nodes in batch: {batch_t0.num_nodes}")
print(f"Node feature dimensions: {batch_t0.x.shape}")
print(f"Edge feature dimensions: {batch_t0.edge_attr.shape}")
print(f"Number of edges: {batch_t0.edge_index.shape[1]}")

print("\n=== BATCH T1 (Time + lag_time) ===")
print(f"Batch size: {batch_t1.num_graphs}")
print(f"Total nodes in batch: {batch_t1.num_nodes}")
print(f"Node feature dimensions: {batch_t1.x.shape}")
print(f"Edge feature dimensions: {batch_t1.edge_attr.shape}")
print(f"Number of edges: {batch_t1.edge_index.shape[1]}")

# Look at the first graph in both batches
graph_idx = 0

# T0 - First graph
t0_node_mask = batch_t0.batch == graph_idx
t0_edge_mask = torch.isin(batch_t0.edge_index[0], torch.nonzero(t0_node_mask).squeeze())

t0_first_graph_nodes = batch_t0.x[t0_node_mask]
t0_first_graph_edges = batch_t0.edge_index[:, t0_edge_mask]
t0_first_graph_edge_attr = batch_t0.edge_attr[t0_edge_mask]

# T1 - First graph
t1_node_mask = batch_t1.batch == graph_idx
t1_edge_mask = torch.isin(batch_t1.edge_index[0], torch.nonzero(t1_node_mask).squeeze())

t1_first_graph_nodes = batch_t1.x[t1_node_mask]
t1_first_graph_edges = batch_t1.edge_index[:, t1_edge_mask]
t1_first_graph_edge_attr = batch_t1.edge_attr[t1_edge_mask]

print("\n=== First Graph Comparison ===")
print("T0 (Current Time):")
print(f"Number of nodes: {t0_first_graph_nodes.shape[0]}")
print(f"Number of edges: {t0_first_graph_edges.shape[1]}")

print("\nT1 (Time + lag_time):")
print(f"Number of nodes: {t1_first_graph_nodes.shape[0]}")
print(f"Number of edges: {t1_first_graph_edges.shape[1]}")

# Analyze node features for both batches
t0_node_features_mean = batch_t0.x.mean(dim=0)
t0_node_features_std = batch_t0.x.std(dim=0)

t1_node_features_mean = batch_t1.x.mean(dim=0)
t1_node_features_std = batch_t1.x.std(dim=0)

print("\n=== Node Feature Statistics ===")
print("T0 (Current Time):")
print(f"Mean: {t0_node_features_mean[:5]}... (showing first 5)")
print(f"Std: {t0_node_features_std[:5]}... (showing first 5)")

print("\nT1 (Time + lag_time):")
print(f"Mean: {t1_node_features_mean[:5]}... (showing first 5)")
print(f"Std: {t1_node_features_std[:5]}... (showing first 5)")

# Analyze edge features (Gaussian expanded distances) for both batches
t0_edge_features_mean = batch_t0.edge_attr.mean(dim=0)
t0_edge_features_std = batch_t0.edge_attr.std(dim=0)

t1_edge_features_mean = batch_t1.edge_attr.mean(dim=0)
t1_edge_features_std = batch_t1.edge_attr.std(dim=0)

print("\n=== Edge Feature Statistics ===")
print("T0 (Current Time):")
print(f"Mean: {t0_edge_features_mean[:5]}... (showing first 5)")
print(f"Std: {t0_edge_features_std[:5]}... (showing first 5)")

print("\nT1 (Time + lag_time):")
print(f"Mean: {t1_edge_features_mean[:5]}... (showing first 5)")
print(f"Std: {t1_edge_features_std[:5]}... (showing first 5)")

# Compare edge feature distributions directly
print("\n=== Edge Feature Distribution Comparison ===")
t0_edge_dist = batch_t0.edge_attr.mean(dim=0)
t1_edge_dist = batch_t1.edge_attr.mean(dim=0)
diff = (t1_edge_dist - t0_edge_dist).abs()
max_diff_idx = torch.argmax(diff)
print(f"Maximum difference in Gaussian channel {max_diff_idx}: {diff[max_diff_idx].item():.4f}")
print(f"Average absolute difference across channels: {diff.mean().item():.4f}")

# Compare structural differences in the first graph's edges
print("\n=== First Graph Edge Structure ===")
# Count common edges between t0 and t1
t0_edge_set = {(int(src), int(dst)) for src, dst in zip(t0_first_graph_edges[0], t0_first_graph_edges[1])}
t1_edge_set = {(int(src), int(dst)) for src, dst in zip(t1_first_graph_edges[0], t1_first_graph_edges[1])}

common_edges = t0_edge_set.intersection(t1_edge_set)
t0_unique = t0_edge_set - t1_edge_set
t1_unique = t1_edge_set - t0_edge_set

print(f"T0 edges: {len(t0_edge_set)}")
print(f"T1 edges: {len(t1_edge_set)}")
print(f"Common edges: {len(common_edges)} ({len(common_edges)/len(t0_edge_set)*100:.1f}% of t0 edges)")
print(f"Edges unique to T0: {len(t0_unique)}")
print(f"Edges unique to T1: {len(t1_unique)}")

# Check if node features are the same or different
node_features_equal = torch.allclose(batch_t0.x, batch_t1.x)
print(f"\nNode features are {'the same' if node_features_equal else 'different'} between T0 and T1")
if not node_features_equal:
    print(f"Average node feature difference: {(batch_t0.x - batch_t1.x).abs().mean().item():.4f}")

