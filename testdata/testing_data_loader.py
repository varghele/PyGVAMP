# Import arguments parser

import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.utils.pipe_utils import find_trajectory_files


def create_test_args():
    """Create a simple argument namespace for testing"""
    args = argparse.Namespace()

    # Basic settings
    args.encoder_type = 'schnet'
    #args.encoder_type = 'meta'

    # Data settings
    args.protein_name = 'ATR'
    #args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/')
    #args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/')
    #args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/r0/')
    args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/')
    args.file_pattern = '*.xtc'
    #args.file_pattern = '*.dcd'
    #args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    #args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein.pdb')
    #args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ATR/prot.pdb')
    args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/topol.pdb')
    args.selection = 'name CA'
    args.stride = 10
    args.lag_time = 20.0
    args.n_neighbors = 7
    args.node_embedding_dim = 16
    args.gaussian_expansion_dim = 12 # TODO: This is edge dim!!!

    # SchNet encoder settings
    args.node_dim = 16
    args.edge_dim = 12
    args.hidden_dim = 32
    args.output_dim = 16
    args.n_interactions = 4
    args.activation = 'tanh'
    args.use_attention = True

    # Meta encoder settings
    """args.meta_node_dim = 16
    args.meta_edge_dim = 12 # TODO: Gaussian expansion dim
    args.meta_global_dim = 32
    args.meta_num_node_mlp_layers = 2
    args.meta_num_edge_mlp_layers = 2
    args.meta_num_global_mlp_layers = 2
    args.meta_hidden_dim = 64
    args.meta_output_dim = 32
    args.meta_num_meta_layers = 3
    args.meta_embedding_type = 'global'  # choices: 'node', 'global', 'combined'
    args.meta_activation = 'leaky_relu'
    args.meta_norm = 'None'
    args.meta_dropout = 0.0"""

    # Classifier settings
    args.n_states = 5
    args.clf_hidden_dim = 32
    args.clf_num_layers = 2
    args.clf_dropout = 0.0
    args.clf_activation = 'leaky_relu'
    args.clf_norm = 'LayerNorm' # 'BatchNorm' #

    # Embedding settings
    args.use_embedding = True
    args.embedding_in_dim = 42
    args.embedding_hidden_dim = 32
    args.embedding_out_dim = 16
    args.embedding_num_layers = 2
    args.embedding_dropout = 0.0
    args.embedding_act = 'leaky_relu'
    args.embedding_norm = 'LayerNorm' # 'BatchNorm' #

    # Training settings
    args.epochs = 10
    args.batch_size = 1024
    args.lr = 0.0005
    args.weight_decay = 1e-5
    args.clip_grad = None
    args.cpu = False  # Use CPU for testing

    # Testing settngs
    args.max_tau = 200

    # Output settings
    args.output_dir = 'area54'
    args.cache_dir = 'area54/cache'
    args.use_cache = False
    args.save_every = 0  # Don't save intermediates
    args.run_name = 'test_run_new'

    return args

args = create_test_args()

def create_dataset_and_loader(args):
    """Create dataset and data loader"""
    # Getting all trajectories in traj directory
    traj_files = find_trajectory_files(args.traj_dir, file_pattern=args.file_pattern)

    print("Creating dataset...")
    dataset = VAMPNetDataset(
        trajectory_files=traj_files,
        topology_file=args.top,
        lag_time=args.lag_time,
        n_neighbors=args.n_neighbors,
        node_embedding_dim=args.node_embedding_dim,
        gaussian_expansion_dim=args.gaussian_expansion_dim,
        selection=args.selection,
        stride=args.stride,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache
    )

    print(f"Dataset created with {len(dataset)} samples")

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Turn off shuffling for debugging
        pin_memory=torch.cuda.is_available() and not args.cpu
    )

    return dataset, loader


def inspect_dataloader(loader, num_batches=5, visualize=False):
    """
    Inspect the contents of a DataLoader.

    Parameters
    ----------
    loader : DataLoader
        The DataLoader to inspect.
    num_batches : int, default=5
        Number of batches to inspect.
    visualize : bool, default=False
        Whether to visualize the data (e.g., node features, edge features).
    """
    print("\n=== Inspecting DataLoader ===")
    print(f"Number of batches in DataLoader: {len(loader)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Shuffle: {loader.shuffle if hasattr(loader, 'shuffle') else 'Unknown'}")

    # Iterate over the DataLoader
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")

        # Unpack the batch (assuming VAMPNetDataset produces time-lagged pairs)
        data_t0, data_t1 = batch

        # Print basic information about the batch
        print(f"Batch T0:")
        print(f"  Number of graphs: {data_t0.num_graphs}")
        print(f"  Total nodes: {data_t0.num_nodes}")
        print(f"  Node feature shape: {data_t0.x.shape}")
        print(f"  Edge feature shape: {data_t0.edge_attr.shape}")
        print(f"  Number of edges: {data_t0.edge_index.shape[1]}")

        print(f"Batch T1:")
        print(f"  Number of graphs: {data_t1.num_graphs}")
        print(f"  Total nodes: {data_t1.num_nodes}")
        print(f"  Node feature shape: {data_t1.x.shape}")
        print(f"  Edge feature shape: {data_t1.edge_attr.shape}")
        print(f"  Number of edges: {data_t1.edge_index.shape[1]}")

        # Check for NaNs or Infs in node and edge features
        for batch_name, data in zip(["T0", "T1"], [data_t0, data_t1]):
            if torch.isnan(data.x).any():
                print(f"  ⚠️ NaN detected in node features of Batch {batch_name}")
            if torch.isinf(data.x).any():
                print(f"  ⚠️ Inf detected in node features of Batch {batch_name}")
            if torch.isnan(data.edge_attr).any():
                print(f"  ⚠️ NaN detected in edge features of Batch {batch_name}")
            if torch.isinf(data.edge_attr).any():
                print(f"  ⚠️ Inf detected in edge features of Batch {batch_name}")

        # Visualize the data if requested
        if visualize:
            visualize_graph(data_t0, title=f"Batch {batch_idx + 1} - T0")
            visualize_graph(data_t1, title=f"Batch {batch_idx + 1} - T1")

    print("\n=== DataLoader Inspection Complete ===")


def visualize_graph(data, title="Graph Visualization"):
    """
    Visualize a graph using NetworkX and Matplotlib.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph data to visualize.
    title : str, default="Graph Visualization"
        Title for the plot.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx

    # Convert PyG graph to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def visualize_graph(data, title="Graph Visualization"):
    """
    Visualize a graph using NetworkX and Matplotlib.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph data to visualize.
    title : str, default="Graph Visualization"
        Title for the plot.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx

    # Convert PyG graph to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    dataset, loader = create_dataset_and_loader(args)
    inspect_dataloader(loader, num_batches=5, visualize=False)