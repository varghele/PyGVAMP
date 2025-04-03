import torch
import cProfile
import pstats
from pstats import SortKey
import time
from torch_geometric.data import Data, Batch
from layers.cfconv_newnew import SchNetEncoder

def create_test_data(num_graphs=5, nodes_per_graph=40, edges_per_node=800,
                     node_dim=32, edge_dim=16, device='cuda'):
    """
    Create synthetic graph data for testing SchNet performance.

    Parameters:
    -----------
    num_graphs : int
        Number of graphs in the batch
    nodes_per_graph : int
        Number of nodes per graph
    edges_per_node : int
        Average number of edges per node
    node_dim : int
        Node feature dimension
    edge_dim : int
        Edge feature dimension
    device : str
        Device to place tensors on

    Returns:
    --------
    torch_geometric.data.Batch
        Batched graph data for testing
    """
    data_list = []
    total_nodes = 0
    total_edges = 0

    for graph_idx in range(num_graphs):
        # Number of nodes in this graph
        n_nodes = nodes_per_graph

        # Create random node features
        x = torch.randn(n_nodes, node_dim)

        # Create edge connections (approximately edges_per_node for each node)
        edge_index = []
        for src in range(n_nodes):
            # For each node, connect to a random selection of other nodes
            n_edges = edges_per_node
            targets = torch.randperm(n_nodes)[:n_edges]
            for target in targets:
                edge_index.append([src, target])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        num_edges = edge_index.size(1)

        # Create random edge features (normalized to simulate RBF expansion)
        edge_attr = torch.randn(num_edges, edge_dim)
        edge_norms = edge_attr.norm(dim=1, keepdim=True)
        edge_attr = edge_attr / (edge_norms + 1e-8)

        # Create batch assignment
        batch = torch.full((n_nodes,), graph_idx, dtype=torch.long)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch
        )

        data_list.append(data)
        total_nodes += n_nodes
        total_edges += num_edges

    # Create batched data
    batch_data = Batch.from_data_list(data_list)

    # Move to specified device
    if device:
        batch_data = batch_data.to(device)

    print(f"Created test data with {num_graphs} graphs, {total_nodes} nodes, {total_edges} edges")
    print(f"Node features: {batch_data.x.shape}, Edge features: {batch_data.edge_attr.shape}")

    return batch_data


def profile_schnet(model, data, n_runs=10, device='cuda'):
    """
    Profile the SchNet model execution.

    Parameters:
    -----------
    model : SchNetEncoder
        SchNet model to profile
    data : torch_geometric.data.Batch
        Batched graph data
    n_runs : int
        Number of runs for averaging performance
    device : str
        Device to run on

    Returns:
    --------
    tuple
        (profiler, stats) for further analysis
    """
    # Move model to device
    model = model.to(device)

    # Ensure data is on the right device
    if data.x.device != torch.device(device):
        data = data.to(device)

    # Warm up
    with torch.no_grad():
        model(data.x, data.edge_index, data.edge_attr, data.batch)

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Run multiple times for better data
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(data.x, data.edge_index, data.edge_attr, data.batch)

    total_time = time.time() - start_time

    # Stop profiling
    profiler.disable()

    # Print summary
    print(f"Average execution time: {total_time / n_runs:.4f} seconds")

    # Get detailed stats
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Print top 20 time-consuming functions

    return profiler, stats


# Create test data with different sizes
test_configs = [
    # Small graph
    {"num_graphs": 2, "nodes_per_graph": 10, "edges_per_node": 3},
    # Medium graph
    {"num_graphs": 5, "nodes_per_graph": 30, "edges_per_node": 5},
    # Large graph
    {"num_graphs": 10, "nodes_per_graph": 300, "edges_per_node": 20},
]

# Choose a device based on availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Test with different configurations
for config in test_configs:
    print("\n" + "=" * 50)
    print(f"Testing with {config['num_graphs']} graphs, {config['nodes_per_graph']} nodes per graph")
    print("=" * 50)

    # Create test data
    test_data = create_test_data(
        num_graphs=config['num_graphs'],
        nodes_per_graph=config['nodes_per_graph'],
        edges_per_node=config['edges_per_node'],
        node_dim=32,
        edge_dim=16,
        device=device
    )

    # Create model
    encoder = SchNetEncoder(
        node_dim=32,
        edge_dim=16,
        hidden_dim=64,
        output_dim=32,
        n_interactions=3,
        use_attention=True
    )

    # Profile model
    profile_results = profile_schnet(encoder, test_data, n_runs=5, device=device)

# You can also visualize the profiling results with a tool like snakeviz
# Uncomment the following lines if you have snakeviz installed
# import tempfile
# with tempfile.NamedTemporaryFile(suffix='.prof', delete=False) as tmp:
#     profiler_dump_path = tmp.name
# profile_results[0].dump_stats(profiler_dump_path)
# print(f"Profile data saved to {profiler_dump_path}")
# print(f"View with: snakeviz {profiler_dump_path}")
