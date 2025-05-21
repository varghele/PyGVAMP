import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D


def _compute_gaussian_expanded_distances(edge_distances, n_gaussians=16, min_dist=0.0, max_dist=5.0):
    """
    Transform distances into Gaussian basis functions.
    """
    # Make sure distances are at least 2D (for single distances)
    if edge_distances.dim() == 1:
        edge_distances = edge_distances.unsqueeze(-1)

    # Create centered gaussians along the distance range
    offset = torch.linspace(min_dist, max_dist, n_gaussians)
    coeff = -0.5 / (0.5 ** 2)  # Assuming sigma=0.5 for all gaussians

    # Compute gaussians for all distances
    expanded = torch.exp(coeff * (edge_distances.unsqueeze(-1) - offset) ** 2)

    return expanded


def create_graph_from_coords(coords, n_neighbors=10):
    """
    Create a graph representation from 3D coordinates.
    """
    n_atoms = coords.shape[0]

    # Calculate pairwise distances
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [n_atoms, n_atoms, 3]
    distances = torch.sqrt((diff ** 2).sum(dim=2))  # [n_atoms, n_atoms]

    # Create a mask to identify self-connections (diagonal elements)
    diag_mask = torch.eye(n_atoms, dtype=torch.bool, device=distances.device)

    # Set self-distances to -1
    distances[diag_mask] = -1.0

    # Create a mask for valid distances (excluding self-connections)
    valid_mask = ~diag_mask

    # For each node, get indices of the k-nearest neighbors (excluding self)
    nn_indices = []
    distance_values = []

    for i in range(n_atoms):
        # Get distances from node i to all other nodes
        node_distances = distances[i]
        # Mask out the self-connection
        valid_distances = node_distances[valid_mask[i]]
        # Get indices of valid nodes (excluding self)
        valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
        # Get top-k nearest neighbors
        dist_vals, top_k_indices = torch.topk(valid_distances, min(n_neighbors, len(valid_distances)), largest=False)
        # Map back to original indices
        node_nn_indices = valid_indices[top_k_indices]
        # Add to lists
        nn_indices.append(node_nn_indices)
        distance_values.append(dist_vals)

    # Stack indices for all nodes
    nn_indices = torch.stack(nn_indices)
    distance_values = torch.stack(distance_values)

    # Create a set to track all edges
    edge_set = set()

    # First, collect all the original directional edges
    for i in range(n_atoms):
        for j in nn_indices[i]:
            edge_set.add((i, j.item()))

    # Create a list of directional edges (no bidirectionality)
    directional_edges = []
    for source, target in edge_set:
        directional_edges.append((target, source))

    # Convert to tensors for source and target indices
    # THIS PART IS CRUCIAL - switching source and target
    source_indices = torch.tensor([edge[0] for edge in directional_edges], device=distances.device)
    target_indices = torch.tensor([edge[1] for edge in directional_edges], device=distances.device)

    # Create the edge_index tensor
    edge_index = torch.stack([source_indices, target_indices], dim=0)

    # Get edge distances
    edge_distances = torch.zeros(len(directional_edges))
    for idx, (i, j) in enumerate(directional_edges):
        edge_distances[idx] = distances[i, j]

    # Compute Gaussian expanded edge features
    edge_attr = _compute_gaussian_expanded_distances(edge_distances)

    # Generate simple one-hot node features
    node_attr = torch.zeros(n_atoms, n_atoms)  # One-hot encoding (n_atoms × n_atoms)
    for i in range(n_atoms):
        node_attr[i, i] = 1.0

    # Create PyG Data object
    graph = Data(
        x=node_attr,
        pos=coords,  # Store original coords for visualization
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_atoms,
        directional_pairs=directional_edges  # Store original directional pairs
    )

    return graph, {
        'nn_indices': nn_indices,
        'distances': distance_values,
        'edge_set': edge_set,
        'directional_edges': directional_edges,
        'edge_index': edge_index
    }


def generate_sample_coordinates(n_atoms=20, noise_scale=0.3):
    """
    Generate 3D coordinates for testing.
    """
    # Create a regular arrangement of atoms (e.g., a simple cubic lattice)
    n_per_side = int(np.ceil(n_atoms ** (1 / 3)))
    grid_points = []

    for x in range(n_per_side):
        for y in range(n_per_side):
            for z in range(n_per_side):
                if len(grid_points) < n_atoms:
                    grid_points.append([x, y, z])

    coords = torch.tensor(grid_points, dtype=torch.float32)

    # Add some random noise to make it more interesting
    coords += torch.randn_like(coords) * noise_scale

    return coords


def create_networkx_graph(graph_data, info):
    """
    Create a NetworkX graph for visualization.
    """
    G = nx.DiGraph()  # Directed graph to show edge directions

    # Add nodes with positions
    for i in range(graph_data.num_nodes):
        G.add_node(i, pos=graph_data.pos[i].numpy())

    # Add edges with their actual direction from directional_pairs
    for source, target in info['directional_edges']:
        G.add_edge(source, target)

    # Add edges based on the edge_index (for comparison)
    edge_indices = graph_data.edge_index.t().numpy()
    for i, (source, target) in enumerate(edge_indices):
        G.add_edge(int(source), int(target), edge_index=True)

    return G


def visualize_graph(graph_data, info):
    """
    Create visualizations to understand edge directionality.
    """
    # Convert to NetworkX for visualization
    G = create_networkx_graph(graph_data, info)
    pos = nx.get_node_attributes(G, 'pos')

    # Plot original 3D arrangement
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("3D Node Positions")

    # Plot nodes
    ax1.scatter(
        graph_data.pos[:, 0],
        graph_data.pos[:, 1],
        graph_data.pos[:, 2],
        c='blue',
        s=100,
        alpha=0.8
    )

    # Add node labels
    for i in range(graph_data.num_nodes):
        ax1.text(
            graph_data.pos[i, 0],
            graph_data.pos[i, 1],
            graph_data.pos[i, 2],
            f"{i}",
            fontsize=8
        )

    # Plot 2D projection of directional edges (original neighbors)
    ax2 = fig.add_subplot(132)
    ax2.set_title("Original k-NN Graph\n(Directional)")

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos={i: pos[i][:2] for i in G.nodes()},  # Project to 2D
        node_size=300,
        node_color='lightblue',
        alpha=0.8
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G,
        pos={i: pos[i][:2] for i in G.nodes()},
        font_size=8
    )

    # Draw directional edges (source→target in the original k-NN sense)
    edge_list = [(s, t) for s, t in info['directional_edges']]
    nx.draw_networkx_edges(
        G.edge_subgraph(edge_list),
        pos={i: pos[i][:2] for i in G.nodes()},
        width=1.0,
        alpha=0.5,
        arrowsize=15,
        edge_color='blue'
    )

    # Plot 2D projection with actual edge_index direction (as used in PyG)
    ax3 = fig.add_subplot(133)
    ax3.set_title("PyG Edge Indices\n(Source→Target)")

    # Draw nodes again
    nx.draw_networkx_nodes(
        G,
        pos={i: pos[i][:2] for i in G.nodes()},
        node_size=300,
        node_color='lightblue',
        alpha=0.8
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G,
        pos={i: pos[i][:2] for i in G.nodes()},
        font_size=8
    )

    # Draw edges with PyG convention (edge_index[0]→edge_index[1])
    edge_list = [(int(s), int(t)) for s, t in graph_data.edge_index.t().numpy()]
    nx.draw_networkx_edges(
        G.edge_subgraph(edge_list),
        pos={i: pos[i][:2] for i in G.nodes()},
        width=1.0,
        alpha=0.5,
        arrowsize=15,
        edge_color='red'
    )

    plt.tight_layout()
    plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
    print("Graph visualization saved as 'graph_visualization.png'")

    # Generate detailed analysis
    generate_edge_analysis(graph_data, info)

    return fig


def generate_edge_analysis(graph_data, info):
    """
    Generate detailed analysis of edge directions.
    """
    n_atoms = graph_data.num_nodes
    n_neighbors = info['nn_indices'].shape[1]

    # Create a figure to visualize specific node neighborhoods
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Select a few nodes to analyze in detail
    sample_nodes = [0, 1, min(5, n_atoms - 1), min(10, n_atoms - 1)]

    for ax_idx, node_idx in enumerate(sample_nodes):
        if node_idx >= n_atoms:
            continue

        ax = axes[ax_idx]
        ax.set_title(f"Node {node_idx} Neighbors")

        # Get the node's k nearest neighbors
        neighbors = info['nn_indices'][node_idx].numpy()

        # Use a subset of nodes for clarity
        nodes_to_show = list(set([node_idx] + neighbors.tolist()))

        # Create a subgraph
        subG = nx.DiGraph()

        # Get positions for nodes
        for i in nodes_to_show:
            subG.add_node(i, pos=graph_data.pos[i].numpy()[:2])  # Use 2D projection

        # Add edges from original directed pairs
        for s, t in info['directional_edges']:
            if s == node_idx and t in nodes_to_show:
                subG.add_edge(s, t, color='blue', style='solid')

        # Add edges from edge_index
        edge_indices = graph_data.edge_index.t().numpy()
        for source, target in edge_indices:
            source, target = int(source), int(target)
            if target == node_idx and source in nodes_to_show:
                # This is an incoming edge to our node in PyG terms
                subG.add_edge(source, target, color='red', style='dashed')

        # Get positions
        pos = nx.get_node_attributes(subG, 'pos')

        # Draw center node
        nx.draw_networkx_nodes(
            subG,
            pos,
            nodelist=[node_idx],
            node_color='yellow',
            node_size=500,
            alpha=0.8,
            ax=ax
        )

        # Draw neighbor nodes
        nx.draw_networkx_nodes(
            subG,
            pos,
            nodelist=[n for n in nodes_to_show if n != node_idx],
            node_color='lightblue',
            node_size=300,
            alpha=0.8,
            ax=ax
        )

        # Draw node labels
        nx.draw_networkx_labels(
            subG,
            pos,
            font_size=10,
            ax=ax
        )

        # Draw original directed edges
        blue_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get('color') == 'blue']
        nx.draw_networkx_edges(
            subG,
            pos,
            edgelist=blue_edges,
            edge_color='blue',
            width=1.5,
            alpha=0.7,
            arrowstyle='-|>',
            arrowsize=15,
            ax=ax,
            label='Original Direction (kNN)'
        )

        # Draw PyG edge_index edges
        red_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get('color') == 'red']
        nx.draw_networkx_edges(
            subG,
            pos,
            edgelist=red_edges,
            edge_color='red',
            width=1.5,
            style='dashed',
            alpha=0.7,
            arrowstyle='-|>',
            arrowsize=15,
            ax=ax,
            label='PyG Direction (edge_index)'
        )

        # Add explanatory text
        neighbors_text = ", ".join([str(n) for n in neighbors])
        ax.text(
            0.5, -0.1,
            f"Node {node_idx} k-nearest neighbors: {neighbors_text}",
            transform=ax.transAxes,
            ha='center',
            fontsize=9
        )

        # Add legend
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('node_neighborhood_analysis.png', dpi=300, bbox_inches='tight')
    print("Node neighborhood analysis saved as 'node_neighborhood_analysis.png'")

    # Create a summary table
    print("\n===== Edge Direction Analysis =====")
    print(f"Total nodes: {n_atoms}")
    print(f"Neighbors per node (k): {n_neighbors}")
    print(f"Original directional edges: {len(info['directional_edges'])}")
    print(f"Edge index pairs: {graph_data.edge_index.shape[1]}")

    # Analyze edge reversal
    original_edges = set(info['directional_edges'])
    edge_index_pairs = set((int(s), int(t)) for s, t in graph_data.edge_index.t().numpy())

    # Check if edges are reversed
    reversed_count = 0
    for s, t in original_edges:
        if (t, s) in edge_index_pairs:
            reversed_count += 1

    print(
        f"Edges with reversed direction: {reversed_count} of {len(original_edges)} ({reversed_count / len(original_edges) * 100:.1f}%)")

    # Check if exactly reversed
    exact_reversal = len(original_edges) == len(edge_index_pairs) and reversed_count == len(original_edges)
    print(f"Perfect reversal of all edges: {'Yes' if exact_reversal else 'No'}")

    return {
        'total_nodes': n_atoms,
        'neighbors_per_node': n_neighbors,
        'original_edges': len(original_edges),
        'edge_index_pairs': len(edge_index_pairs),
        'reversed_count': reversed_count,
        'perfect_reversal': exact_reversal
    }


def main():
    """
    Main function to test graph creation and edge directions.
    """
    print("===== Graph Edge Direction Analysis =====\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate a small dataset
    n_atoms = 15  # Small enough to visualize clearly
    n_neighbors = 3

    print(f"Generating sample coordinates for {n_atoms} atoms with {n_neighbors} neighbors each")
    coords = generate_sample_coordinates(n_atoms)

    # Create graph
    print("Creating graph from coordinates")
    graph, info = create_graph_from_coords(coords, n_neighbors=n_neighbors)

    print(f"Graph created with {graph.num_nodes} nodes and {graph.edge_index.shape[1]} edges")

    # Analyze and visualize
    print("\nAnalyzing edge directions...")
    fig = visualize_graph(graph, info)

    # Print conclusions
    print("\n===== Key Findings =====")

    if info['edge_set']:
        source_example = next(iter(info['edge_set']))
        print(f"Original edge example: Node {source_example[0]} → Node {source_example[1]}")

        edge_index_example = graph.edge_index[:, 0].numpy()
        print(
            f"PyG edge_index example: edge_index[0, 0]={edge_index_example[0]} → edge_index[1, 0]={graph.edge_index[1, 0].item()}")

        # Check if directions are reversed
        edge_idx_first = (graph.edge_index[0, 0].item(), graph.edge_index[1, 0].item())
        is_reversed = edge_idx_first[0] == source_example[1] and edge_idx_first[1] == source_example[0]

        if is_reversed:
            print("\nVERIFIED: Edge directions ARE REVERSED in edge_index compared to original k-NN relationships")
            print("  - In the original k-NN: Node i's neighbors are nodes that i connects TO")
            print("  - In edge_index: The neighbors are the SOURCE nodes, connecting TO node i (the target)")
        else:
            print("\nNOTE: Edge directions have a different pattern than simple reversal")
            print("  - Further investigation recommended")

    # Draw PyTorch Geometric schema
    print("\nPyTorch Geometric Edge Direction Convention:")
    print("  edge_index[0] = source nodes (message senders)")
    print("  edge_index[1] = target nodes (message receivers)")

    return graph, info


if __name__ == "__main__":
    graph, info = main()
