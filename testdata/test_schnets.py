import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter

# Import the CFConv components from both implementations
from classic_schnet import CFConv as ClassicCFConv
from pygv.encoder.schnet_wo_embed import CFConv as PygCFConv


def test_cfconv_with_output_comparison():
    """
    Compare the CFConv components from both implementations,
    including detailed analysis of their output and attention values.
    """
    print("\n=== Testing CFConv Components with Output Comparison ===")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters for testing
    edge_channels = 16  # n_gaussians in classic / edge_channels in PyG
    hidden_channels = 32  # n_filters in classic / hidden_channels in PyG
    node_dim = hidden_channels  # Important: use hidden_channels to ensure compatibility with classic

    print(f"Parameters: edge_channels={edge_channels}, hidden_channels={hidden_channels}, node_dim={node_dim}")

    # Create test instances
    classic_cfconv = ClassicCFConv(
        n_gaussians=edge_channels,
        n_filters=hidden_channels,
        activation=torch.nn.Tanh(),
        use_attention=False
    )

    pyg_cfconv = PygCFConv(
        in_channels=node_dim,
        out_channels=hidden_channels,
        edge_channels=edge_channels,
        hidden_channels=hidden_channels,
        activation='tanh',
        use_attention=False
    )

    # Print parameter counts
    classic_params = sum(p.numel() for p in classic_cfconv.parameters())
    pyg_params = sum(p.numel() for p in pyg_cfconv.parameters())
    print(f"Classic CFConv parameters: {classic_params}")
    print(f"PyG CFConv parameters: {pyg_params}")

    # Generate test data
    print("\nGenerating test data...")

    # Set up a simple scenario with 2 graphs
    batch_size = 32
    n_atoms = 300  # Start with a small number for easier comparison
    n_neighbors = 20

    # For classic implementation
    features_classic = torch.randn(batch_size, n_atoms, hidden_channels)
    rbf_expansion = torch.randn(batch_size, n_atoms, n_neighbors, edge_channels)
    neighbor_list = torch.zeros(batch_size, n_atoms, n_neighbors, dtype=torch.long)

    # Fill neighbor list systematically for easier analysis
    for b in range(batch_size):
        for i in range(n_atoms):
            for j in range(n_neighbors):
                # Simple neighbor assignment pattern: (i+j) % n_atoms
                neighbor_list[b, i, j] = (i + j + 1) % n_atoms

    # For PyG implementation
    # Create corresponding PyG data
    pyg_nodes = []
    pyg_edges_src = []
    pyg_edges_dst = []
    pyg_edge_attr = []

    # Build PyG data manually to ensure exact correspondence with classic data
    for b in range(batch_size):
        for i in range(n_atoms):
            # Add node with its features
            pyg_nodes.append(features_classic[b, i])

            # Add edges for this node
            for j in range(n_neighbors):
                neighbor = neighbor_list[b, i, j].item()

                # Source -> destination edge
                pyg_edges_src.append(i + b * n_atoms)  # Offset by batch
                pyg_edges_dst.append(neighbor + b * n_atoms)  # Offset by batch

                # Edge attributes from rbf_expansion
                pyg_edge_attr.append(rbf_expansion[b, i, j])

    # Convert to PyG tensor format
    x_pyg = torch.stack(pyg_nodes)
    edge_index = torch.stack([torch.tensor(pyg_edges_src), torch.tensor(pyg_edges_dst)])
    edge_attr = torch.stack(pyg_edge_attr)

    print(f"Classic features shape: {features_classic.shape}")
    print(f"Classic RBF expansion shape: {rbf_expansion.shape}")
    print(f"Classic neighbor list shape: {neighbor_list.shape}")
    print(f"PyG features shape: {x_pyg.shape}")
    print(f"PyG edge index shape: {edge_index.shape}")
    print(f"PyG edge attr shape: {edge_attr.shape}")

    # Run forward passes
    print("\nRunning forward passes...")

    # Classic forward pass
    classic_output, classic_attention = classic_cfconv(features_classic, rbf_expansion, neighbor_list)
    print(f"Classic CFConv output shape: {classic_output.shape}")
    print(f"Classic CFConv attention shape: {classic_attention.shape if classic_attention is not None else None}")

    # PyG forward pass
    pyg_output, pyg_attention = pyg_cfconv(x_pyg, edge_index, edge_attr)
    print(f"PyG CFConv output shape: {pyg_output.shape}")
    print(f"PyG CFConv attention shape: {pyg_attention.shape if pyg_attention is not None else None}")

    # Convert PyG output to classic format for comparison
    pyg_output_reshaped = torch.zeros_like(classic_output)
    for b in range(batch_size):
        for i in range(n_atoms):
            node_idx = i + b * n_atoms
            pyg_output_reshaped[b, i] = pyg_output[node_idx]

    # Compare outputs
    print("\n=== Output Comparison ===")
    output_diff = torch.abs(classic_output - pyg_output_reshaped)
    output_mean_diff = output_diff.mean().item()
    output_max_diff = output_diff.max().item()
    output_rel_diff = output_diff.mean() / (torch.abs(classic_output).mean() + 1e-8)

    print(f"Mean absolute difference: {output_mean_diff:.6f}")
    print(f"Max absolute difference: {output_max_diff:.6f}")
    print(f"Relative difference: {output_rel_diff.item():.6f}")

    # Compare attention values (if available)
    if classic_attention is not None and pyg_attention is not None:
        print("\n=== Attention Comparison ===")

        # Need to reformat PyG attention to match classic format
        # PyG attention is a flat vector of length (total edges)
        # Classic attention is [batch_size, n_atoms, n_neighbors]
        pyg_attention_reshaped = torch.zeros_like(classic_attention)

        edge_idx = 0
        for b in range(batch_size):
            for i in range(n_atoms):
                for j in range(n_neighbors):
                    pyg_attention_reshaped[b, i, j] = pyg_attention[edge_idx]
                    edge_idx += 1

        # Compare attention values
        attention_diff = torch.abs(classic_attention - pyg_attention_reshaped)
        attention_mean_diff = attention_diff.mean().item()
        attention_max_diff = attention_diff.max().item()
        attention_rel_diff = attention_diff.mean() / (torch.abs(classic_attention).mean() + 1e-8)

        print(f"Attention mean absolute difference: {attention_mean_diff:.6f}")
        print(f"Attention max absolute difference: {attention_max_diff:.6f}")
        print(f"Attention relative difference: {attention_rel_diff.item():.6f}")

        # Visualize attention patterns for the first graph
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Classic Implementation Attention")
        plt.imshow(classic_attention[0].detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.xlabel("Neighbor Index")
        plt.ylabel("Node Index")

        plt.subplot(1, 2, 2)
        plt.title("PyG Implementation Attention")
        plt.imshow(pyg_attention_reshaped[0].detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.xlabel("Neighbor Index")
        plt.ylabel("Node Index")

        plt.tight_layout()
        plt.savefig("attention_comparison.png")
        print("Attention visualization saved as 'attention_comparison.png'")

    # Timing comparison
    print("\nRunning timing comparison...")
    n_runs = 20

    # Warm up
    for _ in range(5):
        _ = classic_cfconv(features_classic, rbf_expansion, neighbor_list)
        _ = pyg_cfconv(x_pyg, edge_index, edge_attr)

    # Classic timing
    classic_times = []
    for _ in range(n_runs):
        start = time.time()
        _ = classic_cfconv(features_classic, rbf_expansion, neighbor_list)
        classic_times.append(time.time() - start)

    # PyG timing
    pyg_times = []
    for _ in range(n_runs):
        start = time.time()
        _ = pyg_cfconv(x_pyg, edge_index, edge_attr)
        pyg_times.append(time.time() - start)

    # Report results
    avg_classic = sum(classic_times) / n_runs
    avg_pyg = sum(pyg_times) / n_runs
    speedup = avg_classic / avg_pyg if avg_pyg > 0 else float('inf')

    print(f"Classic CFConv avg time: {avg_classic:.6f}s")
    print(f"PyG CFConv avg time: {avg_pyg:.6f}s")
    print(f"Speedup factor: {speedup:.2f}x")

    # Create performance comparison plot
    plt.figure(figsize=(10, 6))

    # Performance bar chart
    plt.subplot(1, 2, 1)
    plt.bar(['Classic', 'PyG'], [avg_classic, avg_pyg])
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison')

    # Output difference visualization
    plt.subplot(1, 2, 2)
    diff_matrix = output_diff[0].detach().numpy()  # First batch
    plt.imshow(diff_matrix, cmap='hot', aspect='auto')
    plt.colorbar(label='Absolute Difference')
    plt.title('Output Difference (First Batch)')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Node Index')

    plt.tight_layout()
    plt.savefig('cfconv_comparison.png')
    print("Performance comparison saved as 'cfconv_comparison.png'")

    # Export sample values for detailed inspection
    print("\n=== Sample Values for Further Inspection ===")
    sample_node = 0
    sample_feature = 0

    print(f"Node {sample_node}, Feature {sample_feature}:")
    print(f"  Classic output: {classic_output[0, sample_node, sample_feature].item():.6f}")
    print(f"  PyG output: {pyg_output_reshaped[0, sample_node, sample_feature].item():.6f}")
    print(f"  Difference: {output_diff[0, sample_node, sample_feature].item():.6f}")

    if classic_attention is not None:
        print(f"\nAttention for Node {sample_node}, Neighbor 0:")
        print(f"  Classic attention: {classic_attention[0, sample_node, 0].item():.6f}")
        print(f"  PyG attention: {pyg_attention_reshaped[0, sample_node, 0].item():.6f}")

    print("\nTest completed.")


if __name__ == "__main__":
    test_cfconv_with_output_comparison()
