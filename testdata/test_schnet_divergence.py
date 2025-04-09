import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import softmax

# Import the CFConv components from both implementations
from classic_schnet import CFConv as ClassicCFConv
from pygv.encoder.schnet_wo_embed import CFConv as PygCFConv
import torch.nn.functional as F

def trace_cfconv_step_by_step():
    """
    Trace through each step of both CFConv implementations to identify where they diverge.
    """
    print("=== Detailed Step-by-Step Comparison of CFConv Implementations ===\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters for testing
    edge_channels = 16  # n_gaussians in classic / edge_channels in PyG
    hidden_channels = 32  # n_filters in classic / hidden_channels in PyG
    node_dim = hidden_channels  # Use same dimension to avoid projection

    print(f"Parameters: edge_channels={edge_channels}, hidden_channels={hidden_channels}, node_dim={node_dim}")

    # Create test instances with IDENTICAL weight initialization
    # This requires modifying the initialization to use the same seed
    classic_cfconv = ClassicCFConv(
        n_gaussians=edge_channels,
        n_filters=hidden_channels,
        activation=torch.nn.Tanh(),
        use_attention=False  # Test without attention first to isolate core differences
    )

    pyg_cfconv = PygCFConv(
        in_channels=node_dim,
        out_channels=hidden_channels,
        edge_channels=edge_channels,
        hidden_channels=hidden_channels,
        activation='tanh',
        use_attention=False
    )

    # Create small test data for precise comparison
    batch_size = 1
    n_atoms = 3  # Tiny system for detailed inspection
    n_neighbors = 2

    # 1. Generate tensors with specific values for comparison
    print("\n1. Creating test data with controlled values...")

    # Create identical starting features
    features_classic = torch.ones((batch_size, n_atoms, node_dim))
    features_classic[0, 0, :] = 1.0  # First atom features
    features_classic[0, 1, :] = 2.0  # Second atom features
    features_classic[0, 2, :] = 3.0  # Third atom features

    # Create identical edge attributes
    rbf_expansion = torch.ones((batch_size, n_atoms, n_neighbors, edge_channels))
    # Add some variation
    for i in range(edge_channels):
        rbf_expansion[0, :, :, i] = (i + 1) * 0.1

    # Simple neighbor list: atom 0 connects to 1,2; atom 1 connects to 0,2; atom 2 connects to 0,1
    neighbor_list = torch.zeros((batch_size, n_atoms, n_neighbors), dtype=torch.long)
    neighbor_list[0, 0, 0] = 1  # Atom 0 -> Atom 1
    neighbor_list[0, 0, 1] = 2  # Atom 0 -> Atom 2
    neighbor_list[0, 1, 0] = 0  # Atom 1 -> Atom 0
    neighbor_list[0, 1, 1] = 2  # Atom 1 -> Atom 2
    neighbor_list[0, 2, 0] = 0  # Atom 2 -> Atom 0
    neighbor_list[0, 2, 1] = 1  # Atom 2 -> Atom 1

    # Create corresponding PyG data
    x_pyg = torch.stack([
        features_classic[0, 0],  # Atom 0
        features_classic[0, 1],  # Atom 1
        features_classic[0, 2],  # Atom 2
    ], dim=0)

    # Create edge index matching the neighbor list
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],  # Source nodes
        [1, 2, 0, 2, 0, 1]  # Target nodes
    ], dtype=torch.long)

    # Create edge attributes matching the rbf_expansion
    edge_attr = torch.stack([
        rbf_expansion[0, 0, 0],  # Edge 0->1
        rbf_expansion[0, 0, 1],  # Edge 0->2
        rbf_expansion[0, 1, 0],  # Edge 1->0
        rbf_expansion[0, 1, 1],  # Edge 1->2
        rbf_expansion[0, 2, 0],  # Edge 2->0
        rbf_expansion[0, 2, 1],  # Edge 2->1
    ], dim=0)

    # Print shapes
    print(f"Classic features shape: {features_classic.shape}")
    print(f"Classic RBF expansion shape: {rbf_expansion.shape}")
    print(f"Classic neighbor list shape: {neighbor_list.shape}")
    print(f"PyG features shape: {x_pyg.shape}")
    print(f"PyG edge index shape: {edge_index.shape}")
    print(f"PyG edge attr shape: {edge_attr.shape}")

    # 2. Extract and compare weights
    print("\n2. Comparing model parameters...")

    # Extract filter generator weights from classic
    classic_filter_weights1 = list(classic_cfconv.filter_generator[0].parameters())[0].data.clone()
    classic_filter_weights2 = list(classic_cfconv.filter_generator[2].parameters())[0].data.clone()
    classic_filter_bias1 = list(classic_cfconv.filter_generator[0].parameters())[1].data.clone()
    classic_filter_bias2 = list(classic_cfconv.filter_generator[2].parameters())[1].data.clone()

    # Extract filter network weights from PyG
    pyg_filter_weights1 = list(pyg_cfconv.filter_network.lins[0].parameters())[0].data.clone()
    pyg_filter_weights2 = list(pyg_cfconv.filter_network.lins[1].parameters())[0].data.clone()
    pyg_filter_bias1 = list(pyg_cfconv.filter_network.lins[0].parameters())[1].data.clone()
    pyg_filter_bias2 = list(pyg_cfconv.filter_network.lins[1].parameters())[1].data.clone()

    # Initialize models with identical weights (optional)
    with torch.no_grad():
        # Set weights to be identical (small fixed values for easier debugging)
        init_weight1 = torch.ones_like(classic_filter_weights1) * 0.1
        init_weight2 = torch.ones_like(classic_filter_weights2) * 0.2
        init_bias1 = torch.ones_like(classic_filter_bias1) * 0.01
        init_bias2 = torch.ones_like(classic_filter_bias2) * 0.02

        # Classic model weights
        classic_cfconv.filter_generator[0].weight.data.copy_(init_weight1)
        classic_cfconv.filter_generator[2].weight.data.copy_(init_weight2)
        classic_cfconv.filter_generator[0].bias.data.copy_(init_bias1)
        classic_cfconv.filter_generator[2].bias.data.copy_(init_bias2)

        # PyG model weights
        pyg_cfconv.filter_network.lins[0].weight.data.copy_(init_weight1)
        pyg_cfconv.filter_network.lins[1].weight.data.copy_(init_weight2)
        pyg_cfconv.filter_network.lins[0].bias.data.copy_(init_bias1)
        pyg_cfconv.filter_network.lins[1].bias.data.copy_(init_bias2)

    print("  Models initialized with identical weights.")

    # 3. TRACE - Step by step execution
    print("\n3. Tracing execution step-by-step...")

    # CLASSIC implementation trace
    print("\nCLASSIC Implementation Trace:")

    # Step 3.1: Edge feature normalization
    print("3.1 Edge feature normalization:")
    rbf_norm = rbf_expansion / (rbf_expansion.norm(dim=-1, keepdim=True) + 1e-8)
    print(f"  Normalized RBF expansion shape: {rbf_norm.shape}")
    print(f"  First normalized value: {rbf_norm[0, 0, 0, 0].item():.6f}")

    # Step 3.2: Filter generation
    print("3.2 Filter generation:")
    # First layer
    layer1_out = torch.matmul(rbf_norm.to(torch.float32), classic_cfconv.filter_generator[0].weight.t())
    layer1_out += classic_cfconv.filter_generator[0].bias
    # Activation
    layer1_act = torch.tanh(layer1_out)
    # Second layer
    conv_filter = torch.matmul(layer1_act, classic_cfconv.filter_generator[2].weight.t())
    conv_filter += classic_cfconv.filter_generator[2].bias
    print(f"  Filter shape: {conv_filter.shape}")
    print(f"  First filter value: {conv_filter[0, 0, 0, 0].item():.6f}")

    # Step 3.3: Gather neighbor features
    print("3.3 Gather neighbor features:")
    neighbor_list_reshaped = neighbor_list.reshape(-1, n_atoms * n_neighbors, 1)
    neighbor_list_expanded = neighbor_list_reshaped.expand(-1, -1, features_classic.size(2))
    neighbor_features = torch.gather(features_classic, 1, neighbor_list_expanded)
    neighbor_features = neighbor_features.reshape(batch_size, n_atoms, n_neighbors, -1)
    print(f"  Neighbor features shape: {neighbor_features.shape}")
    print(f"  First neighbor feature: {neighbor_features[0, 0, 0, 0].item():.6f}")

    # Step 3.4: Apply filters
    print("3.4 Apply filters:")
    conv_features = neighbor_features * conv_filter
    print(f"  Convolved features shape: {conv_features.shape}")
    print(f"  First convolved feature: {conv_features[0, 0, 0, 0].item():.6f}")

    # Step 3.5: Aggregation
    print("3.5 Aggregation:")
    classic_aggregated = conv_features.sum(dim=2)
    print(f"  Aggregated features shape: {classic_aggregated.shape}")
    print(f"  First aggregated feature: {classic_aggregated[0, 0, 0].item():.6f}")
    print(f"  First node aggregated features: {classic_aggregated[0, 0, :5].tolist()}")

    # PyG implementation trace
    print("\nPyG Implementation Trace:")

    # Step 4.1: Edge feature normalization
    print("4.1 Edge feature normalization:")
    edge_attr_norm = edge_attr / (edge_attr.norm(dim=1, keepdim=True) + 1e-8)
    print(f"  Normalized edge attr shape: {edge_attr_norm.shape}")
    print(f"  First normalized value: {edge_attr_norm[0, 0].item():.6f}")

    # Step 4.2: Filter generation
    print("4.2 Filter generation:")
    # First layer
    layer1_out_pyg = torch.matmul(edge_attr_norm, pyg_cfconv.filter_network.lins[0].weight.t())
    layer1_out_pyg += pyg_cfconv.filter_network.lins[0].bias
    # Activation
    layer1_act_pyg = torch.tanh(layer1_out_pyg)
    # Second layer
    layer2_out_pyg = torch.matmul(layer1_act_pyg, pyg_cfconv.filter_network.lins[1].weight.t())
    layer2_out_pyg += pyg_cfconv.filter_network.lins[1].bias
    # Activation (PyG uses activation after both layers while classic doesn't use it after second)
    edge_weights_pyg = layer2_out_pyg
    print(f"  Edge weights shape: {edge_weights_pyg.shape}")
    print(f"  First edge weight: {edge_weights_pyg[0, 0].item():.6f}")

    # Step 4.3: Message computation
    print("4.3 Message computation:")
    # Get source node features
    src_idx = edge_index[0]
    x_j = x_pyg[src_idx]
    print(f"  Source node features shape: {x_j.shape}")
    print(f"  First source feature: {x_j[0, 0].item():.6f}")

    # Apply filters to source features
    messages_pyg = x_j * edge_weights_pyg
    print(f"  Messages shape: {messages_pyg.shape}")
    print(f"  First message: {messages_pyg[0, 0].item():.6f}")

    # Step 4.4: Aggregation
    print("4.4 Aggregation:")
    # Manually reproduce the aggregation step
    tgt_idx = edge_index[1]
    agg_dict = {}
    for i, (src, tgt) in enumerate(zip(src_idx, tgt_idx)):
        if tgt.item() not in agg_dict:
            agg_dict[tgt.item()] = []
        agg_dict[tgt.item()].append(messages_pyg[i])

    # Aggregate messages for each target node
    pyg_aggregated = torch.zeros_like(x_pyg)
    for tgt, msgs in agg_dict.items():
        pyg_aggregated[tgt] = torch.stack(msgs).sum(dim=0)

    print(f"  Aggregated features shape: {pyg_aggregated.shape}")
    print(f"  First aggregated feature: {pyg_aggregated[0, 0].item():.6f}")
    print(f"  First node aggregated features: {pyg_aggregated[0, :5].tolist()}")

    # 5. Compare outputs
    print("\n5. Comparing final outputs:")

    # Use the real functions
    with torch.no_grad():
        classic_output, _ = classic_cfconv(features_classic, rbf_expansion, neighbor_list)
        pyg_output, _ = pyg_cfconv(x_pyg, edge_index, edge_attr)

    # Format PyG output to match classic shape for comparison
    pyg_output_reshaped = torch.zeros_like(classic_output)
    pyg_output_reshaped[0, 0] = pyg_output[0]
    pyg_output_reshaped[0, 1] = pyg_output[1]
    pyg_output_reshaped[0, 2] = pyg_output[2]

    # Compare outputs
    print(f"Classic output shape: {classic_output.shape}")
    print(f"PyG output shape: {pyg_output.shape}")
    print(f"Reshaped PyG output shape: {pyg_output_reshaped.shape}")

    # Compare first node outputs in detail
    print("\nDetailed comparison for first node:")
    print(f"Classic output: {classic_output[0, 0, :5].tolist()}")
    print(f"PyG output: {pyg_output[0, :5].tolist()}")

    # Calculate differences
    diff = torch.abs(classic_output - pyg_output_reshaped)
    print(f"\nDifference statistics:")
    print(f"  Mean absolute difference: {diff.mean().item():.6f}")
    print(f"  Max absolute difference: {diff.max().item():.6f}")
    print(f"  First node difference: {diff[0, 0, :5].tolist()}")

    # 6. Summary of differences
    print("\n6. Key differences identified:")
    print("  a) Normalization dimension: classic uses dim=-1, PyG uses dim=1")
    print(
        "  b) Filter network in PyG applies tanh activation after BOTH layers, while classic only does after the first")
    print("  c) Message passing structure: classic manually gathers neighbor features, PyG uses edge_index")

    # 7. Test with attention and compare attention weights
    print("\n7. Testing with attention enabled:")

    # Create new instances with attention
    classic_cfconv_attn = ClassicCFConv(
        n_gaussians=edge_channels,
        n_filters=hidden_channels,
        activation=torch.nn.Tanh(),
        use_attention=True
    )

    pyg_cfconv_attn = PygCFConv(
        in_channels=node_dim,
        out_channels=hidden_channels,
        edge_channels=edge_channels,
        hidden_channels=hidden_channels,
        activation='tanh',
        use_attention=True
    )

    # Initialize attention vectors with identical weights
    with torch.no_grad():
        init_attn = torch.ones((hidden_channels, 1)) * 0.1
        classic_cfconv_attn.nbr_filter.data.copy_(init_attn)
        pyg_cfconv_attn.attention_vector.data.copy_(init_attn)

        # Also initialize filter networks with the same weights
        classic_cfconv_attn.filter_generator[0].weight.data.copy_(init_weight1)
        classic_cfconv_attn.filter_generator[2].weight.data.copy_(init_weight2)
        classic_cfconv_attn.filter_generator[0].bias.data.copy_(init_bias1)
        classic_cfconv_attn.filter_generator[2].bias.data.copy_(init_bias2)

        pyg_cfconv_attn.filter_network.lins[0].weight.data.copy_(init_weight1)
        pyg_cfconv_attn.filter_network.lins[1].weight.data.copy_(init_weight2)
        pyg_cfconv_attn.filter_network.lins[0].bias.data.copy_(init_bias1)
        pyg_cfconv_attn.filter_network.lins[1].bias.data.copy_(init_bias2)

    # Run with attention
    with torch.no_grad():
        classic_output_attn, classic_attention = classic_cfconv_attn(features_classic, rbf_expansion, neighbor_list)
        pyg_output_attn, pyg_attention = pyg_cfconv_attn(x_pyg, edge_index, edge_attr)

    # Compare attention mechanisms
    print("\nAttention comparison:")
    print(f"Classic attention shape: {classic_attention.shape}")
    print(f"PyG attention shape: {None if pyg_attention is None else pyg_attention.shape}")

    # Compare attention mechanisms
    print("\nAttention comparison:")
    print(f"Classic attention shape: {classic_attention.shape}")
    print(f"PyG attention shape: {None if pyg_attention is None else pyg_attention.shape}")

    if pyg_attention is not None:
        # Get edge indices
        sources, targets = edge_index

        # Print raw PyG attention values for all edges
        print("\nRaw PyG attention values for all edges:")
        for edge_idx, (src, tgt, att) in enumerate(zip(sources, targets, pyg_attention)):
            print(f"  Edge {edge_idx}: {src.item()} → {tgt.item()}: {att.item():.6f}")

        # Group and normalize by target node (PyG's approach)
        print("\nNormalizing by TARGET node (PyG's approach):")
        target_normalized = torch.zeros_like(pyg_attention)

        for target_node in range(3):  # 3 nodes in your test
            # Find all edges where this node is the target
            target_mask = (targets == target_node)
            if not target_mask.any():
                continue

            # Get indices of edges pointing to this target
            target_edge_indices = torch.where(target_mask)[0]

            # Get attention scores for these edges
            target_attention = pyg_attention[target_mask]
            print(f"  Target node {target_node} incoming edges: {target_edge_indices.tolist()}")
            print(f"  Raw attention values: {target_attention.tolist()}")

            # Apply softmax normalization
            norm_attention = F.softmax(target_attention, dim=0)
            print(f"  Normalized attention: {norm_attention.tolist()}")

            # Store normalized values
            target_normalized[target_mask] = norm_attention

        # Group and normalize by source node (classic's approach)
        print("\nNormalizing by SOURCE node (classic's approach):")
        source_normalized = torch.zeros_like(pyg_attention)

        for source_node in range(3):  # 3 nodes in your test
            # Find all edges where this node is the source
            source_mask = (sources == source_node)
            if not source_mask.any():
                continue

            # Get indices of edges from this source
            source_edge_indices = torch.where(source_mask)[0]

            # Get attention scores for these edges
            source_attention = pyg_attention[source_mask]
            print(f"  Source node {source_node} outgoing edges: {source_edge_indices.tolist()}")
            print(f"  Raw attention values: {source_attention.tolist()}")

            # Apply softmax normalization
            norm_attention = F.softmax(source_attention, dim=0)
            print(f"  Normalized attention: {norm_attention.tolist()}")

            # Store normalized values
            source_normalized[source_mask] = norm_attention

        # Reshape PyG's normalized attention (by source) to classic format
        pyg_reshaped = torch.zeros_like(classic_attention)

        for source_node in range(3):
            # Get outgoing edges from this source
            source_mask = (sources == source_node)
            source_edges = torch.where(source_mask)[0]

            # Map normalized values to classic format
            for i, edge_idx in enumerate(source_edges):
                pyg_reshaped[0, source_node, i] = source_normalized[edge_idx]

        # Compare classic with PyG (normalized by source and reshaped)
        print("\nDetailed comparison of attention weights (normalized by source):")
        for node_idx in range(3):
            print(f"\nNode {node_idx} attention weights:")
            print(f"  Classic normalized: {classic_attention[0, node_idx].tolist()}")
            print(f"  PyG normalized by source: {pyg_reshaped[0, node_idx].tolist()}")

            # Calculate differences
            diff = torch.abs(classic_attention[0, node_idx] - pyg_reshaped[0, node_idx])
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"  Max difference: {max_diff:.8f}")
            print(f"  Mean difference: {mean_diff:.8f}")

        # Overall difference
        att_diff = torch.abs(classic_attention - pyg_reshaped)
        print(f"\nOverall attention difference:")
        print(f"  Mean difference: {att_diff.mean().item():.8f}")
        print(f"  Max difference: {att_diff.max().item():.8f}")

    # 8. Visualizations to help understand differences
    print("\n8. Creating visualizations...")

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Compare normalized edge features
    axs[0, 0].plot(rbf_norm[0, 0, 0].detach().numpy(), label='Classic (first edge)')
    axs[0, 0].plot(edge_attr_norm[0].detach().numpy(), label='PyG (first edge)')
    axs[0, 0].set_title('Edge Feature Normalization')
    axs[0, 0].set_xlabel('Feature Dimension')
    axs[0, 0].set_ylabel('Normalized Value')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Plot 2: Compare filter weights
    axs[0, 1].plot(conv_filter[0, 0, 0].detach().numpy(), label='Classic Filter')
    axs[0, 1].plot(edge_weights_pyg[0].detach().numpy(), label='PyG Filter')
    axs[0, 1].set_title('Generated Filter Weights')
    axs[0, 1].set_xlabel('Feature Dimension')
    axs[0, 1].set_ylabel('Weight Value')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: Compare outputs
    axs[1, 0].plot(classic_output[0, 0].detach().numpy(), label='Classic Output (node 0)')
    axs[1, 0].plot(pyg_output[0].detach().numpy(), label='PyG Output (node 0)')
    axs[1, 0].set_title('Output Comparison (First Node)')
    axs[1, 0].set_xlabel('Feature Dimension')
    axs[1, 0].set_ylabel('Output Value')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Plot 4: Difference heatmap
    im = axs[1, 1].imshow(diff[0].detach().numpy(), cmap='hot')
    axs[1, 1].set_title('Output Difference Heatmap')
    axs[1, 1].set_xlabel('Feature Dimension')
    axs[1, 1].set_ylabel('Node Index')
    plt.colorbar(im, ax=axs[1, 1], label='Absolute Difference')

    plt.tight_layout()
    plt.savefig('cfconv_detailed_comparison.png')
    print("Visualization saved as 'cfconv_detailed_comparison.png'")


if __name__ == "__main__":
    trace_cfconv_step_by_step()
