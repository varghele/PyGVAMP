import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter

# Import the CFConv components from both implementations
from classic_schnet import CFConv as ClassicCFConv
from pygv.encoder.schnet_wo_embed_v2 import CFConv as PygCFConv


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

    # After initializing both models
    print("\nForcing identical weights...")

    # Verify layer counts match
    classic_layers = [l for l in classic_cfconv.filter_generator if isinstance(l, torch.nn.Linear)]
    pyg_layers = pyg_cfconv.filter_network.lins

    if len(classic_layers) == len(pyg_layers):
        print(f"Both models have {len(classic_layers)} linear layers - copying weights")

        # Copy weights from classic to PyG
        for i, (classic_layer, pyg_layer) in enumerate(zip(classic_layers, pyg_layers)):
            pyg_layer.weight.data.copy_(classic_layer.weight.data)
            pyg_layer.bias.data.copy_(classic_layer.bias.data)
            print(f"  Copied layer {i} weights")
    else:
        print(f"ERROR: Layer count mismatch - Classic: {len(classic_layers)}, PyG: {len(pyg_layers)}")

    # Generate test data (same as before)
    batch_size = 32
    n_atoms = 300
    n_neighbors = 20

    # For classic implementation
    features_classic = torch.randn(batch_size, n_atoms, hidden_channels)
    rbf_expansion = torch.randn(batch_size, n_atoms, n_neighbors, edge_channels)
    neighbor_list = torch.zeros(batch_size, n_atoms, n_neighbors, dtype=torch.long)

    # Fill neighbor list systematically
    for b in range(batch_size):
        for i in range(n_atoms):
            for j in range(n_neighbors):
                neighbor_list[b, i, j] = (i + j + 1) % n_atoms

    # For PyG implementation
    pyg_nodes = []
    pyg_edges_src = []
    pyg_edges_dst = []
    pyg_edge_attr = []

    # Build PyG data manually to ensure proper bidirectional edges
    for b in range(batch_size):
        for i in range(n_atoms):
            pyg_nodes.append(features_classic[b, i])
            for j in range(n_neighbors):
                neighbor = neighbor_list[b, i, j].item()

                # Create bidirectional edges (both source->target and target->source)
                # Edge 1: source->target (i->neighbor)
                pyg_edges_src.append(neighbor + b * n_atoms)  # Neighbor is the source
                pyg_edges_dst.append(i + b * n_atoms)  # i is the target
                pyg_edge_attr.append(rbf_expansion[b, i, j])

                # Edge 2: target->source (neighbor->i) to ensure bidirectionality
                # Note: You'll need to ensure proper edge attributes are passed for this direction

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

    # ===== TRACING STEP 1: NORMALIZATION OF EDGE ATTRIBUTES =====
    print("\n===== STAGE 1: EDGE ATTRIBUTE NORMALIZATION =====")

    # Classic implementation
    rbf_norm_classic = rbf_expansion / (rbf_expansion.norm(dim=-1, keepdim=True) + 1e-8)
    print(f"Classic RBF norm shape: {rbf_norm_classic.shape}")
    print(
        f"Classic RBF norm stats - mean: {rbf_norm_classic.mean().item():.6f}, std: {rbf_norm_classic.std().item():.6f}")
    print(f"Classic RBF norm sample [0,0,0]: {rbf_norm_classic[0, 0, 0, 0]:.6f}")

    # PyG implementation
    edge_attr_norm = edge_attr / (edge_attr.norm(dim=-1, keepdim=True) + 1e-8)
    print(f"PyG edge attr norm shape: {edge_attr_norm.shape}")
    print(
        f"PyG edge attr norm stats - mean: {edge_attr_norm.mean().item():.6f}, std: {edge_attr_norm.std().item():.6f}")
    print(f"PyG edge attr norm sample [0]: {edge_attr_norm[0, 0]:.6f}")

    # Check normalization difference
    classic_norm = rbf_norm_classic.reshape(-1, edge_channels)
    norm_diff = torch.abs(classic_norm[:edge_attr_norm.size(0)] - edge_attr_norm).mean().item()
    print(f"Normalization difference (first {edge_attr_norm.size(0)} edges): {norm_diff:.6f}")

    # ===== TRACING STEP 2: FILTER NETWORK OUTPUT =====
    print("\n===== STAGE 2: FILTER NETWORK OUTPUT =====")

    # Classic implementation
    conv_filter_classic = classic_cfconv.filter_generator(rbf_norm_classic.to(torch.float32))
    print(f"Classic conv filter shape: {conv_filter_classic.shape}")
    print(
        f"Classic conv filter stats - mean: {conv_filter_classic.mean().item():.6f}, std: {conv_filter_classic.std().item():.6f}")
    print(f"Classic conv filter sample [0,0,0,0]: {conv_filter_classic[0, 0, 0, 0]:.6f}")

    # PyG implementation
    conv_filter_pyg = pyg_cfconv.filter_network(edge_attr_norm)
    print(f"PyG conv filter shape: {conv_filter_pyg.shape}")
    print(f"PyG conv filter stats - mean: {conv_filter_pyg.mean().item():.6f}, std: {conv_filter_pyg.std().item():.6f}")
    print(f"PyG conv filter sample [0,0]: {conv_filter_pyg[0, 0]:.6f}")

    # Flatten classic filters for comparison (first batch only)
    classic_filter_flat = conv_filter_classic[0].reshape(-1, hidden_channels)
    flat_size = min(classic_filter_flat.size(0), conv_filter_pyg.size(0))
    filter_diff = torch.abs(classic_filter_flat[:flat_size] - conv_filter_pyg[:flat_size]).mean().item()
    print(f"Filter network output difference (first {flat_size} entries): {filter_diff:.6f}")

    # Show weights of filter network for both implementations
    print("\nFilter Network Weight Comparison:")

    # Classic filter network weights (first layer)
    if hasattr(classic_cfconv.filter_generator[0], 'weight'):
        classic_w1 = classic_cfconv.filter_generator[0].weight
        print(f"Classic filter network first layer weight shape: {classic_w1.shape}")
        print(
            f"Classic filter network first layer weight stats - mean: {classic_w1.mean().item():.6f}, std: {classic_w1.std().item():.6f}")

    # PyG filter network weights (first layer)
    if hasattr(pyg_cfconv.filter_network.lins[0], 'weight'):
        pyg_w1 = pyg_cfconv.filter_network.lins[0].weight
        print(f"PyG filter network first layer weight shape: {pyg_w1.shape}")
        print(
            f"PyG filter network first layer weight stats - mean: {pyg_w1.mean().item():.6f}, std: {pyg_w1.std().item():.6f}")

    # ===== TRACING STEP 3: NEIGHBOR FEATURES GATHERING =====
    print("\n===== STAGE 3: NEIGHBOR FEATURES GATHERING =====")

    # Classic implementation
    neighbor_list_reshaped = neighbor_list.reshape(-1, n_atoms * n_neighbors, 1).expand(-1, -1,
                                                                                        features_classic.size(2))
    neighbor_features_classic = torch.gather(features_classic, 1, neighbor_list_reshaped)
    neighbor_features_classic = neighbor_features_classic.reshape(batch_size, n_atoms, n_neighbors, -1)

    print(f"Classic neighbor features shape: {neighbor_features_classic.shape}")
    print(
        f"Classic neighbor features stats - mean: {neighbor_features_classic.mean().item():.6f}, std: {neighbor_features_classic.std().item():.6f}")
    print(f"Classic neighbor features sample [0,0,0,0]: {neighbor_features_classic[0, 0, 0, 0]:.6f}")

    # PyG implementation - compare message passing function
    # We need to manually extract the x_j values used in the message function
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    x_j = x_pyg[source_nodes]  # This is what's used in the message function
    x_i = x_pyg[target_nodes]

    print(f"PyG x_j shape: {x_j.shape}")
    print(f"PyG x_j stats - mean: {x_j.mean().item():.6f}, std: {x_j.std().item():.6f}")
    print(f"PyG x_j sample [0,0]: {x_j[0, 0]:.6f}")

    # Extract a comparable subset from both implementations
    # For classic: neighbor_features_classic[0,0,0] is the features of the first neighbor of first node in first batch
    # For PyG: Need to find the corresponding edge

    # Find the first edge in PyG data for node 0 in batch 0
    first_edge_idx = None
    for i, (src, tgt) in enumerate(zip(source_nodes, target_nodes)):
        if tgt.item() == 0:  # First node in first batch
            first_edge_idx = i
            break

    if first_edge_idx is not None:
        classic_first_neighbor = neighbor_list[0, 0, 0].item()
        pyg_first_neighbor = source_nodes[first_edge_idx].item()

        print(f"\nNeighbor comparison for node 0 in batch 0:")
        print(f"Classic first neighbor index: {classic_first_neighbor}")
        print(f"PyG first neighbor index: {pyg_first_neighbor}")

        classic_neighbor_features = features_classic[0, classic_first_neighbor]
        pyg_neighbor_features = x_j[first_edge_idx]

        print(f"Classic neighbor features: {classic_neighbor_features[:3]}...")
        print(f"PyG neighbor features: {pyg_neighbor_features[:3]}...")

    # ===== TRACING STEP 4: MESSAGE CREATION (APPLYING FILTERS) =====
    print("\n===== STAGE 4: MESSAGE CREATION =====")

    # Classic implementation
    conv_features_classic = neighbor_features_classic * conv_filter_classic
    print(f"Classic conv features shape: {conv_features_classic.shape}")
    print(
        f"Classic conv features stats - mean: {conv_features_classic.mean().item():.6f}, std: {conv_features_classic.std().item():.6f}")

    # PyG implementation - manual message computation
    messages_pyg = x_j * conv_filter_pyg
    print(f"PyG messages shape: {messages_pyg.shape}")
    print(f"PyG messages stats - mean: {messages_pyg.mean().item():.6f}, std: {messages_pyg.std().item():.6f}")

    # ===== TRACING STEP 5: AGGREGATION =====
    print("\n===== STAGE 5: AGGREGATION =====")

    # Classic implementation
    aggregated_classic = conv_features_classic.sum(dim=2)
    print(f"Classic aggregated shape: {aggregated_classic.shape}")
    print(
        f"Classic aggregated stats - mean: {aggregated_classic.mean().item():.6f}, std: {aggregated_classic.std().item():.6f}")

    # PyG implementation - simulate scatter sum
    unique_targets = torch.unique(target_nodes)
    aggregated_pyg = torch.zeros((len(unique_targets), hidden_channels), device=messages_pyg.device)

    for i, target in enumerate(unique_targets):
        mask = target_nodes == target
        target_messages = messages_pyg[mask]
        aggregated_pyg[i] = target_messages.sum(dim=0)

    print(f"PyG aggregated shape: {aggregated_pyg.shape}")
    print(f"PyG aggregated stats - mean: {aggregated_pyg.mean().item():.6f}, std: {aggregated_pyg.std().item():.6f}")

    # ===== ACTUAL FORWARD PASS =====
    print("\n===== FULL FORWARD PASS =====")

    # Run actual forward passes
    classic_output, classic_attention = classic_cfconv(features_classic, rbf_expansion, neighbor_list)
    pyg_output, pyg_attention = pyg_cfconv(x_pyg, edge_index, edge_attr)

    print(f"Classic final output shape: {classic_output.shape}")
    print(f"Classic output stats - mean: {classic_output.mean().item():.6f}, std: {classic_output.std().item():.6f}")

    print(f"PyG final output shape: {pyg_output.shape}")
    print(f"PyG output stats - mean: {pyg_output.mean().item():.6f}, std: {pyg_output.std().item():.6f}")

    # Convert PyG output to classic format for final comparison
    pyg_output_reshaped = torch.zeros_like(classic_output)
    for b in range(batch_size):
        for i in range(n_atoms):
            node_idx = i + b * n_atoms
            if node_idx < pyg_output.size(0):  # Safety check
                pyg_output_reshaped[b, i] = pyg_output[node_idx]

    # Final comparison
    output_diff = torch.abs(classic_output - pyg_output_reshaped)
    output_mean_diff = output_diff.mean().item()
    output_max_diff = output_diff.max().item()

    print(f"\nFinal comparison - mean diff: {output_mean_diff:.6f}, max diff: {output_max_diff:.6f}")

    # Print potential causes of difference
    if output_mean_diff > 0.01:
        print("\nPotential causes of divergence:")

        # Check filter network output difference
        if filter_diff > 0.01:
            print("- Filter networks produce different outputs (weights not aligned)")

        # Check normalization
        if norm_diff > 0.01:
            print("- Edge attribute normalization differs")

        # Check if the neighborhood structures might be different
        print("- Message passing/aggregation mechanisms differ in how they handle edges")
        print("- PyG uses scatter operation while classic uses tensor indexing")

    return classic_output, pyg_output_reshaped

if __name__ == "__main__":
    test_cfconv_with_output_comparison()
