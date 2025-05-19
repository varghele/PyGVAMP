import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import importlib.util
import sys


# Load the implementations
def load_module_from_path(name, path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load our debug implementation
try:
    debug_module = load_module_from_path("debug_vampnet_new", "debug_vampnet_new.py")
    DebugClassicInteractionBlock = debug_module.ClassicInteractionBlock
    DebugGCNInteraction = debug_module.GCNInteraction
    DebugCFConv = debug_module.CFConv
    DebugMLP = debug_module.MLP
    DebugSchNetEncoderNoEmbed = debug_module.SchNetEncoderNoEmbed
    print("Successfully loaded debug implementations from debug_vamp_new.py")
except Exception as e:
    print(f"Error loading debug_vamp_new.py: {e}")
    raise

# Load the actual implementation from pygv
try:
    from pygv.encoder.schnet_wo_embed_v2 import GCNInteraction, CFConv, MLP, SchNetEncoderNoEmbed

    print("Successfully loaded implementations from pygv.encoder.schnet_wo_embed_v2")
except Exception as e:
    print(f"Error importing from pygv.encoder.schnet_wo_embed_v2: {e}")
    raise


def generate_test_data(batch_size=2, n_atoms=20, n_neighbors=10, n_features=32, n_gaussians=16):
    """
    Generate test data for comparing SchNet implementations
    """
    print(f"Generating test data with {batch_size} batches, {n_atoms} atoms, {n_features} features")

    # Create data for classic implementation
    classic_features = torch.randn(batch_size, n_atoms, n_features)
    classic_rbf = torch.randn(batch_size, n_atoms, n_neighbors, n_gaussians)

    # Create neighbor list - each atom has n_neighbors
    classic_neighbors = torch.zeros(batch_size, n_atoms, n_neighbors, dtype=torch.long)

    # For each atom, select n_neighbors other atoms as neighbors
    for b in range(batch_size):
        for i in range(n_atoms):
            for j in range(n_neighbors):
                # Choose neighbors in a cyclic pattern
                classic_neighbors[b, i, j] = (i + j + 1) % n_atoms

    # Create PyG compatible data
    pyg_nodes = []
    edge_src = []
    edge_dst = []
    edge_attr = []
    batch_indices = []

    for b in range(batch_size):
        # Add nodes for this batch
        for i in range(n_atoms):
            # Node features
            pyg_nodes.append(classic_features[b, i])
            batch_indices.append(b)  # Track batch index for each node

            # Create edges for each neighbor in classical neighbor list
            for j in range(n_neighbors):
                neighbor_idx = classic_neighbors[b, i, j].item()
                # Source is neighbor, destination is current node (matching message passing convention)
                src = neighbor_idx + b * n_atoms  # Offset by batch
                dst = i + b * n_atoms  # Offset by batch

                edge_src.append(src)
                edge_dst.append(dst)
                edge_attr.append(classic_rbf[b, i, j])

    # Convert lists to tensors
    pyg_x = torch.stack(pyg_nodes)
    pyg_edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    pyg_edge_attr = torch.stack(edge_attr)
    pyg_batch = torch.tensor(batch_indices, dtype=torch.long)

    print(f"Classic features shape: {classic_features.shape}")
    print(f"Classic RBF shape: {classic_rbf.shape}")
    print(f"Classic neighbors shape: {classic_neighbors.shape}")
    print(f"PyG node features shape: {pyg_x.shape}")
    print(f"PyG edge index shape: {pyg_edge_index.shape}")
    print(f"PyG edge attr shape: {pyg_edge_attr.shape}")
    print(f"PyG batch indices shape: {pyg_batch.shape}")

    return {
        'classic': (classic_features, classic_rbf, classic_neighbors),
        'pyg': (pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch),
        'batch_size': batch_size,
        'n_atoms': n_atoms,
        'n_features': n_features,
        'n_gaussians': n_gaussians,
        'n_neighbors': n_neighbors
    }


def initialize_models(test_data):
    """
    Initialize both implementations with identical parameters
    """
    n_features = test_data['n_features']
    n_gaussians = test_data['n_gaussians']

    # Initialize debug models
    debug_classic = DebugClassicInteractionBlock(
        n_inputs=n_features,
        n_gaussians=n_gaussians,
        n_filters=n_features,
        activation=torch.nn.Tanh()
    )

    debug_interaction = DebugGCNInteraction(
        in_channels=n_features,
        edge_channels=n_gaussians,
        hidden_channels=n_features,
        activation='tanh',
        use_attention=True
    )

    debug_schnet = DebugSchNetEncoderNoEmbed(
        node_dim=n_features,
        edge_dim=n_gaussians,
        hidden_dim=n_features,
        output_dim=n_features,
        n_interactions=1,
        activation='tanh',
        use_attention=True
    )

    # Initialize production models
    prod_interaction = GCNInteraction(
        in_channels=n_features,
        edge_channels=n_gaussians,
        hidden_channels=n_features,
        activation='tanh',
        use_attention=True
    )

    prod_schnet = SchNetEncoderNoEmbed(
        node_dim=n_features,
        edge_dim=n_gaussians,
        hidden_dim=n_features,
        output_dim=n_features,
        n_interactions=1,
        activation='tanh',
        use_attention=True
    )

    # Set all models to evaluation mode
    debug_classic.eval()
    debug_interaction.eval()
    debug_schnet.eval()
    prod_interaction.eval()
    prod_schnet.eval()

    return {
        'debug_classic': debug_classic,
        'debug_interaction': debug_interaction,
        'debug_schnet': debug_schnet,
        'prod_interaction': prod_interaction,
        'prod_schnet': prod_schnet
    }


def copy_weights(models):
    """
    Copy weights from debug classic to all other models
    """
    debug_classic = models['debug_classic']
    debug_interaction = models['debug_interaction']
    debug_schnet = models['debug_schnet']
    prod_interaction = models['prod_interaction']
    prod_schnet = models['prod_schnet']

    print("Copying weights from debug classic to all models...")

    # First copy to debug interaction
    debug_interaction.initial_dense.weight.data = debug_classic.initial_dense[0].weight.data.clone()

    debug_interaction.cfconv.filter_network.lins[0].weight.data = debug_classic.cfconv.filter_generator[
        0].weight.data.clone()
    debug_interaction.cfconv.filter_network.lins[0].bias.data = debug_classic.cfconv.filter_generator[
        0].bias.data.clone()
    debug_interaction.cfconv.filter_network.lins[1].weight.data = debug_classic.cfconv.filter_generator[
        2].weight.data.clone()
    debug_interaction.cfconv.filter_network.lins[1].bias.data = debug_classic.cfconv.filter_generator[
        2].bias.data.clone()

    debug_interaction.cfconv.attention_vector.data = debug_classic.cfconv.nbr_filter.data.clone()

    debug_interaction.output_layer.lins[0].weight.data = debug_classic.output_dense[0].weight.data.clone()
    debug_interaction.output_layer.lins[0].bias.data = debug_classic.output_dense[0].bias.data.clone()
    debug_interaction.output_layer.lins[1].weight.data = debug_classic.output_dense[2].weight.data.clone()
    debug_interaction.output_layer.lins[1].bias.data = debug_classic.output_dense[2].bias.data.clone()

    # Copy to debug schnet's first interaction
    debug_schnet.interactions[0].initial_dense.weight.data = debug_classic.initial_dense[0].weight.data.clone()

    debug_schnet.interactions[0].cfconv.filter_network.lins[0].weight.data = debug_classic.cfconv.filter_generator[
        0].weight.data.clone()
    debug_schnet.interactions[0].cfconv.filter_network.lins[0].bias.data = debug_classic.cfconv.filter_generator[
        0].bias.data.clone()
    debug_schnet.interactions[0].cfconv.filter_network.lins[1].weight.data = debug_classic.cfconv.filter_generator[
        2].weight.data.clone()
    debug_schnet.interactions[0].cfconv.filter_network.lins[1].bias.data = debug_classic.cfconv.filter_generator[
        2].bias.data.clone()

    debug_schnet.interactions[0].cfconv.attention_vector.data = debug_classic.cfconv.nbr_filter.data.clone()

    debug_schnet.interactions[0].output_layer.lins[0].weight.data = debug_classic.output_dense[0].weight.data.clone()
    debug_schnet.interactions[0].output_layer.lins[0].bias.data = debug_classic.output_dense[0].bias.data.clone()
    debug_schnet.interactions[0].output_layer.lins[1].weight.data = debug_classic.output_dense[2].weight.data.clone()
    debug_schnet.interactions[0].output_layer.lins[1].bias.data = debug_classic.output_dense[2].bias.data.clone()

    # Copy to production interaction
    prod_interaction.initial_dense.weight.data = debug_classic.initial_dense[0].weight.data.clone()

    prod_interaction.cfconv.filter_network.lins[0].weight.data = debug_classic.cfconv.filter_generator[
        0].weight.data.clone()
    prod_interaction.cfconv.filter_network.lins[0].bias.data = debug_classic.cfconv.filter_generator[
        0].bias.data.clone()
    prod_interaction.cfconv.filter_network.lins[1].weight.data = debug_classic.cfconv.filter_generator[
        2].weight.data.clone()
    prod_interaction.cfconv.filter_network.lins[1].bias.data = debug_classic.cfconv.filter_generator[
        2].bias.data.clone()

    prod_interaction.cfconv.attention_vector.data = debug_classic.cfconv.nbr_filter.data.clone()

    prod_interaction.output_layer.lins[0].weight.data = debug_classic.output_dense[0].weight.data.clone()
    prod_interaction.output_layer.lins[0].bias.data = debug_classic.output_dense[0].bias.data.clone()
    prod_interaction.output_layer.lins[1].weight.data = debug_classic.output_dense[2].weight.data.clone()
    prod_interaction.output_layer.lins[1].bias.data = debug_classic.output_dense[2].bias.data.clone()

    # Copy to production schnet's first interaction
    prod_schnet.interactions[0].initial_dense.weight.data = debug_classic.initial_dense[0].weight.data.clone()

    prod_schnet.interactions[0].cfconv.filter_network.lins[0].weight.data = debug_classic.cfconv.filter_generator[
        0].weight.data.clone()
    prod_schnet.interactions[0].cfconv.filter_network.lins[0].bias.data = debug_classic.cfconv.filter_generator[
        0].bias.data.clone()
    prod_schnet.interactions[0].cfconv.filter_network.lins[1].weight.data = debug_classic.cfconv.filter_generator[
        2].weight.data.clone()
    prod_schnet.interactions[0].cfconv.filter_network.lins[1].bias.data = debug_classic.cfconv.filter_generator[
        2].bias.data.clone()

    prod_schnet.interactions[0].cfconv.attention_vector.data = debug_classic.cfconv.nbr_filter.data.clone()

    prod_schnet.interactions[0].output_layer.lins[0].weight.data = debug_classic.output_dense[0].weight.data.clone()
    prod_schnet.interactions[0].output_layer.lins[0].bias.data = debug_classic.output_dense[0].bias.data.clone()
    prod_schnet.interactions[0].output_layer.lins[1].weight.data = debug_classic.output_dense[2].weight.data.clone()
    prod_schnet.interactions[0].output_layer.lins[1].bias.data = debug_classic.output_dense[2].bias.data.clone()

    print("Weight copying complete")
    return models


def run_models_and_compare(models, test_data):
    """
    Run all models and compare their outputs
    """
    debug_classic = models['debug_classic']
    debug_interaction = models['debug_interaction']
    debug_schnet = models['debug_schnet']
    prod_interaction = models['prod_interaction']
    prod_schnet = models['prod_schnet']

    # Extract test data
    classic_input = test_data['classic']
    pyg_input = test_data['pyg']
    pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch = pyg_input

    batch_size = test_data['batch_size']
    n_atoms = test_data['n_atoms']

    # Set SchNet models to not use residual for direct comparison
    if hasattr(debug_schnet, 'use_residual'):
        debug_schnet.use_residual = False

    # Run all models
    with torch.no_grad():
        # Debug models
        debug_classic_output, _ = debug_classic(*classic_input)
        debug_interaction_output, _ = debug_interaction(pyg_x, pyg_edge_index, pyg_edge_attr)
        debug_schnet_output, (debug_node_features, _, debug_deltas) = debug_schnet(pyg_x, pyg_edge_index, pyg_edge_attr,
                                                                                   pyg_batch)

        # Production models
        prod_interaction_output, _ = prod_interaction(pyg_x, pyg_edge_index, pyg_edge_attr)
        prod_schnet_output, prod_schnet_info = prod_schnet(pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch)

        # Extract node features from production SchNet
        if isinstance(prod_schnet_info, tuple) and len(prod_schnet_info) >= 1:
            prod_node_features = prod_schnet_info[0]
        else:
            print("Warning: Unexpected output format from production SchNet")
            prod_node_features = None

    # Convert PyG outputs to classic shape for comparison
    debug_interaction_reshaped = torch.zeros_like(debug_classic_output)
    debug_delta_reshaped = torch.zeros_like(debug_classic_output)
    prod_interaction_reshaped = torch.zeros_like(debug_classic_output)

    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(debug_interaction_output):
                debug_interaction_reshaped[b, i] = debug_interaction_output[idx]
            if debug_deltas is not None and idx < len(debug_deltas[0]):
                debug_delta_reshaped[b, i] = debug_deltas[0][idx]  # First interaction's delta
            if idx < len(prod_interaction_output):
                prod_interaction_reshaped[b, i] = prod_interaction_output[idx]

    # Compare debug classic vs debug interaction
    debug_vs_debug_interaction_diff = torch.abs(debug_classic_output - debug_interaction_reshaped)
    debug_vs_debug_interaction_mean_diff = debug_vs_debug_interaction_diff.mean().item()
    debug_vs_debug_interaction_max_diff = debug_vs_debug_interaction_diff.max().item()

    # Compare debug classic vs prod interaction
    debug_vs_prod_interaction_diff = torch.abs(debug_classic_output - prod_interaction_reshaped)
    debug_vs_prod_interaction_mean_diff = debug_vs_prod_interaction_diff.mean().item()
    debug_vs_prod_interaction_max_diff = debug_vs_prod_interaction_diff.max().item()

    # Compare debug interaction vs prod interaction
    debug_interaction_vs_prod_diff = torch.abs(debug_interaction_reshaped - prod_interaction_reshaped)
    debug_interaction_vs_prod_mean_diff = debug_interaction_vs_prod_diff.mean().item()
    debug_interaction_vs_prod_max_diff = debug_interaction_vs_prod_diff.max().item()

    # Print comparison results
    print("\n===== Comparison Results =====")

    print("\nDebug classic vs Debug interaction:")
    print(f"Mean absolute difference: {debug_vs_debug_interaction_mean_diff:.8f}")
    print(f"Maximum absolute difference: {debug_vs_debug_interaction_max_diff:.8f}")

    print("\nDebug classic vs Production interaction:")
    print(f"Mean absolute difference: {debug_vs_prod_interaction_mean_diff:.8f}")
    print(f"Maximum absolute difference: {debug_vs_prod_interaction_max_diff:.8f}")

    print("\nDebug interaction vs Production interaction:")
    print(f"Mean absolute difference: {debug_interaction_vs_prod_mean_diff:.8f}")
    print(f"Maximum absolute difference: {debug_interaction_vs_prod_max_diff:.8f}")

    # Test with residual connections
    if hasattr(debug_schnet, 'use_residual'):
        debug_schnet.use_residual = True

    with torch.no_grad():
        debug_schnet_output_with_res, (debug_node_features_with_res, _, _) = debug_schnet(
            pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch)

        prod_schnet_output_with_res, prod_schnet_info_with_res = prod_schnet(
            pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch)

        if isinstance(prod_schnet_info_with_res, tuple) and len(prod_schnet_info_with_res) >= 1:
            prod_node_features_with_res = prod_schnet_info_with_res[0]
        else:
            prod_node_features_with_res = None

    # Compare SchNet node features if available
    if debug_node_features_with_res is not None and prod_node_features_with_res is not None:
        # Reshape to classic format
        debug_node_features_reshaped = torch.zeros_like(debug_classic_output)
        prod_node_features_reshaped = torch.zeros_like(debug_classic_output)

        for b in range(batch_size):
            for i in range(n_atoms):
                idx = i + b * n_atoms
                if idx < len(debug_node_features_with_res):
                    debug_node_features_reshaped[b, i] = debug_node_features_with_res[idx]
                if idx < len(prod_node_features_with_res):
                    prod_node_features_reshaped[b, i] = prod_node_features_with_res[idx]

        # Compare debug vs prod SchNet node features
        schnet_node_diff = torch.abs(debug_node_features_reshaped - prod_node_features_reshaped)
        schnet_node_mean_diff = schnet_node_diff.mean().item()
        schnet_node_max_diff = schnet_node_diff.max().item()

        print("\nDebug SchNet node features vs Production SchNet node features:")
        print(f"Mean absolute difference: {schnet_node_mean_diff:.8f}")
        print(f"Maximum absolute difference: {schnet_node_max_diff:.8f}")

    # Compare global outputs
    if debug_schnet_output_with_res is not None and prod_schnet_output_with_res is not None:
        schnet_global_diff = torch.abs(debug_schnet_output_with_res - prod_schnet_output_with_res)
        schnet_global_mean_diff = schnet_global_diff.mean().item()
        schnet_global_max_diff = schnet_global_diff.max().item()

        print("\nDebug SchNet global output vs Production SchNet global output:")
        print(f"Mean absolute difference: {schnet_global_mean_diff:.8f}")
        print(f"Maximum absolute difference: {schnet_global_max_diff:.8f}")

    return {
        'debug_vs_debug_interaction': debug_vs_debug_interaction_mean_diff,
        'debug_vs_prod_interaction': debug_vs_prod_interaction_mean_diff,
        'debug_interaction_vs_prod': debug_interaction_vs_prod_mean_diff,
        'schnet_node_mean_diff': schnet_node_mean_diff if 'schnet_node_mean_diff' in locals() else None,
        'schnet_global_mean_diff': schnet_global_mean_diff if 'schnet_global_mean_diff' in locals() else None
    }


def time_all_implementations(models, test_data, n_runs=10):
    """
    Time all implementations and compare performance
    """
    # Extract test data
    classic_input = test_data['classic']
    pyg_input = test_data['pyg']
    pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch = pyg_input

    # Extract models
    debug_classic = models['debug_classic']
    debug_interaction = models['debug_interaction']
    debug_schnet = models['debug_schnet']
    prod_interaction = models['prod_interaction']
    prod_schnet = models['prod_schnet']

    # Timing for debug classic
    debug_classic_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = debug_classic(*classic_input)
        debug_classic_times.append(time.time() - start)

    # Timing for debug interaction
    debug_interaction_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = debug_interaction(pyg_x, pyg_edge_index, pyg_edge_attr)
        debug_interaction_times.append(time.time() - start)

    # Timing for debug schnet
    debug_schnet_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = debug_schnet(pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch)
        debug_schnet_times.append(time.time() - start)

    # Timing for prod interaction
    prod_interaction_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = prod_interaction(pyg_x, pyg_edge_index, pyg_edge_attr)
        prod_interaction_times.append(time.time() - start)

    # Timing for prod schnet
    prod_schnet_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = prod_schnet(pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch)
        prod_schnet_times.append(time.time() - start)

    # Calculate averages
    avg_debug_classic = sum(debug_classic_times) / n_runs
    avg_debug_interaction = sum(debug_interaction_times) / n_runs
    avg_debug_schnet = sum(debug_schnet_times) / n_runs
    avg_prod_interaction = sum(prod_interaction_times) / n_runs
    avg_prod_schnet = sum(prod_schnet_times) / n_runs

    # Print timing results
    print("\n===== Performance Comparison =====")
    print(f"Debug classic: {avg_debug_classic:.6f} seconds (baseline)")
    print(f"Debug interaction: {avg_debug_interaction:.6f} seconds ({avg_debug_classic / avg_debug_interaction:.2f}x)")
    print(f"Debug SchNet: {avg_debug_schnet:.6f} seconds ({avg_debug_classic / avg_debug_schnet:.2f}x)")
    print(
        f"Production interaction: {avg_prod_interaction:.6f} seconds ({avg_debug_classic / avg_prod_interaction:.2f}x)")
    print(f"Production SchNet: {avg_prod_schnet:.6f} seconds ({avg_debug_classic / avg_prod_schnet:.2f}x)")

    return {
        'debug_classic': avg_debug_classic,
        'debug_interaction': avg_debug_interaction,
        'debug_schnet': avg_debug_schnet,
        'prod_interaction': avg_prod_interaction,
        'prod_schnet': avg_prod_schnet
    }


def main():
    """Main function to run all comparisons"""
    print("===== Starting SchNet Implementation Comparison =====")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate test data
    test_data = generate_test_data(batch_size=2, n_atoms=20, n_neighbors=10)

    # Initialize all models
    models = initialize_models(test_data)

    # Copy weights for fair comparison
    models = copy_weights(models)

    # Run all models and compare outputs
    comparison_results = run_models_and_compare(models, test_data)

    # Time all implementations
    timing_results = time_all_implementations(models, test_data)

    # Print final summary
    print("\n===== Final Summary =====")
    if all(diff < 1e-4 for diff in comparison_results.values() if diff is not None):
        print("✓ All implementations produce practically identical outputs")
    else:
        print("✗ There are differences between implementations")

    fastest = min(timing_results, key=timing_results.get)
    print(f"Fastest implementation: {fastest} ({timing_results[fastest]:.6f}s)")

    return {
        'comparison': comparison_results,
        'timing': timing_results
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()
