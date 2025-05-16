import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from torch.nn import Tanh

# Import the model implementations
# Adjust these imports based on your project structure
from pygv.encoder.schnet_wo_embed_v2 import GCNInteraction as PyGInteraction
from pygv.encoder.schnet_wo_embed_v2 import CFConv as PygCFConv
from pygv.encoder.schnet_wo_embed_v2 import MLP

# Import classic implementation - adjust as needed
from classic_schnet import CFConv as ContinuousFilterConv, LinearLayer


# Define the classic InteractionBlock directly here to avoid import issues
class ClassicInteractionBlock(torch.nn.Module):
    """
    SchNet interaction block as described by Schütt et al. (2018).
    """

    def __init__(self, n_inputs, n_gaussians, n_filters, activation=Tanh()):
        super(ClassicInteractionBlock, self).__init__()

        # Initial dense layer
        self.initial_dense = torch.nn.Sequential(
            *LinearLayer.create(n_inputs, n_filters, bias=False, activation=None)
        )

        # Continuous filter convolution
        self.cfconv = ContinuousFilterConv(
            n_gaussians=n_gaussians,
            n_filters=n_filters,
            activation=activation
        )

        # Output layers
        output_layers = []
        output_layers.extend(LinearLayer.create(n_filters, n_filters, bias=True, activation=activation))
        output_layers.extend(LinearLayer.create(n_filters, n_filters, bias=True))
        self.output_dense = torch.nn.Sequential(*output_layers)

    def forward(self, features, rbf_expansion, neighbor_list):
        init_feature_output = self.initial_dense(features)
        conv_output, attn = self.cfconv(init_feature_output.to(torch.float32),
                                        rbf_expansion.to(torch.float32),
                                        neighbor_list)
        output_features = self.output_dense(conv_output).to(torch.float32)
        return output_features, attn


def generate_test_data(batch_size=2, n_atoms=20, n_neighbors=10, n_features=32, n_gaussians=16):
    """
    Generate test data for both implementations
    """
    print(f"Generating test data with {batch_size} batches, {n_atoms} atoms, {n_features} features")

    # Create data for classic implementation
    classic_features = torch.randn(batch_size, n_atoms, n_features)
    classic_rbf = torch.randn(batch_size, n_atoms, n_neighbors, n_gaussians)

    # Create neighbor list - each atom has n_neighbors
    classic_neighbors = torch.zeros(batch_size, n_atoms, n_neighbors, dtype=torch.long)

    # For each atom, select n_neighbors other atoms as neighbors
    # We'll use a deterministic pattern for testing reproducibility
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

    for b in range(batch_size):
        # Add nodes for this batch
        for i in range(n_atoms):
            # Node features
            pyg_nodes.append(classic_features[b, i])

            # Create edges for each neighbor in classical neighbor list
            for j in range(n_neighbors):
                neighbor_idx = classic_neighbors[b, i, j].item()

                # Convert edges for PyG format
                # PyG uses standard message-passing convention: source -> target
                # The trick is to get edge direction right: classic has neighbors receiving
                # messages, so we need to reverse the direction for PyG
                src = neighbor_idx + b * n_atoms  # Source is the neighbor (reversed from classic)
                dst = i + b * n_atoms  # Destination is current node

                edge_src.append(src)
                edge_dst.append(dst)
                edge_attr.append(classic_rbf[b, i, j])

    # Convert lists to tensors
    pyg_x = torch.stack(pyg_nodes)
    pyg_edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    pyg_edge_attr = torch.stack(edge_attr)

    print(f"Classic features shape: {classic_features.shape}")
    print(f"Classic RBF shape: {classic_rbf.shape}")
    print(f"Classic neighbors shape: {classic_neighbors.shape}")
    print(f"PyG node features shape: {pyg_x.shape}")
    print(f"PyG edge index shape: {pyg_edge_index.shape}")
    print(f"PyG edge attr shape: {pyg_edge_attr.shape}")

    return {
        'classic': (classic_features, classic_rbf, classic_neighbors),
        'pyg': (pyg_x, pyg_edge_index, pyg_edge_attr),
        'batch_size': batch_size,
        'n_atoms': n_atoms,
        'n_features': n_features,
        'n_gaussians': n_gaussians,
        'n_neighbors': n_neighbors
    }


def initialize_models(test_data):
    """
    Initialize both model implementations with matching parameters
    """
    n_features = test_data['n_features']
    n_gaussians = test_data['n_gaussians']

    # Initialize classic model
    classic_model = ClassicInteractionBlock(
        n_inputs=n_features,
        n_gaussians=n_gaussians,
        n_filters=n_features,  # Use n_features to match PyG model dimensions
        activation=torch.nn.Tanh()
    )

    # Initialize PyG model with matching parameters
    pyg_model = PyGInteraction(
        in_channels=n_features,
        edge_channels=n_gaussians,
        hidden_channels=n_features,  # Same as n_filters in classic model
        activation='tanh',
        use_attention=True
    )

    # Set both models to evaluation mode
    classic_model.eval()
    pyg_model.eval()

    return classic_model, pyg_model


def copy_weights(classic_model, pyg_model):
    """
    Copy weights from classic model to PyG model to ensure comparable outputs
    """
    print("Copying weights from classic model to PyG model...")

    # Copy initial dense layer weights
    pyg_model.initial_dense.weight.data = classic_model.initial_dense[0].weight.data.clone()

    # Copy CFConv filter network weights
    # Classic: filter_generator has [0] and [2] as LinearLayers
    # PyG: filter_network.lins contains the Linear layers
    pyg_model.cfconv.filter_network.lins[0].weight.data = classic_model.cfconv.filter_generator[0].weight.data.clone()
    pyg_model.cfconv.filter_network.lins[0].bias.data = classic_model.cfconv.filter_generator[0].bias.data.clone()
    pyg_model.cfconv.filter_network.lins[1].weight.data = classic_model.cfconv.filter_generator[2].weight.data.clone()
    pyg_model.cfconv.filter_network.lins[1].bias.data = classic_model.cfconv.filter_generator[2].bias.data.clone()

    # Copy attention weights if present
    if hasattr(classic_model.cfconv, 'nbr_filter') and hasattr(pyg_model.cfconv, 'attention_vector'):
        pyg_model.cfconv.attention_vector.data = classic_model.cfconv.nbr_filter.data.clone()

    # Copy output layers
    # Classic: output_dense is Sequential with layers at even indices (0, 2, etc.)
    # PyG: output_layer.lins contains the Linear layers
    pyg_model.output_layer.lins[0].weight.data = classic_model.output_dense[0].weight.data.clone()
    pyg_model.output_layer.lins[0].bias.data = classic_model.output_dense[0].bias.data.clone()
    pyg_model.output_layer.lins[1].weight.data = classic_model.output_dense[2].weight.data.clone()
    pyg_model.output_layer.lins[1].bias.data = classic_model.output_dense[2].bias.data.clone()

    print("Weight copying complete")
    return pyg_model


def compare_outputs(classic_output, pyg_output, test_data):
    """
    Compare the outputs of classic and PyG models and visualize differences
    """
    batch_size = test_data['batch_size']
    n_atoms = test_data['n_atoms']

    # Reshape PyG output to match classic shape
    pyg_reshaped = torch.zeros_like(classic_output)

    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(pyg_output):
                pyg_reshaped[b, i] = pyg_output[idx]

    # Compute statistics on differences
    abs_diff = torch.abs(classic_output - pyg_reshaped)
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()
    relative_diff = (abs_diff / (torch.abs(classic_output) + 1e-10)).mean().item()

    print(f"Mean absolute difference: {mean_diff:.8f}")
    print(f"Maximum absolute difference: {max_diff:.8f}")
    print(f"Mean relative difference: {relative_diff:.8f}")

    # Visualize the first batch's output
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(classic_output[0].detach().numpy())
    plt.title("Classic Model Output")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(pyg_reshaped[0].detach().numpy())
    plt.title("PyG Model Output")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(abs_diff[0].detach().numpy())
    plt.title("Absolute Difference")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)
    print("Saved output comparison visualization to 'model_comparison.png'")

    return {
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'relative_diff': relative_diff
    }


def run_test(n_rounds=1):
    """
    Run a complete test with multiple rounds and summarize results
    """
    results = []

    for round_idx in range(n_rounds):
        print(f"\n===== Test Round {round_idx + 1}/{n_rounds} =====")

        # Generate test data
        test_data = generate_test_data()

        # Initialize models with matching parameters
        classic_model, pyg_model = initialize_models(test_data)

        # Copy weights to ensure comparable outputs
        pyg_model = copy_weights(classic_model, pyg_model)

        # Extract data
        classic_input = test_data['classic']
        pyg_x, pyg_edge_index, pyg_edge_attr = test_data['pyg']

        # Measure performance
        start_time = time.time()
        with torch.no_grad():
            classic_output, classic_attn = classic_model(*classic_input)
        classic_time = time.time() - start_time

        start_time = time.time()
        with torch.no_grad():
            pyg_output, pyg_attn = pyg_model(pyg_x, pyg_edge_index, pyg_edge_attr)
        pyg_time = time.time() - start_time

        print(f"Classic implementation time: {classic_time:.6f} seconds")
        print(f"PyG implementation time: {pyg_time:.6f} seconds")
        print(f"Speedup factor: {classic_time / pyg_time:.2f}x")

        # Compare results
        diff_stats = compare_outputs(classic_output, pyg_output, test_data)

        # Store results
        results.append({
            'classic_time': classic_time,
            'pyg_time': pyg_time,
            'speedup': classic_time / pyg_time,
            **diff_stats
        })

    # Compute average results
    avg_results = {
        'avg_classic_time': sum(r['classic_time'] for r in results) / n_rounds,
        'avg_pyg_time': sum(r['pyg_time'] for r in results) / n_rounds,
        'avg_speedup': sum(r['speedup'] for r in results) / n_rounds,
        'avg_mean_diff': sum(r['mean_diff'] for r in results) / n_rounds,
        'avg_max_diff': sum(r['max_diff'] for r in results) / n_rounds,
        'avg_relative_diff': sum(r['relative_diff'] for r in results) / n_rounds
    }

    print("\n===== Summary =====")
    print(f"Average classic time: {avg_results['avg_classic_time']:.6f} seconds")
    print(f"Average PyG time: {avg_results['avg_pyg_time']:.6f} seconds")
    print(f"Average speedup: {avg_results['avg_speedup']:.2f}x")
    print(f"Average mean difference: {avg_results['avg_mean_diff']:.8f}")
    print(f"Average max difference: {avg_results['avg_max_diff']:.8f}")
    print(f"Average relative difference: {avg_results['avg_relative_diff']:.8f}")

    if avg_results['avg_mean_diff'] < 1e-5:
        print("\nPASS: Outputs are practically identical!")
    elif avg_results['avg_mean_diff'] < 1e-3:
        print("\nWARNING: Small differences detected. Check for numerical precision issues.")
    else:
        print("\nFAIL: Significant differences detected between implementations.")

    return results, avg_results


def analyze_layer_by_layer(test_data):
    """
    Perform layer-by-layer comparison of the two implementations
    """
    print("\n===== Layer-by-Layer Analysis =====")

    # Extract data
    classic_features, classic_rbf, classic_neighbors = test_data['classic']
    pyg_x, pyg_edge_index, pyg_edge_attr = test_data['pyg']
    batch_size = test_data['batch_size']
    n_atoms = test_data['n_atoms']

    # Initialize models
    classic_model, pyg_model = initialize_models(test_data)
    pyg_model = copy_weights(classic_model, pyg_model)

    # 1. Initial Dense Layer
    print("\n1. Initial Dense Layer")
    classic_init = classic_model.initial_dense(classic_features)
    pyg_init = pyg_model.initial_dense(pyg_x)

    # Reshape PyG output for comparison
    pyg_init_reshaped = torch.zeros_like(classic_init)
    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(pyg_init):
                pyg_init_reshaped[b, i] = pyg_init[idx]

    init_diff = torch.abs(classic_init - pyg_init_reshaped).mean().item()
    print(f"Initial Dense Layer Output Difference: {init_diff:.8f}")

    # 2. CFConv Layer
    print("\n2. CFConv Layer")
    with torch.no_grad():
        classic_conv, classic_attn = classic_model.cfconv(classic_init, classic_rbf, classic_neighbors)
        pyg_conv, pyg_attn = pyg_model.cfconv(pyg_init, pyg_edge_index, pyg_edge_attr)

    # Reshape PyG output
    pyg_conv_reshaped = torch.zeros_like(classic_conv)
    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(pyg_conv):
                pyg_conv_reshaped[b, i] = pyg_conv[idx]

    conv_diff = torch.abs(classic_conv - pyg_conv_reshaped).mean().item()
    print(f"CFConv Layer Output Difference: {conv_diff:.8f}")

    # Compare attention if available
    if classic_attn is not None and pyg_attn is not None:
        # Reshape PyG attention to match classic attention
        # This is complex as attention structure differs between implementations
        print("Attention output shapes:")
        print(f"Classic attention shape: {classic_attn.shape if classic_attn is not None else None}")
        print(f"PyG attention shape: {pyg_attn.shape if pyg_attn is not None else None}")

    # 3. Output Layer
    print("\n3. Output Layer")
    with torch.no_grad():
        classic_output = classic_model.output_dense(classic_conv)
        pyg_output = pyg_model.output_layer(pyg_conv)

    # Reshape PyG output
    pyg_output_reshaped = torch.zeros_like(classic_output)
    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(pyg_output):
                pyg_output_reshaped[b, i] = pyg_output[idx]

    output_diff = torch.abs(classic_output - pyg_output_reshaped).mean().item()
    print(f"Output Layer Output Difference: {output_diff:.8f}")

    print("\n===== Probable Sources of Differences =====")
    if init_diff > 1e-5:
        print("- Initial dense layer implementation differs")
    if conv_diff > 1e-5:
        print("- CFConv layer has different behavior")
        print("  - Check message passing direction")
        print("  - Check attention mechanism implementation")
    if output_diff > init_diff and output_diff > conv_diff:
        print("- Output layer implementation amplifies differences")

    return {
        'init_diff': init_diff,
        'conv_diff': conv_diff,
        'output_diff': output_diff
    }


if __name__ == "__main__":
    print("\n===== SchNet Interaction Block Implementation Comparison =====")
    print("Comparing classic implementation with PyG implementation")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run the main test
    try:
        results, avg_results = run_test(n_rounds=1)

        # Generate test data for layer-by-layer analysis
        test_data = generate_test_data()

        # Detailed layer-by-layer analysis
        layer_diffs = analyze_layer_by_layer(test_data)

        print("\n===== SUMMARY =====")
        print(f"PyG Implementation Performance: {avg_results['avg_speedup']:.2f}x faster than classic")

        if avg_results['avg_relative_diff'] < 0.01:
            print("Implementations are functionally equivalent with acceptable numerical differences.")
        else:
            print("Significant differences detected. See the layer analysis for debugging.")
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()
