import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool
import time
import os

# Import your model implementations here
try:
    # Try to import from your module structure
    from pygv.encoder.schnet_wo_embed_v2 import GCNInteraction, CFConv, MLP, SchNetEncoderNoEmbed
except ImportError:
    print("Could not import PyG models from pygv module. Using local definitions.")
    # If import fails, define them locally for testing (make sure these match your actual implementations)


# Define the classic InteractionBlock directly to avoid import issues
class ClassicInteractionBlock(torch.nn.Module):
    """SchNet interaction block as described by Schütt et al. (2018)."""

    def __init__(self, n_inputs, n_gaussians, n_filters, activation=torch.nn.Tanh()):
        super(ClassicInteractionBlock, self).__init__()

        # Dynamically import LinearLayer if not available
        if 'LinearLayer' not in globals():
            # Define a simple LinearLayer to avoid dependency
            class LinearLayer:
                @staticmethod
                def create(d_in, d_out, bias=True, activation=None):
                    layers = [torch.nn.Linear(d_in, d_out, bias=bias)]
                    if activation is not None:
                        layers.append(activation)
                    return layers

            globals()['LinearLayer'] = LinearLayer

        # Define classic ContinuousFilterConv if not available
        if 'ContinuousFilterConv' not in globals():
            # Define a minimal version for testing
            class ContinuousFilterConv(torch.nn.Module):
                def __init__(self, n_gaussians, n_filters, activation, normalization_layer=None):
                    super(ContinuousFilterConv, self).__init__()
                    self.filter_generator = torch.nn.Sequential(
                        torch.nn.Linear(n_gaussians, n_filters),
                        activation,
                        torch.nn.Linear(n_filters, n_filters)
                    )
                    self.normalization_layer = normalization_layer
                    self.use_attention = True
                    if self.use_attention:
                        self.nbr_filter = torch.nn.Parameter(torch.Tensor(n_filters, 1))
                        torch.nn.init.xavier_uniform_(self.nbr_filter)

                def forward(self, features, rbf_expansion, neighbor_list):
                    # Simplified implementation for testing
                    batch_size, n_atoms, _ = features.shape
                    n_neighbors = neighbor_list.shape[2]

                    # Generate filters
                    filters = self.filter_generator(rbf_expansion)

                    # Gather neighbor features
                    neighbor_list_expanded = neighbor_list.reshape(batch_size, n_atoms * n_neighbors, 1)
                    neighbor_list_expanded = neighbor_list_expanded.expand(-1, -1, features.shape[2])
                    neighbor_features = torch.gather(features, 1, neighbor_list_expanded)
                    neighbor_features = neighbor_features.reshape(batch_size, n_atoms, n_neighbors, -1)

                    # Apply filters
                    filtered_features = neighbor_features * filters

                    # Apply attention if used
                    if self.use_attention:
                        attention = torch.matmul(filtered_features, self.nbr_filter).squeeze(-1)
                        attention = torch.softmax(attention, dim=-1)
                        output = torch.einsum('bij,bijc->bic', attention, filtered_features)
                        return output, attention
                    else:
                        # Sum over neighbors
                        output = filtered_features.sum(dim=2)
                        return output, None

            globals()['ContinuousFilterConv'] = ContinuousFilterConv

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
    Generate test data for comparing all SchNet implementations
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

    # For the full SchNet encoder, create complete model input
    # Data for old model (with nbr_adj_dist and nbr_adj_list)
    old_model_data = torch.cat([
        classic_rbf.reshape(batch_size, n_atoms, -1),  # flatten rbf expansion
        classic_neighbors.reshape(batch_size, n_atoms, -1)  # flatten neighbor list
    ], dim=-1)

    print(f"Classic features shape: {classic_features.shape}")
    print(f"Classic RBF shape: {classic_rbf.shape}")
    print(f"Classic neighbors shape: {classic_neighbors.shape}")
    print(f"PyG node features shape: {pyg_x.shape}")
    print(f"PyG edge index shape: {pyg_edge_index.shape}")
    print(f"PyG edge attr shape: {pyg_edge_attr.shape}")
    print(f"PyG batch indices shape: {pyg_batch.shape}")

    return {
        'classic': (classic_features, classic_rbf, classic_neighbors),
        'pyg_interaction': (pyg_x, pyg_edge_index, pyg_edge_attr),
        'pyg_full': (pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch),
        'old_model': old_model_data,
        'batch_size': batch_size,
        'n_atoms': n_atoms,
        'n_features': n_features,
        'n_gaussians': n_gaussians,
        'n_neighbors': n_neighbors
    }


def initialize_models(test_data):
    """
    Initialize all model implementations with equivalent parameters
    """
    n_features = test_data['n_features']
    n_gaussians = test_data['n_gaussians']
    n_atoms = test_data['n_atoms']

    # Initialize classic model
    classic_model = ClassicInteractionBlock(
        n_inputs=n_features,
        n_gaussians=n_gaussians,
        n_filters=n_features,
        activation=torch.nn.Tanh()
    )

    # Initialize PyG interaction model
    pyg_interaction = GCNInteraction(
        in_channels=n_features,
        edge_channels=n_gaussians,
        hidden_channels=n_features,
        activation='tanh',
        use_attention=True
    )

    # Initialize full SchNet encoder
    pyg_schnet = SchNetEncoderNoEmbed(
        node_dim=n_features,
        edge_dim=n_gaussians,
        hidden_dim=n_features,
        output_dim=n_features,  # Match output dimension for comparison
        n_interactions=1,  # Use single interaction for direct comparison
        activation='tanh',
        use_attention=True
    )

    # Set all models to evaluation mode
    classic_model.eval()
    pyg_interaction.eval()
    pyg_schnet.eval()

    return {
        'classic': classic_model,
        'pyg_interaction': pyg_interaction,
        'pyg_schnet': pyg_schnet,
    }


def copy_weights_between_models(models):
    """
    Copy weights from classic model to PyG models to ensure comparable outputs
    """
    classic_model = models['classic']
    pyg_interaction = models['pyg_interaction']
    pyg_schnet = models['pyg_schnet']

    print("Copying weights from classic model to PyG models...")

    # Copy weights to PyG interaction model
    # Initial dense layer
    pyg_interaction.initial_dense.weight.data = classic_model.initial_dense[0].weight.data.clone()

    # CFConv filter network
    pyg_interaction.cfconv.filter_network.lins[0].weight.data = classic_model.cfconv.filter_generator[
        0].weight.data.clone()
    pyg_interaction.cfconv.filter_network.lins[0].bias.data = classic_model.cfconv.filter_generator[0].bias.data.clone()
    pyg_interaction.cfconv.filter_network.lins[1].weight.data = classic_model.cfconv.filter_generator[
        2].weight.data.clone()
    pyg_interaction.cfconv.filter_network.lins[1].bias.data = classic_model.cfconv.filter_generator[2].bias.data.clone()

    # Attention weights
    if hasattr(classic_model.cfconv, 'nbr_filter') and hasattr(pyg_interaction.cfconv, 'attention_vector'):
        pyg_interaction.cfconv.attention_vector.data = classic_model.cfconv.nbr_filter.data.clone()

    # Output layers
    pyg_interaction.output_layer.lins[0].weight.data = classic_model.output_dense[0].weight.data.clone()
    pyg_interaction.output_layer.lins[0].bias.data = classic_model.output_dense[0].bias.data.clone()
    pyg_interaction.output_layer.lins[1].weight.data = classic_model.output_dense[2].weight.data.clone()
    pyg_interaction.output_layer.lins[1].bias.data = classic_model.output_dense[2].bias.data.clone()

    # Copy weights to full SchNet encoder (first interaction layer)
    # Initial dense layer
    pyg_schnet.interactions[0].initial_dense.weight.data = classic_model.initial_dense[0].weight.data.clone()

    # CFConv filter network
    pyg_schnet.interactions[0].cfconv.filter_network.lins[0].weight.data = classic_model.cfconv.filter_generator[
        0].weight.data.clone()
    pyg_schnet.interactions[0].cfconv.filter_network.lins[0].bias.data = classic_model.cfconv.filter_generator[
        0].bias.data.clone()
    pyg_schnet.interactions[0].cfconv.filter_network.lins[1].weight.data = classic_model.cfconv.filter_generator[
        2].weight.data.clone()
    pyg_schnet.interactions[0].cfconv.filter_network.lins[1].bias.data = classic_model.cfconv.filter_generator[
        2].bias.data.clone()

    # Attention weights
    if hasattr(classic_model.cfconv, 'nbr_filter') and hasattr(pyg_schnet.interactions[0].cfconv, 'attention_vector'):
        pyg_schnet.interactions[0].cfconv.attention_vector.data = classic_model.cfconv.nbr_filter.data.clone()

    # Output layers
    pyg_schnet.interactions[0].output_layer.lins[0].weight.data = classic_model.output_dense[0].weight.data.clone()
    pyg_schnet.interactions[0].output_layer.lins[0].bias.data = classic_model.output_dense[0].bias.data.clone()
    pyg_schnet.interactions[0].output_layer.lins[1].weight.data = classic_model.output_dense[2].weight.data.clone()
    pyg_schnet.interactions[0].output_layer.lins[1].bias.data = classic_model.output_dense[2].bias.data.clone()

    print("Weight copying complete")
    return models


def compare_outputs(outputs, test_data):
    """
    Compare the outputs of all models and visualize differences
    """
    classic_output = outputs['classic']
    pyg_interaction_output = outputs['pyg_interaction']
    pyg_schnet_output = outputs['pyg_schnet_node_features']  # Node features before pooling

    batch_size = test_data['batch_size']
    n_atoms = test_data['n_atoms']

    # Reshape PyG interaction output to match classic shape
    pyg_interaction_reshaped = torch.zeros_like(classic_output)

    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(pyg_interaction_output):
                pyg_interaction_reshaped[b, i] = pyg_interaction_output[idx]

    # Reshape PyG SchNet encoder output to match classic shape
    pyg_schnet_reshaped = torch.zeros_like(classic_output)

    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(pyg_schnet_output):
                pyg_schnet_reshaped[b, i] = pyg_schnet_output[idx]

    # Compute statistics on differences
    interaction_diff = torch.abs(classic_output - pyg_interaction_reshaped)
    interaction_mean_diff = interaction_diff.mean().item()
    interaction_max_diff = interaction_diff.max().item()

    schnet_diff = torch.abs(classic_output - pyg_schnet_reshaped)
    schnet_mean_diff = schnet_diff.mean().item()
    schnet_max_diff = schnet_diff.max().item()

    print("\n=== Classic vs PyG Interaction ===")
    print(f"Mean absolute difference: {interaction_mean_diff:.8f}")
    print(f"Maximum absolute difference: {interaction_max_diff:.8f}")

    print("\n=== Classic vs PyG SchNet Encoder ===")
    print(f"Mean absolute difference: {schnet_mean_diff:.8f}")
    print(f"Maximum absolute difference: {schnet_max_diff:.8f}")

    # Create visualization directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)

    # Visualize the outputs
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 3, 1)
    plt.imshow(classic_output[0].detach().numpy())
    plt.title("Classic Output")
    plt.colorbar()

    plt.subplot(3, 3, 2)
    plt.imshow(pyg_interaction_reshaped[0].detach().numpy())
    plt.title("PyG Interaction Output")
    plt.colorbar()

    plt.subplot(3, 3, 3)
    plt.imshow(interaction_diff[0].detach().numpy())
    plt.title("Classic vs PyG Interaction Diff")
    plt.colorbar()

    plt.subplot(3, 3, 4)
    plt.imshow(classic_output[0].detach().numpy())
    plt.title("Classic Output")
    plt.colorbar()

    plt.subplot(3, 3, 5)
    plt.imshow(pyg_schnet_reshaped[0].detach().numpy())
    plt.title("PyG SchNet Output")
    plt.colorbar()

    plt.subplot(3, 3, 6)
    plt.imshow(schnet_diff[0].detach().numpy())
    plt.title("Classic vs PyG SchNet Diff")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("visualizations/model_comparison.png", dpi=300)
    print("Saved output comparison visualization to 'visualizations/model_comparison.png'")

    return {
        'interaction_mean_diff': interaction_mean_diff,
        'interaction_max_diff': interaction_max_diff,
        'schnet_mean_diff': schnet_mean_diff,
        'schnet_max_diff': schnet_max_diff
    }


def time_performance(models, test_data, n_runs=10):
    """
    Time the performance of each model implementation
    """
    print("\n=== Performance Timing ===")

    # Extract test data
    classic_input = test_data['classic']
    pyg_interaction_input = test_data['pyg_interaction']
    pyg_full_input = test_data['pyg_full']

    # Extract models
    classic_model = models['classic']
    pyg_interaction = models['pyg_interaction']
    pyg_schnet = models['pyg_schnet']

    # Time classic model
    classic_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = classic_model(*classic_input)
        classic_times.append(time.time() - start)

    # Time PyG interaction model
    pyg_interaction_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = pyg_interaction(*pyg_interaction_input)
        pyg_interaction_times.append(time.time() - start)

    # Time full PyG SchNet encoder
    pyg_schnet_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = pyg_schnet(*pyg_full_input)
        pyg_schnet_times.append(time.time() - start)

    # Calculate average times
    avg_classic_time = sum(classic_times) / n_runs
    avg_pyg_interaction_time = sum(pyg_interaction_times) / n_runs
    avg_pyg_schnet_time = sum(pyg_schnet_times) / n_runs

    print(f"Classic Model: {avg_classic_time:.6f} seconds")
    print(
        f"PyG Interaction: {avg_pyg_interaction_time:.6f} seconds (speedup: {avg_classic_time / avg_pyg_interaction_time:.2f}x)")
    print(
        f"PyG SchNet Encoder: {avg_pyg_schnet_time:.6f} seconds (speedup: {avg_classic_time / avg_pyg_schnet_time:.2f}x)")

    # Create bar chart of performance
    plt.figure(figsize=(10, 6))
    plt.bar(['Classic', 'PyG Interaction', 'PyG SchNet'],
            [avg_classic_time, avg_pyg_interaction_time, avg_pyg_schnet_time],
            color=['blue', 'green', 'orange'])
    plt.title('Performance Comparison')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')  # Use log scale to better see differences

    # Add text labels with speedup
    plt.text(0, avg_classic_time, f"{avg_classic_time:.6f}s",
             ha='center', va='bottom')
    plt.text(1, avg_pyg_interaction_time,
             f"{avg_pyg_interaction_time:.6f}s\n({avg_classic_time / avg_pyg_interaction_time:.2f}x)",
             ha='center', va='bottom')
    plt.text(2, avg_pyg_schnet_time, f"{avg_pyg_schnet_time:.6f}s\n({avg_classic_time / avg_pyg_schnet_time:.2f}x)",
             ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("visualizations/performance_comparison.png", dpi=300)
    print("Saved performance comparison to 'visualizations/performance_comparison.png'")

    return {
        'classic_time': avg_classic_time,
        'pyg_interaction_time': avg_pyg_interaction_time,
        'pyg_schnet_time': avg_pyg_schnet_time
    }


def analyze_architecture_differences():
    """
    Analyze the architectural differences between the three implementations
    """
    print("\n=== Architectural Comparison ===")

    differences = [
        {
            "component": "Overall Architecture",
            "classic": "Custom dense tensor operations with manual neighbor handling",
            "pyg_interaction": "Uses PyG sparse graph representation with message passing",
            "pyg_schnet": "Full encoder with multiple interaction blocks and global pooling"
        },
        {
            "component": "Data Structure",
            "classic": "Dense tensors [batch, n_atoms, features], Explicit neighbor lists",
            "pyg_interaction": "Nodes: [n_nodes, features], Edges: COO format sparse adjacency",
            "pyg_schnet": "Same as PyG interaction plus batch assignment for pooling"
        },
        {
            "component": "Initial Layer",
            "classic": "LinearLayer wrapper with custom initialization",
            "pyg_interaction": "Standard nn.Linear without activation",
            "pyg_schnet": "Uses same structure as pyg_interaction per block"
        },
        {
            "component": "Filter Generation",
            "classic": "Custom sequential layers with manual tensor manipulation",
            "pyg_interaction": "MLP with consistent API",
            "pyg_schnet": "Same as pyg_interaction per block"
        },
        {
            "component": "Message Passing",
            "classic": "Manual gather operations on dense tensors",
            "pyg_interaction": "PyG's MessagePassing abstraction with built-in scatter",
            "pyg_schnet": "Same as pyg_interaction with additional residual connections"
        },
        {
            "component": "Attention Mechanism",
            "classic": "Manual implementation with matmul and softmax",
            "pyg_interaction": "PyG's optimized implementation",
            "pyg_schnet": "Same as pyg_interaction per block"
        },
        {
            "component": "Output Transformation",
            "classic": "Sequential layers built with LinearLayer",
            "pyg_interaction": "MLP with configurable layers",
            "pyg_schnet": "Per-block same as pyg_interaction, plus final global pooling"
        },
        {
            "component": "Residual Connection",
            "classic": "Added externally after block output",
            "pyg_interaction": "No residual in the block itself",
            "pyg_schnet": "Explicitly adds residuals: h = h + delta"
        }
    ]

    # Print as a table
    header = f"{'Component':<25} {'Classic Model':<40} {'PyG Interaction':<40} {'PyG SchNet Encoder':<40}"
    print('-' * len(header))
    print(header)
    print('-' * len(header))

    for diff in differences:
        print(f"{diff['component']:<25} {diff['classic']:<40} {diff['pyg_interaction']:<40} {diff['pyg_schnet']:<40}")

    return differences


def run_full_comparison_old():
    """
    Run a comprehensive comparison of all three implementations
    """
    print("\n===== SchNet Implementation Comparison =====")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate test data
    test_data = generate_test_data(batch_size=2, n_atoms=20, n_neighbors=10)

    # Initialize models
    models = initialize_models(test_data)

    # Copy weights from classic to PyG models for fair comparison
    models = copy_weights_between_models(models)

    # Run all models on test data
    with torch.no_grad():
        # Run classic model
        classic_output, classic_attn = models['classic'](*test_data['classic'])

        # Run PyG interaction model
        pyg_interaction_output, pyg_interaction_attn = models['pyg_interaction'](*test_data['pyg_interaction'])

        # Run PyG SchNet encoder
        pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch = test_data['pyg_full']
        pyg_schnet_output, (pyg_schnet_node_features, _) = models['pyg_schnet'](pyg_x, pyg_edge_index, pyg_edge_attr,
                                                                                pyg_batch)

    outputs = {
        'classic': classic_output,
        'pyg_interaction': pyg_interaction_output,
        'pyg_schnet_output': pyg_schnet_output,
        'pyg_schnet_node_features': pyg_schnet_node_features
    }

    # Compare outputs
    diff_stats = compare_outputs(outputs, test_data)

    # Time performance
    perf_stats = time_performance(models, test_data)

    # Analyze architecture differences
    arch_diffs = analyze_architecture_differences()

    print("\n===== Conclusions =====")
    if diff_stats['interaction_mean_diff'] < 1e-5 and diff_stats['schnet_mean_diff'] < 1e-5:
        print("✅ All implementations produce equivalent output when properly configured.")
    else:
        print("⚠️ There are differences between implementations that should be investigated.")

    fastest = min(perf_stats, key=perf_stats.get)
    print(f"Fastest implementation: {fastest.replace('_time', '')}")

    print("\nRecommendation:")
    if perf_stats['pyg_schnet_time'] < perf_stats['classic_time'] * 1.2:
        print("Use the PyG SchNet Encoder as it offers the most features and good performance.")
    elif perf_stats['pyg_interaction_time'] < perf_stats['classic_time']:
        print("Use the PyG Interaction block for best balance of performance and compatibility.")
    else:
        print("Use the classic implementation if compatibility with existing code is important.")

    return {
        'models': models,
        'outputs': outputs,
        'diff_stats': diff_stats,
        'perf_stats': perf_stats,
        'arch_diffs': arch_diffs
    }


def run_full_comparison():
    """Run a fixed comparison that correctly aligns outputs"""
    # Generate test data
    test_data = generate_test_data(batch_size=2, n_atoms=20, n_neighbors=10)

    # Initialize models
    models = initialize_models(test_data)

    # Copy weights
    models = copy_weights_between_models(models)

    # Run models with proper output extraction
    with torch.no_grad():
        # Run classic model
        classic_output, classic_attn = models['classic'](*test_data['classic'])

        # Run PyG interaction model
        pyg_interaction_output, pyg_interaction_attn = models['pyg_interaction'](*test_data['pyg_interaction'])

        # For PyG SchNet, capture intermediate outputs
        pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch = test_data['pyg_full']

        # Manually execute the SchNet encoder to capture delta before adding residual
        initial_h = pyg_x  # Save initial node features

        # Run just the first interaction (without applying residual)
        delta, attention = models['pyg_schnet'].interactions[0](pyg_x, pyg_edge_index, pyg_edge_attr)

        # Now delta is directly comparable to classic_output
        pyg_schnet_delta = delta

    # Reshape PyG interaction output
    batch_size = test_data['batch_size']
    n_atoms = test_data['n_atoms']

    pyg_interaction_reshaped = torch.zeros_like(classic_output)
    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(pyg_interaction_output):
                pyg_interaction_reshaped[b, i] = pyg_interaction_output[idx]

    # Reshape PyG SchNet delta output
    pyg_schnet_delta_reshaped = torch.zeros_like(classic_output)
    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            if idx < len(pyg_schnet_delta):
                pyg_schnet_delta_reshaped[b, i] = pyg_schnet_delta[idx]

    # Compare outputs
    interaction_diff = torch.abs(classic_output - pyg_interaction_reshaped)
    interaction_mean_diff = interaction_diff.mean().item()
    interaction_max_diff = interaction_diff.max().item()

    # Compare with delta instead of node_features
    schnet_diff = torch.abs(classic_output - pyg_schnet_delta_reshaped)
    schnet_mean_diff = schnet_diff.mean().item()
    schnet_max_diff = schnet_diff.max().item()

    print("\n=== Classic vs PyG Interaction ===")
    print(f"Mean absolute difference: {interaction_mean_diff:.8f}")
    print(f"Maximum absolute difference: {interaction_max_diff:.8f}")

    print("\n=== Classic vs PyG SchNet Delta (before residual) ===")
    print(f"Mean absolute difference: {schnet_mean_diff:.8f}")
    print(f"Maximum absolute difference: {schnet_max_diff:.8f}")

    return {
        'interaction_mean_diff': interaction_mean_diff,
        'interaction_max_diff': interaction_max_diff,
        'schnet_mean_diff': schnet_mean_diff,
        'schnet_max_diff': schnet_max_diff
    }


def run_detailed_debug():
    """Run detailed debugging to find the exact mismatch point"""
    print("\n===== Detailed SchNet Debugging =====")

    # Generate test data
    test_data = generate_test_data(batch_size=2, n_atoms=20, n_neighbors=10)

    # Initialize models
    models = initialize_models(test_data)
    models = copy_weights_between_models(models)

    # Extract inputs
    classic_features, classic_rbf, classic_neighbors = test_data['classic']
    pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch = test_data['pyg_full']

    # First get classic output for reference
    with torch.no_grad():
        classic_output, classic_attn = models['classic'](*test_data['classic'])

    # Now let's trace through PyG schnet step by step
    with torch.no_grad():
        schnet = models['pyg_schnet']

        # STEP 1: Initial state
        print("\nSTEP 1: Initial node features")
        h0 = pyg_x.clone()  # Initial node features
        print(f"  Initial PyG features shape: {h0.shape}")

        # STEP 2: First interaction block (before residual)
        print("\nSTEP 2: First interaction output (before residual)")
        interaction0 = schnet.interactions[0]
        delta0, attn0 = interaction0(h0, pyg_edge_index, pyg_edge_attr)
        print(f"  PyG interaction delta shape: {delta0.shape}")

        # Compare with classic output
        batch_size = test_data['batch_size']
        n_atoms = test_data['n_atoms']

        # Convert delta to classic shape
        delta_classic = torch.zeros_like(classic_output)
        for b in range(batch_size):
            for i in range(n_atoms):
                idx = i + b * n_atoms
                if idx < len(delta0):
                    delta_classic[b, i] = delta0[idx]

        # Compare
        diff1 = (delta_classic - classic_output).abs().mean()
        print(f"  Mean diff between classic output and delta: {diff1:.8f}")

        # STEP 3: After residual connection
        print("\nSTEP 3: After residual connection")
        h1 = h0 + delta0  # Apply residual connection

        # Convert h1 to classic shape
        h1_classic = torch.zeros_like(classic_output)
        for b in range(batch_size):
            for i in range(n_atoms):
                idx = i + b * n_atoms
                if idx < len(h1):
                    h1_classic[b, i] = h1[idx]

        # Compare with: classic_features + classic_output
        classic_after_residual = classic_features + classic_output
        diff2 = (h1_classic - classic_after_residual).abs().mean()
        print(f"  Mean diff after residual: {diff2:.8f}")

        # Extract values for inspection
        print("\nSample values for first node:")
        print(f"  Classic original features: {classic_features[0, 0, :5]}...")
        print(f"  PyG original features: {h0[0, :5]}...")
        print(f"  Classic delta: {classic_output[0, 0, :5]}...")
        print(f"  PyG delta: {delta0[0, :5]}...")
        print(f"  Classic after residual: {classic_after_residual[0, 0, :5]}...")
        print(f"  PyG after residual: {h1[0, :5]}...")

    return {
        'h0': h0,
        'delta0': delta0,
        'h1': h1,
        'classic_output': classic_output,
        'diff1': diff1,
        'diff2': diff2
    }


if __name__ == "__main__":
    try:
        results = run_full_comparison()
        run_detailed_debug()
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        import traceback

        traceback.print_exc()
