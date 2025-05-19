import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# Create directory for outputs if it doesn't exist
os.makedirs("cfconv_comparison", exist_ok=True)


# Helper functions for initialization
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.fill_(0.0)


# MLP implementation for both CFConv versions
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 act='tanh', norm=None, plain_last=False):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Create activation function
        if isinstance(act, str):
            if act == 'relu':
                self.activation = nn.ReLU()
            elif act == 'leaky_relu':
                self.activation = nn.LeakyReLU(0.1)
            elif act == 'elu':
                self.activation = nn.ELU()
            elif act == 'tanh':
                self.activation = nn.Tanh()
            else:
                raise ValueError(f"Unknown activation: {act}")
        else:
            self.activation = act

        # Create normalization layers if specified
        self.norm = norm

        # Create MLP
        self.lins = nn.ModuleList()

        # Special case for single layer
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            # First layer
            self.lins.append(nn.Linear(in_channels, hidden_channels))

            # Middle layers
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))

            # Last layer
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.plain_last = plain_last

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            # Apply activation and normalization to all but the last layer (if plain_last=True)
            if i < len(self.lins) - 1 or not self.plain_last:
                x = self.activation(x)
                if self.norm is not None:
                    x = self.norm(x)
        return x


# CFConv implementation using MessagePassing
class MessagePassingCFConv(MessagePassing):
    """
    Continuous Filter Convolution for PyTorch Geometric graph data using MessagePassing.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_channels,
                 hidden_channels=None,
                 activation='tanh',
                 use_attention=True,
                 aggr='add'):

        super(MessagePassingCFConv, self).__init__(aggr=aggr)

        # Set default hidden channels if not provided
        if hidden_channels is None:
            hidden_channels = out_channels

        # Set channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        # Convert string activation to module if needed
        self.activation = self._get_activation(activation) if isinstance(activation, str) else activation

        # Filter network transforms edge features to weights
        self.filter_network = MLP(
            in_channels=edge_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            act=self.activation,
            norm=None,
            plain_last=True
        ).apply(init_weights)

        # Add node projection layer if input and output dimensions differ
        self.has_node_projection = (in_channels != out_channels)
        if self.has_node_projection:
            self.node_projection = nn.Linear(in_channels, out_channels)

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention_vector = nn.Parameter(torch.Tensor(out_channels, 1))
            nn.init.xavier_uniform_(self.attention_vector, gain=1.414)

        # For storing attention weights
        self._attention_weights = None

    def _get_activation(self, activation_name):
        """Convert activation name to module"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations.get(activation_name.lower(), nn.Tanh())

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of CFConv layer.
        """
        # Normalize edge attributes for numerical stability
        edge_attr_norm = edge_attr / (edge_attr.norm(dim=-1, keepdim=True) + 1e-8)

        # Generate weights from edge attributes
        edge_weights = self.filter_network(edge_attr_norm)

        # Clear previous attention weights
        self._attention_weights = None

        # Project input features to output dimension if needed
        if self.has_node_projection:
            x = self.node_projection(x)

        # Message passing
        out = self.propagate(edge_index, x=x, edge_weights=edge_weights)

        return out, self._attention_weights

    def message(self, x_j, edge_weights, edge_index_i):
        """
        Optimized message function that uses PyG's built-in softmax for better performance.
        """
        # Apply edge weights to source node features (keep dimension check for debugging)
        if x_j.size(1) != edge_weights.size(1):
            print(f"Dimension mismatch - x_j: {x_j.shape}, edge_weights: {edge_weights.shape}")

        # Compute messages by element-wise multiplication
        messages = x_j * edge_weights

        # If using attention, compute attention weights with PyG's optimized functions
        if self.use_attention:
            # Compute attention score for each edge
            attention = torch.matmul(messages, self.attention_vector).squeeze(-1)

            # Use PyG's optimized softmax instead of manual per-node implementation
            normalized_attention = softmax(attention, edge_index_i)

            # Apply attention weights to messages
            messages = messages * normalized_attention.view(-1, 1)

            # Store raw attention weights for later access
            self._attention_weights = attention

        return messages

    def update(self, aggr_out):
        """No additional update step needed"""
        return aggr_out


# CFConv implementation using manual operations
class ManualCFConv(torch.nn.Module):
    """Continuous Filter Convolution using manual operations."""

    def __init__(self, in_channels, out_channels, edge_channels, hidden_channels, activation='tanh',
                 use_attention=True):
        super(ManualCFConv, self).__init__()

        # Set channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert string activation to module if needed
        if isinstance(activation, str):
            if activation == 'tanh':
                self.activation = torch.nn.Tanh()
            else:
                self.activation = torch.nn.ReLU()
        else:
            self.activation = activation

        # Filter network transforms edge features to weights
        self.filter_network = MLP(
            in_channels=edge_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            act=self.activation
        )

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention_vector = torch.nn.Parameter(torch.Tensor(out_channels, 1))
            torch.nn.init.xavier_uniform_(self.attention_vector, gain=1.414)

    def forward(self, x, edge_index, edge_attr):
        """Forward pass with calculations similar to classic CFConv."""
        # Generate weights from edge attributes
        edge_weights = self.filter_network(edge_attr)

        # Extract source and target nodes
        source = edge_index[0]
        target = edge_index[1]

        # Get feature vectors for source nodes
        source_features = x[source]

        # Compute messages as product of source features and edge weights
        messages = source_features * edge_weights

        # Initialize attention weights
        attention_weights = None

        # Apply attention if enabled
        if self.use_attention:
            # Compute attention scores
            attn_scores = torch.matmul(messages, self.attention_vector).squeeze(-1)

            # Group by target node for softmax
            target_nodes = torch.unique(target)
            attention_weights = torch.zeros_like(attn_scores)

            # Apply softmax per target node
            for node in target_nodes:
                mask = (target == node)
                node_scores = attn_scores[mask]
                node_attention = torch.softmax(node_scores, dim=0)
                attention_weights[mask] = node_attention

            # Apply attention to messages
            messages = messages * attention_weights.unsqueeze(-1)

        # Sum messages for each target node
        output = torch.zeros((x.size(0), self.out_channels), device=x.device)
        for i in range(len(source)):
            output[target[i]] += messages[i]

        return output, attention_weights


# Generate synthetic graph data for testing
def generate_test_data(num_nodes=100, edge_density=0.1, feature_dim=32, edge_attr_dim=16, device='cpu'):
    """Generate synthetic graph data for testing"""
    # Generate node features
    x = torch.randn(num_nodes, feature_dim, device=device)

    # Generate edges (random graph)
    num_edges = int(num_nodes * num_nodes * edge_density)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    # Generate edge attributes
    edge_attr = torch.randn(num_edges, edge_attr_dim, device=device)

    return x, edge_index, edge_attr


# Functional comparison between implementations
def compare_outputs(message_passing_output, manual_output):
    """Compare outputs from both implementations"""
    # Compare the actual node embeddings
    diff = torch.abs(message_passing_output[0] - manual_output[0])
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    # Compare attention weights if available
    attn_diff = None
    if message_passing_output[1] is not None and manual_output[1] is not None:
        attn_diff = torch.abs(message_passing_output[1] - manual_output[1])
        attn_mean_diff = attn_diff.mean().item()
        attn_max_diff = attn_diff.max().item()
    else:
        attn_mean_diff = float('nan')
        attn_max_diff = float('nan')

    return {
        'output_mean_diff': mean_diff,
        'output_max_diff': max_diff,
        'attn_mean_diff': attn_mean_diff,
        'attn_max_diff': attn_max_diff
    }


# Performance comparison between implementations
def benchmark_performance(message_passing_cfconv, manual_cfconv, x, edge_index, edge_attr, n_runs=100):
    """Benchmark performance of both implementations"""
    # Ensure models are in eval mode for fair comparison
    message_passing_cfconv.eval()
    manual_cfconv.eval()

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            message_passing_cfconv(x, edge_index, edge_attr)
            manual_cfconv(x, edge_index, edge_attr)

    # Time MessagePassing implementation
    mp_times = []
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad():
            message_passing_cfconv(x, edge_index, edge_attr)
        mp_times.append(time.time() - start_time)

    # Time manual implementation
    manual_times = []
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad():
            manual_cfconv(x, edge_index, edge_attr)
        manual_times.append(time.time() - start_time)

    avg_mp_time = sum(mp_times) / n_runs
    avg_manual_time = sum(manual_times) / n_runs
    speedup = avg_manual_time / avg_mp_time

    return {
        'avg_message_passing_time': avg_mp_time,
        'avg_manual_time': avg_manual_time,
        'speedup': speedup,
        'mp_times': mp_times,
        'manual_times': manual_times
    }


# Visualization of outputs and performance
def visualize_comparison(x, message_passing_output, manual_output, perf_stats):
    """Create visualizations to compare outputs and performance"""
    # Create directory for visualizations
    os.makedirs("cfconv_comparison/visualizations", exist_ok=True)

    # 1. Output comparison visualization
    plt.figure(figsize=(15, 5))

    # Plot a sample of node features
    sample_size = min(10, x.size(0))
    feature_idx = 0  # Feature dimension to visualize

    plt.subplot(1, 3, 1)
    plt.bar(range(sample_size), message_passing_output[0][:sample_size, feature_idx].detach().cpu().numpy())
    plt.title("MessagePassing Output")
    plt.xlabel("Node Index")
    plt.ylabel(f"Feature {feature_idx} Value")

    plt.subplot(1, 3, 2)
    plt.bar(range(sample_size), manual_output[0][:sample_size, feature_idx].detach().cpu().numpy())
    plt.title("Manual Output")
    plt.xlabel("Node Index")

    plt.subplot(1, 3, 3)
    diff = torch.abs(message_passing_output[0] - manual_output[0])
    plt.bar(range(sample_size), diff[:sample_size, feature_idx].detach().cpu().numpy())
    plt.title("Absolute Difference")
    plt.xlabel("Node Index")

    plt.tight_layout()
    plt.savefig("cfconv_comparison/visualizations/output_comparison.png", dpi=300)
    plt.close()

    # 2. Performance comparison
    plt.figure(figsize=(10, 6))

    # Plot average times
    plt.bar(['MessagePassing', 'Manual'],
            [perf_stats['avg_message_passing_time'], perf_stats['avg_manual_time']])

    # Add text labels
    plt.text(0, perf_stats['avg_message_passing_time'],
             f"{perf_stats['avg_message_passing_time']:.6f}s",
             ha='center', va='bottom')

    plt.text(1, perf_stats['avg_manual_time'],
             f"{perf_stats['avg_manual_time']:.6f}s",
             ha='center', va='bottom')

    # Add speedup annotation
    plt.annotate(f"Speedup: {perf_stats['speedup']:.2f}x",
                 xy=(0.5, max(perf_stats['avg_message_passing_time'], perf_stats['avg_manual_time']) * 0.9),
                 ha='center', va='center',
                 bbox=dict(boxstyle='round', fc='yellow', alpha=0.3))

    plt.title("Performance Comparison")
    plt.ylabel("Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig("cfconv_comparison/visualizations/performance_comparison.png", dpi=300)
    plt.close()

    # 3. Performance distribution
    plt.figure(figsize=(10, 6))

    plt.hist(perf_stats['mp_times'], alpha=0.5, bins=30, label='MessagePassing')
    plt.hist(perf_stats['manual_times'], alpha=0.5, bins=30, label='Manual')

    plt.axvline(perf_stats['avg_message_passing_time'], color='blue', linestyle='--',
                label=f'MessagePassing Mean: {perf_stats["avg_message_passing_time"]:.6f}s')
    plt.axvline(perf_stats['avg_manual_time'], color='orange', linestyle='--',
                label=f'Manual Mean: {perf_stats["avg_manual_time"]:.6f}s')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Execution Times')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    plt.savefig("cfconv_comparison/visualizations/performance_distribution.png", dpi=300)
    plt.close()


# Code structure comparison
def compare_code_structure():
    """Analyze and compare the code structure of both implementations"""

    comparison = {
        "Base Class": {
            "MessagePassing": "Inherits from PyG's MessagePassing class",
            "Manual": "Standard torch.nn.Module"
        },
        "Message Function": {
            "MessagePassing": "Uses PyG's message/propagate framework",
            "Manual": "Manually extracts source/target nodes and computes messages"
        },
        "Aggregation": {
            "MessagePassing": "Handled by PyG's propagate mechanism",
            "Manual": "Manual accumulation with output[target] += message"
        },
        "Attention": {
            "MessagePassing": "Uses PyG's softmax for target-grouped normalization",
            "Manual": "Iterates through unique target nodes for softmax"
        },
        "Performance Optimizations": {
            "MessagePassing": "Has JIT compiled helper functions and utilizes PyG's optimized operations",
            "Manual": "Pure PyTorch operations without specialized optimizations"
        },
        "Edge Feature Normalization": {
            "MessagePassing": "Normalizes edge features for numerical stability",
            "Manual": "No normalization of edge features"
        },
        "Memory Efficiency": {
            "MessagePassing": "Better due to PyG's optimized sparse operations",
            "Manual": "Manual indexing may create more temporary tensors"
        }
    }

    return comparison


# Main function to run all comparisons
def compare_cfconv_implementations(use_cuda=False):
    """Run comprehensive comparison between CFConv implementations"""
    print("===== CFConv Implementation Comparison =====\n")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f"Using device: {device}")

    # Parameters for test
    node_dim = 32
    edge_dim = 16
    hidden_dim = 64
    out_dim = 32

    # Generate test data
    print("Generating test data...")
    x, edge_index, edge_attr = generate_test_data(
        num_nodes=1000,
        edge_density=0.01,
        feature_dim=node_dim,
        edge_attr_dim=edge_dim,
        device=device
    )
    print(f"Generated data - Nodes: {x.shape}, Edges: {edge_index.shape}, Edge attr: {edge_attr.shape}")

    # Initialize both implementations
    print("Initializing CFConv implementations...")
    mp_cfconv = MessagePassingCFConv(
        in_channels=node_dim,
        out_channels=out_dim,
        edge_channels=edge_dim,
        hidden_channels=hidden_dim,
        activation='tanh',
        use_attention=True
    ).to(device)

    manual_cfconv = ManualCFConv(
        in_channels=node_dim,
        out_channels=out_dim,
        edge_channels=edge_dim,
        hidden_channels=hidden_dim,
        activation='tanh',
        use_attention=True
    ).to(device)

    # Copy weights to ensure fair comparison
    print("Copying weights for fair comparison...")

    # Copy filter network weights
    manual_cfconv.filter_network.lins[0].weight.data = mp_cfconv.filter_network.lins[0].weight.data.clone()
    manual_cfconv.filter_network.lins[0].bias.data = mp_cfconv.filter_network.lins[0].bias.data.clone()
    manual_cfconv.filter_network.lins[1].weight.data = mp_cfconv.filter_network.lins[1].weight.data.clone()
    manual_cfconv.filter_network.lins[1].bias.data = mp_cfconv.filter_network.lins[1].bias.data.clone()

    # Copy attention vector
    manual_cfconv.attention_vector.data = mp_cfconv.attention_vector.data.clone()

    # Run both implementations
    print("Running both implementations...")
    with torch.no_grad():
        mp_output = mp_cfconv(x, edge_index, edge_attr)
        manual_output = manual_cfconv(x, edge_index, edge_attr)

    # Compare outputs
    print("Comparing outputs...")
    output_comparison = compare_outputs(mp_output, manual_output)
    print(f"Output mean difference: {output_comparison['output_mean_diff']:.8f}")
    print(f"Output max difference: {output_comparison['output_max_diff']:.8f}")
    print(f"Attention mean difference: {output_comparison['attn_mean_diff']:.8f}")
    print(f"Attention max difference: {output_comparison['attn_max_diff']:.8f}")

    # Benchmark performance
    print("\nBenchmarking performance...")
    perf_stats = benchmark_performance(mp_cfconv, manual_cfconv, x, edge_index, edge_attr)
    print(f"MessagePassing avg time: {perf_stats['avg_message_passing_time']:.6f} seconds")
    print(f"Manual avg time: {perf_stats['avg_manual_time']:.6f} seconds")
    print(f"Speedup: {perf_stats['speedup']:.2f}x")

    # Visualize results
    print("\nCreating visualizations...")
    visualize_comparison(x, mp_output, manual_output, perf_stats)

    # Compare code structure
    print("\nComparing code structure...")
    code_comparison = compare_code_structure()
    for category, details in code_comparison.items():
        print(f"\n{category}:")
        print(f"  MessagePassing: {details['MessagePassing']}")
        print(f"  Manual: {details['Manual']}")

    # Generate final summary
    print("\n===== Summary =====")
    if output_comparison['output_mean_diff'] < 1e-5:
        print("✅ Both implementations produce functionally identical outputs")
    elif output_comparison['output_mean_diff'] < 1e-3:
        print("⚠️ Minor differences in outputs (likely due to floating point precision)")
    else:
        print("❌ Significant differences in outputs")

    if perf_stats['speedup'] > 1.1:
        print(f"✅ MessagePassing implementation is {perf_stats['speedup']:.2f}x faster")
    elif perf_stats['speedup'] < 0.9:
        print(f"❌ Manual implementation is {1 / perf_stats['speedup']:.2f}x faster")
    else:
        print("✅ Both implementations have comparable performance")

    print("\nRecommendation:")
    if output_comparison['output_mean_diff'] < 1e-3 and perf_stats['speedup'] > 1.0:
        print("Use the MessagePassing implementation for better performance and cleaner code structure")
    elif output_comparison['output_mean_diff'] < 1e-3 and perf_stats['speedup'] < 1.0:
        print("Use the Manual implementation for better performance")
    else:
        print("Investigate differences further before choosing an implementation")

    return {
        'data': {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        },
        'models': {
            'message_passing': mp_cfconv,
            'manual': manual_cfconv
        },
        'outputs': {
            'message_passing': mp_output,
            'manual': manual_output
        },
        'comparisons': {
            'output': output_comparison,
            'performance': perf_stats,
            'code': code_comparison
        }
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run the comparison
    try:
        results = compare_cfconv_implementations()
    except Exception as e:
        print(f"Error in comparison: {str(e)}")
        import traceback

        traceback.print_exc()
