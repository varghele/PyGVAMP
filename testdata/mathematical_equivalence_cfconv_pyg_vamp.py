import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.optim import Adam
from torch_geometric.utils import softmax

# Import your specific CFConv implementation
from pygv.encoder.schnet_wo_embed_v2 import CFConv as PyGVCFConv

# Create directory for outputs
os.makedirs("cfconv_comparison", exist_ok=True)


# Reference implementation of CFConv
class ReferenceCFConv(nn.Module):
    """Reference implementation of Continuous Filter Convolution with attention"""

    def __init__(self, in_channels, out_channels, edge_channels, hidden_channels=None,
                 activation='tanh', use_attention=True):
        super().__init__()

        # Set default hidden channels if not provided
        if hidden_channels is None:
            hidden_channels = out_channels

        # Store channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        # Convert string activation to module if needed
        if isinstance(activation, str):
            if activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            else:
                self.activation = nn.Tanh()
        else:
            self.activation = activation

        # Create filter network
        self.filter_network = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            self.activation,
            nn.Linear(hidden_channels, out_channels)
        )

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention_vector = nn.Parameter(torch.Tensor(out_channels, 1))
            nn.init.xavier_uniform_(self.attention_vector, gain=1.414)

    def forward(self, x, edge_index, edge_attr):
        # Extract source and target nodes
        source, target = edge_index

        # Get source node features
        source_features = x[source]

        # Normalize edge attributes
        edge_attr_norm = edge_attr / (edge_attr.norm(dim=-1, keepdim=True) + 1e-8)

        # Generate filters
        edge_filters = self.filter_network(edge_attr_norm)

        # Apply filters to source features
        messages = source_features * edge_filters

        # Apply attention if enabled
        if self.use_attention:
            # Compute attention scores
            attention_scores = torch.matmul(messages, self.attention_vector).squeeze(-1)

            # Normalize with softmax per target node
            attention_weights = softmax(attention_scores, target)

            # Apply attention weights
            messages = messages * attention_weights.view(-1, 1)
        else:
            attention_weights = None

        # Initialize output tensor
        output = torch.zeros(x.size(0), self.out_channels, device=x.device)

        # Aggregate messages
        for i in range(source.size(0)):
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


# Simple training function to test mathematical equivalence
def train_and_compare(pygv_cfconv, reference_cfconv, x, edge_index, edge_attr, epochs=10, lr=0.01):
    """Train both models and compare their outputs and gradients"""

    # Create optimizers
    pygv_optimizer = Adam(pygv_cfconv.parameters(), lr=lr)
    ref_optimizer = Adam(reference_cfconv.parameters(), lr=lr)

    # Create loss function (simple MSE against a target)
    target = torch.randn_like(x)

    # Store losses for comparison
    pygv_losses = []
    ref_losses = []
    output_diffs = []
    grad_diffs = []

    print(f"\n{'Epoch':^6} | {'PyGV Loss':^12} | {'Ref Loss':^12} | {'Out Diff':^12} | {'Grad Diff':^12}")
    print("-" * 65)

    for epoch in range(epochs):
        # Forward pass for PyGV model
        pygv_optimizer.zero_grad()
        pygv_output, pygv_attn = pygv_cfconv(x, edge_index, edge_attr)
        pygv_loss = F.mse_loss(pygv_output, target)
        pygv_loss.backward()

        # Forward pass for reference model
        ref_optimizer.zero_grad()
        ref_output, ref_attn = reference_cfconv(x, edge_index, edge_attr)
        ref_loss = F.mse_loss(ref_output, target)
        ref_loss.backward()

        # Compare outputs
        output_diff = (pygv_output - ref_output).abs().mean().item()

        # Compare gradients
        grad_diff = 0.0
        grad_count = 0

        # Compare filter network gradients
        for p1, p2 in zip(pygv_cfconv.filter_network.parameters(), reference_cfconv.filter_network.parameters()):
            if p1.grad is not None and p2.grad is not None:
                grad_diff += (p1.grad - p2.grad).abs().mean().item()
                grad_count += 1

        # Compare attention vector gradients
        if pygv_cfconv.use_attention and reference_cfconv.use_attention:
            attn_grad_diff = (
                        pygv_cfconv.attention_vector.grad - reference_cfconv.attention_vector.grad).abs().mean().item()
            grad_diff += attn_grad_diff
            grad_count += 1

        grad_diff = grad_diff / max(1, grad_count)  # Average gradient difference

        # Store values
        pygv_losses.append(pygv_loss.item())
        ref_losses.append(ref_loss.item())
        output_diffs.append(output_diff)
        grad_diffs.append(grad_diff)

        # Print progress
        print(f"{epoch + 1:^6} | {pygv_loss.item():^12.6f} | {ref_loss.item():^12.6f} | "
              f"{output_diff:^12.6f} | {grad_diff:^12.6f}")

        # Update weights
        pygv_optimizer.step()
        ref_optimizer.step()

    # Final forward pass for comparison
    with torch.no_grad():
        final_pygv_output, final_pygv_attn = pygv_cfconv(x, edge_index, edge_attr)
        final_ref_output, final_ref_attn = reference_cfconv(x, edge_index, edge_attr)

    # Compare final attention weights
    if final_pygv_attn is not None and final_ref_attn is not None:
        attn_diff = (final_pygv_attn - final_ref_attn).abs().mean().item()
    else:
        attn_diff = float('nan')

    return {
        'pygv_losses': pygv_losses,
        'ref_losses': ref_losses,
        'output_diffs': output_diffs,
        'grad_diffs': grad_diffs,
        'final_output_diff': (final_pygv_output - final_ref_output).abs().mean().item(),
        'final_attn_diff': attn_diff,
        'final_pygv_output': final_pygv_output,
        'final_ref_output': final_ref_output,
        'final_pygv_attn': final_pygv_attn,
        'final_ref_attn': final_ref_attn
    }


# Visualize training results
def visualize_results(results):
    """Visualize training results"""
    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(results['pygv_losses'], label='PyGV CFConv')
    plt.plot(results['ref_losses'], label='Reference CFConv')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot differences
    plt.subplot(2, 2, 2)
    plt.plot(results['output_diffs'], label='Output Difference')
    plt.plot(results['grad_diffs'], label='Gradient Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Difference')
    plt.title('Differences During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot final outputs (first 10 nodes, first 5 features)
    plt.subplot(2, 2, 3)
    num_nodes = min(10, results['final_pygv_output'].size(0))
    num_features = min(5, results['final_pygv_output'].size(1))

    for f in range(num_features):
        plt.plot(results['final_pygv_output'][:num_nodes, f].cpu().numpy(), 'o-',
                 label=f'PyGV Feature {f}', alpha=0.7)
        plt.plot(results['final_ref_output'][:num_nodes, f].cpu().numpy(), 's--',
                 label=f'Ref Feature {f}', alpha=0.7)

    plt.xlabel('Node Index')
    plt.ylabel('Feature Value')
    plt.title('Final Output Comparison (Sample)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot attention weights if available
    plt.subplot(2, 2, 4)
    if results['final_pygv_attn'] is not None and results['final_ref_attn'] is not None:
        num_edges = min(20, results['final_pygv_attn'].size(0))

        plt.plot(results['final_pygv_attn'][:num_edges].cpu().numpy(), 'o-',
                 label='PyGV Attention', alpha=0.7)
        plt.plot(results['final_ref_attn'][:num_edges].cpu().numpy(), 's--',
                 label='Ref Attention', alpha=0.7)

        plt.xlabel('Edge Index')
        plt.ylabel('Attention Weight')
        plt.title('Final Attention Weight Comparison (Sample)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Attention weights not available',
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig('cfconv_comparison/training_comparison.png', dpi=300)
    plt.close()


def main():
    """Main function to test mathematical equivalence"""
    print("===== CFConv Mathematical Equivalence Test =====")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters
    node_dim = 32
    edge_dim = 16
    hidden_dim = 64

    # Generate test data
    print("Generating test data...")
    x, edge_index, edge_attr = generate_test_data(
        num_nodes=500,  # Medium-sized graph for testing
        edge_density=0.01,
        feature_dim=node_dim,
        edge_attr_dim=edge_dim,
        device=device
    )

    print(f"Graph: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

    # Initialize models
    print("Initializing models...")
    pygv_cfconv = PyGVCFConv(
        in_channels=node_dim,
        out_channels=node_dim,  # Same in/out dimensions for simplicity
        edge_channels=edge_dim,
        hidden_channels=hidden_dim,
        activation='tanh',
        use_attention=True
    ).to(device)

    reference_cfconv = ReferenceCFConv(
        in_channels=node_dim,
        out_channels=node_dim,
        edge_channels=edge_dim,
        hidden_channels=hidden_dim,
        activation='tanh',
        use_attention=True
    ).to(device)

    # Copy weights for fair comparison
    print("Copying weights for initialization...")

    # Copy filter network weights
    # Note: Structure may differ, so we need to match correctly
    if hasattr(pygv_cfconv, 'filter_network') and hasattr(reference_cfconv, 'filter_network'):
        # Assuming both use a Sequential with Linear layers
        pygv_layers = [m for m in pygv_cfconv.filter_network.modules() if isinstance(m, nn.Linear)]
        ref_layers = [m for m in reference_cfconv.filter_network.modules() if isinstance(m, nn.Linear)]

        if len(pygv_layers) >= 2 and len(ref_layers) >= 2:
            # Copy first layer
            ref_layers[0].weight.data.copy_(pygv_layers[0].weight.data)
            ref_layers[0].bias.data.copy_(pygv_layers[0].bias.data)

            # Copy second layer
            ref_layers[1].weight.data.copy_(pygv_layers[1].weight.data)
            ref_layers[1].bias.data.copy_(pygv_layers[1].bias.data)

    # Copy attention vector
    if (hasattr(pygv_cfconv, 'attention_vector') and
            hasattr(reference_cfconv, 'attention_vector')):
        reference_cfconv.attention_vector.data.copy_(pygv_cfconv.attention_vector.data)

    # Initial forward pass to check output shape
    print("Checking initial outputs...")
    with torch.no_grad():
        pygv_output, pygv_attn = pygv_cfconv(x, edge_index, edge_attr)
        ref_output, ref_attn = reference_cfconv(x, edge_index, edge_attr)

    print(f"PyGV output shape: {pygv_output.shape}")
    print(f"Reference output shape: {ref_output.shape}")

    initial_diff = (pygv_output - ref_output).abs().mean().item()
    print(f"Initial output difference: {initial_diff:.6f}")

    # Train and compare
    print("\nTraining both models to verify mathematical equivalence...")
    results = train_and_compare(
        pygv_cfconv,
        reference_cfconv,
        x, edge_index, edge_attr,
        epochs=50,
        lr=0.001
    )

    # Visualize results
    print("\nVisualizing results...")
    visualize_results(results)

    # Final analysis
    print("\n===== Final Analysis =====")
    print(f"Final output difference: {results['final_output_diff']:.6f}")

    if not np.isnan(results['final_attn_diff']):
        print(f"Final attention weight difference: {results['final_attn_diff']:.6f}")

    # Check mathematical equivalence
    if results['final_output_diff'] < 1e-5:
        print("\n✅ MATHEMATICALLY EQUIVALENT: The implementations produce identical outputs")
    elif results['final_output_diff'] < 1e-3:
        print("\n⚠️ NUMERICALLY CLOSE: The implementations have minor numerical differences")
    else:
        print("\n❌ DIFFERENT BEHAVIOR: The implementations show significant differences")

    # Compare learning curves
    final_pygv_loss = results['pygv_losses'][-1]
    final_ref_loss = results['ref_losses'][-1]
    loss_diff = abs(final_pygv_loss - final_ref_loss)

    if loss_diff < 1e-5:
        print("✅ IDENTICAL LEARNING: Both models learn the same function")
    elif loss_diff < 1e-3:
        print("⚠️ SIMILAR LEARNING: Models learn very similar functions")
    else:
        print("❌ DIFFERENT LEARNING: Models learn different functions")

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        import traceback

        traceback.print_exc()
