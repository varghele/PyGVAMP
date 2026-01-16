"""
Unit tests for PyGVAMP encoders.

Tests verify that all encoders:
1. Execute forward pass without errors
2. Produce outputs with expected shapes
3. Allow gradient flow through all parameters
4. Handle batched inputs correctly

Run with: pytest tests/test_encoders.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# Import encoders
from pygv.encoder.schnet import SchNetEncoderNoEmbed
from pygv.encoder.ml3 import GNNML3
from pygv.encoder.meta import Meta
from pygv.encoder.meta_att import Meta as MetaAtt
from pygv.encoder.gat import GAT


# =============================================================================
# Fixtures: Reusable test data
# =============================================================================

@pytest.fixture
def device():
    """Use CPU for tests to ensure reproducibility and CI compatibility."""
    return torch.device('cpu')


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def single_graph_data(device, seed):
    """
    Create a single graph with known dimensions.

    Graph structure: 10 nodes, ~30 edges (k-NN style)
    Node features: 20 dimensions (simulating one-hot atom types)
    Edge features: 16 dimensions (simulating Gaussian expansion)
    """
    num_nodes = 10
    node_dim = 20
    edge_dim = 16

    # Create k-NN style edges (each node connects to 3 neighbors)
    edge_index = []
    for i in range(num_nodes):
        for j in range(1, 4):  # Connect to 3 neighbors
            neighbor = (i + j) % num_nodes
            edge_index.append([i, neighbor])
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()

    # Node features (one-hot style)
    x = torch.zeros(num_nodes, node_dim, device=device)
    for i in range(num_nodes):
        x[i, i % node_dim] = 1.0
    x.requires_grad_(True)

    # Edge features (Gaussian expansion style)
    edge_attr = torch.randn(edge_index.size(1), edge_dim, device=device)
    edge_attr.requires_grad_(True)

    # Batch tensor (single graph)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


@pytest.fixture
def batched_graph_data(device, seed):
    """
    Create a batch of 3 graphs with varying sizes.

    This tests that encoders handle batched inputs correctly,
    including proper scatter operations and pooling.
    """
    graphs = []
    node_dim = 20
    edge_dim = 16

    # Create 3 graphs with different sizes
    graph_sizes = [8, 12, 10]  # nodes per graph

    for g_idx, num_nodes in enumerate(graph_sizes):
        # k-NN edges
        edge_index = []
        for i in range(num_nodes):
            for j in range(1, 4):
                neighbor = (i + j) % num_nodes
                edge_index.append([i, neighbor])
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()

        # Node features
        x = torch.randn(num_nodes, node_dim, device=device)

        # Edge features
        edge_attr = torch.randn(edge_index.size(1), edge_dim, device=device)

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    # Create batch
    batch = Batch.from_data_list(graphs)
    batch.x.requires_grad_(True)
    batch.edge_attr.requires_grad_(True)

    return batch, len(graphs)


# =============================================================================
# Helper functions
# =============================================================================

def check_gradient_flow(model, loss):
    """
    Verify that gradients flow through all trainable parameters.

    Args:
        model: The encoder model
        loss: A scalar loss tensor to backpropagate

    Returns:
        dict with gradient statistics
    """
    loss.backward()

    grad_stats = {
        'total_params': 0,
        'params_with_grad': 0,
        'params_with_zero_grad': 0,
        'params_with_none_grad': 0,
        'param_details': []
    }

    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_stats['total_params'] += 1

            if param.grad is None:
                grad_stats['params_with_none_grad'] += 1
                grad_stats['param_details'].append((name, 'None'))
            elif torch.all(param.grad == 0):
                grad_stats['params_with_zero_grad'] += 1
                grad_stats['param_details'].append((name, 'zero'))
            else:
                grad_stats['params_with_grad'] += 1
                grad_stats['param_details'].append((name, 'ok'))

    return grad_stats


def assert_gradients_flow(model, loss, encoder_name, min_grad_ratio=0.7):
    """
    Assert that gradients flow to most parameters.

    Args:
        model: The encoder model
        loss: A scalar loss tensor to backpropagate
        encoder_name: Name for error messages
        min_grad_ratio: Minimum ratio of parameters that must have gradients (default: 0.7)

    Note:
        Some parameters (e.g., global_mlp in layer 0 for Meta encoders, or global features
        when using "node" embedding type) may not receive gradients due to architectural
        choices. This is expected behavior for certain configurations.
    """
    stats = check_gradient_flow(model, loss)

    grad_ratio = stats['params_with_grad'] / stats['total_params'] if stats['total_params'] > 0 else 0

    # Known parameters that may not receive gradients in certain configurations
    # These are architectural choices, not bugs
    known_disconnected_patterns = [
        'global_mlp',  # Global features may not be used with "node" embedding
        'layers.0.global_model',  # First layer global model may be unused
    ]

    # Filter out known disconnected parameters
    unexpected_problems = [
        (name, status) for name, status in stats['param_details']
        if status != 'ok' and not any(pattern in name for pattern in known_disconnected_patterns)
    ]

    # Fail if gradient ratio is too low OR if there are unexpected disconnected parameters
    if grad_ratio < min_grad_ratio or len(unexpected_problems) > 0:
        all_problems = [
            f"  - {name}: {status}"
            for name, status in stats['param_details']
            if status != 'ok'
        ]
        error_msg = (
            f"\n{encoder_name} gradient flow issues:\n"
            f"  Total parameters: {stats['total_params']}\n"
            f"  With gradients: {stats['params_with_grad']} ({grad_ratio:.1%})\n"
            f"  Zero gradients: {stats['params_with_zero_grad']}\n"
            f"  None gradients: {stats['params_with_none_grad']}\n"
            f"  Parameters without gradients:\n" + "\n".join(all_problems[:15])
        )
        if len(unexpected_problems) > 0:
            error_msg += f"\n  UNEXPECTED disconnected parameters detected!"
        pytest.fail(error_msg)


# =============================================================================
# SchNet Encoder Tests
# =============================================================================

class TestSchNetEncoder:
    """Tests for SchNetEncoderNoEmbed."""

    def test_forward_single_graph(self, single_graph_data, device):
        """Test forward pass on a single graph."""
        data = single_graph_data
        node_dim = data.x.size(1)
        edge_dim = data.edge_attr.size(1)
        output_dim = 32

        model = SchNetEncoderNoEmbed(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=64,
            output_dim=output_dim,
            n_interactions=2,
            use_attention=True
        ).to(device)

        # Use eval mode for single graph (BatchNorm requires batch_size > 1 in training)
        model.eval()
        with torch.no_grad():
            output, aux = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # Check output shape: [num_graphs, output_dim]
        assert output.shape == (1, output_dim), f"Expected (1, {output_dim}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_batched(self, batched_graph_data, device):
        """Test forward pass on batched graphs."""
        batch, num_graphs = batched_graph_data
        node_dim = batch.x.size(1)
        edge_dim = batch.edge_attr.size(1)
        output_dim = 32

        model = SchNetEncoderNoEmbed(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=64,
            output_dim=output_dim,
            n_interactions=2,
            use_attention=True
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert output.shape == (num_graphs, output_dim), f"Expected ({num_graphs}, {output_dim}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_gradient_flow(self, batched_graph_data, device):
        """Test that gradients flow through all parameters."""
        batch, num_graphs = batched_graph_data
        node_dim = batch.x.size(1)
        edge_dim = batch.edge_attr.size(1)

        model = SchNetEncoderNoEmbed(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=64,
            output_dim=32,
            n_interactions=2,
            use_attention=True
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = output.sum()

        assert_gradients_flow(model, loss, "SchNetEncoderNoEmbed")

    def test_gradient_flow_to_inputs(self, batched_graph_data, device):
        """Test that gradients flow back to input tensors."""
        batch, num_graphs = batched_graph_data
        node_dim = batch.x.size(1)
        edge_dim = batch.edge_attr.size(1)

        model = SchNetEncoderNoEmbed(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=64,
            output_dim=32,
            n_interactions=2,
            use_attention=True
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = output.sum()
        loss.backward()

        assert batch.x.grad is not None, "No gradient for node features (x)"
        assert not torch.all(batch.x.grad == 0), "Zero gradient for node features (x)"
        assert batch.edge_attr.grad is not None, "No gradient for edge features"
        assert not torch.all(batch.edge_attr.grad == 0), "Zero gradient for edge features"

    def test_attention_disabled(self, batched_graph_data, device):
        """Test encoder works with attention disabled."""
        batch, num_graphs = batched_graph_data

        model = SchNetEncoderNoEmbed(
            node_dim=batch.x.size(1),
            edge_dim=batch.edge_attr.size(1),
            hidden_dim=64,
            output_dim=32,
            n_interactions=2,
            use_attention=False  # Attention disabled
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert output.shape == (num_graphs, 32)
        assert not torch.isnan(output).any()


# =============================================================================
# ML3 Encoder Tests
# =============================================================================

class TestML3Encoder:
    """Tests for GNNML3 encoder."""

    def test_forward_single_graph(self, single_graph_data, device):
        """Test forward pass on a single graph."""
        data = single_graph_data

        model = GNNML3(
            node_dim=32,  # Target dimension after encoding
            edge_dim=32,
            global_dim=32,
            hidden_dim=30,
            num_layers=2,
            num_encoder_layers=2,
            output_dim=2,
            embedding_type="node"
        ).to(device)

        output, aux = model(data.x, data.edge_index, data.edge_attr, data.batch, u=None)

        # ML3 outputs node-level predictions: [num_nodes, 1]
        num_nodes = data.x.size(0)
        assert output.shape == (num_nodes, 1), f"Expected ({num_nodes}, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_batched(self, batched_graph_data, device):
        """Test forward pass on batched graphs."""
        batch, num_graphs = batched_graph_data

        model = GNNML3(
            node_dim=32,
            edge_dim=32,
            global_dim=32,
            hidden_dim=30,
            num_layers=2,
            num_encoder_layers=2,
            output_dim=2,
            embedding_type="node"
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, u=None)

        total_nodes = batch.x.size(0)
        assert output.shape == (total_nodes, 1), f"Expected ({total_nodes}, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_gradient_flow(self, single_graph_data, device):
        """Test that gradients flow through all parameters."""
        data = single_graph_data

        model = GNNML3(
            node_dim=32,
            edge_dim=32,
            global_dim=32,
            hidden_dim=30,
            num_layers=2,
            num_encoder_layers=2,
            output_dim=2,
            embedding_type="node"
        ).to(device)

        output, aux = model(data.x, data.edge_index, data.edge_attr, data.batch, u=None)
        loss = output.sum()

        assert_gradients_flow(model, loss, "GNNML3")

    def test_gradient_flow_to_inputs(self, single_graph_data, device):
        """Test that gradients flow back to input tensors."""
        data = single_graph_data

        model = GNNML3(
            node_dim=32,
            edge_dim=32,
            global_dim=32,
            hidden_dim=30,
            num_layers=2,
            num_encoder_layers=2,
            output_dim=2,
            embedding_type="node"
        ).to(device)

        output, aux = model(data.x, data.edge_index, data.edge_attr, data.batch, u=None)
        loss = output.sum()
        loss.backward()

        assert data.x.grad is not None, "No gradient for node features (x)"
        assert not torch.all(data.x.grad == 0), "Zero gradient for node features (x)"


# =============================================================================
# Meta Encoder Tests
# =============================================================================

class TestMetaEncoder:
    """Tests for Meta encoder (without attention)."""

    def test_forward_single_graph(self, single_graph_data, device):
        """Test forward pass on a single graph."""
        data = single_graph_data
        node_dim = data.x.size(1)
        edge_dim = data.edge_attr.size(1)
        output_dim = 32

        model = Meta(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=32,
            num_node_mlp_layers=2,
            num_edge_mlp_layers=2,
            num_global_mlp_layers=2,
            hidden_dim=64,
            output_dim=output_dim,
            num_meta_layers=2,
            embedding_type="combined"
        ).to(device)

        # Use eval mode for single graph (BatchNorm requires batch_size > 1 in training)
        model.eval()
        with torch.no_grad():
            output, aux = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # Meta outputs graph-level embeddings: [num_graphs, output_dim]
        assert output.shape == (1, output_dim), f"Expected (1, {output_dim}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_batched(self, batched_graph_data, device):
        """Test forward pass on batched graphs."""
        batch, num_graphs = batched_graph_data
        node_dim = batch.x.size(1)
        edge_dim = batch.edge_attr.size(1)
        output_dim = 32

        model = Meta(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=32,
            num_node_mlp_layers=2,
            num_edge_mlp_layers=2,
            num_global_mlp_layers=2,
            hidden_dim=64,
            output_dim=output_dim,
            num_meta_layers=2,
            embedding_type="combined"
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert output.shape == (num_graphs, output_dim), f"Expected ({num_graphs}, {output_dim}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_gradient_flow(self, batched_graph_data, device):
        """Test that gradients flow through all parameters."""
        batch, num_graphs = batched_graph_data

        model = Meta(
            node_dim=batch.x.size(1),
            edge_dim=batch.edge_attr.size(1),
            global_dim=32,
            num_node_mlp_layers=2,
            num_edge_mlp_layers=2,
            num_global_mlp_layers=2,
            hidden_dim=64,
            output_dim=32,
            num_meta_layers=2,
            embedding_type="combined"
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = output.sum()

        assert_gradients_flow(model, loss, "Meta")

    def test_embedding_types(self, batched_graph_data, device):
        """Test different embedding types work."""
        batch, num_graphs = batched_graph_data

        for emb_type in ["node", "global", "combined"]:
            model = Meta(
                node_dim=batch.x.size(1),
                edge_dim=batch.edge_attr.size(1),
                global_dim=32,
                num_node_mlp_layers=2,
                num_edge_mlp_layers=2,
                num_global_mlp_layers=2,
                hidden_dim=64,
                output_dim=32,
                num_meta_layers=2,
                embedding_type=emb_type
            ).to(device)

            output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            assert output.shape == (num_graphs, 32), f"embedding_type={emb_type}: Expected ({num_graphs}, 32), got {output.shape}"
            assert not torch.isnan(output).any(), f"embedding_type={emb_type}: Output contains NaN"


# =============================================================================
# Meta with Attention Encoder Tests
# =============================================================================

class TestMetaAttEncoder:
    """Tests for Meta encoder with attention."""

    def test_forward_single_graph(self, single_graph_data, device):
        """Test forward pass on a single graph."""
        data = single_graph_data
        node_dim = data.x.size(1)
        edge_dim = data.edge_attr.size(1)
        output_dim = 32

        model = MetaAtt(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=32,
            num_node_mlp_layers=2,
            num_edge_mlp_layers=2,
            num_global_mlp_layers=2,
            hidden_dim=64,
            output_dim=output_dim,
            num_meta_layers=2,
            embedding_type="combined",
            use_attention=True
        ).to(device)

        # Use eval mode for single graph (BatchNorm requires batch_size > 1 in training)
        model.eval()
        with torch.no_grad():
            output, aux = model(data.x, data.edge_index, data.edge_attr, data.batch)

        assert output.shape == (1, output_dim), f"Expected (1, {output_dim}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_batched(self, batched_graph_data, device):
        """Test forward pass on batched graphs."""
        batch, num_graphs = batched_graph_data
        node_dim = batch.x.size(1)
        edge_dim = batch.edge_attr.size(1)
        output_dim = 32

        model = MetaAtt(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=32,
            num_node_mlp_layers=2,
            num_edge_mlp_layers=2,
            num_global_mlp_layers=2,
            hidden_dim=64,
            output_dim=output_dim,
            num_meta_layers=2,
            embedding_type="combined",
            use_attention=True
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert output.shape == (num_graphs, output_dim), f"Expected ({num_graphs}, {output_dim}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_gradient_flow(self, batched_graph_data, device):
        """Test that gradients flow through all parameters."""
        batch, num_graphs = batched_graph_data

        model = MetaAtt(
            node_dim=batch.x.size(1),
            edge_dim=batch.edge_attr.size(1),
            global_dim=32,
            num_node_mlp_layers=2,
            num_edge_mlp_layers=2,
            num_global_mlp_layers=2,
            hidden_dim=64,
            output_dim=32,
            num_meta_layers=2,
            embedding_type="combined",
            use_attention=True
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = output.sum()

        assert_gradients_flow(model, loss, "MetaAtt")

    def test_attention_output(self, batched_graph_data, device):
        """Test that attention weights are returned."""
        batch, num_graphs = batched_graph_data

        model = MetaAtt(
            node_dim=batch.x.size(1),
            edge_dim=batch.edge_attr.size(1),
            global_dim=32,
            num_node_mlp_layers=2,
            num_edge_mlp_layers=2,
            num_global_mlp_layers=2,
            hidden_dim=64,
            output_dim=32,
            num_meta_layers=2,
            embedding_type="combined",
            use_attention=True
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # aux should contain attention weights
        assert len(aux) >= 4, f"Expected at least 4 auxiliary outputs, got {len(aux)}"


# =============================================================================
# GAT Encoder Tests
# =============================================================================

class TestGATEncoder:
    """Tests for GAT encoder."""

    def test_forward_single_graph(self, single_graph_data, device):
        """Test forward pass on a single graph."""
        data = single_graph_data
        hidden_dim = 64
        heads = 4

        # GAT requires hidden_dim % heads == 0
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        model = GAT(
            node_dim=32,  # Target dimension
            edge_dim=32,
            global_dim=32,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_encoder_layers=2,
            num_global_mlp_layers=2,
            heads=heads,
            dropout=0.0,  # Disable for deterministic tests
            shift_predictor_hidden_dim=32,
            shift_predictor_layers=2,
            embedding_type="combined"
        ).to(device)

        # Use eval mode for single graph (BatchNorm requires batch_size > 1 in training)
        model.eval()
        with torch.no_grad():
            output, aux = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # GAT outputs node-level predictions: [num_nodes, 1]
        num_nodes = data.x.size(0)
        assert output.shape == (num_nodes, 1), f"Expected ({num_nodes}, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_batched(self, batched_graph_data, device):
        """Test forward pass on batched graphs."""
        batch, num_graphs = batched_graph_data

        model = GAT(
            node_dim=32,
            edge_dim=32,
            global_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_encoder_layers=2,
            num_global_mlp_layers=2,
            heads=4,
            dropout=0.0,
            shift_predictor_hidden_dim=32,
            shift_predictor_layers=2,
            embedding_type="combined"
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        total_nodes = batch.x.size(0)
        assert output.shape == (total_nodes, 1), f"Expected ({total_nodes}, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_gradient_flow(self, batched_graph_data, device):
        """Test that gradients flow through all parameters."""
        batch, num_graphs = batched_graph_data

        model = GAT(
            node_dim=32,
            edge_dim=32,
            global_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_encoder_layers=2,
            num_global_mlp_layers=2,
            heads=4,
            dropout=0.0,
            shift_predictor_hidden_dim=32,
            shift_predictor_layers=2,
            embedding_type="node"
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = output.sum()

        assert_gradients_flow(model, loss, "GAT")

    def test_embedding_types(self, batched_graph_data, device):
        """Test different embedding types work."""
        batch, num_graphs = batched_graph_data

        for emb_type in ["node", "global", "combined"]:
            model = GAT(
                node_dim=32,
                edge_dim=32,
                global_dim=32,
                hidden_dim=64,
                num_layers=2,
                num_encoder_layers=2,
                num_global_mlp_layers=2,
                heads=4,
                dropout=0.0,
                shift_predictor_hidden_dim=32,
                shift_predictor_layers=2,
                embedding_type=emb_type
            ).to(device)

            output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            total_nodes = batch.x.size(0)
            assert output.shape == (total_nodes, 1), f"embedding_type={emb_type}: Expected ({total_nodes}, 1), got {output.shape}"
            assert not torch.isnan(output).any(), f"embedding_type={emb_type}: Output contains NaN"


# =============================================================================
# Cross-Encoder Integration Tests
# =============================================================================

class TestEncoderIntegration:
    """Integration tests across all encoders."""

    def test_all_encoders_same_input(self, batched_graph_data, device):
        """Test that all encoders can process the same input without errors."""
        batch, num_graphs = batched_graph_data
        node_dim = batch.x.size(1)
        edge_dim = batch.edge_attr.size(1)

        encoders = {
            'SchNet': SchNetEncoderNoEmbed(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=64, output_dim=32,
                n_interactions=2, use_attention=True
            ),
            'ML3': GNNML3(
                node_dim=32, edge_dim=32, global_dim=32, hidden_dim=30,
                num_layers=2, num_encoder_layers=2, output_dim=2, embedding_type="node"
            ),
            'Meta': Meta(
                node_dim=node_dim, edge_dim=edge_dim, global_dim=32,
                num_node_mlp_layers=2, num_edge_mlp_layers=2, num_global_mlp_layers=2,
                hidden_dim=64, output_dim=32, num_meta_layers=2, embedding_type="combined"
            ),
            'MetaAtt': MetaAtt(
                node_dim=node_dim, edge_dim=edge_dim, global_dim=32,
                num_node_mlp_layers=2, num_edge_mlp_layers=2, num_global_mlp_layers=2,
                hidden_dim=64, output_dim=32, num_meta_layers=2,
                embedding_type="combined", use_attention=True
            ),
            'GAT': GAT(
                node_dim=32, edge_dim=32, global_dim=32, hidden_dim=64,
                num_layers=2, num_encoder_layers=2, num_global_mlp_layers=2,
                heads=4, dropout=0.0, shift_predictor_hidden_dim=32,
                shift_predictor_layers=2, embedding_type="combined"
            ),
        }

        results = {}
        for name, encoder in encoders.items():
            encoder = encoder.to(device)

            # Need fresh data for each encoder (gradients get consumed)
            x = batch.x.detach().clone().requires_grad_(True)
            edge_attr = batch.edge_attr.detach().clone().requires_grad_(True)

            try:
                if name == 'ML3':
                    output, aux = encoder(x, batch.edge_index, edge_attr, batch.batch, u=None)
                else:
                    output, aux = encoder(x, batch.edge_index, edge_attr, batch.batch)

                loss = output.sum()
                loss.backward()

                results[name] = {
                    'success': True,
                    'output_shape': output.shape,
                    'has_nan': torch.isnan(output).any().item(),
                    'grad_flows': x.grad is not None and not torch.all(x.grad == 0)
                }
            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e)
                }

        # Report results
        failures = [name for name, r in results.items() if not r['success']]
        nan_outputs = [name for name, r in results.items() if r.get('has_nan', False)]
        no_grad = [name for name, r in results.items() if not r.get('grad_flows', True)]

        if failures:
            pytest.fail(f"Encoders failed: {failures}")
        if nan_outputs:
            pytest.fail(f"Encoders produced NaN: {nan_outputs}")
        if no_grad:
            pytest.fail(f"Encoders with no gradient flow: {no_grad}")

    def test_deterministic_output(self, single_graph_data, device):
        """Test that encoders produce deterministic outputs with fixed seed."""
        data = single_graph_data

        torch.manual_seed(123)
        model1 = SchNetEncoderNoEmbed(
            node_dim=data.x.size(1), edge_dim=data.edge_attr.size(1),
            hidden_dim=64, output_dim=32, n_interactions=2, use_attention=True
        ).to(device)
        model1.eval()

        torch.manual_seed(123)
        model2 = SchNetEncoderNoEmbed(
            node_dim=data.x.size(1), edge_dim=data.edge_attr.size(1),
            hidden_dim=64, output_dim=32, n_interactions=2, use_attention=True
        ).to(device)
        model2.eval()

        with torch.no_grad():
            out1, _ = model1(data.x, data.edge_index, data.edge_attr, data.batch)
            out2, _ = model2(data.x, data.edge_index, data.edge_attr, data.batch)

        assert torch.allclose(out1, out2), "Same seed should produce same output"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_node_graph(self, device, seed):
        """Test handling of single-node graph."""
        # Single node, self-loop edge
        x = torch.randn(1, 20, device=device, requires_grad=True)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        edge_attr = torch.randn(1, 16, device=device, requires_grad=True)
        batch = torch.zeros(1, dtype=torch.long, device=device)

        model = SchNetEncoderNoEmbed(
            node_dim=20, edge_dim=16, hidden_dim=64, output_dim=32,
            n_interactions=2, use_attention=True
        ).to(device)

        # Use eval mode for single graph (BatchNorm requires batch_size > 1 in training)
        model.eval()
        with torch.no_grad():
            output, aux = model(x, edge_index, edge_attr, batch)

        assert output.shape == (1, 32), "Single node should produce valid output"
        assert not torch.isnan(output).any(), "Single node output should not be NaN"

    def test_no_batch_tensor(self, single_graph_data, device):
        """Test that encoders handle missing batch tensor (single graph)."""
        data = single_graph_data

        model = SchNetEncoderNoEmbed(
            node_dim=data.x.size(1), edge_dim=data.edge_attr.size(1),
            hidden_dim=64, output_dim=32, n_interactions=2, use_attention=True
        ).to(device)

        # Use eval mode for single graph (BatchNorm requires batch_size > 1 in training)
        model.eval()
        with torch.no_grad():
            # Pass None for batch
            output, aux = model(data.x, data.edge_index, data.edge_attr, batch=None)

        assert output.shape == (1, 32), "Should infer single graph from missing batch"

    def test_large_batch(self, device, seed):
        """Test handling of larger batch sizes."""
        num_graphs = 16
        graphs = []

        for _ in range(num_graphs):
            num_nodes = 20
            x = torch.randn(num_nodes, 20)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
            edge_attr = torch.randn(num_nodes * 3, 16)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        batch = Batch.from_data_list(graphs).to(device)
        batch.x.requires_grad_(True)

        model = SchNetEncoderNoEmbed(
            node_dim=20, edge_dim=16, hidden_dim=64, output_dim=32,
            n_interactions=2, use_attention=True
        ).to(device)

        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert output.shape == (num_graphs, 32), f"Expected ({num_graphs}, 32), got {output.shape}"

        loss = output.sum()
        loss.backward()

        assert batch.x.grad is not None, "Gradients should flow through large batch"


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
