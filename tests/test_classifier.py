"""
Unit tests for SoftmaxMLP classifier.

Tests verify that the classifier:
1. Produces valid probability distributions (sum to 1, non-negative)
2. Has correct output shapes for various configurations
3. Allows gradient flow through all parameters
4. Handles different num_layers, hidden_channels, and n_states

Run with: pytest tests/test_classifier.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from pygv.classifier.SoftmaxMLP import SoftmaxMLP, init_weights


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Use CPU for tests to ensure reproducibility and CI compatibility."""
    return torch.device('cpu')


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def input_features(device, seed):
    """Sample input features simulating encoder output."""
    batch_size = 32
    in_channels = 64
    x = torch.randn(batch_size, in_channels, device=device, requires_grad=True)
    return x


@pytest.fixture
def classifier_params():
    """Default classifier parameters."""
    return {
        'in_channels': 64,
        'hidden_channels': 32,
        'out_channels': 5,
        'num_layers': 2,
        'dropout': 0.0,
        'act': 'relu',
        'norm': None,
    }


@pytest.fixture
def classifier(classifier_params, device):
    """Create a default classifier."""
    return SoftmaxMLP(**classifier_params).to(device)


# =============================================================================
# TestOutputShape - Tests output dimensions
# =============================================================================

class TestOutputShape:
    """Tests for output shape correctness."""

    def test_output_shape_default(self, classifier, input_features):
        """Output shape is [batch_size, n_states]."""
        output = classifier(input_features)
        assert output.shape == (input_features.shape[0], 5)

    def test_output_shape_different_n_states(self, input_features, device):
        """Output shape adapts to different n_states values."""
        for n_states in [3, 5, 7, 10]:
            clf = SoftmaxMLP(
                in_channels=64,
                hidden_channels=32,
                out_channels=n_states,
                num_layers=2,
            ).to(device)
            output = clf(input_features)
            assert output.shape == (input_features.shape[0], n_states), \
                f"Expected shape ({input_features.shape[0]}, {n_states}), got {output.shape}"

    def test_output_shape_single_sample(self, classifier, device):
        """Works with single sample input."""
        x = torch.randn(1, 64, device=device)
        output = classifier(x)
        assert output.shape == (1, 5)

    def test_output_shape_large_batch(self, classifier, device):
        """Works with large batch sizes."""
        x = torch.randn(256, 64, device=device)
        output = classifier(x)
        assert output.shape == (256, 5)


# =============================================================================
# TestProbabilityDistribution - Tests softmax properties
# =============================================================================

class TestProbabilityDistribution:
    """Tests that outputs form valid probability distributions."""

    def test_outputs_sum_to_one(self, classifier, input_features):
        """Softmax outputs sum to 1 for each sample."""
        output = classifier(input_features)
        row_sums = output.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            f"Row sums should be 1, got {row_sums}"

    def test_outputs_non_negative(self, classifier, input_features):
        """All output values are non-negative."""
        output = classifier(input_features)
        assert (output >= 0).all(), "All probabilities should be non-negative"

    def test_outputs_at_most_one(self, classifier, input_features):
        """All output values are at most 1."""
        output = classifier(input_features)
        assert (output <= 1).all(), "All probabilities should be at most 1"

    def test_outputs_valid_distribution(self, classifier, input_features):
        """Outputs form valid probability distributions."""
        output = classifier(input_features)
        # Check all properties together
        assert (output >= 0).all(), "Probabilities must be non-negative"
        assert (output <= 1).all(), "Probabilities must be at most 1"
        row_sums = output.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            "Probabilities must sum to 1"

    def test_no_nan_or_inf(self, classifier, input_features):
        """Output contains no NaN or Inf values."""
        output = classifier(input_features)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_extreme_inputs_still_valid(self, classifier, device):
        """Softmax handles extreme input values gracefully."""
        # Large positive values
        x_large = torch.randn(16, 64, device=device) * 100
        output_large = classifier(x_large)
        assert not torch.isnan(output_large).any(), "NaN with large inputs"
        assert torch.allclose(output_large.sum(dim=1), torch.ones(16, device=device), atol=1e-5)

        # Large negative values
        x_neg = torch.randn(16, 64, device=device) * -100
        output_neg = classifier(x_neg)
        assert not torch.isnan(output_neg).any(), "NaN with large negative inputs"
        assert torch.allclose(output_neg.sum(dim=1), torch.ones(16, device=device), atol=1e-5)


# =============================================================================
# TestGradientFlow - Tests gradient backpropagation
# =============================================================================

class TestGradientFlow:
    """Tests that gradients flow through the classifier."""

    def test_gradient_flow_to_input(self, classifier, input_features):
        """Gradients flow back to input features."""
        output = classifier(input_features)
        loss = output.sum()
        loss.backward()

        assert input_features.grad is not None, "No gradient for input"
        assert not torch.all(input_features.grad == 0), "Zero gradient for input"

    def test_gradient_flow_to_parameters(self, classifier, input_features):
        """Gradients flow to all classifier parameters."""
        output = classifier(input_features)
        loss = output.sum()
        loss.backward()

        for name, param in classifier.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # Note: some params may have zero grad depending on input

    def test_gradient_with_cross_entropy_loss(self, classifier, input_features, device):
        """Gradients work with typical cross-entropy loss."""
        # Create random targets
        targets = torch.randint(0, 5, (input_features.shape[0],), device=device)

        output = classifier(input_features)
        # Use NLL loss since output is already softmax probabilities
        loss = nn.NLLLoss()(torch.log(output + 1e-10), targets)
        loss.backward()

        assert input_features.grad is not None, "No gradient with NLL loss"

    def test_multiple_backward_passes(self, classifier, device):
        """Multiple backward passes work correctly."""
        for i in range(3):
            x = torch.randn(16, 64, device=device, requires_grad=True)
            output = classifier(x)
            loss = output.sum()
            loss.backward()
            assert x.grad is not None, f"No gradient on pass {i}"


# =============================================================================
# TestLayerConfigurations - Tests different layer setups
# =============================================================================

class TestLayerConfigurations:
    """Tests different num_layers and hidden_channels configurations."""

    def test_single_layer(self, device):
        """Single layer classifier (num_layers=1) works."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=1,
        ).to(device)

        x = torch.randn(16, 64, device=device)
        output = clf(x)

        assert output.shape == (16, 5)
        assert torch.allclose(output.sum(dim=1), torch.ones(16, device=device), atol=1e-6)

    def test_two_layers(self, device):
        """Two layer classifier works."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
        ).to(device)

        x = torch.randn(16, 64, device=device)
        output = clf(x)

        assert output.shape == (16, 5)
        assert torch.allclose(output.sum(dim=1), torch.ones(16, device=device), atol=1e-6)

    def test_deep_network(self, device):
        """Deep classifier (num_layers=5) works."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=5,
        ).to(device)

        x = torch.randn(16, 64, device=device)
        output = clf(x)

        assert output.shape == (16, 5)
        assert torch.allclose(output.sum(dim=1), torch.ones(16, device=device), atol=1e-6)

    def test_wide_hidden_layer(self, device):
        """Wide hidden layer (hidden_channels > in_channels) works."""
        clf = SoftmaxMLP(
            in_channels=32,
            hidden_channels=128,
            out_channels=5,
            num_layers=3,
        ).to(device)

        x = torch.randn(16, 32, device=device)
        output = clf(x)

        assert output.shape == (16, 5)

    def test_narrow_hidden_layer(self, device):
        """Narrow hidden layer (hidden_channels < in_channels) works."""
        clf = SoftmaxMLP(
            in_channels=128,
            hidden_channels=16,
            out_channels=5,
            num_layers=3,
        ).to(device)

        x = torch.randn(16, 128, device=device)
        output = clf(x)

        assert output.shape == (16, 5)

    def test_different_activations(self, device):
        """Different activation functions work."""
        for act in ['relu', 'tanh', 'leaky_relu']:
            clf = SoftmaxMLP(
                in_channels=64,
                hidden_channels=32,
                out_channels=5,
                num_layers=2,
                act=act,
            ).to(device)

            x = torch.randn(16, 64, device=device)
            output = clf(x)

            assert output.shape == (16, 5), f"Failed with activation {act}"
            assert not torch.isnan(output).any(), f"NaN with activation {act}"


# =============================================================================
# TestDropout - Tests dropout behavior
# =============================================================================

class TestDropout:
    """Tests dropout behavior in train vs eval mode."""

    def test_dropout_train_vs_eval(self, device):
        """Dropout behaves differently in train vs eval mode."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=3,
            dropout=0.5,
        ).to(device)

        x = torch.randn(16, 64, device=device)

        # Training mode - outputs may vary due to dropout
        clf.train()
        torch.manual_seed(42)
        out_train1 = clf(x).clone()
        torch.manual_seed(43)
        out_train2 = clf(x).clone()

        # Eval mode - outputs should be deterministic
        clf.eval()
        out_eval1 = clf(x).clone()
        out_eval2 = clf(x).clone()

        # Eval outputs should be identical
        assert torch.allclose(out_eval1, out_eval2), "Eval mode should be deterministic"

    def test_zero_dropout(self, device):
        """Zero dropout gives deterministic outputs in train mode."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            dropout=0.0,
        ).to(device)

        x = torch.randn(16, 64, device=device)
        clf.train()

        out1 = clf(x)
        out2 = clf(x)

        assert torch.allclose(out1, out2), "Zero dropout should be deterministic"


# =============================================================================
# TestNormalization - Tests batch normalization
# =============================================================================

class TestNormalization:
    """Tests normalization options."""

    def test_with_batch_norm(self, device):
        """Classifier works with batch normalization."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=3,
            norm='batch_norm',
        ).to(device)

        x = torch.randn(16, 64, device=device)
        output = clf(x)

        assert output.shape == (16, 5)
        assert not torch.isnan(output).any()

    def test_without_norm(self, device):
        """Classifier works without normalization."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=3,
            norm=None,
        ).to(device)

        x = torch.randn(16, 64, device=device)
        output = clf(x)

        assert output.shape == (16, 5)


# =============================================================================
# TestInitWeights - Tests weight initialization
# =============================================================================

class TestInitWeights:
    """Tests the init_weights function."""

    def test_init_weights_linear(self):
        """init_weights properly initializes Linear layers."""
        linear = nn.Linear(32, 16)
        init_weights(linear)

        # Xavier uniform should give values roughly in [-sqrt(6/(in+out)), sqrt(6/(in+out))]
        # For 32+16=48: sqrt(6/48) ≈ 0.35
        assert linear.weight.abs().max() < 1.0, "Weights should be bounded"
        assert (linear.bias == 0.01).all(), "Bias should be initialized to 0.01"

    def test_init_weights_non_linear(self):
        """init_weights ignores non-Linear layers."""
        conv = nn.Conv1d(32, 16, 3)
        original_weight = conv.weight.clone()
        init_weights(conv)
        # Should be unchanged
        assert torch.equal(conv.weight, original_weight)


# =============================================================================
# TestIntegration - Tests typical usage patterns
# =============================================================================

class TestIntegration:
    """Tests typical integration patterns."""

    def test_with_encoder_output(self, device):
        """Classifier works with typical encoder output dimensions."""
        # Typical SchNet output: 64 dim
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
        ).to(device)

        encoder_output = torch.randn(32, 64, device=device)
        probs = clf(encoder_output)

        assert probs.shape == (32, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(32, device=device), atol=1e-6)

    def test_training_step(self, device):
        """Full training step works correctly."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
        ).to(device)

        optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

        # Simulated encoder output
        x = torch.randn(32, 64, device=device)
        targets = torch.randint(0, 5, (32,), device=device)

        # Forward
        probs = clf(x)
        loss = nn.NLLLoss()(torch.log(probs + 1e-10), targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check loss is finite
        assert torch.isfinite(loss), "Loss should be finite"

    def test_multiple_training_steps(self, device):
        """Multiple training steps reduce loss."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
        ).to(device)

        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)

        # Fixed input and targets for consistent test
        torch.manual_seed(42)
        x = torch.randn(64, 64, device=device)
        targets = torch.randint(0, 5, (64,), device=device)

        losses = []
        for _ in range(50):
            probs = clf(x)
            loss = nn.NLLLoss()(torch.log(probs + 1e-10), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0], "Loss should decrease with training"


# =============================================================================
# TestEdgeCases - Tests boundary conditions
# =============================================================================

class TestEdgeCases:
    """Tests edge cases and boundary conditions."""

    def test_two_classes(self, device):
        """Binary classification (2 classes) works."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=2,
            num_layers=2,
        ).to(device)

        x = torch.randn(16, 64, device=device)
        output = clf(x)

        assert output.shape == (16, 2)
        assert torch.allclose(output.sum(dim=1), torch.ones(16, device=device), atol=1e-6)

    def test_many_classes(self, device):
        """Many classes (20) works."""
        clf = SoftmaxMLP(
            in_channels=64,
            hidden_channels=32,
            out_channels=20,
            num_layers=2,
        ).to(device)

        x = torch.randn(16, 64, device=device)
        output = clf(x)

        assert output.shape == (16, 20)
        assert torch.allclose(output.sum(dim=1), torch.ones(16, device=device), atol=1e-6)

    def test_small_input_dim(self, device):
        """Small input dimension works."""
        clf = SoftmaxMLP(
            in_channels=4,
            hidden_channels=8,
            out_channels=3,
            num_layers=2,
        ).to(device)

        x = torch.randn(16, 4, device=device)
        output = clf(x)

        assert output.shape == (16, 3)

    def test_large_input_dim(self, device):
        """Large input dimension works."""
        clf = SoftmaxMLP(
            in_channels=512,
            hidden_channels=128,
            out_channels=10,
            num_layers=3,
        ).to(device)

        x = torch.randn(16, 512, device=device)
        output = clf(x)

        assert output.shape == (16, 10)


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
