"""
Equivalence tests: our ML3 implementation vs the original ml3_repo.

Verifies that SpectConvWithAttention (attention=False) and ML3Layer
produce identical outputs to the original SpectConv and ML3Layer
given the same weights and inputs.

The original reference classes are embedded directly in this file
(copied from ml3_repo/libs/spect_conv.py) so the test is self-contained.

Run with: pytest tests/test_ml3_equivalence.py -v
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

# Import our implementation
from pygv.encoder.ml3 import SpectConvWithAttention, ML3Layer as OurML3Layer


# =============================================================================
# Original reference implementation (from ml3_repo/libs/spect_conv.py)
# =============================================================================

def _orig_glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def _orig_zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class OrigSpectConv(MessagePassing):
    """Original SpectConv from ml3_repo/libs/spect_conv.py (lines 23-103)."""

    def __init__(self, in_channels, out_channels, K=1, selfconn=True,
                 depthwise=False, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(OrigSpectConv, self).__init__(**kwargs)
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise = depthwise
        self.selfconn = selfconn
        if self.selfconn:
            K = K + 1
        if self.depthwise:
            self.DSweight = Parameter(torch.Tensor(K, in_channels))
            self.nsup = K
            K = 1
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        _orig_glorot(self.weight)
        _orig_zeros(self.bias)
        if self.depthwise:
            _orig_zeros(self.DSweight)

    def forward(self, x, edge_index, edge_attr, edge_weight=None,
                batch=None, lambda_max=None):
        Tx_0 = x
        out = 0
        if not self.depthwise:
            enditr = self.weight.size(0)
            if self.selfconn:
                out = torch.matmul(Tx_0, self.weight[-1])
                enditr -= 1
            for i in range(0, enditr):
                h = self.propagate(edge_index, x=Tx_0, norm=edge_attr[:, i], size=None)
                out = out + torch.matmul(h, self.weight[i])
        else:
            enditr = self.nsup
            if self.selfconn:
                out = Tx_0 * self.DSweight[-1]
                enditr -= 1
            out = out + (1 + self.DSweight[0:1, :]) * self.propagate(
                edge_index, x=Tx_0, norm=edge_attr[:, 0], size=None)
            for i in range(1, enditr):
                out = out + self.DSweight[i:i + 1, :] * self.propagate(
                    edge_index, x=Tx_0, norm=edge_attr[:, i], size=None)
            out = torch.matmul(out, self.weight[0])
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class OrigML3Layer(nn.Module):
    """Original ML3Layer from ml3_repo/libs/spect_conv.py (lines 182-212)."""

    def __init__(self, learnedge, nedgeinput, nedgeoutput, ninp, nout1, nout2):
        super(OrigML3Layer, self).__init__()
        self.learnedge = learnedge
        self.nout2 = nout2
        if self.learnedge:
            self.fc1_1 = nn.Linear(nedgeinput, 2 * nedgeinput, bias=False)
            self.fc1_2 = nn.Linear(nedgeinput, 2 * nedgeinput, bias=False)
            self.fc1_3 = nn.Linear(nedgeinput, 2 * nedgeinput, bias=False)
            self.fc1_4 = nn.Linear(4 * nedgeinput, nedgeoutput, bias=False)
        else:
            nedgeoutput = nedgeinput
        self.conv1 = OrigSpectConv(ninp, nout1, nedgeoutput, selfconn=False)
        if nout2 > 0:
            self.fc11 = nn.Linear(ninp, nout2)
            self.fc12 = nn.Linear(ninp, nout2)

    def forward(self, x, edge_index, edge_attr):
        if self.learnedge:
            tmp = torch.cat([F.relu(self.fc1_1(edge_attr)),
                             torch.tanh(self.fc1_2(edge_attr)) * torch.tanh(self.fc1_3(edge_attr))], 1)
            edge_attr = F.relu(self.fc1_4(tmp))
        if self.nout2 > 0:
            x = torch.cat([F.relu(self.conv1(x, edge_index, edge_attr)),
                           torch.tanh(self.fc11(x)) * torch.tanh(self.fc12(x))], 1)
        else:
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        return x


# =============================================================================
# Helpers
# =============================================================================

def _make_test_graph(n_nodes=10, node_dim=8, edge_dim=4, seed=42):
    """Create a deterministic test graph."""
    torch.manual_seed(seed)
    # Ring + random edges
    src = list(range(n_nodes))
    dst = [(i + 1) % n_nodes for i in range(n_nodes)]
    # Add a few extra edges
    src += [0, 2, 4]
    dst += [3, 5, 7]
    # Bidirectional
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    x = torch.randn(n_nodes, node_dim)
    edge_attr = torch.randn(edge_index.size(1), edge_dim)
    return x, edge_index, edge_attr


def _copy_spectconv_weights(orig, ours):
    """Copy weights from original SpectConv to our SpectConvWithAttention."""
    ours.weight.data.copy_(orig.weight.data)
    if orig.bias is not None and ours.bias is not None:
        ours.bias.data.copy_(orig.bias.data)


def _copy_ml3layer_weights(orig, ours):
    """Copy weights from original ML3Layer to our ML3Layer."""
    # Copy SpectConv weights
    _copy_spectconv_weights(orig.conv1, ours.conv)

    # Copy edge transform weights
    if orig.learnedge:
        ours.edge_transform['linear1'].weight.data.copy_(orig.fc1_1.weight.data)
        ours.edge_transform['linear2'].weight.data.copy_(orig.fc1_2.weight.data)
        ours.edge_transform['linear3'].weight.data.copy_(orig.fc1_3.weight.data)
        ours.edge_transform['linear4'].weight.data.copy_(orig.fc1_4.weight.data)

    # Copy skip connection weights
    if orig.nout2 > 0:
        ours.skip_connection['linear1'].weight.data.copy_(orig.fc11.weight.data)
        ours.skip_connection['linear1'].bias.data.copy_(orig.fc11.bias.data)
        ours.skip_connection['linear2'].weight.data.copy_(orig.fc12.weight.data)
        ours.skip_connection['linear2'].bias.data.copy_(orig.fc12.bias.data)


# =============================================================================
# SpectConv Equivalence
# =============================================================================

class TestSpectConvEquivalence:
    """Verify our SpectConvWithAttention (no attention) == original SpectConv."""

    @pytest.mark.parametrize("selfconn", [True, False])
    def test_basic_equivalence(self, selfconn):
        """Same weights, same input -> identical output."""
        torch.manual_seed(0)
        in_ch, out_ch, K = 8, 16, 4

        orig = OrigSpectConv(in_ch, out_ch, K=K, selfconn=selfconn)
        ours = SpectConvWithAttention(in_ch, out_ch, K=K, selfconn=selfconn,
                                      use_attention=False)

        _copy_spectconv_weights(orig, ours)

        x, edge_index, _ = _make_test_graph(node_dim=in_ch, edge_dim=K)
        edge_attr = torch.randn(edge_index.size(1), K)

        orig.eval()
        ours.eval()

        with torch.no_grad():
            out_orig = orig(x, edge_index, edge_attr)
            out_ours, att = ours(x, edge_index, edge_attr)

        assert att is None, "Attention should be None when disabled"
        assert torch.allclose(out_orig, out_ours, atol=1e-6), \
            f"Max diff: {(out_orig - out_ours).abs().max():.2e}"

    def test_different_inputs(self):
        """Equivalence holds across multiple random inputs."""
        torch.manual_seed(0)
        in_ch, out_ch, K = 6, 12, 3

        orig = OrigSpectConv(in_ch, out_ch, K=K, selfconn=True)
        ours = SpectConvWithAttention(in_ch, out_ch, K=K, selfconn=True,
                                      use_attention=False)
        _copy_spectconv_weights(orig, ours)
        orig.eval()
        ours.eval()

        for seed in range(10):
            x, edge_index, _ = _make_test_graph(
                n_nodes=8 + seed, node_dim=in_ch, edge_dim=K, seed=seed
            )
            edge_attr = torch.randn(edge_index.size(1), K)

            with torch.no_grad():
                out_orig = orig(x, edge_index, edge_attr)
                out_ours, _ = ours(x, edge_index, edge_attr)

            assert torch.allclose(out_orig, out_ours, atol=1e-6), \
                f"Seed {seed}: max diff {(out_orig - out_ours).abs().max():.2e}"

    def test_gradient_equivalence(self):
        """Gradients w.r.t. input match between implementations."""
        torch.manual_seed(0)
        in_ch, out_ch, K = 8, 16, 4

        orig = OrigSpectConv(in_ch, out_ch, K=K, selfconn=True)
        ours = SpectConvWithAttention(in_ch, out_ch, K=K, selfconn=True,
                                      use_attention=False)
        _copy_spectconv_weights(orig, ours)

        x_base, edge_index, _ = _make_test_graph(node_dim=in_ch, edge_dim=K)
        edge_attr = torch.randn(edge_index.size(1), K)

        x_orig = x_base.clone().requires_grad_(True)
        x_ours = x_base.clone().requires_grad_(True)

        out_orig = orig(x_orig, edge_index, edge_attr)
        out_ours, _ = ours(x_ours, edge_index, edge_attr)

        out_orig.sum().backward()
        out_ours.sum().backward()

        assert torch.allclose(x_orig.grad, x_ours.grad, atol=1e-6), \
            f"Input grad max diff: {(x_orig.grad - x_ours.grad).abs().max():.2e}"

    def test_weight_gradient_equivalence(self):
        """Gradients w.r.t. weight parameters match."""
        torch.manual_seed(0)
        in_ch, out_ch, K = 8, 16, 4

        orig = OrigSpectConv(in_ch, out_ch, K=K, selfconn=True)
        ours = SpectConvWithAttention(in_ch, out_ch, K=K, selfconn=True,
                                      use_attention=False)
        _copy_spectconv_weights(orig, ours)

        x, edge_index, _ = _make_test_graph(node_dim=in_ch, edge_dim=K)
        edge_attr = torch.randn(edge_index.size(1), K)

        out_orig = orig(x, edge_index, edge_attr)
        out_ours, _ = ours(x, edge_index, edge_attr)

        out_orig.sum().backward()
        out_ours.sum().backward()

        assert torch.allclose(orig.weight.grad, ours.weight.grad, atol=1e-6), \
            f"Weight grad max diff: {(orig.weight.grad - ours.weight.grad).abs().max():.2e}"
        assert torch.allclose(orig.bias.grad, ours.bias.grad, atol=1e-6), \
            f"Bias grad max diff: {(orig.bias.grad - ours.bias.grad).abs().max():.2e}"


# =============================================================================
# ML3Layer Equivalence
# =============================================================================

class TestML3LayerEquivalence:
    """Verify our ML3Layer (no attention) == original ML3Layer."""

    @pytest.mark.parametrize("learnedge", [True, False])
    @pytest.mark.parametrize("nout2", [0, 8])
    def test_ml3layer_equivalence(self, learnedge, nout2):
        """Same weights, same input -> identical output."""
        torch.manual_seed(0)
        ninp, nout1 = 8, 16
        nedge = 4

        orig = OrigML3Layer(
            learnedge=learnedge, nedgeinput=nedge, nedgeoutput=nedge,
            ninp=ninp, nout1=nout1, nout2=nout2
        )
        ours = OurML3Layer(
            learnedge=learnedge, nedgeinput=nedge, nedgeoutput=nedge,
            ninp=ninp, nout1=nout1, nout2=nout2, use_attention=False
        )

        _copy_ml3layer_weights(orig, ours)
        orig.eval()
        ours.eval()

        x, edge_index, _ = _make_test_graph(node_dim=ninp, edge_dim=nedge)
        edge_attr = torch.randn(edge_index.size(1), nedge)

        with torch.no_grad():
            out_orig = orig(x, edge_index, edge_attr)
            out_ours, att = ours(x, edge_index, edge_attr)

        assert att is None, "Attention should be None when disabled"
        assert out_orig.shape == out_ours.shape, \
            f"Shape mismatch: {out_orig.shape} vs {out_ours.shape}"
        assert torch.allclose(out_orig, out_ours, atol=1e-6), \
            f"Max diff: {(out_orig - out_ours).abs().max():.2e}"

    def test_ml3layer_gradient_equivalence(self):
        """Gradients through ML3Layer match between implementations."""
        torch.manual_seed(0)
        ninp, nout1, nout2, nedge = 8, 16, 8, 4

        orig = OrigML3Layer(
            learnedge=True, nedgeinput=nedge, nedgeoutput=nedge,
            ninp=ninp, nout1=nout1, nout2=nout2
        )
        ours = OurML3Layer(
            learnedge=True, nedgeinput=nedge, nedgeoutput=nedge,
            ninp=ninp, nout1=nout1, nout2=nout2, use_attention=False
        )
        _copy_ml3layer_weights(orig, ours)

        x_base, edge_index, _ = _make_test_graph(node_dim=ninp, edge_dim=nedge)
        edge_attr = torch.randn(edge_index.size(1), nedge)

        x_orig = x_base.clone().requires_grad_(True)
        x_ours = x_base.clone().requires_grad_(True)

        out_orig = orig(x_orig, edge_index, edge_attr)
        out_ours, _ = ours(x_ours, edge_index, edge_attr)

        out_orig.sum().backward()
        out_ours.sum().backward()

        assert torch.allclose(x_orig.grad, x_ours.grad, atol=1e-6), \
            f"Input grad max diff: {(x_orig.grad - x_ours.grad).abs().max():.2e}"

    def test_stacked_layers_equivalence(self):
        """Three stacked ML3Layers (like original GNNML3) match."""
        torch.manual_seed(0)
        nout1, nout2, nedge = 16, 8, 4
        nin = nout1 + nout2
        ninp = 8  # initial node features

        # Build original stack
        orig_layers = nn.ModuleList([
            OrigML3Layer(learnedge=True, nedgeinput=nedge, nedgeoutput=nedge,
                         ninp=ninp, nout1=nout1, nout2=nout2),
            OrigML3Layer(learnedge=True, nedgeinput=nedge, nedgeoutput=nedge,
                         ninp=nin, nout1=nout1, nout2=nout2),
            OrigML3Layer(learnedge=True, nedgeinput=nedge, nedgeoutput=nedge,
                         ninp=nin, nout1=nout1, nout2=nout2),
        ])

        # Build our stack
        our_layers = nn.ModuleList([
            OurML3Layer(learnedge=True, nedgeinput=nedge, nedgeoutput=nedge,
                        ninp=ninp, nout1=nout1, nout2=nout2, use_attention=False),
            OurML3Layer(learnedge=True, nedgeinput=nedge, nedgeoutput=nedge,
                        ninp=nin, nout1=nout1, nout2=nout2, use_attention=False),
            OurML3Layer(learnedge=True, nedgeinput=nedge, nedgeoutput=nedge,
                        ninp=nin, nout1=nout1, nout2=nout2, use_attention=False),
        ])

        # Copy weights
        for orig_l, our_l in zip(orig_layers, our_layers):
            _copy_ml3layer_weights(orig_l, our_l)

        orig_layers.eval()
        our_layers.eval()

        x, edge_index, _ = _make_test_graph(node_dim=ninp, edge_dim=nedge)
        edge_attr = torch.randn(edge_index.size(1), nedge)

        with torch.no_grad():
            h_orig = x.clone()
            for layer in orig_layers:
                h_orig = layer(h_orig, edge_index, edge_attr)

            h_ours = x.clone()
            for layer in our_layers:
                h_ours, _ = layer(h_ours, edge_index, edge_attr)

        assert torch.allclose(h_orig, h_ours, atol=1e-5), \
            f"Stacked layers max diff: {(h_orig - h_ours).abs().max():.2e}"


# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
