"""
Comprehensive unit tests for the ML3 encoder.

Tests cover: basic sanity, permutation invariance, gradient flow,
WL expressiveness, determinism, learnability, edge modes, attention,
and VAMP integration.

Run with: pytest tests/test_ml3_encoder.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import TUDataset

from pygv.encoder.ml3 import ML3Encoder
from pygv.scores.vamp_score_v0 import VAMPScore


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def mutag_dataset():
    """MUTAG dataset: 7 node features, 4 edge features."""
    return TUDataset(root="/tmp/TUDataset", name="MUTAG")


@pytest.fixture
def ml3_factory():
    """Factory returning an ML3Encoder with configurable parameters."""
    def _make(node_dim=7, edge_dim=4, hidden_dim=30, output_dim=32,
              num_layers=3, activation='relu', use_attention=True,
              edge_mode='gaussian', nout1=30, nout2=2, **kwargs):
        return ML3Encoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            activation=activation,
            use_attention=use_attention,
            edge_mode=edge_mode,
            nout1=nout1,
            nout2=nout2,
            **kwargs,
        )
    return _make


# =============================================================================
# Helpers
# =============================================================================

def _prepare_mutag_graph(data, edge_dim=4):
    """Ensure a MUTAG graph has edge_attr."""
    if data.edge_attr is None:
        data = data.clone()
        data.edge_attr = torch.ones(data.edge_index.size(1), edge_dim)
    return data


def _make_cycle(n, node_dim=7, edge_dim=4):
    """Create an n-cycle graph with constant features."""
    src = list(range(n))
    dst = [(i + 1) % n for i in range(n)]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    x = torch.ones(n, node_dim)
    edge_attr = torch.ones(edge_index.size(1), edge_dim)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _permute_graph(data):
    """Return a node-permuted copy of data."""
    n = data.x.size(0)
    perm = torch.randperm(n)
    new_x = data.x[perm]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n)
    new_edge_index = inv_perm[data.edge_index]
    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=data.edge_attr.clone(),
    )
    return new_data


# =============================================================================
# 1. Basic Sanity
# =============================================================================

class TestBasicSanity:

    def test_single_graph_forward(self, mutag_dataset, ml3_factory):
        """Single MUTAG graph -> shape (1, output_dim), dtype float32."""
        data = _prepare_mutag_graph(mutag_dataset[0])
        model = ml3_factory()
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x, data.edge_index, data.edge_attr)
        assert out.shape == (1, 32)
        assert out.dtype == torch.float32

    def test_batched_forward(self, mutag_dataset, ml3_factory):
        """Batch 8 MUTAG graphs -> shape (8, output_dim)."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(8)]
        batch = Batch.from_data_list(graphs)
        model = ml3_factory()
        model.eval()
        with torch.no_grad():
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert out.shape == (8, 32)

    def test_no_nan_inf(self, mutag_dataset, ml3_factory):
        """50 MUTAG graphs -> all outputs finite."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(min(50, len(mutag_dataset)))]
        batch = Batch.from_data_list(graphs)
        model = ml3_factory()
        model.eval()
        with torch.no_grad():
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_non_degenerate_output(self, mutag_dataset, ml3_factory):
        """20 graphs -> more than 1 unique row in output."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(20)]
        batch = Batch.from_data_list(graphs)
        model = ml3_factory()
        model.eval()
        with torch.no_grad():
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        unique_rows = torch.unique(out, dim=0)
        assert unique_rows.size(0) > 1, "All graph embeddings are identical"


# =============================================================================
# 2. Permutation Invariance
# =============================================================================

class TestPermutationInvariance:

    def test_permutation_invariance(self, mutag_dataset, ml3_factory):
        """Graph-level output is invariant to node permutation."""
        torch.manual_seed(0)
        model = ml3_factory()
        model.eval()

        for i in range(10):
            data = _prepare_mutag_graph(mutag_dataset[i])
            perm_data = _permute_graph(data)

            with torch.no_grad():
                out_orig, _ = model(data.x, data.edge_index, data.edge_attr)
                out_perm, _ = model(perm_data.x, perm_data.edge_index, perm_data.edge_attr)

            assert torch.allclose(out_orig, out_perm, atol=1e-5), \
                f"Graph {i}: permuted output differs (max diff {(out_orig - out_perm).abs().max():.2e})"


# =============================================================================
# 3. Gradient Flow
# =============================================================================

class TestGradientFlow:

    def test_all_params_receive_gradients(self, mutag_dataset, ml3_factory):
        """Forward + backward -> no param has grad=None or all-zero."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(8)]
        batch = Batch.from_data_list(graphs)
        model = ml3_factory()

        out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out.sum().backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name}: grad is None"
                assert not torch.all(p.grad == 0), f"{name}: grad is all zeros"

    def test_gradient_magnitudes_reasonable(self, mutag_dataset, ml3_factory):
        """No grad norm > 1e6."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(16)]
        batch = Batch.from_data_list(graphs)
        model = ml3_factory()

        out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out.sum().backward()

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_norm = p.grad.norm().item()
                assert grad_norm <= 1e6, f"{name}: grad norm {grad_norm:.2e} > 1e6"


# =============================================================================
# 4. WL Expressiveness
# =============================================================================

class TestWLExpressiveness:

    def test_different_structure_graphs(self, ml3_factory):
        """Star graph vs path graph -> different embeddings (basic 1-WL)."""
        torch.manual_seed(42)
        model = ml3_factory(use_attention=True)
        model.eval()

        n, node_dim, edge_dim = 6, 7, 4

        # Star graph: node 0 connected to all others
        src_s = [0] * (n - 1) + list(range(1, n))
        dst_s = list(range(1, n)) + [0] * (n - 1)
        star = Data(
            x=torch.ones(n, node_dim),
            edge_index=torch.tensor([src_s, dst_s], dtype=torch.long),
            edge_attr=torch.ones(len(src_s), edge_dim),
        )

        # Path graph: 0-1-2-3-4-5
        src_p, dst_p = [], []
        for i in range(n - 1):
            src_p += [i, i + 1]
            dst_p += [i + 1, i]
        path = Data(
            x=torch.ones(n, node_dim),
            edge_index=torch.tensor([src_p, dst_p], dtype=torch.long),
            edge_attr=torch.ones(len(src_p), edge_dim),
        )

        with torch.no_grad():
            out_star, _ = model(star.x, star.edge_index, star.edge_attr)
            out_path, _ = model(path.x, path.edge_index, path.edge_attr)

        assert not torch.allclose(out_star, out_path, atol=1e-4), \
            "Star and path graphs should produce different embeddings"

    @pytest.mark.xfail(reason="3-WL test: may require spectral mode or specific architecture "
                               "tuning to distinguish graphs beyond 1-WL")
    def test_3wl_expressiveness(self, ml3_factory):
        """Graphs indistinguishable by 1-WL but distinguishable by 3-WL.

        Uses the Rook's 4x4 graph vs Shrikhande graph — a classic pair that
        1-WL cannot distinguish but higher-order WL can.
        Here we use a simpler proxy: C6 vs 2xC3 (6-cycle vs two triangles).
        """
        c6 = _make_cycle(6)

        # 2xC3: two disjoint triangles as single graph
        ei1 = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]], dtype=torch.long)
        ei2 = torch.tensor([[3, 4, 5, 4, 5, 3], [4, 5, 3, 3, 4, 5]], dtype=torch.long)
        edge_index = torch.cat([ei1, ei2], dim=1)
        x = torch.ones(6, 7)
        edge_attr = torch.ones(edge_index.size(1), 4)
        two_c3 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        distinguished = False
        for seed in range(10):
            torch.manual_seed(seed)
            model = ml3_factory()
            model.eval()
            with torch.no_grad():
                out_c6, _ = model(c6.x, c6.edge_index, c6.edge_attr)
                out_2c3, _ = model(two_c3.x, two_c3.edge_index, two_c3.edge_attr)
            if not torch.allclose(out_c6, out_2c3, atol=1e-4):
                distinguished = True
                break

        assert distinguished, "Could not distinguish C6 from 2xC3 across 10 seeds"


# =============================================================================
# 5. Determinism
# =============================================================================

class TestDeterminism:

    def test_determinism_eval_mode(self, mutag_dataset, ml3_factory):
        """model.eval(), same graph twice -> bitwise identical output."""
        torch.manual_seed(99)
        model = ml3_factory()
        model.eval()

        data = _prepare_mutag_graph(mutag_dataset[0])
        with torch.no_grad():
            out1, _ = model(data.x, data.edge_index, data.edge_attr)
            out2, _ = model(data.x, data.edge_index, data.edge_attr)

        assert torch.equal(out1, out2), "Eval-mode outputs are not bitwise identical"


# =============================================================================
# 6. Learnability
# =============================================================================

class TestLearnability:

    def test_loss_decreases_mutag(self, mutag_dataset, ml3_factory):
        """Train on 100 MUTAG graphs for 20 epochs -> loss decreases by >=20%."""
        torch.manual_seed(42)
        n_graphs = min(100, len(mutag_dataset))
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(n_graphs)]
        batch = Batch.from_data_list(graphs)
        labels = torch.tensor([mutag_dataset[i].y.item() for i in range(n_graphs)])

        model = ml3_factory()
        head = nn.Linear(32, 2)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None
        for epoch in range(20):
            optimizer.zero_grad()
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = head(out)
            loss = criterion(logits, labels)
            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
            loss.backward()
            optimizer.step()

        assert final_loss < 0.8 * initial_loss, \
            f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_overfit_tiny_batch(self, mutag_dataset, ml3_factory):
        """5 graphs, 200 steps -> >=80% accuracy."""
        torch.manual_seed(42)
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(5)]
        batch = Batch.from_data_list(graphs)
        labels = torch.tensor([mutag_dataset[i].y.item() for i in range(5)])

        model = ml3_factory()
        head = nn.Linear(32, 2)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for _ in range(200):
            optimizer.zero_grad()
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = head(out)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = head(out).argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        assert acc >= 0.8, f"Could not overfit 5 graphs: accuracy={acc:.0%}"


# =============================================================================
# 7. Edge Modes
# =============================================================================

class TestEdgeModes:

    def test_gaussian_mode(self, mutag_dataset, ml3_factory):
        """Gaussian edge mode produces finite output."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(8)]
        batch = Batch.from_data_list(graphs)
        model = ml3_factory(edge_mode='gaussian')
        model.eval()
        with torch.no_grad():
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert out.shape == (8, 32)
        assert torch.isfinite(out).all()

    def test_spectral_mode(self, ml3_factory):
        """Spectral edge mode produces finite output on synthetic graphs."""
        torch.manual_seed(42)
        graphs = []
        for _ in range(4):
            n = 8
            x = torch.randn(n, 7)
            # Create connected graph
            src = list(range(n))
            dst = [(i + 1) % n for i in range(n)]
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            edge_attr = torch.randn(edge_index.size(1), 4)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        batch = Batch.from_data_list(graphs)
        model = ml3_factory(edge_mode='spectral', nfreq=5)
        model.eval()
        with torch.no_grad():
            out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

    def test_both_modes_finite(self, ml3_factory):
        """Both edge modes produce finite, non-degenerate outputs."""
        torch.manual_seed(0)
        n = 6
        x = torch.randn(n, 7)
        src = list(range(n))
        dst = [(i + 1) % n for i in range(n)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        edge_attr = torch.randn(edge_index.size(1), 4)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        for mode in ['gaussian', 'spectral']:
            torch.manual_seed(0)
            model = ml3_factory(edge_mode=mode, nfreq=5)
            model.eval()
            with torch.no_grad():
                out, _ = model(data.x, data.edge_index, data.edge_attr)
            assert torch.isfinite(out).all(), f"mode={mode}: output not finite"


# =============================================================================
# 8. Attention
# =============================================================================

class TestAttention:

    def test_attention_disabled_works(self, mutag_dataset, ml3_factory):
        """Encoder works correctly with attention disabled."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(8)]
        batch = Batch.from_data_list(graphs)
        model = ml3_factory(use_attention=False)
        model.eval()
        with torch.no_grad():
            out, (h, attentions) = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert out.shape == (8, 32)
        assert torch.isfinite(out).all()
        assert len(attentions) == 0, "Should have no attention weights when disabled"

    def test_attention_weights_returned(self, mutag_dataset, ml3_factory):
        """Attention weights are returned with correct shapes."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(8)]
        batch = Batch.from_data_list(graphs)
        n_layers = 3
        model = ml3_factory(use_attention=True, num_layers=n_layers)
        model.eval()
        with torch.no_grad():
            out, (h, attentions) = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert len(attentions) == n_layers, \
            f"Expected {n_layers} attention tensors, got {len(attentions)}"

        num_edges = batch.edge_index.size(1)
        for i, att in enumerate(attentions):
            assert att.shape == (num_edges,), \
                f"Attention layer {i}: expected ({num_edges},), got {att.shape}"

    def test_attention_sums_to_one(self, mutag_dataset, ml3_factory):
        """Attention weights sum to ~1 for each target node."""
        graphs = [_prepare_mutag_graph(mutag_dataset[i]) for i in range(8)]
        batch = Batch.from_data_list(graphs)
        model = ml3_factory(use_attention=True)
        model.eval()
        with torch.no_grad():
            out, (h, attentions) = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        att = attentions[0]
        target_nodes = batch.edge_index[1]

        for target in torch.unique(target_nodes)[:10]:
            mask = target_nodes == target
            att_sum = att[mask].sum().item()
            assert abs(att_sum - 1.0) < 1e-4, \
                f"Attention for target node {target} sums to {att_sum:.6f}, expected ~1.0"


# =============================================================================
# 9. VAMP Integration
# =============================================================================

class TestVAMPIntegration:

    def test_vamp2_score_with_ml3(self, ml3_factory):
        """ML3 embeddings -> VAMP2 score -> finite positive scalar, gradients flow."""
        torch.manual_seed(42)
        output_dim = 32
        model = ml3_factory(output_dim=output_dim)
        vamp = VAMPScore(method='VAMP2')

        n_graphs = 16
        graphs_t, graphs_tlag = [], []
        for _ in range(n_graphs):
            n_nodes = torch.randint(5, 15, (1,)).item()
            n_edges = n_nodes * 3
            ei = torch.randint(0, n_nodes, (2, n_edges))
            graphs_t.append(Data(
                x=torch.randn(n_nodes, 7),
                edge_index=ei,
                edge_attr=torch.randn(n_edges, 4),
            ))
            graphs_tlag.append(Data(
                x=torch.randn(n_nodes, 7),
                edge_index=ei.clone(),
                edge_attr=torch.randn(n_edges, 4),
            ))

        batch_t = Batch.from_data_list(graphs_t)
        batch_tlag = Batch.from_data_list(graphs_tlag)

        emb_t, _ = model(batch_t.x, batch_t.edge_index, batch_t.edge_attr, batch_t.batch)
        emb_tlag, _ = model(batch_tlag.x, batch_tlag.edge_index, batch_tlag.edge_attr, batch_tlag.batch)

        score = vamp(emb_t, emb_tlag)
        loss = -score

        assert torch.isfinite(score), f"VAMP2 score is not finite: {score.item()}"
        assert score.item() > 0, f"VAMP2 score should be positive, got {score.item()}"

        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients flowed through ML3 encoder from VAMP2 loss"


# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
