"""
Tests for Phase 3 — warm-starting retrains.

Covers the 7 tests listed in IMPLEMENTATION_PLAN.md Phase 3:
  * preserves encoder weights
  * replaces classifier (output dim changes)
  * preserves encoder-side BN running stats
  * resets reversible score module (RevVAMPNet only)
  * optimizer reinit picks up new classifier params
  * off by default in config
  * short training converges / doesn't catastrophically fail
"""

import copy
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from pygv.classifier.SoftmaxMLP import SoftmaxMLP
from pygv.config.base_config import BaseConfig
from pygv.encoder.schnet import SchNetEncoderNoEmbed
from pygv.scores.reversible_score import ReversibleVAMPScore
from pygv.scores.vamp_score_v0 import VAMPScore
from pygv.vampnet.rev_vampnet import RevVAMPNet
from pygv.vampnet.vampnet import VAMPNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODE_DIM = 8
EDGE_DIM = 4
HIDDEN = 16
OUT = 12


def _make_encoder() -> SchNetEncoderNoEmbed:
    """SchNet encoder -- has BatchNorm via the default PyG MLP norm, which is
    required for the BN running-stats test."""
    torch.manual_seed(0)
    return SchNetEncoderNoEmbed(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=HIDDEN,
        output_dim=OUT,
        n_interactions=2,
    )


def _make_classifier(out_channels: int = 5) -> SoftmaxMLP:
    return SoftmaxMLP(
        in_channels=OUT,
        hidden_channels=HIDDEN,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.1,
        act="relu",
        norm="batch_norm",
    )


def _make_vampnet(n_classes: int = 5) -> VAMPNet:
    return VAMPNet(
        encoder=_make_encoder(),
        vamp_score=VAMPScore(epsilon=1e-6, mode="regularize"),
        classifier_module=_make_classifier(n_classes),
    )


def _make_rev_vampnet(n_classes: int = 5) -> RevVAMPNet:
    return RevVAMPNet(
        encoder=_make_encoder(),
        rev_score=ReversibleVAMPScore(n_states=n_classes, epsilon=1e-6),
        classifier_module=_make_classifier(n_classes),
    )


def _random_graph_batch(n_graphs: int = 4, n_nodes: int = 6) -> Data:
    """Small batched graph input for forward-pass sanity checks."""
    torch.manual_seed(1)
    datas = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes, NODE_DIM)
        # simple ring topology
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        edge_attr = torch.randn(edge_index.size(1), EDGE_DIM)
        datas.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return Batch.from_data_list(datas)


def _snapshot_state(module: nn.Module) -> dict:
    """Deep-copy of state_dict, detached from the module (safe to compare later)."""
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _states_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# 1. Encoder weights preserved
# ---------------------------------------------------------------------------

def test_warm_restart_preserves_encoder_weights():
    model = _make_vampnet(n_classes=5)
    encoder_before = _snapshot_state(model.encoder)
    model.warm_restart_with_new_k(3)
    encoder_after = _snapshot_state(model.encoder)
    assert _states_equal(encoder_before, encoder_after), (
        "Encoder weights must be byte-identical after warm-restart"
    )


def test_warm_restart_preserves_embedding_module_when_present():
    """If a model has an embedding_module, it must survive the restart."""
    model = VAMPNet(
        encoder=_make_encoder(),
        vamp_score=VAMPScore(epsilon=1e-6, mode="regularize"),
        embedding_in_dim=NODE_DIM,
        embedding_out_dim=NODE_DIM,
        embedding_hidden_dim=HIDDEN,
        classifier_module=_make_classifier(5),
    )
    assert model.embedding_module is not None
    emb_before = _snapshot_state(model.embedding_module)
    model.warm_restart_with_new_k(3)
    emb_after = _snapshot_state(model.embedding_module)
    assert _states_equal(emb_before, emb_after)


# ---------------------------------------------------------------------------
# 2. Classifier replaced, output dim updated
# ---------------------------------------------------------------------------

def test_warm_restart_replaces_classifier():
    model = _make_vampnet(n_classes=10)
    old_classifier = model.classifier_module
    model.warm_restart_with_new_k(7)
    new_classifier = model.classifier_module

    assert new_classifier is not old_classifier, "classifier module must be a new object"
    assert new_classifier.out_channels == 7
    # The final Linear's output features must match new_k
    assert new_classifier.final_layer[0].out_features == 7


def test_warm_restart_rejects_k_less_than_2():
    model = _make_vampnet(n_classes=5)
    with pytest.raises(ValueError, match="new_k must be >= 2"):
        model.warm_restart_with_new_k(1)


# ---------------------------------------------------------------------------
# 3. Encoder BN running stats preserved
# ---------------------------------------------------------------------------

def test_warm_restart_preserves_encoder_bn_running_stats():
    """Run a few forward passes in train mode to populate encoder BN running
    stats, then warm-restart and verify those stats are unchanged."""
    model = _make_vampnet(n_classes=5)
    model.train()
    for _ in range(3):
        batch = _random_graph_batch()
        _ = model(batch, apply_classifier=False)  # only encoder path

    # Snapshot any BN-style buffers inside the encoder before restart
    bn_snap_before = {}
    for name, mod in model.encoder.named_modules():
        if hasattr(mod, "running_mean") and mod.running_mean is not None:
            bn_snap_before[f"{name}.running_mean"] = mod.running_mean.detach().clone()
            bn_snap_before[f"{name}.running_var"] = mod.running_var.detach().clone()
            bn_snap_before[f"{name}.num_batches_tracked"] = (
                mod.num_batches_tracked.detach().clone()
            )

    # There must actually be some BN buffers in the encoder for this test
    # to have any signal — SchNet's default MLP uses batch_norm, so this
    # should yield several entries.
    assert len(bn_snap_before) > 0, (
        "Encoder has no BN running-stats buffers; test infrastructure broken "
        "(expected SchNet encoder to include BN via default PyG MLP norm)"
    )

    model.warm_restart_with_new_k(3)

    bn_snap_after = {}
    for name, mod in model.encoder.named_modules():
        if hasattr(mod, "running_mean") and mod.running_mean is not None:
            bn_snap_after[f"{name}.running_mean"] = mod.running_mean.detach().clone()
            bn_snap_after[f"{name}.running_var"] = mod.running_var.detach().clone()
            bn_snap_after[f"{name}.num_batches_tracked"] = (
                mod.num_batches_tracked.detach().clone()
            )

    for k, v in bn_snap_before.items():
        assert torch.equal(v, bn_snap_after[k]), (
            f"Encoder BN buffer '{k}' must not change during warm-restart"
        )


# ---------------------------------------------------------------------------
# 4. Reversible score reset for RevVAMPNet
# ---------------------------------------------------------------------------

def test_warm_restart_revvampnet_resets_reversible():
    model = _make_rev_vampnet(n_classes=10)
    # Perturb the score params so we can tell if they're reset to fresh init
    with torch.no_grad():
        model.rev_score.log_stationary.fill_(3.14)
        model.rev_score.rate_matrix_weights.fill_(2.71)

    model.warm_restart_with_new_k(6)

    # New score module, new shape
    assert model.rev_score.n_states == 6
    assert model.rev_score.log_stationary.shape == (6,)
    assert model.rev_score.rate_matrix_weights.shape == (6, 6)

    # Fresh init is zeros (per ReversibleVAMPScore.__init__); the perturbed
    # 3.14 / 2.71 must be gone.
    assert torch.allclose(model.rev_score.log_stationary,
                          torch.zeros(6))
    assert torch.allclose(model.rev_score.rate_matrix_weights,
                          torch.zeros(6, 6))
    # Epsilon preserved
    assert model.rev_score.epsilon == pytest.approx(1e-6)


def test_warm_restart_revvampnet_preserves_encoder_weights():
    model = _make_rev_vampnet(n_classes=10)
    enc_before = _snapshot_state(model.encoder)
    model.warm_restart_with_new_k(4)
    enc_after = _snapshot_state(model.encoder)
    assert _states_equal(enc_before, enc_after)


# ---------------------------------------------------------------------------
# 5. Optimizer reinit picks up new classifier params
# ---------------------------------------------------------------------------

def test_warm_restart_optimizer_reinit_required():
    """After warm-restart, model.parameters() must iterate over the NEW
    classifier's parameters, so a fresh optimizer picks them up."""
    model = _make_vampnet(n_classes=10)
    before_ids = {id(p) for p in model.classifier_module.parameters()}
    model.warm_restart_with_new_k(5)
    after_ids = {id(p) for p in model.classifier_module.parameters()}

    # Disjoint: none of the old classifier's tensors remain
    assert before_ids.isdisjoint(after_ids)

    # Model.parameters() now includes the new ones
    all_ids = {id(p) for p in model.parameters()}
    assert after_ids.issubset(all_ids), (
        "Fresh classifier parameters must appear in model.parameters() "
        "so a freshly-constructed optimizer sees them"
    )

    # Fresh optimizer can be built without error and has nonzero param count
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_param_groups_with_tensors = sum(len(g["params"]) for g in opt.param_groups)
    assert n_param_groups_with_tensors > 0


# ---------------------------------------------------------------------------
# 6. Off by default
# ---------------------------------------------------------------------------

def test_baseconfig_warm_start_default_false():
    cfg = BaseConfig()
    assert cfg.warm_start_retrains is False


# ---------------------------------------------------------------------------
# 7. Short training converges (does not catastrophically break)
# ---------------------------------------------------------------------------

def test_warm_restart_model_still_trainable():
    """Warm-restart a model, then run a few gradient steps.  Loss must be
    finite and decrease (or at least not explode to NaN/inf)."""
    torch.manual_seed(42)
    model = _make_vampnet(n_classes=10)
    model.train()

    # Brief pre-training so encoder BN has nontrivial running stats
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        batch_t0 = _random_graph_batch()
        batch_t1 = _random_graph_batch(n_graphs=4, n_nodes=6)
        opt.zero_grad()
        probs_t0, _ = model(batch_t0, apply_classifier=True)
        probs_t1, _ = model(batch_t1, apply_classifier=True)
        loss = model.vamp_score.loss(probs_t0, probs_t1)
        loss.backward()
        opt.step()

    # Warm-restart to smaller k
    model.warm_restart_with_new_k(4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        batch_t0 = _random_graph_batch()
        batch_t1 = _random_graph_batch(n_graphs=4, n_nodes=6)
        opt.zero_grad()
        probs_t0, _ = model(batch_t0, apply_classifier=True)
        probs_t1, _ = model(batch_t1, apply_classifier=True)
        loss = model.vamp_score.loss(probs_t0, probs_t1)
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    # Model did not diverge into NaN territory
    final_params = [p.detach() for p in model.parameters()]
    for p in final_params:
        assert torch.isfinite(p).all(), "Model parameters diverged to NaN/inf during post-warm-restart training"


def test_warm_restart_model_output_shape_matches_new_k():
    """After warm-restart, the classifier must produce (batch, new_k) outputs."""
    model = _make_vampnet(n_classes=10)
    batch = _random_graph_batch()
    probs_before, _ = model(batch, apply_classifier=True)
    assert probs_before.shape[-1] == 10

    model.warm_restart_with_new_k(3)
    model.eval()
    probs_after, _ = model(batch, apply_classifier=True)
    assert probs_after.shape[-1] == 3
