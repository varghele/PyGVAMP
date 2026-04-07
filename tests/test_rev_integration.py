"""
Integration tests for RevGraphVAMP pipeline integration.

Tests verify:
1. create_model() produces correct model type based on reversible flag
2. Analysis detects RevVAMPNet and extracts learned K
3. Config reversible field roundtrips

Run with: pytest tests/test_rev_integration.py -v
"""

import pytest
import torch
import numpy as np
import argparse

from pygv.vampnet.vampnet import VAMPNet
from pygv.vampnet.rev_vampnet import RevVAMPNet
from pygv.scores.reversible_score import ReversibleVAMPScore
from pygv.config.base_config import BaseConfig
from pygv.utils.analysis import extract_learned_transition_matrix
from pygv.encoder.schnet import SchNetEncoderNoEmbed
from pygv.classifier.SoftmaxMLP import SoftmaxMLP


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def seed():
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


def _make_train_args(reversible=False, encoder_type='schnet'):
    """Create minimal args namespace for create_model()."""
    args = argparse.Namespace(
        # Encoder
        encoder_type=encoder_type,
        node_dim=16,
        edge_dim=16,
        hidden_dim=32,
        output_dim=32,
        n_interactions=2,
        activation='relu',
        use_attention=False,
        edge_norm_eps=1e-8,
        # Classifier
        n_states=5,
        clf_hidden_dim=32,
        clf_num_layers=2,
        clf_dropout=0.0,
        clf_activation='relu',
        clf_norm=None,
        # Embedding
        use_embedding=False,
        embedding_in_dim=None,
        embedding_hidden_dim=64,
        embedding_out_dim=32,
        embedding_num_layers=2,
        embedding_dropout=0.0,
        embedding_act='relu',
        embedding_norm=None,
        # Training
        lag_time=1.0,
        training_jitter=1e-6,
        vamp_epsilon=1e-6,
        cpu=True,
        # Reversible
        reversible=reversible,
    )
    return args


# =============================================================================
# Tests
# =============================================================================

def test_create_rev_model_via_training_pipeline(seed):
    """create_model() with reversible=True should return RevVAMPNet."""
    from pygv.pipe.training import create_model

    args = _make_train_args(reversible=True)
    model = create_model(args)

    assert isinstance(model, RevVAMPNet), f"Expected RevVAMPNet, got {type(model).__name__}"
    assert hasattr(model, 'get_transition_matrix')
    assert hasattr(model, 'get_stationary_distribution')
    assert hasattr(model, 'rev_score')


def test_create_standard_model_unchanged(seed):
    """create_model() with reversible=False should return standard VAMPNet."""
    from pygv.pipe.training import create_model

    args = _make_train_args(reversible=False)
    model = create_model(args)

    assert isinstance(model, VAMPNet), f"Expected VAMPNet, got {type(model).__name__}"
    assert not isinstance(model, RevVAMPNet), "Should not be RevVAMPNet"


def test_analysis_detects_rev_model(seed):
    """extract_learned_transition_matrix should work with RevVAMPNet."""
    encoder = SchNetEncoderNoEmbed(
        node_dim=16, edge_dim=16, hidden_dim=32, output_dim=32, n_interactions=2
    )
    classifier = SoftmaxMLP(
        in_channels=32, hidden_channels=32, out_channels=5, num_layers=2
    )
    rev_score = ReversibleVAMPScore(n_states=5)
    model = RevVAMPNet(encoder=encoder, rev_score=rev_score, classifier_module=classifier)

    assert isinstance(model, RevVAMPNet)

    K, pi = extract_learned_transition_matrix(model)

    # Check K is row-stochastic
    assert K.shape == (5, 5)
    assert np.all(K >= 0), "K must be non-negative"
    np.testing.assert_allclose(K.sum(axis=1), np.ones(5), atol=1e-5)

    # Check detailed balance
    for i in range(5):
        for j in range(5):
            np.testing.assert_allclose(
                pi[i] * K[i, j], pi[j] * K[j, i], atol=1e-5,
                err_msg=f"Detailed balance violated at ({i},{j})"
            )

    # Check pi
    assert pi.shape == (5,)
    assert np.all(pi > 0)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-6)


def test_analysis_detects_standard_model(seed):
    """Standard VAMPNet should not be detected as RevVAMPNet."""
    from pygv.scores.vamp_score_v0 import VAMPScore

    encoder = SchNetEncoderNoEmbed(
        node_dim=16, edge_dim=16, hidden_dim=32, output_dim=32, n_interactions=2
    )
    classifier = SoftmaxMLP(
        in_channels=32, hidden_channels=32, out_channels=5, num_layers=2
    )
    vamp_score = VAMPScore(epsilon=1e-6)
    model = VAMPNet(encoder=encoder, vamp_score=vamp_score, classifier_module=classifier)

    assert not isinstance(model, RevVAMPNet)

    with pytest.raises(TypeError):
        extract_learned_transition_matrix(model)


def test_config_reversible_field(seed):
    """Config reversible field should default to False and roundtrip through to_dict/from_dict."""
    config = BaseConfig()
    assert config.reversible is False, "reversible should default to False"

    config.reversible = True
    d = config.to_dict()
    assert d['reversible'] is True

    # Roundtrip
    restored = BaseConfig.from_dict(d)
    assert restored.reversible is True
