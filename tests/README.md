# PyGVAMP Test Suite

## Quick Start

```bash
# Run all unit tests
pytest tests/ -v

# Run integration tests (requires real MD data)
pytest tests/ -v -m integration

# Run everything
pytest tests/ -v --tb=short

# Run specific file or class
pytest tests/test_encoders.py -v
pytest tests/test_encoders.py::TestSchNetEncoder -v

# Useful options
pytest tests/ -v -x              # Stop on first failure
pytest tests/ -v --lf            # Re-run last failed
pytest tests/ -v -k "gradient"   # Pattern matching
pytest tests/ -v -s              # Show print output
```

## Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_encoders.py` | 26 | SchNet, ML3, Meta, MetaAtt, GAT: forward pass, gradients, batching, edge cases |
| `test_gin_encoder.py` | 18 | GIN encoder: sanity, permutation invariance, WL expressiveness, attention, learnability |
| `test_ml3_encoder.py` | 30 | ML3 encoder: sanity, gradients, WL expressiveness, edge modes, attention, VAMP integration |
| `test_ml3_equivalence.py` | — | ML3 equivalence tests against original repo (2 xfail) |
| `test_vamp_score.py` | 31 | VAMP score computation, gradients, math properties, regularization modes |
| `test_vamp_score_numpy.py` | 25 | NumPy reference validation of covariances, Koopman, VAMP1/2/E |
| `test_vampnet_model.py` | 28+1skip | Full model: forward, probabilities, training, save/load, attention, configs |
| `test_dataset.py` | 53 | VAMPNetDataset: graph construction, node/edge features, time-lagged pairs, caching |
| `test_classifier.py` | 33 | SoftmaxMLP: probability outputs, gradient flow, dropout, batch norm |
| `test_config.py` | 58 | Configuration: defaults, presets, registry, serialization, inheritance |
| `test_training.py` | 30 | Training: steps, loops, checkpointing, early stopping, device placement |
| `test_analysis.py` | 43 | Analysis: state assignment, transition matrices, attention maps, numerical stability |
| `test_ck_its.py` | 45 | Chapman-Kolmogorov tests, implied timescales, Koopman operator |
| `test_state_merging.py` | — | State merging utilities |
| `test_state_diagnostics.py` | — | State diagnostics (eigenvalue gap, population, JSD) |
| `test_interactive_report.py` | — | Interactive HTML report generation |
| `test_pipeline_integration.py` | 19+7 | Pipeline unit tests + 7 integration tests (SchNet/GIN/ML3, .xtc/.dcd) |

**Total: 504+ passed, 1 skipped, 2 xfailed** (unit tests only; Meta encoder tests are expected failures)

## Integration Tests

Integration tests use real MD trajectory data and are marked with `@pytest.mark.integration`:

```bash
pytest tests/ -v -m integration     # Run only integration tests
pytest tests/ -v -m "not integration"  # Skip integration tests
```

They test the full pipeline (preparation → training → analysis) with:
- **SchNet + .xtc** (AB42 protein)
- **GIN + .dcd** (NTL9 protein)
- **ML3 + .xtc** (AB42 protein)
- Preparation phase, lag time validation, resume

## Known Bugs Found During Testing

1. **`vampnet.py:134` — classifier_module overwrite**: `self.add_module('classifier_module', classifier_module)` overwrites auto-created classifiers with the (None) parameter. Workaround: pass `classifier_module` explicitly.

2. **`vampnet.py:741` — get_config() AttributeError**: Assumes `self.classifier_module.final_layer.out_features` exists, but `final_layer` is a Sequential. Workaround: don't call `get_config()` with classifier.

## Writing New Tests

See `TESTING_GUIDE.md` in this directory for best practices, templates, and conventions.
