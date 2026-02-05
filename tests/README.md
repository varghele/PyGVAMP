# PyGVAMP Test Suite

This directory contains unit tests for the PyGVAMP pipeline.

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run encoder tests only
pytest tests/test_encoders.py -v

# Run VAMP score tests only
pytest tests/test_vamp_score.py -v

# Run specific test class
pytest tests/test_encoders.py::TestSchNetEncoder -v

# Run with short traceback
pytest tests/ -v --tb=short
```

## Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_encoders.py` | 26 | Encoder forward pass, gradient flow, batching |
| `test_vamp_score.py` | 31 | VAMP score computation, gradients, math properties |
| `test_vampnet_model.py` | 28 (+1 skip) | Full model integration, training, save/load |
| `test_dataset.py` | 53 | VAMPNetDataset graph construction, caching, time-lagged pairs |
| `test_classifier.py` | 33 | SoftmaxMLP probability outputs, gradient flow, configurations |
| `test_config.py` | 58 | Configuration defaults, presets, serialization, registry |
| `test_training.py` | 30 | Training pipeline, checkpointing, early stopping, device placement |

**Total: 259 passed, 1 skipped**

## Test Coverage by Encoder

### SchNet (`TestSchNetEncoder`) - 5 tests
- Forward pass (single graph, batched)
- Gradient flow through parameters
- Gradient flow to inputs
- Attention disabled mode

### ML3 (`TestML3Encoder`) - 4 tests
- Forward pass (single graph, batched)
- Gradient flow through parameters
- Gradient flow to inputs

### Meta (`TestMetaEncoder`) - 4 tests
- Forward pass (single graph, batched)
- Gradient flow through parameters
- All embedding types (node, global, combined)

### MetaAtt (`TestMetaAttEncoder`) - 4 tests
- Forward pass (single graph, batched)
- Gradient flow through parameters
- Attention weights returned

### GAT (`TestGATEncoder`) - 4 tests
- Forward pass (single graph, batched)
- Gradient flow through parameters
- All embedding types

### Integration (`TestEncoderIntegration`) - 2 tests
- All encoders process same input
- Deterministic outputs with fixed seed

### Edge Cases (`TestEdgeCases`) - 3 tests
- Single node graph handling
- Missing batch tensor
- Large batch sizes (16 graphs)

## VAMP Score Tests

### Forward Pass (`TestVAMPScoreForward`) - 4 tests
- VAMP1, VAMP2, VAMPE methods
- Loss is negative of score

### Mathematical Properties (`TestVAMPScoreMathProperties`) - 4 tests
- Score >= 1 (includes constant singular function)
- Correlated data scores higher than uncorrelated
- Perfect correlation produces highest score

### Gradient Flow (`TestVAMPScoreGradients`) - 4 tests
- Gradients flow through VAMP1, VAMP2, VAMPE
- Loss gradients point in correct direction

### Covariance (`TestCovarianceComputation`) - 4 tests
- Correct shapes
- Symmetry of auto-covariance matrices
- Positive definiteness
- Softmax output handling

### Regularization Modes (`TestRegularizationModes`) - 3 tests
- Truncation mode
- Regularization mode
- Modes produce similar results

### Edge Cases (`TestEdgeCases`) - 7 tests
- Small/large batch sizes
- Many states
- Input validation errors

### Numerical Stability (`TestNumericalStability`) - 3 tests
- Near-zero variance
- Large values
- Deterministic results

### Training Integration (`TestTrainingIntegration`) - 2 tests
- Optimization improves score
- Batch consistency

## VAMPNet Model Tests

### Forward Pass (`TestVAMPNetForward`) - 4 tests
- Single graph and batched forward pass
- Return features option
- Forward without classifier

### Probability Outputs (`TestProbabilityOutputs`) - 4 tests
- Probabilities sum to 1
- Non-negative and at most 1
- Valid distribution

### Gradient Flow (`TestGradientFlow`) - 3 tests
- Gradients flow to encoder and classifier
- Gradients work with VAMP loss

### Training Step (`TestTrainingStep`) - 3 tests
- Single and multiple training steps
- Training improves VAMP score

### Attention (`TestAttentionExtraction`) - 2 tests
- Attention extraction returns tuple
- Features have correct shape

### Configuration (`TestModelConfiguration`) - 5 tests
- Model has encoder, classifier, vamp_score
- Different n_states work

### Save/Load (`TestSaveLoad`) - 3 tests
- Save and load complete model
- Save with metadata

### Different Encoders (`TestDifferentEncoders`) - 2 tests
- Works with Meta encoder
- Output dimension inference

### Edge Cases (`TestEdgeCases`) - 2 tests (+1 skipped)
- Eval vs train mode
- Deterministic eval mode

## VAMPNetDataset Tests

### Graph Construction (`TestGraphConstruction`) - 8 tests
- Correct number of nodes
- k-NN edge count
- No self-edges
- Edge attribute shape and values
- Valid PyG Data object
- Deterministic construction

### Node Features (`TestNodeFeatures`) - 6 tests
- One-hot encoding shape, diagonal, off-diagonal
- Amino acid labels shape and range
- Amino acid properties shape

### Gaussian Expansion (`TestGaussianExpansion`) - 4 tests
- Output shape
- Non-negative and bounded values
- Deterministic results

### Time-Lagged Pairs (`TestTimeLaggedPairs`) - 8 tests
- Continuous mode pair count and offset
- Non-continuous mode boundary handling
- Short trajectory handling
- Getitem returns tuple of different graphs

### Lag Time Validation (`TestLagTimeValidation`) - 4 tests
- Valid/invalid lag time handling
- Lag frames calculation
- Stride effects

### Caching (`TestCaching`) - 6 tests
- Cache filename format
- Save on first run
- Load on second run
- Config storage and mismatch warnings
- use_cache=False behavior

### Dataset Interface (`TestDatasetInterface`) - 5 tests
- len() returns int
- Valid and negative index access
- get_graph() and get_frames_dataset()

### Edge Cases (`TestEdgeCases`) - 4 tests
- Single frame trajectory
- Few atoms (< n_neighbors)
- Empty selection raises error
- Missing file handling

### Frames Dataset (`TestFramesDataset`) - 4 tests
- Length and single graph return
- Pairs mode
- Encoding override

### Precompute & Distance (`TestPrecomputeGraphs`, `TestDistanceRange`) - 4 tests
- Graph precomputation
- Distance range determination

## Training Pipeline Tests

### Setup (`TestSetupOutputDirectory`) - 5 tests
- Creates directories
- Returns correct paths
- Run name generation

### Config (`TestSaveConfig`) - 3 tests
- Saves config file
- Contains parameters
- Values correct

### Training Step (`TestTrainingStep`) - 4 tests
- Forward pass
- Loss computation
- Backward pass
- Optimizer step

### Training Loop (`TestTrainingLoop`) - 4 tests
- Fit completes
- Validation
- Returns history
- Training improves score

### Checkpointing (`TestCheckpointing`) - 3 tests
- Saves checkpoints
- Saves final model
- Checkpoint loadable

### Early Stopping (`TestEarlyStopping`) - 1 test
- Early stopping triggers

### Gradient Clipping (`TestGradientClipping`) - 1 test
- Gradient clipping applied

### Different Encoders (`TestDifferentEncoders`) - 2 tests
- Training with SchNet
- Training with Meta

### Device Placement (`TestDevicePlacement`) - 2 tests
- Training on CPU
- Training on GPU (skipped if unavailable)

### Edge Cases (`TestEdgeCases`) - 3 tests
- Single epoch
- Small batch size
- No save directory

### Optimizer Options (`TestOptimizerOptions`) - 2 tests
- Custom optimizer
- Different learning rates

## Known Behaviors

The gradient flow tests document that `global_mlp` parameters in Meta/MetaAtt/GAT encoders may not receive gradients when using certain embedding types (e.g., "node"). This is expected architectural behavior.

## Bugs Found During Testing

See `tests_todo.md` for documented bugs discovered during test development.

## Adding New Tests

See `TESTING_GUIDE.md` in the project root for unit test best practices.