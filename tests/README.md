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
| `test_analysis.py` | 43 | Analysis utilities, transition matrices, attention maps |
| `test_ck_its.py` | 45 | Chapman-Kolmogorov test, implied timescales, Koopman operator |
| `test_pipeline_integration.py` | 51 | Pipeline utilities, caching, argument parsing |

**Total: 398 passed, 1 skipped**

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

## Analysis Tests

### State Assignment (`TestStateAssignment`) - 4 tests
- Argmax gives state assignments
- Clear states match highest probability
- Values in valid range
- Unique states count

### Transition Matrix (`TestTransitionMatrix`) - 6 tests
- Correct shape
- Rows sum to 1
- Values in [0, 1]
- No-self has zero diagonal
- Short trajectory returns identity
- Lag frames calculation

### State Edge Attention Maps (`TestStateEdgeAttentionMaps`) - 5 tests
- Attention maps shape
- State populations sum to 1
- Populations non-negative
- Maps non-negative
- Saves files

### Extract Residue Indices (`TestExtractResidueIndices`) - 6 tests
- Returns tuple
- Indices are list
- Names are list
- Same length
- Names contain residue info
- Empty selection raises error

### Analyze VAMPNet Outputs (`TestAnalyzeVAMPNetOutputs`) - 9 tests
- Returns tuple of four
- Probs shape
- Embeddings shape
- Returns numpy by default
- Returns tensors when requested
- Creates output files
- Edge attentions/indices are lists
- Probs are valid distribution

### Probability Properties (`TestProbabilityProperties`) - 4 tests
- Non-negative
- At most one
- Sum to one
- Clear state has high probability

### Edge Cases (`TestEdgeCases`) - 4 tests
- Single frame probabilities
- Two-state system
- Many states system
- Missing attention handled

### Numerical Stability (`TestNumericalStability`) - 3 tests
- Very small probabilities
- Equal probabilities
- Deterministic results

### Output Tests (`TestMetadataOutput`, `TestStateCountsOutput`) - 2 tests
- Metadata file contains info
- State counts file format

## Chapman-Kolmogorov & Implied Timescales Tests

### Koopman Operator (`TestEstimateKoopmanOp`) - 7 tests
- Correct shape
- Lag=0 is identity
- Handles list of trajectories
- Skips short trajectories
- Eigenvalues bounded
- Has unit eigenvalue
- Different lag times differ

### Chapman-Kolmogorov Test (`TestChapmanKolmogorovTest`) - 7 tests
- Returns list of [predicted, estimated]
- Output shapes
- Initial is identity
- Predicted uses matrix power
- Probabilities in valid range
- Two-state system
- Markovian data agreement

### CK Plotting (`TestCKPlotting`) - 3 tests
- Creates file
- Returns figure and axes
- Axes shape matches states

### CK Analysis (`TestRunCKAnalysis`) - 3 tests
- Returns dict
- Creates folder
- Results contain arrays

### Implied Timescales (`TestGetITS`) - 7 tests
- Returns tuple
- Array shape (n_states-1, n_lags)
- Excludes stationary process
- Returns lag times
- Positive values
- Two-state system
- Stride affects frames

### ITS Plotting (`TestITSPlotting`) - 3 tests
- Creates file
- Returns path
- Subset of states

### ITS Analysis (`TestAnalyzeImpliedTimescales`) - 5 tests
- Returns dict
- Creates folder
- Saves data and summary
- Result keys

### Edge Cases (`TestEdgeCases`) - 4 tests
- Single lag time
- Many states
- Short trajectory
- Empty trajectory list

### Numerical Stability (`TestNumericalStability`) - 3 tests
- Near-degenerate eigenvalues
- Deterministic results
- Large trajectory

### Mathematical Properties (`TestMathematicalProperties`) - 3 tests
- Koopman power property
- ITS formula
- Stationary distribution

## Pipeline Integration Tests

### CacheManager (`TestCacheManager`) - 11 tests
- Initialization with config
- Hash generation (determinism, length)
- Hash sensitivity to parameters (traj_dir, stride, n_neighbors, selection)
- Cache file checking

### Pipeline Args (`TestPipelineArgs`) - 19 tests
- Required arguments parsing
- Default values (lag_times, n_states, output_dir, protein_name)
- Multiple values (lag_times, n_states)
- Flags (cache, hurry)
- Optional arguments (preset, model, resume, skip_*, only_analysis)

### Mock Config (`TestMockConfig`) - 4 tests
- to_dict returns dict
- Contains all fields
- YAML creation and validity

### Module Imports (`TestPipelineModuleImports`) - 4 tests
- Caching module importable
- Args module importable
- Training and analysis modules have expected functions

### Edge Cases (`TestCacheEdgeCases`, `TestArgsEdgeCases`) - 6 tests
- Special characters in paths
- Unicode in selection
- Nonexistent cache directory
- Single values as lists

### Summary Format (`TestPipelineSummaryFormat`) - 2 tests
- JSON structure
- Config is dictionary

### Directory Structure (`TestDirectoryStructure`) - 3 tests
- Experiment directory subdirectories
- Training/analysis subdirs per experiment

### Config Serialization (`TestConfigSerialization`) - 2 tests
- YAML roundtrip
- JSON serializable

## Known Behaviors

The gradient flow tests document that `global_mlp` parameters in Meta/MetaAtt/GAT encoders may not receive gradients when using certain embedding types (e.g., "node"). This is expected architectural behavior.

## Bugs Found During Testing

See `tests_todo.md` for documented bugs discovered during test development.

## Adding New Tests

See `TESTING_GUIDE.md` in the project root for unit test best practices.