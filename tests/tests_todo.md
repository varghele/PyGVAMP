# PyGVAMP Test Development Roadmap

This document tracks the unit tests to be implemented for the PyGVAMP pipeline.

---

## Current Status

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_encoders.py` | 26 | ✅ Complete |
| `test_vamp_score.py` | 31 | ✅ Complete |
| `test_vampnet_model.py` | 28 (+1 skipped) | ✅ Complete |
| `test_dataset.py` | 53 | ✅ Complete |
| `test_classifier.py` | 33 | ✅ Complete |
| `test_config.py` | 58 | ✅ Complete |
| `test_training.py` | 30 | ✅ Complete |
| `test_analysis.py` | - | ⬜ Not started |
| `test_pipeline_integration.py` | - | ⬜ Not started |
| `test_ck_its.py` | - | ⬜ Not started |

**Total tests: 259 passed, 1 skipped**

---

## High Priority Tests

### 1. `test_vampnet_model.py` ✅ COMPLETE

**Component:** `pygv/vampnet/vampnet.py` (1184 lines)

**Tests implemented (28 tests + 1 skipped):**
- [x] Full model forward pass (graph → softmax probabilities)
- [x] Gradient flow through entire model (encoder → classifier → output)
- [x] Different encoder types work with classifier (SchNet, Meta)
- [x] Output probabilities sum to 1 for each sample
- [x] Output probabilities are all non-negative
- [x] Training step works (forward + VAMP loss + backward)
- [x] Model handles batched graph inputs
- [x] Model produces consistent outputs with fixed seed
- [x] Attention weights are extractable (for analysis phase)
- [x] Save/load model functionality
- [x] Model configuration

---

### 2. `test_classifier.py` ✅ COMPLETE

**Component:** `pygv/classifier/SoftmaxMLP.py`

**Tests implemented (33 tests):**
- [x] Output shape is [batch_size, n_states]
- [x] Softmax outputs sum to 1 for each sample
- [x] All output values are in [0, 1]
- [x] Handles different n_states values (3, 5, 7, 10, 20)
- [x] Gradient flow to input embeddings
- [x] Different hidden layer configurations work (1-5 layers)
- [x] Dropout behavior (train vs eval mode)
- [x] Batch normalization support
- [x] Weight initialization
- [x] Training step integration
- [x] Edge cases (binary classification, many classes)

---

### 3. `test_dataset.py` ✅ COMPLETE

**Component:** `pygv/dataset/vampnet_dataset.py` (811 lines)

**Tests implemented (53 tests):**
- [x] Graph construction produces valid PyG Data objects
- [x] Node features have correct dimensions (one-hot encoding)
- [x] k-NN edges are created correctly (correct number of neighbors)
- [x] Edge features (Gaussian expansion) have correct dimensions
- [x] Time-lagged pairs have correct temporal offset
- [x] Stride parameter works correctly
- [x] Caching works (hash-based, loads from cache on second call)
- [x] Handles multiple trajectory files
- [x] Graph is asymmetric (k-NN, not mutual k-NN)
- [x] Amino acid encoding (labels and properties)
- [x] Continuous vs non-continuous trajectory modes
- [x] Lag time validation
- [x] Edge cases (single frame, few atoms, empty selection)

**Note:** Tests use mocked MDTraj to avoid external file dependencies.

---

## Medium Priority Tests

### 4. `test_config.py` ✅ COMPLETE

**Component:** `pygv/config/` (base_config.py, model_configs/, presets/)

**Tests implemented (58 tests):**
- [x] BaseConfig initializes with valid defaults (all parameter groups)
- [x] SchNetConfig, MetaConfig, ML3Config have correct encoder-specific defaults
- [x] Preset classes exist and are instantiable
- [x] CONFIG_REGISTRY contains all expected entries
- [x] get_config() works with all presets
- [x] get_config() applies overrides correctly
- [x] Invalid preset/parameter raises ValueError
- [x] YAML/JSON serialization and deserialization
- [x] Config update() and merge() methods
- [x] Config inheritance chain
- [x] Dataclass behavior (equality, repr, fields)

---

### 5. `test_training.py` ✅ COMPLETE

**Component:** `pygv/pipe/training.py` (449 lines)

**Tests implemented (30 tests):**
- [x] Single training step completes without error
- [x] Loss decreases over multiple epochs (on simple data)
- [x] Checkpoint saving works
- [x] Checkpoint loading restores model state
- [x] Early stopping triggers correctly
- [x] Gradient clipping works
- [x] Handles device placement (CPU/CUDA)
- [x] Different encoder types (SchNet, Meta)
- [x] Custom optimizer options
- [x] Output directory setup

**Note:** Tests use synthetic graph datasets to avoid external file dependencies.

---

### 6. `test_analysis.py`

**Component:** `pygv/pipe/analysis.py` (352 lines)

**Why important:** Validates that analysis outputs are correct for publication.

**Tests to implement:**
- [ ] State assignment produces valid labels (0 to n_states-1)
- [ ] Transition matrix rows sum to 1
- [ ] Transition matrix values are in [0, 1]
- [ ] Attention maps have correct shape
- [ ] Representative structure extraction works
- [ ] Analysis handles models with different n_states

---

## Lower Priority Tests (Integration)

### 7. `test_pipeline_integration.py`

**Component:** Full pipeline (`pygv/pipe/master_pipeline.py`)

**Why important:** End-to-end validation that all components work together.

**Tests to implement:**
- [ ] Full pipeline runs on synthetic/small test data
- [ ] All expected output files are generated
- [ ] Pipeline summary JSON is valid
- [ ] Pipeline handles interruption gracefully
- [ ] Resume functionality works

**Note:** These are slower integration tests, may want to mark with `@pytest.mark.slow`.

---

### 8. `test_ck_its.py`

**Components:** `pygv/utils/ck.py`, `pygv/utils/its.py`

**Why important:** Validates Markov model analysis tools.

**Tests to implement:**
- [ ] Implied timescales calculation is correct
- [ ] ITS converges for Markovian data
- [ ] Chapman-Kolmogorov test produces valid comparison
- [ ] Handles different lag times correctly
- [ ] Eigenvalue extraction works

---

## Test Data Requirements

Some tests require test data:

| Test | Data Needed | Status |
|------|-------------|--------|
| `test_dataset.py` | Mock MDTraj objects | ✅ Done |
| `test_training.py` | Synthetic graphs | ✅ Done |
| `test_analysis.py` | Model checkpoint | ⬜ Not started |
| `test_pipeline_integration.py` | Trajectory files | ⬜ Not started |

**Options:**
1. Create synthetic test data in fixtures
2. Use small real trajectory files in `testdata/`
3. Mock MDTraj/file loading functions (used for test_dataset.py)

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_vampnet_model.py -v

# Run with coverage
pytest tests/ --cov=pygv --cov-report=term-missing

# Run only fast tests (exclude slow integration tests)
pytest tests/ -v -m "not slow"

# Run tests matching pattern
pytest tests/ -v -k "gradient"
```

---

## Progress Log

| Date | Changes |
|------|---------|
| 2026-01-16 | Created `test_encoders.py` (26 tests) |
| 2026-01-16 | Created `test_vamp_score.py` (31 tests) |
| 2026-01-16 | Created `test_vampnet_model.py` (28 tests + 1 skipped) |
| 2026-02-05 | Created `test_dataset.py` (53 tests) - VAMPNetDataset with mocked MDTraj |
| 2026-02-05 | Created `test_classifier.py` (33 tests) - SoftmaxMLP classifier |
| 2026-02-05 | Created `test_config.py` (58 tests) - Configuration system |
| 2026-02-05 | Created `test_training.py` (30 tests) - Training pipeline with synthetic data |
| | |

## Bugs Found During Testing

The following bugs were discovered while writing tests:

### 1. VAMPNet.__init__ classifier_module bug (line 134)
**File:** `pygv/vampnet/vampnet.py:134`
**Issue:** `self.add_module('classifier_module', classifier_module)` uses the parameter instead of `self.classifier_module`
**Impact:** Auto-created classifiers (via `n_classes` parameter) are overwritten with None
**Workaround:** Pass `classifier_module` explicitly instead of using `n_classes`
**Fix:** Change line 134 to `self.add_module('classifier_module', self.classifier_module)`

### 2. VAMPNet.get_config() bug (line 741)
**File:** `pygv/vampnet/vampnet.py:741`
**Issue:** Assumes `self.classifier_module.final_layer.out_features` exists
**Impact:** Raises AttributeError because `final_layer` is a Sequential, not Linear
**Workaround:** Don't call `get_config()` with classifier present
**Fix:** Iterate through Sequential to find Linear layer's out_features

---

*Last updated: 2026-02-05*
