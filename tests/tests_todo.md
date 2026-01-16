# PyGVAMP Test Development Roadmap

This document tracks the unit tests to be implemented for the PyGVAMP pipeline.

---

## Current Status

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_encoders.py` | 26 | ✅ Complete |
| `test_vamp_score.py` | 31 | ✅ Complete |
| `test_vampnet_model.py` | 28 (+1 skipped) | ✅ Complete |
| `test_classifier.py` | - | ⬜ Not started |
| `test_dataset.py` | - | ⬜ Not started |
| `test_config.py` | - | ⬜ Not started |
| `test_training.py` | - | ⬜ Not started |
| `test_analysis.py` | - | ⬜ Not started |
| `test_pipeline_integration.py` | - | ⬜ Not started |
| `test_ck_its.py` | - | ⬜ Not started |

**Total tests: 85 passed, 1 skipped**

---

## High Priority Tests

### 1. `test_vampnet_model.py`

**Component:** `pygv/vampnet/vampnet.py` (1184 lines)

**Why important:** Tests the full model that ties encoders and classifiers together. This is the core component used in training.

**Tests to implement:**
- [ ] Full model forward pass (graph → softmax probabilities)
- [ ] Gradient flow through entire model (encoder → classifier → output)
- [ ] Different encoder types work with classifier (SchNet, ML3, Meta)
- [ ] Output probabilities sum to 1 for each sample
- [ ] Output probabilities are all non-negative
- [ ] Training step works (forward + VAMP loss + backward)
- [ ] Model handles batched graph inputs
- [ ] Model produces consistent outputs with fixed seed
- [ ] Attention weights are extractable (for analysis phase)

---

### 2. `test_classifier.py`

**Component:** `pygv/classifier/SoftmaxMLP.py`

**Why important:** Ensures classifier outputs valid probability distributions for state assignment.

**Tests to implement:**
- [ ] Output shape is [batch_size, n_states]
- [ ] Softmax outputs sum to 1 for each sample
- [ ] All output values are in [0, 1]
- [ ] Handles different n_states values (3, 5, 7, 10)
- [ ] Gradient flow to input embeddings
- [ ] Different hidden layer configurations work

---

### 3. `test_dataset.py`

**Component:** `pygv/dataset/vampnet_dataset.py` (692 lines)

**Why important:** Validates the MD trajectory → PyG graph conversion pipeline. Errors here propagate to all downstream components.

**Tests to implement:**
- [ ] Graph construction produces valid PyG Data objects
- [ ] Node features have correct dimensions (one-hot encoding)
- [ ] k-NN edges are created correctly (correct number of neighbors)
- [ ] Edge features (Gaussian expansion) have correct dimensions
- [ ] Time-lagged pairs have correct temporal offset
- [ ] Stride parameter works correctly
- [ ] Different atom selections work ("name CA", "protein", etc.)
- [ ] Caching works (hash-based, loads from cache on second call)
- [ ] Handles multiple trajectory files
- [ ] Graph is asymmetric (k-NN, not mutual k-NN)

**Note:** These tests may require mock trajectory data or small test PDB/XTC files.

---

## Medium Priority Tests

### 4. `test_config.py`

**Component:** `pygv/config/base_config.py` (169 lines)

**Why important:** Configuration errors cause hard-to-debug failures downstream.

**Tests to implement:**
- [ ] BaseConfig initializes with valid defaults
- [ ] SchNetConfig has correct encoder-specific defaults
- [ ] Invalid parameter combinations raise errors
- [ ] Preset loading works (small preset)
- [ ] CLI argument overrides work
- [ ] Config can be serialized/deserialized (for reproducibility)

---

### 5. `test_training.py`

**Component:** `pygv/pipe/training.py` (449 lines)

**Why important:** Ensures training loop behaves correctly.

**Tests to implement:**
- [ ] Single training step completes without error
- [ ] Loss decreases over multiple epochs (on simple data)
- [ ] Checkpoint saving works
- [ ] Checkpoint loading restores model state
- [ ] Early stopping triggers correctly
- [ ] Learning rate scheduling works
- [ ] Handles device placement (CPU/CUDA)
- [ ] Grid search over lag_times and n_states works

**Note:** May need to mock or use small synthetic datasets.

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

| Test | Data Needed |
|------|-------------|
| `test_dataset.py` | Small PDB + XTC files (or mock MDTraj objects) |
| `test_training.py` | Pre-built small dataset or synthetic graphs |
| `test_analysis.py` | Trained model checkpoint |
| `test_pipeline_integration.py` | Small trajectory files |

**Options:**
1. Create synthetic test data in fixtures
2. Use small real trajectory files in `testdata/`
3. Mock MDTraj/file loading functions

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

*Last updated: 2026-01-16*
