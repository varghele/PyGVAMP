# Potential Missing Tests

These files contain test snippets that may be useful for future unit test development.
They test functionality that isn't fully covered by the main test suite in `tests/`.

## Files

| File | Tests | Notes |
|------|-------|-------|
| `validate_vamp_score.py` | VAMP score validation against deeptime library | Useful for ensuring our VAMP implementation matches the reference |
| `cfconv_tests.py` | Encoder internal operations (softmax, scatter aggregation) | Tests PyG vs manual implementations |
| `compare_gauss_dist.py` | GaussianDistance edge feature expansion | Tests internals of Gaussian basis functions |
| `test_dcd_loader.py` | Actual MDTraj file loading (not mocked) | Tests real trajectory file parsing |

## Why Kept

1. **validate_vamp_score.py** - Compares our VAMPScore against deeptime's implementation.
   Could be converted to a proper integration test to ensure numerical correctness.

2. **cfconv_tests.py** - Tests that our PyG-based softmax and scatter_add produce
   identical results to manual loops. Useful if encoder internals change.

3. **compare_gauss_dist.py** - Tests GaussianDistance class produces correct
   Gaussian basis function expansions. Could verify edge feature computation.

4. **test_dcd_loader.py** - Tests actual DCD/XTC file loading with MDTraj.
   Current test_dataset.py uses mocked MDTraj; this tests real file I/O.

## Usage

These are not pytest files. To run them:

```bash
python testdata/potential_missing_tests/validate_vamp_score.py
python testdata/potential_missing_tests/cfconv_tests.py
# etc.
```

## Converting to Unit Tests

To add these to the test suite:
1. Extract the core assertions
2. Add proper pytest fixtures
3. Mock external dependencies where appropriate
4. Add to appropriate test file in `tests/`
