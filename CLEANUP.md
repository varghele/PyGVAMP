# Repository Cleanup Checklist

## Immediate — generated/ignored files
- [x] **`.gitignore` is nearly empty** — added `build/`, `dist/`, `*.egg-info/`, `__pycache__/`, `*.pyc`
- [x] **`build/`, `dist/`, `pygv.egg-info/`** — build artifacts, should be gitignored and deleted
- [x] **16 `__pycache__/` dirs** with 163 `.pyc` files — added to `.gitignore` (PyCharm hides these; they're untracked local bytecode)
- [x] **`cudatest.py`** — moved to `local_checks/`

## High priority — dead/legacy code
- [x] **`legacy_areas/`** — its own README says it's obsolete (area51 + area52, replaced by the unified pipeline)
- [x] **`pygv/dataset/legacy/`** — old dataset files superseded by `vampnet_dataset.py`
- [x] **`pygv/utils/ck.py:399`** — function marked `# TODO: Delete`
- [x] **`pygv/clustering/graph2vec.py`** — duplicate code at line 265 (`TODO: Decide on one, delete the other`) and dead function at line 291 (`_create_doc2vec_documents_o`)
- [x] **`pygv/utils/metrics.py`** — empty file (0 bytes)

## Medium priority — unfinished stubs & TODOs
- [x] **`pygv/pipe/caching.py:31`** — `cache_dataset()` stub with `TODO: Implement this` — deleted; `VAMPNetDataset` handles its own caching
- [x] **`pygv/pipe/training.py:207`** — `TODO: IMPLEMENT` for ML3 encoder integration — done, `ML3Encoder` fully wired
- [x] **`pygv/vampnet/vampnet.py:415`** — `get_embeddings()` works fine, removed misleading TODO
- [x] **`experimental_area/compare_states.py`** — standalone script, not integrated into pipeline

## Low priority — docs & misc
- [x] **`GIN_ENCODER_UNIT_TESTS.md`** — staged but looks like a working doc, decide if it belongs in the repo
- [x] **`testdata/plots/`** (~9 MB) — generated plot images, could be regenerated
- [x] **`testdata/potential_missing_tests/`** — old test snippets, candidates for integration or deletion
