# Legacy Areas

This directory contains old experimental code that is no longer part of the main PyGVAMP pipeline.
These files are kept for reference but are not maintained.

## Status: DEPRECATED

**Do not use code from this directory in production.**

---

## Contents

### area51_testing_grounds/

Old Graph2Vec and Weisfeiler-Lehman kernel experiments. These were early explorations
for graph-based molecular representation learning before settling on the current
GNN-based approach.

| File | Description | Status |
|------|-------------|--------|
| `compare_wl_g2v_Pyg.py` | Weisfeiler-Lehman machine comparison with PyG | Obsolete |
| `g2vec_v4.py` | Graph2Vec implementation (Gensim-based) | Obsolete |
| `last_test.py` | Classification tests with Graph2Vec | Obsolete |
| `test_g2vec_v2.py` | Graph2Vec with BIRCH clustering | Obsolete |
| `test_g2vec_v3.py` | Graph2Vec with MiniBatchKMeans | Obsolete |
| `test_g2vec_with_real_data.py` | Graph2Vec on trajectory data | Obsolete |
| `test_viewer.html` | HTML visualization for testing | Obsolete |

**Why deprecated:** Graph2Vec approach was replaced by end-to-end trainable GNNs
(SchNet, Meta, GAT) which provide better representation learning for VAMP objectives.

---

### area52/

Old pipeline testing scripts from early development. These were used to test
individual pipeline components before the unified pipeline was created.

| File | Description | Status |
|------|-------------|--------|
| `anly.py` | Analysis testing script | Replaced by `pygv/pipe/analysis.py` |
| `clust.py` | Clustering example | Obsolete |
| `create_dataset.py` | Dataset creation utilities | Replaced by `pygv/dataset/` |
| `generate_protein_tokens.py` | Protein tokenization | Obsolete |
| `multi_train.py` | Multi-experiment training | Replaced by `pygv/pipe/master_pipeline.py` |
| `prep.py` | Preparation testing | Replaced by `pygv/pipe/preparation.py` |
| `schnetprofile.py` | SchNet profiling script | Obsolete |
| `train.py` | Training testing script | Replaced by `pygv/pipe/training.py` |
| `experiments.csv` | Experiment tracking | Obsolete |
| `experiments_done.csv` | Completed experiments | Obsolete |

**Why deprecated:** All functionality has been moved to the main `pygv/` package
with proper structure, configuration, and testing.

---

## Cleanup Plan

These files can be removed entirely once you've verified no useful code snippets remain.

1. **Can remove immediately:**
   - All `__pycache__` directories (done)
   - `testdata/` directories with cached data (done)
   - `area53/` old prep outputs (done)

2. **Review before removing:**
   - `g2vec_v4.py` - may have useful Gensim patterns
   - `compare_wl_g2v_Pyg.py` - may have useful WL kernel code

3. **Safe to remove:**
   - All other files - functionality exists in main package

---

## Migration Guide

| Old Location | New Location |
|--------------|--------------|
| `area52/train.py` | `pygv/pipe/training.py` |
| `area52/prep.py` | `pygv/pipe/preparation.py` |
| `area52/anly.py` | `pygv/pipe/analysis.py` |
| `area52/create_dataset.py` | `pygv/dataset/vampnet_dataset.py` |
| `area51_testing_grounds/g2vec_*` | Not migrated (different approach) |

---

*Last updated: 2026-02-05*
