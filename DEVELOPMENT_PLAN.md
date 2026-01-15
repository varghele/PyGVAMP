# PyGVAMP Development Plan

This document tracks the current development priorities and progress for cleaning up and completing the PyGVAMP codebase.

---

## Overview

**Goals:**
1. Consolidate duplicate code
2. Complete missing features
3. Reduce technical debt
4. Improve maintainability

---

## Phase 1: Critical Fixes (Blocking Issues)

These issues will cause import errors when loading the package.

### 1.1 Fix Broken Config Imports

**Status:** Pending

**Problem:** `pygv/config/__init__.py` imports classes that don't exist:
- Lines 4: Imports `MetaConfig`, `ML3Config` (commented out in `base_config.py`)
- Lines 6-7: Imports from `presets/medium.py` and `presets/large.py` (files don't exist)

**Solution:**
- [ ] Uncomment `MetaConfig` and `ML3Config` in `base_config.py`
- [ ] Create `presets/medium.py` with `MediumSchNetConfig` and `MediumMetaConfig`
- [ ] Create `presets/large.py` with `LargeSchNetConfig` and `LargeMetaConfig`

### 1.2 Fix Hardcoded CUDA

**Status:** Pending

**Problem:** `training.py:268` uses `model.to('cuda')` instead of device variable

**Solution:**
- [ ] Change `model.to('cuda')` to `model.to(device)` after device determination

---

## Phase 2: Code Consolidation

### 2.1 Merge Duplicate Dataset Files

**Status:** Pending

**Files:**
- `pygv/dataset/vampnet_dataset.py` (692 lines, base)
- `pygv/dataset/vampnet_dataset_with_AA.py` (758 lines, amino acid variant)

**Differences:**
- AA version adds `use_amino_acid_encoding` parameter to `_create_graph_from_frame()`
- AA version adds `get_AA_frames()` method
- AA version loads topology for residue name lookup

**Solution:**
- [ ] Add `use_amino_acid_encoding` flag to main dataset
- [ ] Add `get_AA_frames()` method to main dataset
- [ ] Load topology lazily when AA encoding is used
- [ ] Delete `vampnet_dataset_with_AA.py`

### 2.2 Remove Deleted Modules

**Status:** Pending

**Files staged for deletion (already in git):**
- `psevo/` directory (entire module)
- `viz/` directory (empty)

**Solution:**
- [ ] Commit the deletion (already staged)

---

## Phase 3: Feature Completion

### 3.1 Integrate ML3 Encoder

**Status:** Pending

**Problem:** `training.py:194` returns `encoder = None` for ML3 type

**Working code exists:** `pygv/encoder/ml3.py` has `GNNML3` class

**Solution:**
- [ ] Import `GNNML3` in training.py
- [ ] Instantiate ML3 encoder with proper config parameters
- [ ] Add ML3-specific arguments to `args_train.py`

### 3.2 Complete Config Presets

**Status:** Pending

**Needed files:**
- `pygv/config/presets/medium.py` - standard training configs
- `pygv/config/presets/large.py` - production training configs

**Solution:**
- [ ] Create medium presets with balanced hyperparameters
- [ ] Create large presets with higher capacity settings

---

## Phase 4: Code Quality

### 4.1 Remove Unused Imports

**Status:** Pending

**Known issues:**
- `training.py:10` - `from pymol.querying import distance` unused

**Solution:**
- [ ] Remove unused import

### 4.2 Fix NaN Handling

**Status:** Deferred

**Problem:** `vampnet.py` replaces NaN outputs with zeros (masking the problem)

**Solution:** Investigate root cause of NaN generation

---

## Task Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Fix config imports (blocking) | Low | Critical |
| 2 | Remove deleted modules | Trivial | Low |
| 3 | Fix hardcoded CUDA | Trivial | Medium |
| 4 | Remove unused imports | Trivial | Low |
| 5 | Create preset files | Medium | Medium |
| 6 | Complete MetaConfig/ML3Config | Medium | Medium |
| 7 | Merge dataset files | Medium | Medium |
| 8 | Integrate ML3 encoder | Medium | High |

---

## Progress Tracker

### Completed
- [x] Read and analyze codebase
- [x] Created CODEBASE_SUMMARY.md
- [x] Created DEVELOPMENT_PLAN.md (this file)

### In Progress
- [ ] Phase 1: Critical Fixes

### Pending
- [ ] Phase 2: Code Consolidation
- [ ] Phase 3: Feature Completion
- [ ] Phase 4: Code Quality

---

*Last updated: 2026-01-15*
