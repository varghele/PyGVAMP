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

**Status:** ✅ Complete

**Problem:** `pygv/config/__init__.py` imports classes that don't exist:
- Lines 4: Imports `MetaConfig`, `ML3Config` (commented out in `base_config.py`)
- Lines 6-7: Imports from `presets/medium.py` and `presets/large.py` (files don't exist)

**Solution:**
- [x] Uncomment `MetaConfig` and `ML3Config` in `base_config.py`
- [x] Create `presets/medium.py` with `MediumSchNetConfig` and `MediumMetaConfig`
- [x] Create `presets/large.py` with `LargeSchNetConfig` and `LargeMetaConfig`

### 1.2 Fix Hardcoded CUDA

**Status:** ✅ Complete

**Problem:** `training.py:268` uses `model.to('cuda')` instead of device variable

**Solution:**
- [x] Change `model.to('cuda')` to `model.to(device)` after device determination

---

## Phase 2: Code Consolidation

### 2.1 Merge Duplicate Dataset Files

**Status:** ✅ Complete

**Original files:**
- `pygv/dataset/vampnet_dataset.py` (base) → moved to `legacy/`
- `pygv/dataset/vampnet_dataset_with_AA.py` (amino acid variant) → moved to `legacy/`
- `pygv/dataset/vampnet_dataset_new.py` (unified version) → renamed to `vampnet_dataset.py`

**Solution:**
- [x] Review `vampnet_dataset_new.py` and determine if it should be the canonical version
- [x] Add `use_amino_acid_encoding` flag to main dataset
- [x] Add `get_AA_frames()` method to main dataset
- [x] Load topology lazily when AA encoding is used
- [x] Move old dataset files to `legacy/` folder
- [x] Rename unified dataset to `vampnet_dataset.py`
- [x] Update imports in area51/area52 test files

### 2.2 Remove Deleted Modules

**Status:** ✅ Complete

**Files staged for deletion (already in git):**
- `psevo/` directory (entire module)
- `viz/` directory (empty)

**Solution:**
- [x] Directories have been deleted

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

**Status:** ✅ Complete

**Needed files:**
- `pygv/config/presets/medium.py` - standard training configs
- `pygv/config/presets/large.py` - production training configs

**Solution:**
- [x] Create medium presets with balanced hyperparameters
- [x] Create large presets with higher capacity settings

### 3.3 Non-Continuous Trajectory Support

**Status:** ✅ Complete

**Problem:** All trajectory files were concatenated as one continuous trajectory, causing time-lagged pairs to incorrectly span across trajectory boundaries (e.g., from the end of one simulation to the start of another).

**Solution implemented in `vampnet_dataset_new.py`:**
- [x] Added `continuous` parameter to `__init__()` (default `True` for backward compatibility)
- [x] Track trajectory boundaries in `_process_trajectories()` via `self.trajectory_boundaries`
- [x] Filter cross-boundary pairs in `_create_time_lagged_pairs()` when `continuous=False`
- [x] Updated cache filename to include `cont`/`noncont` suffix
- [x] Updated cache save/load to include `trajectory_boundaries` and `continuous` config
- [x] Added `continuous: bool = True` to `BaseConfig` in `base_config.py`

**Usage:**
```python
# Independent simulations - pairs won't cross trajectory boundaries
dataset = VAMPNetDataset(
    trajectory_files=[...],
    topology_file="protein.pdb",
    lag_time=20.0,
    continuous=False
)
```

---

## Phase 4: Code Quality

### 4.1 Remove Unused Imports

**Status:** ✅ Complete

**Known issues:**
- `training.py:10` - `from pymol.querying import distance` unused

**Solution:**
- [x] Removed unused import

### 4.2 Fix NaN Handling

**Status:** Deferred

**Problem:** `vampnet.py` replaces NaN outputs with zeros (masking the problem)

**Solution:** Investigate root cause of NaN generation

---

## Task Priority

| Priority | Task | Effort | Impact | Status |
|----------|------|--------|--------|--------|
| ~~1~~ | ~~Fix config imports (blocking)~~ | ~~Low~~ | ~~Critical~~ | ✅ Done |
| ~~2~~ | ~~Remove deleted modules~~ | ~~Trivial~~ | ~~Low~~ | ✅ Done |
| ~~3~~ | ~~Fix hardcoded CUDA~~ | ~~Trivial~~ | ~~Medium~~ | ✅ Done |
| ~~4~~ | ~~Remove unused imports~~ | ~~Trivial~~ | ~~Low~~ | ✅ Done |
| ~~5~~ | ~~Create preset files~~ | ~~Medium~~ | ~~Medium~~ | ✅ Done |
| ~~6~~ | ~~Complete MetaConfig/ML3Config~~ | ~~Medium~~ | ~~Medium~~ | ✅ Done |
| ~~7~~ | ~~Non-continuous trajectory support~~ | ~~Medium~~ | ~~High~~ | ✅ Done |
| ~~8~~ | ~~Merge dataset files~~ | ~~Medium~~ | ~~Medium~~ | ✅ Done |
| 9 | Integrate ML3 encoder | Medium | High | Pending |

---

## Progress Tracker

### Completed
- [x] Read and analyze codebase
- [x] Created CODEBASE_SUMMARY.md
- [x] Created DEVELOPMENT_PLAN.md (this file)
- [x] Phase 1: Critical Fixes (config imports, CUDA hardcoding)
- [x] Phase 2.1: Merge dataset files (old files → `legacy/`, unified → `vampnet_dataset.py`)
- [x] Phase 2.2: Remove deleted modules (psevo/, viz/)
- [x] Phase 3.2: Complete config presets (medium.py, large.py)
- [x] Phase 3.3: Non-continuous trajectory support (vampnet_dataset.py, base_config.py)
- [x] Phase 4.1: Remove unused imports (pymol)

### In Progress
- [ ] Phase 3.1: Integrate ML3 encoder

### Pending
- [ ] Phase 4.2: Fix NaN handling (deferred)

---

*Last updated: 2026-02-04*
