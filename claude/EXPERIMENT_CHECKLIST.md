# EXPERIMENT_CHECKLIST.md — PyGVAMP Production Experiments

This is your checklist for running the experiments that go into the paper. Treat it as a living document — check items off, add notes, update timing estimates as you learn the actual numbers.

The experiments fall into three categories with different scientific goals and different protocols. Don't conflate them.

---

## Category 1: Reproduction Sweeps

**Scientific goal:** Demonstrate that PyGVAMP correctly reproduces the published results of GraphVAMPNet and RevGraphVAMP. This validates the framework before any improvement claims are made.

**Protocol:**
- 10 seeds per (system, lag time) combination
- SchNet encoder only
- Fixed k matching the baseline paper's choice
- Fixed lag time matching the baseline paper's choice
- No auto-discovery (turn off the diagnostic-driven retraining)
- No auto-stride (use whatever stride the baseline paper used, or stride matching the preprocessed cache)
- Standard VAMP-2 score for GraphVAMPNet baselines, reversible likelihood loss for RevGraphVAMP baselines

### GraphVAMPNet reproduction

- [ ] **Trp-cage** (Lindorff-Larsen, 208 µs)
  - k = ?? (look up from Ghorbani 2022 paper or supplementary)
  - lag time = ?? ns (look up — likely 20 ns based on what I recall)
  - 10 seeds
  - Compare against: published VAMP-2 score, ITS plot, CK test
  - Estimated wall time: ~2-3 hours total at 10-20 min/seed

- [ ] **Villin** (Lindorff-Larsen, 125 µs)
  - k = ??, lag time = 20 ns (per Fig 4 in Ghorbani 2022)
  - 10 seeds
  - Compare against: published VAMP-2 score, ITS plot, CK test
  - Estimated wall time: ~2-3 hours total

- [ ] **NTL9** (Lindorff-Larsen, 62 µs)
  - k = ??, lag time = ??
  - 10 seeds
  - Compare against: published VAMP-2 score, ITS plot, CK test
  - Estimated wall time: ~2-3 hours total

### RevGraphVAMP reproduction

- [ ] **Alanine dipeptide** (mdshare, 3 trajectories)
  - k = 6 (per RevGraphVAMP paper, matching original VAMPNet)
  - lag time = ?? (look up from Huang 2024)
  - 10 seeds
  - Reversible model
  - Compare against: published VAMP-2 and VAMP-E scores from Table 2
  - Estimated wall time: ~1-2 hours total (small system)

- [ ] **Aβ42-red** (Huang 2024 dataset, 1 ns spacing)
  - k = 4 (per RevGraphVAMP defaults)
  - lag time = ?? (look up — likely matches their `lag_time` arg)
  - 10 seeds
  - Reversible model
  - Compare against: published VAMP scores
  - Estimated wall time: ~3-4 hours total

- [ ] **Aβ42-ox** (Huang 2024 dataset, 1 ns spacing)
  - Same protocol as Aβ42-red
  - 10 seeds
  - Estimated wall time: ~3-4 hours total

**Reproduction success criterion:** PyGVAMP's mean VAMP-2 (over 10 seeds, with 95% CI) should overlap with the published value within their reported confidence interval, or differ by less than 0.05 (a typical CI width for these scores).

**Reproduction failure investigation:** If reproduction fails, before assuming framework bugs, check: (a) trajectory preprocessing (atom selection, alignment, frame stride) matches the baseline; (b) hyperparameters (hidden dims, n_interactions, n_neighbors) match; (c) lag time and k match exactly.

---

## Category 2: Encoder Improvement Sweeps

**Scientific goal:** Demonstrate that PyGVAMP's GIN-with-parallel-attention and ML3 encoders perform competitively or better than SchNet on the same systems. This is the core methodological contribution.

**Protocol:**
- 10 seeds per (system, encoder) combination
- All three encoders: SchNet, GIN (with parallel attention), ML3
- **Matched architecture across encoders:** same hidden dim, same depth, same n_neighbors, same embedding setup. Only the encoder *type* changes.
- Fixed k matching the baseline paper's choice for that system
- Fixed lag time matching the baseline paper's choice for that system
- No auto-discovery
- No auto-stride (or use whatever was used in reproduction for that system, for consistency)

### Per system

- [ ] **Trp-cage:** SchNet, GIN, ML3 × 10 seeds = 30 runs
- [ ] **Villin:** SchNet, GIN, ML3 × 10 seeds = 30 runs
- [ ] **NTL9:** SchNet, GIN, ML3 × 10 seeds = 30 runs
- [ ] **Alanine dipeptide:** SchNet, GIN, ML3 × 10 seeds = 30 runs (reversible)
- [ ] **Aβ42-red:** SchNet, GIN, ML3 × 10 seeds = 30 runs (reversible)
- [ ] **Aβ42-ox:** SchNet, GIN, ML3 × 10 seeds = 30 runs (reversible)

**Total: 180 runs.** At 15 min/run on a 5090 = ~45 hours of wall time. About 2 days continuous, or several overnights.

**Reporting:** A table per system showing mean VAMP-2 ± 95% CI for each encoder. A claim like "GIN improves over SchNet on Trp-cage" requires non-overlapping CIs.

**Note on the reproduction SchNet runs:** The 10 SchNet seeds you ran in Category 1 can also serve as the SchNet baseline in Category 2. You don't need to repeat them — just reuse the results. Saves 60 runs.

---

## Category 3: Multi-Lag-Time Exploration

**Scientific goal:** Use PyGVAMP's auto-stride and auto-discovery features to explore Aβ42 dynamics across multiple timescales, demonstrating the framework's exploratory power. Not a comparison to the baselines — a novel analysis enabled by the framework.

**Protocol:**
- 3 seeds per (system, encoder, lag time) — enough to estimate variance, not enough to claim small differences
- All three encoders
- Lag times: 0.5, 1, 2, 5, 10, 20, 50 ns (or some sensible range — adjust based on what makes sense for Aβ42 dynamics)
- Auto-discovery ON
- Auto-stride ON
- Reversible model
- Aβ42-red and Aβ42-ox

### Runs

- [ ] **Aβ42-red:** SchNet, GIN, ML3 × 7 lag times × 3 seeds = 63 runs
- [ ] **Aβ42-ox:** SchNet, GIN, ML3 × 7 lag times × 3 seeds = 63 runs

**Total: 126 runs.** With auto-stride keeping each run to ~10-20 min, about 30 hours wall time.

**Reporting:**
- k* vs τ plot per (system, encoder) — how does the optimal state count change with lag time?
- VAMP-2 vs τ per encoder — how do the encoders compare across timescales?
- ox vs red comparison: do the two forms show different timescale hierarchies?
- The interactive report (you mentioned it's already built) is the primary deliverable for this category

---

## Pre-Production Checks

Before kicking off any of the above, do these one-shot validations:

- [ ] **All three implementation features tested end-to-end** (per IMPLEMENTATION_PLAN.md Phase 5)
- [ ] **Stride sanity check:** Run one Aβ42 SchNet seed at stride 1 and one at the auto-stride choice (probably stride 5 for τ=10 ns). Compare VAMP-2 — should agree within ~0.05. If not, investigate before running the full sweep.
- [ ] **Warm-start sanity check:** Run one (system, encoder, lag) case with full from-scratch retrains and one with warm-started retrains. Compare final VAMP-2 and final k. Should agree.
- [ ] **Reversible model spot-check:** Verify the reversible model produces a valid (row-stochastic, detailed-balance-satisfying) transition matrix on at least one test case. (Should be automated in tests already, but verify visually on real data.)
- [ ] **Confirm k and lag time choices for each baseline system:** Read the relevant papers/supplementary and write down the exact values. Don't guess.
- [ ] **Trajectory access verified:** All six systems' trajectories are on local disk and preprocessed. (You said this is the case but worth a final check.)
- [ ] **Disk space:** ~180 + 126 + 60 = 366 runs × ~50 MB each (model + analysis output) = ~18 GB. Confirm available.

---

## Suggested Execution Order

1. **Pre-production checks** (1-2 days, mostly waiting on small runs and reading papers)
2. **Reproduction sweeps** first — if these fail, no point doing the rest until the framework is fixed (3-4 days wall time, can run overnight in batches)
3. **Encoder improvement sweeps** — the core contribution (2-3 days wall time)
4. **Multi-lag exploration** — fastest, leave for last (1-2 days)

Total wall-time budget: ~10-12 days of GPU time, spread over 2-3 weeks of calendar time accounting for analysis between batches.

---

## Things to Track Per Run

For each run, save (your pipeline probably already does most of this):

- Final model checkpoint
- Training history (per-epoch train/val VAMP-2 or NLL)
- Final VAMP-2 (and VAMP-E for reversible runs)
- Final transition matrix K
- Final stationary distribution π
- ITS plot data (lag time → eigenvalues)
- CK test data
- State assignments and populations
- Encoder attention weights (if applicable)
- Wall time and seed
- All hyperparameters (CLI args, config snapshot)

For sweeps, also save a master CSV with one row per (system, encoder, lag_time, seed) summarizing the final scores. This makes the reporting tables trivial to generate.

---

## Things NOT to Do

- Don't run all 366 jobs in one shell loop — if anything crashes, you lose the partial results unwritten to disk. Run in batches of 10-30 with explicit save points.
- Don't change hyperparameters mid-sweep. If you discover a better setting, finish the current sweep with the original setting and add the new setting as a separate sweep.
- Don't compare runs across different stride choices or different k choices in the comparison tables — those aren't fair comparisons.
- Don't claim "GIN beats SchNet" if the 95% CIs overlap. Use language like "GIN performs comparably to SchNet."
- Don't skip the reproduction sweeps. The improvement claims have no foundation without them.

---

## When You're Done

Each row in the paper's results tables should be traceable to a specific set of runs in this checklist. Before submitting:

- [ ] Every numerical claim in the paper points to a specific sweep
- [ ] Every figure has its underlying data archived
- [ ] CLI commands and configs for each sweep are committed to the repo
- [ ] A `EXPERIMENTS.md` in the repo describes how to reproduce each sweep

This is the difference between "we got these numbers" and "anyone can verify our numbers."
