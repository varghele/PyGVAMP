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

**Architecture note for strict reproduction:** Ghorbani et al. 2022 used a deliberately small network — 4 graph layers with 16 neurons per layer, 16 Gaussians for edge features. Your default `medium_schnet` preset uses 128 hidden dims and 64 output dims, which is ~8× larger. For **strict reproduction** runs you should override to match their architecture (hidden_dim=16, similar scaling throughout). For **"PyGVAMP with its default preset" reproduction**, use your medium_schnet preset but clearly document the architectural difference in the paper — expect somewhat higher VAMP-2 scores than theirs because the larger model has more capacity. My recommendation: do strict reproduction first to validate the framework, then a "scaled-up" run at your preset to show what PyGVAMP's default delivers.

**Other protocol details from Ghorbani et al. 2022:**
- 70/30 train/validation split (your default is 80/20 — worth matching for strict reproduction)
- 100 epochs training (before early stopping)
- Adam optimizer, lr = 0.0005
- Batch size = 1000
- 10 seeds for error bars (95% CI)

- [ ] **Trp-cage** (Lindorff-Larsen, 208 µs)
  - k = 5, lag time = 20 ns
  - n_neighbors = 7, n_atoms (Cα) = 20
  - batch_size = 1000, lr = 0.0005
  - 10 seeds
  - Compare against: published VAMP-2 = 4.79 ± 0.01 (Table S1), ITS plot (Fig 2A), CK test (Fig 2B)
  - Estimated wall time: ~2-3 hours total at 10-20 min/seed

- [ ] **Villin** (Lindorff-Larsen, 125 µs)
  - k = 4, lag time = 20 ns
  - n_neighbors = 10, n_atoms (Cα) = 35
  - batch_size = 1000, lr = 0.0005
  - 10 seeds
  - Compare against: published VAMP-2 = 3.78 ± 0.02 (Table S1), ITS plot (Fig 4A), CK test (Fig 4B)
  - Estimated wall time: ~2-3 hours total

- [ ] **NTL9** (Lindorff-Larsen, 1.11 ms)
  - k = 5, lag time = 200 ns
  - n_neighbors = 10, n_atoms (Cα) = 39
  - batch_size = 1000, lr = 0.0005
  - 10 seeds
  - Compare against: published VAMP-2 = 4.59 ± 0.09 (Table S1), ITS plot (Fig 6A), CK test (Fig 6B)
  - Estimated wall time: ~2-3 hours total

### RevGraphVAMP reproduction

**Architecture note for strict reproduction:** Huang et al. 2024 use a small network — hidden dim 16, 4 graph layers — and the RevGraphVAMP model has only **6,357 parameters** on Aβ42 (vs. 464,646 for the standard VAMPNets they compare against). This small-model choice is a core part of their story. For strict reproduction, match their architecture; for "PyGVAMP default preset" reproduction, use medium_schnet and note the difference. The parameter count comparison is actually a nice angle to highlight in your paper.

**Confirmed protocol details from Huang et al. 2024:**
- 7:3 train/val split
- Adam optimizer
- Three-phase training strategy (Section 2.3):
  1. Train GASchNet (χ) with VAMP-2 at lr=0.0005 for `epoch_chi` epochs
  2. Fix GASchNet, train U+S constraints with VAMP-E at lr=0.0005 for `epoch_US` epochs
  3. Train full model with VAMP-E at lr=0.0001 for `epoch_all` epochs
- GitHub defaults for Aβ42: pre_train = 300, total = 1000 epochs
- Hardware: NVIDIA GeForce GTX 3090 (your 5090 is meaningfully faster)
- Aβ42 dataset from Löhr et al. 2021 (ref 23), 5119 trajectories, 315 µs total at 250 ps/frame

**Important: Aβ42 ox/red split.** The paper treats Aβ42 as a single dataset and does not report separate ox/red runs. The Löhr et al. 2021 source dataset likely contains both forms, but RevGraphVAMP doesn't split them. **If you want to report ox vs red comparisons, that's a novel contribution** of your work, not a reproduction of RevGraphVAMP's results — frame this carefully in the paper. The RevGraphVAMP reproduction should be on the combined Aβ42 dataset to match their protocol exactly.

**Note on number of seeds:** The paper doesn't explicitly state how many training runs were used for error bars. Error bars in Table 2 are very tight (±0.002 on Aβ42 VAMP-2) which suggests at least 5 runs, probably more. Running 10 seeds gives you stronger error-bar discipline than the baseline, which is a small methodological point worth noting in your paper.

**One unresolved discrepancy:** The paper's Table 1 says Aβ42 uses **40 atoms** while the GitHub `train.py` command uses **`--num-atoms 42`**. Aβ42 has 42 amino acid residues. Worth checking their actual preprocessing code — they may select only 40 Cα positions due to terminal handling, or the paper table might have a typo. Test with both values and see which matches their published VAMP scores.

- [ ] **Alanine dipeptide** (mdshare, 3 trajectories)
  - k = 6, lag time = 20 ps (confirmed in Section 3.1.1)
  - n_atoms = 10 (heavy atoms on main chain), n_neighbors = 5
  - batch_size = 1000
  - lr = [0.0005, 0.0001] — two-phase: 0.0005 for phase 1 (χ training), 0.0001 for phase 3 (full model)
  - hidden_dim = 16, n_graph_layers = 4, n_Gaussians = 16
  - Training: 3 trajectories × 250 ns × 1 ps/frame = 750k frames total
  - 10 seeds (recommended; paper doesn't state count)
  - Reversible model, three-phase training strategy (VAMP-2 then VAMP-E)
  - Compare against: RevGraphVAMP published VAMP-2 = 4.41 ± 0.01, VAMP-E = 4.38 ± 0.01 (Table 2)
  - Estimated wall time: ~1-2 hours total (small system)

- [ ] **Aβ42** (Löhr et al. 2021 dataset, ref 23 in Huang 2024)
  - k = 4, lag time = 10 ns (confirmed in Section 3.2.1)
  - n_atoms = 40 (per paper Table 1) OR 42 (per GitHub train.py command) — discrepancy worth resolving
  - n_neighbors = 10
  - batch_size = 500
  - lr = [0.0005, 0.0001] — two-phase
  - hidden_dim = 16, n_graph_layers = 4, n_Gaussians = 16 (dmin=0, dmax=8, step=0.5)
  - pre_train_epochs = 300, epochs = 1000 (from GitHub script)
  - Dataset: 5,119 trajectories, 1.26M frames total, 250 ps/frame, 315 µs total
  - Validation subset: 1024 trajectories = 273,715 frames
  - State populations: 52.3%, 26.7%, 10.7%, 10.2%
  - 10 seeds (recommended; paper doesn't state count, but error bars are tight)
  - Reversible model
  - Compare against: RevGraphVAMP published VAMP-2 = 3.99 ± 0.002, VAMP-E = 3.99 ± 0.003 (Table 2)
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
- [ ] **Aβ42** (full dataset, matching RevGraphVAMP): SchNet, GIN, ML3 × 10 seeds = 30 runs (reversible)
- [ ] **Aβ42-red** (if you have the split): SchNet, GIN, ML3 × 10 seeds = 30 runs (reversible)
- [ ] **Aβ42-ox** (if you have the split): SchNet, GIN, ML3 × 10 seeds = 30 runs (reversible)

**Total: 120-180 runs depending on whether you include ox/red split.** At 15 min/run on a 5090 = ~30-45 hours of wall time. About 2 days continuous, or several overnights.

**Reporting:** A table per system showing mean VAMP-2 ± 95% CI for each encoder. A claim like "GIN improves over SchNet on Trp-cage" requires non-overlapping CIs.

**Note on the reproduction SchNet runs:** The 10 SchNet seeds you ran in Category 1 can also serve as the SchNet baseline in Category 2. You don't need to repeat them — just reuse the results. Saves 60 runs.

---

## Category 3: Multi-Lag-Time Exploration

**Scientific goal:** Use PyGVAMP's auto-stride and auto-discovery features to explore Aβ42 dynamics across multiple timescales, demonstrating the framework's exploratory power. Not a comparison to the baselines — a novel analysis enabled by the framework. **The ox vs red comparison itself is novel** — RevGraphVAMP doesn't separate them, so any finding here is your contribution, not a reproduction.

**Protocol:**
- 3 seeds per (system, encoder, lag time) — enough to estimate variance, not enough to claim small differences
- All three encoders
- Lag times: 0.5, 1, 2, 5, 10, 20, 50 ns (or some sensible range — adjust based on what makes sense for Aβ42 dynamics; note RevGraphVAMP's chosen τ = 10 ns is in the middle of this range)
- Auto-discovery ON
- Auto-stride ON
- Reversible model
- Aβ42-red and Aβ42-ox (assuming you have them split)

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
