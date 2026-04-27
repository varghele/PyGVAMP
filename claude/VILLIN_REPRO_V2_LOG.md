# VILLIN_REPRO_V2_LOG.md — Villin reproduction v2

Companion to `VILLIN_REPRO_LOG.md`.  Same scientific target — Ghorbani et al.
2022 Table S1 VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0 — but with three
v1-vs-paper deviations rolled back.

## v1 result recap (2026-04-25, seeds 0–9)

| Stat | Value |
|---|---|
| Mean Val VAMP-2 | 3.611 |
| Stdev | 0.053 |
| Range | [3.545, 3.737] |
| Paper target | 3.78 ± 0.02 |
| Gap | −0.17 (outside the ±0.05 success window) |

Per-seed best: 3.5685, 3.5868, 3.5928, 3.6496, 3.7366, 3.5447, 3.5840,
3.5985, 3.6172, 3.6295.  Four seeds bailed at the
`--early_stopping_min_epochs 20` floor; the best seed (04) ran to epoch 63.

## What v2 changes (and why)

| # | Change | v1 setting | v2 setting | Reason |
|---|---|---|---|---|
| 1 | Pre-encoder embedding MLP | `--use_embedding` (35→64→32 ReLU MLP) | `--no_use_embedding` | Paper feeds one-hot directly; SchNet's first linear handles 35→16 — equivalent to `nn.Embedding(35, 16)` |
| 2 | Classifier depth | 2-layer 64-hidden MLP | `--clf_num_layers 1` | Paper describes only a softmax output |
| 3 | Classifier dropout | 0.1 | `--clf_dropout 0` | Not in paper |
| 4 | Classifier norm | `batch_norm` | `--clf_norm none` | Not in paper |
| 5 | Early stopping | patience=8, tol=5e-4, min_epochs=20 | (flags omitted → fully off) | Paper trains 100 epochs; v1 truncated 4/10 seeds at the warmup floor |

Everything else identical to v1: data paths, k=4, lag=20 ns, n_neighbors=10,
hidden_dim=16, n_interactions=4, gaussian_expansion_dim=16, no attention,
no discovery, batch=1000, lr=5e-4, val_split=0.3, weight_decay=1e-5,
seed sweep 0–9.

## CLI plumbing committed alongside v2

`pygv/pipe/args.py` and `pygv/pipe/master_pipeline.main` gained:

- `--use_embedding` / `--no_use_embedding` (paired toggle, mirrors `--use_attention`).
- `--embedding_hidden_dim`, `--embedding_out_dim`, `--embedding_num_layers`,
  `--embedding_act`, `--embedding_dropout`, `--embedding_norm`.
- `--clf_hidden_dim`, `--clf_num_layers`, `--clf_dropout`,
  `--clf_activation`, `--clf_norm`.

`--clf_norm none` and `--embedding_norm none` (literal string) are translated
to Python `None` in `master_pipeline.main` before reaching the `MLP`
constructor.

Early stopping was already opt-in on the training side
(`pygv/vampnet/vampnet.py:1004` short-circuits when `early_stopping is None`),
so disabling it required only dropping the three `--early_stopping_*` flags
from the SLURM script — no code change.

## Submission

```
sbatch --array=0-9%1 cluster_scripts/villin_repro_v2.sh
```

Outputs land in `/mnt/hdd/experiments/villin_repro_v2/seed_NN/`.

Per-task wall time will be longer than v1 because every seed now runs the
full 100 epochs (v1 averaged ~30 because of early stopping).  Estimate
~60–90 min/seed → **~10–15 h** for the full array.

## What we expect to learn

- If the gap closes substantially → the v1 architectural extras (pre-MLP +
  classifier regularization) were the main culprit.
- If the gap closes only a little → early stopping was masking a real
  architectural deficit; further ablation needed (e.g. activation choice
  inside SchNet, AdamW vs Adam, training jitter).
- If the gap doesn't close at all → the architecture matches the paper
  closely enough; the residual gap is data prep, the small-network ceiling,
  or something subtle we haven't yet identified.

## Outputs to compare against

For each seed, best Val VAMP-2 is in:
  `/mnt/hdd/experiments/villin_repro_v2/seed_NN/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for `Loaded best model with score: <X.XXXX>`.

Aggregate: mean ± stdev across the 10 seeds.  Success criterion (per
EXPERIMENT_CHECKLIST): result within 0.05 of 3.78.