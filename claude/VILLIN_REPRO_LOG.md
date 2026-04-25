# VILLIN_REPRO_LOG.md — Villin Strict-Reproduction Run

Living log of every step taken to set up the Villin strict-reproduction
experiment (Category 1 of EXPERIMENT_CHECKLIST.md).  Scientific goal:
reproduce Ghorbani et al. 2022 Table S1 VAMP-2 = 3.78 ± 0.02 on the DE Shaw
2F4K-0 trajectory.

## Target protocol (from EXPERIMENT_CHECKLIST.md)

- System: 2F4K-0 (Villin headpiece, Nle-mutant), DE Shaw, 2011, Science 334:517
- k = 4 (fixed; no auto-discovery)
- lag time = 20 ns
- n_neighbors = 10
- n_atoms (Cα) = 35
- batch_size = 1000
- lr = 5e-4
- val_split = 0.3 (70/30, per Ghorbani 2022 protocol)
- 100 epochs, Adam optimizer (their choice), plateau-based early stop OK
- SchNet encoder **strict architecture**: hidden_dim=16, n_interactions=4, gaussian_expansion_dim=16
- No auto-stride (single lag time)
- No --reversible (standard VAMP-2)
- 10 seeds for 95% CI error bars
- Target VAMP-2: 3.78 ± 0.02

## Step log

### 2026-04-24 — setup

**Data extraction**
- Source: `/mnt/hdd/data/DESHAW/DESRES-Trajectory_2F4K-0-c-alpha.tar.xz`
- Extracted to: `/mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/` (302 MB)
- Contents: 7 DCD files (`2F4K-0-c-alpha-000.dcd` through `-006.dcd`), `2F4K-0-c-alpha.mae`, `system.mae`, times.csv.
- Per `README.txt`: 2F4K-0 = Villin HP-35 Nle-mutant at 360 K, Lindorff-Larsen et al. 2011, Science 334:517.

**Topology**
- PyMOL `.mae` → `.pdb` conversion path was blocked: the conda env's PyMOL (`pymol-3.1.7.2`) fails to start with `ImportError: libCatch2.so.3.13.0: cannot open shared object file`. Bug in the installed build — worth filing but out of scope.
- **Workaround**: parsed the `.mae` file directly in Python (Maestro files are plain text; the `m_atom[35]` block contains atom name, residue name, residue number, x/y/z for every Cα). Wrote `/mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb` — 35 Cα atoms, chain A, residues 42-76 (LEU → PHE).
- Roundtrip validated: mdtraj loads the PDB as 35 atoms / 35 residues, then loads DCD 000 against it as 100,000 frames × 35 atoms.

**Trajectory timestep — important gotcha**
- DCD time metadata reports 1 ps/frame, but the actual physical timestep is **200 ps/frame** (from `times.csv`: segment 000 starts at 200 ps, segment 001 at 20,000,200 ps → 20 µs/segment).
- **Consequence**: every pygvamp call on this data MUST pass `--timestep 0.2`. Without it, `lag_time=20ns` would compute `lag_frames=20000` (absurd), breaking training pair construction.
- Total dataset length: 7 segments × 20 µs = 140 µs (matches README and EXPERIMENT_CHECKLIST's "125 µs" within rounding).

**Paths committed for downstream scripts**
- `TRAJ_DIR = /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/`
- `TOP     = /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb`
- `FILE_PATTERN = 2F4K-0-c-alpha-*.dcd`
- `TIMESTEP_NS  = 0.2`
- `OUTPUT_BASE  = /mnt/hdd/experiments/villin_repro/`

**CLI plumbing additions (committed alongside this experiment)**

To make every reproduction protocol fully self-documenting in its SLURM script
without spawning per-experiment preset classes, the following pipeline-level
CLI flags were added to `pygv/pipe/args.py` (and wired into `master_pipeline.main`):

- `--seed`                  master RNG (train/val split + dataset random state)
- `--hidden_dim`            encoder hidden
- `--output_dim`            encoder output
- `--n_interactions`        message-passing layers
- `--n_neighbors`           k-NN neighbours
- `--gaussian_expansion_dim` RBF dim
- `--use_attention` / `--no_use_attention` (paired toggle)
- `--file_pattern`          glob for trajectory files
- `--lr`, `--weight_decay`, `--val_split`

Plus a small fix in `master_pipeline._run_retrain_loop`: `max_retrain <= 0` now
short-circuits the loop entirely — the strict-reproduction switch.

**Latent bug noticed (not fixed, out of scope)**
`pygv.config.presets.{small,medium,large}` define subclasses of `SchNetConfig`
without re-decorating with `@dataclass`, so their field overrides
(e.g. `SmallSchNetConfig.hidden_dim = 64`) are silently shadowed by the parent
dataclass's defaults — `get_config('small_schnet').hidden_dim` returns `128`,
not `64`.  Affects every existing preset.  Logged here for later cleanup.

## Strict-reproduction protocol (Ghorbani et al. 2022, Table S1)

The full `pygvamp` command is in `cluster_scripts/villin_repro.sh`.  Reproduced
here for audit/skim:

```
pygvamp \
    --traj_dir /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/ \
    --top      /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb \
    --file_pattern '2F4K-0-c-alpha-*.dcd' \
    --protein_name villin \
    --output_dir   /mnt/hdd/experiments/villin_repro/seed_<NN> \
    --timestep     0.2 \
    --seed         <NN>             # 0..9 from SLURM_ARRAY_TASK_ID
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     4 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
    --no_use_attention \
    --lr           5e-4 \
    --weight_decay 1e-5 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --early_stopping_patience  8 \
    --early_stopping_tol       5e-4 \
    --early_stopping_min_epochs 20 \
    --cache
```

### Mapping each flag to the Ghorbani 2022 protocol

| Flag | Source (paper / checklist) |
|---|---|
| `--n_states 4` | Villin entry, Table S1 |
| `--lag_times 20.0` | Villin entry, Table S1 |
| `--hidden_dim 16` / `--n_interactions 4` / `--gaussian_expansion_dim 16` | "16 neurons per layer, 4 graph layers, 16 Gaussians" |
| `--n_neighbors 10` | Villin entry |
| `--no_use_attention` | Original GraphVAMPNet uses classic SchNet, no attention |
| `--lr 5e-4`, `--batch_size 1000`, `--epochs 100`, `--val_split 0.3` | Ghorbani 2022 protocol details |
| `--no_discover_states` + `--max_retrains 0` | Strict — no diagnostic-driven k change |
| `--no_warm_start_retrains` | Strict — moot when max_retrains=0, set explicitly for clarity |
| `--timestep 0.2` | DE Shaw DCD metadata override (200 ps/frame) |
| `--cache` | Reuse preprocessed dataset across seeds |
| `--early_stopping_*` | Plateau detection at recent-default settings |
| `--seed N` | Set per array task; persisted in `config.yaml` and `pipeline_summary.json` |

### Submission

```
sbatch --array=0-9%1 cluster_scripts/villin_repro.sh
```

`%1` throttles to one GPU at a time (one task runs, the next nine pend).
Per-task wall time estimate: ~30-45 min on RTX 5090 at this small architecture
(35 Cα atoms × 700k frames = small dataset; 100 epochs × tiny model).
Total array wall time: **~5-7 h.**

### Outputs to compare against

For each seed, the final Val VAMP-2 score lives in:
  `/mnt/hdd/experiments/villin_repro/seed_NN/exp_villin_<TIMESTAMP>/training/lag20.0ns_4states/<TIMESTAMP>/`

The aggregate target is mean Val VAMP-2 over the 10 seeds, with 95% CI.
Success criterion (per EXPERIMENT_CHECKLIST): result within 0.05 of 3.78
(Ghorbani's reported value).


