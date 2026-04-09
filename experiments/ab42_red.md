# Ab42 Reduced — Experiment Tracker

## System Info

| Property | Value |
|----------|-------|
| Protein | Amyloid-beta 42 (reduced) |
| Topology | `/mnt/hdd/data/ab42/trajectories/red/topol.pdb` |
| Trajectories | `/mnt/hdd/data/ab42/trajectories/red/` |
| Replicates | r1–r5 (+ r1cs–r5cs) |
| Trajectory files | 5119 |
| Selection | `name CA` (42 atoms) |
| Total frames (stride=1) | 1,259,172 |
| Timestep | 250 ps |
| Preset | medium_schnet |

## State Discovery

| Property | Value |
|----------|-------|
| Date | 2026-04-09 |
| Job ID | 339 |
| Stride | 1 |
| Graph2Vec dim | 512 |
| Clustering subsample | 100,000 |
| Best source | umap_2 (silhouette=0.444) |
| **Recommended n_states** | **10** |
| Log | `/mnt/hdd/experiments/logs/disc_339.out` |
| Output | `/mnt/hdd/experiments/ab42_red/discovery/exp_ab42_red_20260409_110532` |

Metric breakdown (umap_2):
- Silhouette: k=6
- Elbow: k=3
- BIC: k=10
- AIC: k=10

### Command
```bash
sbatch cluster_scripts/run_discovery.sh \
    --protein ab42_red \
    --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
    --traj /mnt/hdd/data/ab42/trajectories/red/
```

## Experiments

### Standard (VAMP-2)

| Run | Lag (ns) | Encoder | n_states | Epochs | Train VAMP | Val VAMP | Status | Job ID | Notes |
|-----|----------|---------|----------|--------|------------|----------|--------|--------|-------|
| 0   | 1        | SchNet  | 10       | 50     | —          | —        | running | 342   | exploratory, batch_size=2048, stride=1 |

#### Submit command
```bash
# Single run
sbatch cluster_scripts/run_experiment.sh \
    --protein ab42_red \
    --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
    --traj /mnt/hdd/data/ab42/trajectories/red/ \
    --lag <LAG> --n_states 10 --run <N>

# 10 runs at one lag time
./cluster_scripts/submit_runs.sh \
    --protein ab42_red \
    --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
    --traj /mnt/hdd/data/ab42/trajectories/red/ \
    --lag <LAG> --n_states 10 --n_runs 10
```

### Reversible (RevGraphVAMP)

| Run | Lag (ns) | Encoder | n_states | Epochs | Train VAMP | Val VAMP | Status | Job ID | Notes |
|-----|----------|---------|----------|--------|------------|----------|--------|--------|-------|

#### Submit command
```bash
./cluster_scripts/submit_runs.sh \
    --protein ab42_red \
    --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
    --traj /mnt/hdd/data/ab42/trajectories/red/ \
    --lag <LAG> --n_states 10 --n_runs 10 --reversible
```

## Output Location

`/mnt/hdd/experiments/ab42_red_std/` (standard)
`/mnt/hdd/experiments/ab42_red_rev/` (reversible)

## Notes

- First test run (job 336) used 100 epochs, batch_size=2048, stride=1
- Epoch time ~23 min with 8 CPUs. Future runs use 16 CPUs + `sample_validate_every=100`
- Dataset cache reusable across runs for same lag time
