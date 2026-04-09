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
| Job ID | 338 |
| Stride | 1 |
| Graph2Vec dim | 512 |
| Clustering subsample | 100,000 |
| Best source | pending |
| **Recommended n_states** | **pending** |
| Log | `/mnt/hdd/experiments/logs/disc_338.out` |

### Command
```bash
sbatch cluster_scripts/run_discovery.sh \
    --protein ab42_red \
    --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
    --traj /mnt/hdd/data/ab42/trajectories/red/
```

## Experiments

### Standard (VAMP-2)

| Run | Lag (ns) | n_states | Epochs | VAMP score | Status | Notes |
|-----|----------|----------|--------|------------|--------|-------|
| 0   | 1        | 10       | 100    | ~9.83      | done   | first test run, job 336 |

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

| Run | Lag (ns) | n_states | Epochs | VAMP score | Status | Notes |
|-----|----------|----------|--------|------------|--------|-------|

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
