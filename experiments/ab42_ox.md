# Ab42 Oxidized — Experiment Tracker

## System Info

| Property | Value |
|----------|-------|
| Protein | Amyloid-beta 42 (oxidized) |
| Topology | `/mnt/hdd/data/ab42/trajectories/ox/topol.gro` |
| Trajectories | `/mnt/hdd/data/ab42/trajectories/ox/` |
| Replicates | r1–r3 |
| Trajectory files | 3071 |
| Selection | `name CA` (42 atoms) |
| Timestep | 250 ps (assumed, verify) |
| Preset | medium_schnet |

## State Discovery

| Property | Value |
|----------|-------|
| Date | — |
| Job ID | — |
| **Recommended n_states** | **TBD** |

### Discovery command
```bash
sbatch cluster_scripts/run_discovery.sh \
    --protein ab42_ox \
    --top /mnt/hdd/data/ab42/trajectories/ox/topol.gro \
    --traj /mnt/hdd/data/ab42/trajectories/ox/
```

## Experiments

### Standard (VAMP-2)

| Run | Lag (ns) | n_states | Epochs | VAMP score | Status | Notes |
|-----|----------|----------|--------|------------|--------|-------|

#### Submit command
```bash
# Single run
sbatch cluster_scripts/run_experiment.sh \
    --protein ab42_ox \
    --top /mnt/hdd/data/ab42/trajectories/ox/topol.gro \
    --traj /mnt/hdd/data/ab42/trajectories/ox/ \
    --lag <LAG> --n_states <N> --run <N>

# 10 runs at one lag time
./cluster_scripts/submit_runs.sh \
    --protein ab42_ox \
    --top /mnt/hdd/data/ab42/trajectories/ox/topol.gro \
    --traj /mnt/hdd/data/ab42/trajectories/ox/ \
    --lag <LAG> --n_states <N> --n_runs 10
```

### Reversible (RevGraphVAMP)

| Run | Lag (ns) | n_states | Epochs | VAMP score | Status | Notes |
|-----|----------|----------|--------|------------|--------|-------|

#### Submit command
```bash
./cluster_scripts/submit_runs.sh \
    --protein ab42_ox \
    --top /mnt/hdd/data/ab42/trajectories/ox/topol.gro \
    --traj /mnt/hdd/data/ab42/trajectories/ox/ \
    --lag <LAG> --n_states <N> --n_runs 10 --reversible
```

## Output Location

`/mnt/hdd/experiments/ab42_ox_std/` (standard)
`/mnt/hdd/experiments/ab42_ox_rev/` (reversible)

## Notes

- Discovery not yet run — submit before experiments
- Fewer trajectories than reduced (3071 vs 5119)
