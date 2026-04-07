# PyGVAMP Environment Module

Install PyGVAMP as a shared module on an HPC cluster using [Environment Modules](https://modules.readthedocs.io/) (TCL, v5.x).

## Quick Install

```bash
# As cluster admin:
./install_module.sh \
    --prefix /opt/software/pygvamp/0.9.0 \
    --moduledir /opt/modulefiles

# If your modulefiles directory isn't in MODULEPATH yet:
module use /opt/modulefiles
```

## User Usage

```bash
module load pygvamp/0.9.0
pygvamp --help

# Run full pipeline
pygvamp --traj_dir /path/to/traj --top /path/to/topology.pdb \
    --preset medium_schnet --n_states 5 --lag_times 10 20 50

# SLURM array job (sweep n_states 3-10)
sbatch --array=3-10 example_slurm.sh
```

## What the Installer Does

1. Creates a self-contained conda environment at `<prefix>/conda_env/` with all dependencies (PyTorch, PyG, MDTraj, etc.)
2. Copies PyGVAMP source to `<prefix>/source/` and installs it (`pip install -e`)
3. Generates a modulefile at `<moduledir>/pygvamp/0.9.0` from the template

## Directory Layout

```
<prefix>/pygvamp/0.9.0/
├── conda_env/          # Full conda environment (Python, PyTorch, PyG, ...)
└── source/             # PyGVAMP source code (editable install)

<moduledir>/pygvamp/
└── 0.9.0               # TCL modulefile
```

## Updating

To update to a new version:

```bash
# Install new version alongside the old one
./install_module.sh \
    --prefix /opt/software/pygvamp/1.0.0 \
    --moduledir /opt/modulefiles

# Users can then:
module load pygvamp/1.0.0

# To set a default version, create a .version file:
echo '#%Module1.0
set ModulesVersion "0.9.0"' > /opt/modulefiles/pygvamp/.version
```

## Customization

### Different CUDA version
```bash
./install_module.sh --prefix ... --moduledir ... --cuda 11.8
```

### Reuse existing conda env (e.g., after adding a package)
```bash
./install_module.sh --prefix ... --moduledir ... --skip-env
```

### Manual modulefile editing
The modulefile template is `pygvamp.tcl`. The only value substituted during install is `__PYGVAMP_ROOT__` → the prefix path. Everything else is relative.

## Files

| File | Purpose |
|------|---------|
| `pygvamp.tcl` | Modulefile template (portable, single root variable) |
| `install_module.sh` | Automated installer (creates env + configures modulefile) |
| `example_slurm.sh` | Example SLURM script using the module |
