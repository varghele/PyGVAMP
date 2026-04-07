#%Module1.0
##
## PyGVAMP - Graph-based VAMPNet for MD trajectory analysis
##
## This modulefile is portable: it derives all paths from PYGVAMP_ROOT,
## which is set during installation by install_module.sh.
##
## Usage after installation:
##   module load pygvamp/1.0.0
##   pygvamp --help
##

proc ModulesHelp { } {
    puts stderr "PyGVAMP 1.0.0 - Graph-based VAMPNet for MD trajectory analysis"
    puts stderr ""
    puts stderr "  Converts MD trajectories (.xtc/.dcd) to k-NN graphs and trains"
    puts stderr "  VAMPNet models to learn slow collective variables via VAMP score"
    puts stderr "  optimization. ~50x speedup over the original GraphVAMPNets."
    puts stderr ""
    puts stderr "  Usage:  pygvamp --traj_dir /path/to/traj --top /path/to/top.pdb"
    puts stderr "  Help:   pygvamp --help"
    puts stderr ""
    puts stderr "  Encoders: SchNet (default), GIN, ML3"
    puts stderr "  Presets:  small_schnet, medium_schnet, large_schnet, etc."
}

module-whatis "PyGVAMP 1.0.0 - Graph-based VAMPNet for MD trajectory analysis with PyTorch Geometric"

# Conflict with other versions of this module
conflict pygvamp

# ── Root directory (set during installation) ──────────────────────────
# PYGVAMP_ROOT points to the installation prefix, e.g.:
#   /opt/software/pygvamp/1.0.0
#
# Expected layout under PYGVAMP_ROOT:
#   conda_env/     - self-contained conda environment
#   source/        - PyGVAMP source (pip install -e)
#
# !! This line is replaced by install_module.sh during installation !!
set root "__PYGVAMP_ROOT__"

# ── Validate installation ─────────────────────────────────────────────
if { ![file isdirectory $root] } {
    puts stderr "ERROR: PyGVAMP installation not found at $root"
    puts stderr "       Run install_module.sh to install or check PYGVAMP_ROOT."
    break
}

# ── Environment setup ─────────────────────────────────────────────────

# Conda environment paths
set condaenv "$root/conda_env"

# Prepend conda env bin (gives us python, pygvamp, pytest, etc.)
prepend-path PATH "$condaenv/bin"

# Library paths for compiled extensions (torch, pyg, etc.)
prepend-path LD_LIBRARY_PATH "$condaenv/lib"

# Python can find PyGVAMP source
prepend-path PYTHONPATH "$root/source"

# Make the install root discoverable by other tools
setenv PYGVAMP_ROOT $root
setenv PYGVAMP_VERSION "1.0.0"

# CUDA — let the system CUDA module handle this, but set fallback
# if CUDA_HOME is already set (e.g., by a CUDA module), don't override
if { ![info exists env(CUDA_HOME)] && [file isdirectory "$condaenv/lib/python3.12/site-packages/torch/lib"] } {
    setenv TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;8.6;8.9;9.0"
}
