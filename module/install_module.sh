#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PyGVAMP Module Installer
#
# Installs PyGVAMP as an Environment Modules package on a shared filesystem.
#
# Usage:
#   ./install_module.sh --prefix /opt/software/pygvamp/0.9.0 \
#                       --moduledir /opt/modulefiles
#
# This creates:
#   <prefix>/
#   ├── conda_env/        # Self-contained conda environment
#   └── source/           # PyGVAMP source (editable install)
#
#   <moduledir>/pygvamp/0.9.0   # TCL modulefile
#
# After installation:
#   module load pygvamp/0.9.0
#   pygvamp --help
# ============================================================================

VERSION="1.0.0"
PYTHON_VERSION="3.12"
CUDA_VERSION="12.4"
TORCH_VERSION="2.5.1"

# ── Parse arguments ───────────────────────────────────────────────────

PREFIX=""
MODULEDIR=""
CONDA_CMD="conda"
REPO_DIR=""
SKIP_ENV=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --prefix DIR        Installation prefix (e.g., /opt/software/pygvamp/0.9.0)
  --moduledir DIR     Directory for modulefiles (e.g., /opt/modulefiles)

Optional:
  --repo DIR          Path to PyGVAMP source repo (default: auto-detect from script location)
  --conda CMD         Conda executable (default: conda)
  --cuda VERSION      CUDA version for PyTorch (default: $CUDA_VERSION)
  --python VERSION    Python version (default: $PYTHON_VERSION)
  --skip-env          Skip conda env creation (reuse existing)
  -h, --help          Show this help
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)     PREFIX="$2"; shift 2;;
        --moduledir)  MODULEDIR="$2"; shift 2;;
        --repo)       REPO_DIR="$2"; shift 2;;
        --conda)      CONDA_CMD="$2"; shift 2;;
        --cuda)       CUDA_VERSION="$2"; shift 2;;
        --python)     PYTHON_VERSION="$2"; shift 2;;
        --skip-env)   SKIP_ENV=1; shift;;
        -h|--help)    usage;;
        *)            echo "Unknown option: $1"; usage;;
    esac
done

if [[ -z "$PREFIX" || -z "$MODULEDIR" ]]; then
    echo "ERROR: --prefix and --moduledir are required."
    echo ""
    usage
fi

# ── Detect repo directory ─────────────────────────────────────────────

if [[ -z "$REPO_DIR" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

if [[ ! -f "$REPO_DIR/setup.py" ]]; then
    echo "ERROR: Cannot find PyGVAMP repo at $REPO_DIR (no setup.py found)"
    echo "       Use --repo to specify the path."
    exit 1
fi

echo "============================================"
echo "  PyGVAMP Module Installer v${VERSION}"
echo "============================================"
echo ""
echo "  Repo:       $REPO_DIR"
echo "  Prefix:     $PREFIX"
echo "  Modulefiles: $MODULEDIR"
echo "  Python:     $PYTHON_VERSION"
echo "  CUDA:       $CUDA_VERSION"
echo "  PyTorch:    $TORCH_VERSION"
echo ""

# ── Create installation directories ───────────────────────────────────

CONDA_ENV="$PREFIX/conda_env"
SOURCE_DIR="$PREFIX/source"

mkdir -p "$PREFIX"
mkdir -p "$MODULEDIR/pygvamp"

# ── Step 1: Create conda environment ─────────────────────────────────

if [[ $SKIP_ENV -eq 0 ]]; then
    echo "[1/4] Creating conda environment at $CONDA_ENV ..."

    if [[ -d "$CONDA_ENV" ]]; then
        echo "  WARNING: $CONDA_ENV already exists. Removing..."
        rm -rf "$CONDA_ENV"
    fi

    $CONDA_CMD create -y -p "$CONDA_ENV" python="$PYTHON_VERSION"

    echo "[1/4] Installing dependencies..."

    # Activate the environment for installation
    eval "$($CONDA_CMD shell.bash hook)"
    $CONDA_CMD activate "$CONDA_ENV"

    # PyMOL (optional, for structure visualization)
    $CONDA_CMD install -y -c conda-forge -c schrodinger pymol-bundle 2>/dev/null || \
        echo "  WARNING: pymol-bundle not available, skipping (optional)"

    # Core Python packages
    pip install --no-cache-dir \
        numpy scipy matplotlib pandas scikit-learn \
        tqdm joblib pyyaml jinja2 sympy

    # PyTorch + CUDA
    CUDA_TAG=$(echo "$CUDA_VERSION" | tr -d '.')
    $CONDA_CMD install -y pytorch==${TORCH_VERSION} torchvision torchaudio \
        pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia

    # PyTorch Geometric
    pip install --no-cache-dir torch_geometric
    TORCH_SHORT=$(echo "$TORCH_VERSION" | cut -d. -f1-2)
    pip install --no-cache-dir \
        pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f "https://data.pyg.org/whl/torch-${TORCH_SHORT}.0+cu${CUDA_TAG}.html" \
        2>/dev/null || echo "  WARNING: Some PyG extensions failed (may need manual install)"

    # MD analysis + embeddings
    pip install --no-cache-dir mdtraj gensim

    # RDKit (optional)
    $CONDA_CMD install -y conda-forge::rdkit 2>/dev/null || \
        echo "  WARNING: rdkit not available, skipping (optional)"

    # Optional extras
    pip install --no-cache-dir umap-learn 2>/dev/null || true

    # Testing
    pip install --no-cache-dir pytest

    echo "[1/4] Conda environment created."
else
    echo "[1/4] Skipping conda env creation (--skip-env)."
    eval "$($CONDA_CMD shell.bash hook)"
    $CONDA_CMD activate "$CONDA_ENV"
fi

# ── Step 2: Install PyGVAMP source ────────────────────────────────────

echo "[2/4] Installing PyGVAMP source..."

# Copy source to the install prefix (so it doesn't depend on the build dir)
if [[ -d "$SOURCE_DIR" ]]; then
    rm -rf "$SOURCE_DIR"
fi

# Use git archive if available, otherwise rsync
if command -v git &>/dev/null && [[ -d "$REPO_DIR/.git" ]]; then
    mkdir -p "$SOURCE_DIR"
    (cd "$REPO_DIR" && git archive HEAD | tar -x -C "$SOURCE_DIR")
else
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='.idea' --exclude='local_checks' \
        "$REPO_DIR/" "$SOURCE_DIR/"
fi

# Install in editable mode
pip install --no-cache-dir -e "$SOURCE_DIR"

echo "[2/4] PyGVAMP installed."

# ── Step 3: Generate modulefile ───────────────────────────────────────

echo "[3/4] Generating modulefile..."

MODULEFILE="$MODULEDIR/pygvamp/$VERSION"

# Use the template from the repo, replacing the root placeholder
sed "s|__PYGVAMP_ROOT__|$PREFIX|g" \
    "$REPO_DIR/module/pygvamp.tcl" > "$MODULEFILE"

echo "[3/4] Modulefile written to $MODULEFILE"

# ── Step 4: Verify installation ───────────────────────────────────────

echo "[4/4] Verifying installation..."

ERRORS=0

# Check pygvamp command exists
if [[ -x "$CONDA_ENV/bin/pygvamp" ]]; then
    echo "  OK: pygvamp command found"
else
    echo "  FAIL: pygvamp command not found in $CONDA_ENV/bin/"
    ERRORS=$((ERRORS + 1))
fi

# Check Python imports
if "$CONDA_ENV/bin/python" -c "import pygv; print(f'  OK: pygv {pygv.__version__}')" 2>/dev/null; then
    :
else
    echo "  FAIL: cannot import pygv"
    ERRORS=$((ERRORS + 1))
fi

if "$CONDA_ENV/bin/python" -c "import torch; print(f'  OK: torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" 2>/dev/null; then
    :
else
    echo "  FAIL: cannot import torch"
    ERRORS=$((ERRORS + 1))
fi

if "$CONDA_ENV/bin/python" -c "import torch_geometric; print(f'  OK: torch_geometric {torch_geometric.__version__}')" 2>/dev/null; then
    :
else
    echo "  FAIL: cannot import torch_geometric"
    ERRORS=$((ERRORS + 1))
fi

# Check modulefile
if [[ -f "$MODULEFILE" ]]; then
    echo "  OK: modulefile at $MODULEFILE"
else
    echo "  FAIL: modulefile not found"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "============================================"
if [[ $ERRORS -eq 0 ]]; then
    echo "  Installation successful!"
else
    echo "  Installation completed with $ERRORS error(s)."
fi
echo "============================================"
echo ""
echo "  To use:"
echo "    module load pygvamp/$VERSION"
echo "    pygvamp --help"
echo ""
echo "  Make sure $MODULEDIR is in your MODULEPATH:"
echo "    module use $MODULEDIR"
echo ""
echo "  To run tests:"
echo "    module load pygvamp/$VERSION"
echo "    cd $SOURCE_DIR && pytest tests/ -v --tb=short"
echo ""

exit $ERRORS
