#!/bin/bash
# Build the MACE environment (it cannot share the main one: e3nn conflict) on Ocean. Run it inside a Slurm job.
umask 002
export PYTHONNOUSERSITE=1
set -euo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
ENV=$ROOT/envs/mace
export CONDA_PKGS_DIRS=$ROOT/.conda/pkgs PIP_CACHE_DIR=$ROOT/.pipcache TMPDIR=$ROOT/tmp
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
module load anaconda3
[ -x "$ENV/bin/python" ] || conda create --prefix "$ENV" python=3.12 pip -y
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r "$REPO/requirements-mace.txt"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$ROOT/envs/mace_freeze.txt"
"$ENV/bin/python" -c "import torch, mace; print('torch', torch.__version__, 'mace ok')"
