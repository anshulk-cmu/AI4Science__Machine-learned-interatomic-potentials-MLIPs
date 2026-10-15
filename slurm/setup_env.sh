#!/bin/bash
# Build, or complete and verify, the training environment on Ocean. Run it inside a Slurm job, never on a login node.
umask 002
export PYTHONNOUSERSITE=1
set -euo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
ENV=$ROOT/envs/mlip
export CONDA_PKGS_DIRS=$ROOT/.conda/pkgs PIP_CACHE_DIR=$ROOT/.pipcache TMPDIR=$ROOT/tmp
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
module load anaconda3
[ -x "$ENV/bin/python" ] || conda create --prefix "$ENV" python=3.12 pip -y   # a rerun reuses the environment
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r "$REPO/requirements.txt" -r "$REPO/requirements-training.txt" wandb
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$ROOT/envs/mlip_freeze.txt"
"$ENV/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
