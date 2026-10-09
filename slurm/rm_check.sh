#!/bin/bash
# Check that a CPU-only Regular Memory job works for the project: the environment, the data and the repo are visible.
# Run it inside a Slurm job on the RM-shared partition.
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
PY=$ROOT/envs/mlip/bin/python
hostname; date; echo "partition ${SLURM_JOB_PARTITION:-?}, cpus ${SLURM_CPUS_ON_NODE:-?}, account ${SLURM_JOB_ACCOUNT:-?}"
free -g | head -2
$PY -c "import torch, fairchem.core, pymatgen, wandb; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available(), 'threads', torch.get_num_threads())"
$PY -m pip check
for split in train_4M val; do echo "$split: $(ls $ROOT/data/omol25/$split/*.aselmdb | wc -l) shards"; done
$PY -B -m unittest discover -s "$ROOT/repo/AI4Science-MLIPs/tests" -t "$ROOT/repo/AI4Science-MLIPs" -p 'test_local_routing*.py' 2>&1 | grep -E '^(Ran|OK|FAILED)'
