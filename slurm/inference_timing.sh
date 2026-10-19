#!/bin/bash
# Inference timing of the final pilot checkpoints (scripts/inference_timing.py): Dense, Global and Local-6, seeds 0 to 2,
# on the one GPU of this job, energy only and energy plus forces. Inference only.
# Usage: inference_timing.sh LABEL [REPO]; LABEL names the output (for example v100, h100).
# Results go to $OUT/results, outside the repository, so a later syncrepo cannot overwrite them (guide B9 row 16).
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
LABEL=${1:?usage: inference_timing.sh LABEL [REPO]}
ROOT=/ocean/projects/che260063p/shared
REPO=${2:-$ROOT/repo_timing/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
OUT=$ROOT/runs/inference_timing_${SLURM_JOB_ID:-manual}
WORK=$OUT/work
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
mkdir -p "$OUT/logs" "$OUT/results" "$WORK"
now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(now) [$LABEL] $*" | tee -a "$OUT/summary.txt"; }

# work in a copy (the loader imports training/ and local_routing/ to rebuild the Local-6 model)
cp -r "$REPO/scripts" "$REPO/training" "$REPO/local_routing" "$REPO/configs" "$REPO/manifests" "$WORK/"
cd "$WORK"
NODE=$(hostname -s)
note "node $NODE; job $SLURM_JOB_ID; partition ${SLURM_JOB_PARTITION:-}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}; repo $REPO"
note "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tr '\n' ';')"
note "other compute processes on this GPU before the run: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
note "jobs on node $NODE at start (all users): $(squeue -h -w "$NODE" -o '%i' | wc -l)"
squeue -h -w "$NODE" -o '%i %P %b %C %M' | tee -a "$OUT/summary.txt"
LOG=$OUT/logs/inference_timing_${LABEL}.log
note "START inference_timing"
$PY -u -B scripts/inference_timing.py run --out "$OUT/results/inference_timing_${LABEL}.csv" > "$LOG" 2>&1
RC=$?
note "END   inference_timing exit $RC"
note "jobs on node $NODE at end (all users): $(squeue -h -w "$NODE" -o '%i' | wc -l)"
grep -a -E "^\{'gpu'|candidate rows|parameters|done in|wrote|Error" "$LOG" | cut -c1-400 | tee -a "$OUT/summary.txt"
note "RESULT: $([ $RC -eq 0 ] && echo passed || echo failed)"
exit $RC
