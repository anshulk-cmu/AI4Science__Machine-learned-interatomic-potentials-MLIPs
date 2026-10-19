#!/bin/bash
# Router check of one MoLE pilot run (scripts/router_check.py): router and expert statistics at step 0, step 10,000 and
# final, and the fixed-router ablation on the 5,000 validation structures. Inference only, one GPU.
# Usage: router_check.sh TASK [REPO]; TASK 0 to 5 = global seeds 0, 1, 2, then local6 seeds 0, 1, 2 (the array task id).
# Results go to $OUT/results, outside the repository, so a later syncrepo cannot overwrite them (guide B9 row 16).
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
TASK=${1:?usage: router_check.sh TASK [REPO]}
ROOT=/ocean/projects/che260063p/shared
REPO=${2:-$ROOT/repo_router/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
VARIANT=$([ "$TASK" -lt 3 ] && echo global || echo local6)
SEED=$(( TASK % 3 ))
RUN_DIR=$(ls -d $ROOT/runs/pilot_${VARIANT}_s${SEED}_[0-9]* | grep -E "/pilot_${VARIANT}_s${SEED}_[0-9]+$")
[ "$(echo "$RUN_DIR" | wc -l)" -eq 1 ] || { echo "expected one run folder for $VARIANT seed $SEED, got: $RUN_DIR"; exit 2; }
OUT=$ROOT/runs/router_check_${SLURM_ARRAY_JOB_ID:-manual}
WORK=$OUT/work_${VARIANT}_s${SEED}
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
mkdir -p "$OUT/logs" "$OUT/results" "$WORK"
now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(now) [$VARIANT s$SEED] $*" | tee -a "$OUT/summary.txt"; }

# work in a copy (the loader imports training/ and local_routing/ to rebuild the Local-6 model)
cp -r "$REPO/scripts" "$REPO/training" "$REPO/local_routing" "$REPO/configs" "$REPO/manifests" "$WORK/"
cd "$WORK"
note "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1); run $RUN_DIR; repo $REPO"
LOG=$OUT/logs/router_check_${VARIANT}_s${SEED}.log
note "START router_check"
$PY -u -B scripts/router_check.py run --run-dir "$RUN_DIR" --out-dir "$OUT/results" > "$LOG" 2>&1
RC=$?
note "END   router_check exit $RC"
grep -a -E '^pilot_|^step |^  [a-z]+: |wrote|Error' "$LOG" | cut -c1-300 | tee -a "$OUT/summary.txt"
note "RESULT: $([ $RC -eq 0 ] && echo passed || echo failed)"
exit $RC
