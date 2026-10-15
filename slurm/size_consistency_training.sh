#!/bin/bash
# Size consistency along training: Delta for the Al and Ni slabs at every saved checkpoint (step 0 to final) of the
# nine pilot runs, with scripts/size_consistency_along_training.py. Inference only, one GPU.
# Results go to $OUT/work/results, outside the repository, so a later syncrepo cannot overwrite them (guide B9 row 16).
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
OUT=$ROOT/runs/size_consistency_training_${SLURM_JOB_ID:-manual}
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
mkdir -p "$OUT/logs"
now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(now) $*" | tee -a "$OUT/summary.txt"; }

# work in a copy (the loader imports training/ and local_routing/ to rebuild the Local-6 model)
WORK=$OUT/work
mkdir -p "$WORK/results"
cp -r "$REPO/audit" "$REPO/scripts" "$REPO/training" "$REPO/local_routing" "$WORK/"
cd "$WORK"
note "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1); repo $REPO; results $WORK/results"
note "START size_consistency_training"
$PY -u -B scripts/size_consistency_along_training.py --out-dir "$WORK/results" > "$OUT/logs/size_consistency_training.log" 2>&1
RC=$?
note "END   size_consistency_training exit $RC"
grep -a -E 'runs:|wrote|self-check|FAILED|Error' "$OUT/logs/size_consistency_training.log" | cut -c1-300 | tee -a "$OUT/summary.txt"
note "RESULT: $([ $RC -eq 0 ] && echo passed || echo failed)"
exit $RC
