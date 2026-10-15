#!/bin/bash
# Rerun the ORB-v3 null tests and neon probe (float64) after the ASE cache fix. ORB builds its graph with NVIDIA Warp, which
# with CUDA 13 needs a GPU of compute capability 7.5 or higher: use an L40S (l40s-48) or an H100, not a V100.
# The new files are written to $OUT/work/results, not into the repository.
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
OUT=$ROOT/runs/rerun_orb_${SLURM_JOB_ID:-manual}
LOGS=$OUT/logs
SUMMARY=$OUT/summary.txt
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
mkdir -p "$LOGS" "$ROOT/cache"
FAILED=0
now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(now) $*" | tee -a "$SUMMARY"; }
stage() {
    local name=$1; shift
    note "START $name"
    "$@" > "$LOGS/$name.log" 2>&1
    local rc=$?
    note "END   $name exit $rc"
    [ $rc -ne 0 ] && FAILED=1
    return $rc
}
WORK=$OUT/work
mkdir -p "$WORK/results"
cp -r "$REPO/audit" "$WORK/audit"
cd "$WORK/audit"
note "writing results to $WORK/results"
stage nulls_orb-v3 $PY -u nulls.py --family orb --model orb-v3
stage neon_orb-v3 $PY -u neon_probe.py --family orb --model orb-v3
grep -a 'sigma_E' "$LOGS/nulls_orb-v3.log" | tee -a "$SUMMARY"
ls -l ../results | tee -a "$SUMMARY"
note "RESULT: $([ $FAILED -eq 0 ] && echo all stages passed || echo at least one stage failed)"
exit $FAILED
