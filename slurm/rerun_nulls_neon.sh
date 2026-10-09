#!/bin/bash
# Rerun the null tests and the neon probe for the float64 models after the ASE cache fix (the first versions of the
# committed null files compared a cached result with itself). The old files are moved to results/old_before_cachefix.
# UMA-M ran in float32 on a cloud GPU and is not rerun. ORB runs on an L40S (slurm/rerun_orb.sh). UMA weights come from the private folder $UMA (gated model).
# Every stage writes its own log under $OUT/logs; the job exits 1 if any stage failed. Run it on a V100-32.
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
UMA=/ocean/projects/che260063p/akumar45/uma
PYM=$ROOT/envs/mlip/bin/python
PYC=$ROOT/envs/mace/bin/python
OUT=$ROOT/runs/rerun_${SLURM_JOB_ID:-manual}
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

cd "$REPO/audit"
OLD=$REPO/results/old_before_cachefix
mkdir -p "$OLD"
for f in ../results/nulls_*.csv ../results/neon_*.csv; do
    case "$f" in *uma-m-1p1*|*orb-v3*) continue ;; esac
    [ -e "$f" ] && mv "$f" "$OLD/"
done
note "old null and neon files moved to $OLD"

stage nulls_medium $PYC -u nulls.py --family mace --model medium
stage neon_medium $PYC -u neon_probe.py --family mace --model medium
# ORB is not run here: its graph builder (NVIDIA Warp with CUDA 13) needs sm_75 or higher and fails on a V100; see rerun_orb.sh
for T in omat omol; do
    for L in uma-s-1p1 uma-s-1p2; do
        stage nulls_${L}_$T $PYM -u nulls.py --family uma --model $UMA/$L.pt --label $L --task $T
        stage neon_${L}_$T $PYM -u neon_probe.py --family uma --model $UMA/$L.pt --label $L --task $T
    done
    stage nulls_uma-s-1p2-v1.1_$T $PYM -u nulls.py --family uma --model $UMA/uma-s-1p2.pt --label uma-s-1p2-v1.1 --task $T --model-version 1.1
    stage nulls_uma-s-1p1-v1_$T $PYM -u nulls.py --family uma --model $UMA/uma-s-1p1.pt --label uma-s-1p1-v1 --task $T --model-version 1.0
done
for f in "$LOGS"/nulls_*.log; do echo "$(basename $f .log): $(grep -a 'sigma_E' $f | tail -1)" | tee -a "$SUMMARY"; done
ls ../results/nulls_*.csv ../results/neon_*.csv | tee -a "$SUMMARY"
note "RESULT: $([ $FAILED -eq 0 ] && echo all stages passed || echo at least one stage failed)"
exit $FAILED
