#!/bin/bash
# Rerun the null tests of UMA-M-1.1 in float32 with the ASE cache fix. The committed UMA-M null files were made on a cloud GPU
# before the fix, so their repeat test N1 compares a cached result with itself, and they lack the large pads of N4.
# The new files are written to $OUT/work/results; the old ones stay in the repository until they are replaced.
# Same scheme as rerun_nulls_neon.sh. The weights are read from the private folder $UMA (gated model). Run it on a V100-32:
# the peak GPU memory was 7.8 GiB on the cloud GPU, and the host needs about 15 GiB of RAM to load the 10.4 GiB checkpoint.
# Every stage writes its own log under $OUT/logs; the job exits 1 if any stage failed.
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
UMA=/ocean/projects/che260063p/akumar45/uma
PYM=$ROOT/envs/mlip/bin/python
OUT=$ROOT/runs/rerun_uma_m_${SLURM_JOB_ID:-manual}
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

# work in a copy so that syncing the repository to Bridges-2 later cannot overwrite the new files
WORK=$OUT/work
mkdir -p "$WORK/results"
cp -r "$REPO/audit" "$WORK/audit"
cd "$WORK/audit"
note "writing results to $WORK/results; checkpoint $(ls -l $UMA/uma-m-1p1.pt | awk '{print $5}') bytes"
stage gpu nvidia-smi

for T in omat omol; do
    stage nulls_uma-m-1p1_$T $PYM -u nulls.py --family uma --model $UMA/uma-m-1p1.pt --label uma-m-1p1 --task $T --dtype float32
done
for f in "$LOGS"/nulls_*.log; do echo "$(basename $f .log): $(grep -a 'sigma_E' $f | tail -1)" | tee -a "$SUMMARY"; done
ls -l ../results | tee -a "$SUMMARY"
note "RESULT: $([ $FAILED -eq 0 ] && echo all stages passed || echo at least one stage failed)"
exit $FAILED
