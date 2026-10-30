#!/bin/bash
# CPU job for the long-run lists (docs/data-lists.md): redraw the lists and compare hashes, check that fairchem builds exactly
# the listed rows (check_pilot_data.py verify, through pilot_data_overrides.sh), measure both lists (check_pilot_data.py stats)
# and compare them with the pilot lists (make_long_manifests.py compare). The existing index files in $OMOL_DATA_ROOT/pilot
# cover every row of each split, so no new index file is needed. Writes only under $RUN. Exits 1 if any stage failed.
# Usage: prep_long_data.sh [REPO]
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
export OMOL_DATA_ROOT=$ROOT/data/omol25 OMOL_CACHE_DIR=$ROOT/cache
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
RUN=$ROOT/runs/prep_long_${SLURM_JOB_ID:-manual}
LOGS=$RUN/logs
SUMMARY=$RUN/summary.txt
WORKERS=${SLURM_NTASKS:-4}
mkdir -p "$LOGS" "$OMOL_CACHE_DIR"
cd "$REPO" || exit 1
FAILED=0
now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(now) $*" | tee -a "$SUMMARY"; }
stage() {   # stage NAME COMMAND...: full output to $LOGS/NAME.log, exit status to the summary
    local name=$1; shift
    note "START $name"
    "$@" > "$LOGS/$name.log" 2>&1
    local rc=$?
    note "END   $name exit $rc (log: $LOGS/$name.log)"
    [ $rc -ne 0 ] && FAILED=1
    return $rc
}

PILOT_TRAIN_LIST=$REPO/manifests/long_train_1200k.txt PILOT_VAL_LIST=$REPO/manifests/select_val_5k.txt source "$REPO/slurm/pilot_data_overrides.sh"
note "prep long data: repo $REPO, $WORKERS workers, lists $(basename "$TRAIN_LIST") and $(basename "$VAL_LIST")"

stage redraw $PY -u -B scripts/make_long_manifests.py write --out-dir "$RUN/redraw"
for f in long_train_1200k.txt select_val_5k.txt; do
    a=$(sha256sum < "$REPO/manifests/$f" | cut -c1-64); b=$(sha256sum < "$RUN/redraw/$f" 2>/dev/null | cut -c1-64)
    if [ "$a" = "$b" ]; then note "PASS  redraw on this node gives the same $f ($a)"; else note "FAIL  redraw differs for $f"; FAILED=1; fi
done

stage verify_samples $PY -u -B scripts/check_pilot_data.py verify --samples manifests/long_samples.json \
    --train-rows "$TRAIN_LIST" --val-rows "$VAL_LIST" "${DATA_OVERRIDES[@]}"
grep -a -E 'PASS|FAIL|RESULT' "$LOGS/verify_samples.log" | tee -a "$SUMMARY"
stage stats_long $PY -u -B scripts/check_pilot_data.py stats --src "$OMOL_DATA_ROOT/train_4M" --rows "$TRAIN_LIST" \
    --out "$RUN/stats_long_1200k.json" --sids "$RUN/sids_long_1200k.txt" --workers "$WORKERS"
stage stats_select $PY -u -B scripts/check_pilot_data.py stats --src "$OMOL_DATA_ROOT/val" --rows "$VAL_LIST" \
    --out "$RUN/stats_select_5k.json" --sids "$RUN/sids_select_5k.txt" --workers "$WORKERS"
stage compare $PY -u -B scripts/make_long_manifests.py compare --data-root "$OMOL_DATA_ROOT" --out "$RUN/compare.json" \
    --train-sources "$OMOL_DATA_ROOT/pilot/sids_train_200k.txt" --workers "$WORKERS"
note "RESULT: $([ $FAILED -eq 0 ] && echo all stages passed || echo at least one stage failed)"
exit $FAILED
