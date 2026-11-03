#!/bin/bash
# CPU job for the 70/20/10 split (docs/data-split.md). 1: read the group keys of every row of train_4M and val into
# $OMOL_DATA_ROOT/split/keys_<split>.npz (skipped when the file exists). 2, when the code copy holds the lists: redraw them on
# this node and compare hashes, check that fairchem builds exactly the listed rows (train from split_train_70, validation
# from split_val_monitor_5k, both read from train_4M), and run make_split_manifests.py check. Writes only under
# $OMOL_DATA_ROOT/split and $RUN. Exits 1 if any stage failed.
# Usage: prep_split_data.sh [REPO]
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
export OMOL_DATA_ROOT=$ROOT/data/omol25 OMOL_CACHE_DIR=$ROOT/cache
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
OUT=$OMOL_DATA_ROOT/split
RUN=$ROOT/runs/prep_split_${SLURM_JOB_ID:-manual}
LOGS=$RUN/logs
SUMMARY=$RUN/summary.txt
WORKERS=${SLURM_NTASKS:-4}
mkdir -p "$LOGS" "$OUT" "$OMOL_CACHE_DIR"
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

note "prep split data: repo $REPO, $WORKERS workers, tables in $OUT"
for split in train_4M val; do
    if [ -f "$OUT/keys_$split.npz" ]; then note "keys_$split.npz exists, not read again"; continue; fi
    stage "extract_$split" $PY -u -B scripts/make_split_manifests.py extract --data-root "$OMOL_DATA_ROOT" --split $split \
        --out "$OUT/keys_$split.npz" --workers "$WORKERS" && tail -1 "$LOGS/extract_$split.log" | tee -a "$SUMMARY"
done

if [ -f manifests/split_samples.json ]; then
    stage redraw $PY -u -B scripts/make_split_manifests.py write --table "$OUT/keys_train_4M.npz" --out-dir "$RUN/redraw" \
        --assignment "$RUN/redraw/split_assignment.npz"
    for f in $(cd manifests && ls split_*.txt) split_samples.json; do
        a=$(sha256sum < "manifests/$f" | cut -c1-64); b=$(sha256sum < "$RUN/redraw/$f" 2>/dev/null | cut -c1-64)
        if [ "$a" = "$b" ]; then note "PASS  redraw on this node gives the same $f ($a)"; else note "FAIL  redraw differs for $f"; FAILED=1; fi
    done
    PILOT_TRAIN_LIST=$REPO/manifests/split_train_70.txt PILOT_VAL_LIST=$REPO/manifests/split_val_monitor_5k.txt \
        PILOT_VAL_SPLIT=train_4M source "$REPO/slurm/pilot_data_overrides.sh"
    stage verify_samples $PY -u -B scripts/check_pilot_data.py verify --samples manifests/split_samples.json \
        --train-rows "$TRAIN_LIST" --val-rows "$VAL_LIST" "${DATA_OVERRIDES[@]}"
    grep -a -E 'PASS|FAIL|RESULT' "$LOGS/verify_samples.log" | tee -a "$SUMMARY"
    # The same lists with the validation rows read from the official val folder must fail the check.
    PILOT_TRAIN_LIST=$TRAIN_LIST PILOT_VAL_LIST=$VAL_LIST PILOT_VAL_SPLIT=val source "$REPO/slurm/pilot_data_overrides.sh"
    $PY -u -B scripts/check_pilot_data.py verify --samples manifests/split_samples.json --train-rows "$TRAIN_LIST" \
        --val-rows "$VAL_LIST" "${DATA_OVERRIDES[@]}" > "$LOGS/verify_wrong_folder.log" 2>&1
    if [ $? -ne 0 ] && grep -q "FAIL  split_val_monitor_5k.txt: rows read from the train_4M folder" "$LOGS/verify_wrong_folder.log"; then
        note "PASS  verify fails when the validation rows are read from val (log: $LOGS/verify_wrong_folder.log)"
    else note "FAIL  verify did not catch validation rows read from val"; FAILED=1; fi
    stage check $PY -u -B scripts/make_split_manifests.py check --table "$OUT/keys_train_4M.npz" \
        --val-table "$OUT/keys_val.npz" --out "$RUN/checks.json"
    grep -a -E 'PASS|FAIL|RESULT' "$LOGS/check.log" | tee -a "$SUMMARY"
else
    note "no manifests/split_samples.json in this code copy: lists, verify and check skipped"
fi
note "RESULT: $([ $FAILED -eq 0 ] && echo all stages passed || echo at least one stage failed)"
exit $FAILED
