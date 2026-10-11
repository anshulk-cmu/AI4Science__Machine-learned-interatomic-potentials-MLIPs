#!/bin/bash
# One-time CPU job before the pilots: the index files for the fixed sample lists, a check that fairchem builds exactly
# those datasets, and statistics of the training rows (neighbours, force and energy scales, elements, charges, spins).
# Results: $OMOL_DATA_ROOT/pilot/ (index files, stats_*.json, structure ids) and $RUN/summary.txt with the full logs.
# Reads the data and writes only under $OMOL_DATA_ROOT/pilot and $RUN. The job exits 1 if any stage failed.
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
export OMOL_DATA_ROOT=$ROOT/data/omol25 OMOL_CACHE_DIR=$ROOT/cache
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
RUN=$ROOT/runs/prep_pilot_${SLURM_JOB_ID:-manual}
LOGS=$RUN/logs
SUMMARY=$RUN/summary.txt
OUT=$OMOL_DATA_ROOT/pilot
WORKERS=${SLURM_NTASKS:-4}
mkdir -p "$LOGS" "$OUT" "$OMOL_CACHE_DIR"
cd "$REPO"
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

source "$REPO/slurm/pilot_data_overrides.sh"
note "prep pilot data: repo $REPO, $WORKERS workers, lists $(basename "$TRAIN_LIST") and $(basename "$VAL_LIST")"
stage prep $PY -u -B scripts/check_pilot_data.py prep && cat "$LOGS/prep.log" | tee -a "$SUMMARY"
stage verify_samples $PY -u -B scripts/check_pilot_data.py verify --train-rows "$TRAIN_LIST" --val-rows "$VAL_LIST" "${DATA_OVERRIDES[@]}"
grep -a -E 'PASS|FAIL|RESULT' "$LOGS/verify_samples.log" | tee -a "$SUMMARY"
stage stats_train $PY -u -B scripts/check_pilot_data.py stats --src "$OMOL_DATA_ROOT/train_4M" --rows "$TRAIN_LIST" \
    --out "$OUT/stats_train_200k.json" --sids "$OUT/sids_train_200k.txt" --workers "$WORKERS"
stage stats_val $PY -u -B scripts/check_pilot_data.py stats --src "$OMOL_DATA_ROOT/val" --rows "$VAL_LIST" \
    --out "$OUT/stats_val_5k.json" --sids "$OUT/sids_val_5k.txt" --workers "$WORKERS"
for f in stats_train_200k stats_val_5k; do
    [ -f "$OUT/$f.json" ] && grep -E '"(structures|atoms|mean_atoms_per_structure|mean_neighbors_per_atom|max_neighbors_per_atom|force_norm_rms_per_atom|force_component_rms|energy_residual_per_atom_mean|energy_residual_per_atom_rms)"' "$OUT/$f.json" | sed "s/^/$f /" | tee -a "$SUMMARY"
done
note "RESULT: $([ $FAILED -eq 0 ] && echo all stages passed || echo at least one stage failed)"
exit $FAILED
