#!/bin/bash
# One segment of a long run (docs/launch-checklist.md; the rules are docs/resume-test.md Section 6). slurm/submit_chain.sh
# chains the segments with afterany. Every segment runs this script with the same settings and the same run folder, and
# fairchem resumes from the last complete checkpoint because the run id is pinned (+job.timestamp_id).
# Usage: long_run.sh dense|global|local6|local12|element|local_global|global_frozen|dense_scaled
# Required, no default: LONG_STEPS (optimizer steps of the whole run), LONG_SEED, LONG_EVAL_EVERY, LONG_CKPT_EVERY,
#   LONG_SEGMENT_HOURS (the Slurm time limit of this segment, 1 to 48).
# Optional: LONG_STOP_MARGIN_SEC (1800: training stops this long before the limit, so this script can still write its markers),
#   LONG_SHARED (the project's shared folder), LONG_REPO, LONG_RUNS_DIR, LONG_PY.
# Data (docs/data-split.md): LONG_TRAIN_LIST (manifests/split_train_70.txt) for training, LONG_VAL_LIST
#   (manifests/split_val_monitor_5k.txt) for the validation every LONG_EVAL_EVERY steps, its rows read from the LONG_VAL_SPLIT
#   folder (train_4M). The test rows (split_test_10.txt) are never read here.
# Run folder <runs>/long_<variant>_s<seed>_<steps>: settings.txt (written once, checked by every later segment), summary.txt,
#   segments.txt (ordered training logs and W&B files, read by scripts/collect_final_metrics.py), logs/, train/<name>/checkpoints.
# The run is finished only when checkpoints/final/.metadata exists. Then this segment runs the size-consistency check and writes
#   DONE. A segment that finds DONE exits at once. A real failure writes FAILED, and every later segment exits at once.
# Exit 0: stopped by time as planned, or the run is done. 1: failure. 2: bad settings.
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
VARIANT=${1:-}
case "$VARIANT" in
    dense_scaled|dense|global|global_frozen) MIN_MIB=31000 ;;
    local6|element|local_global) MIN_MIB=44000 ;;
    local12) MIN_MIB=79000 ;;
    *) echo "usage: long_run.sh dense|global|local6|local12|element|local_global|global_frozen|dense_scaled"; exit 2 ;;
esac
for v in LONG_STEPS LONG_SEED LONG_EVAL_EVERY LONG_CKPT_EVERY LONG_SEGMENT_HOURS; do
    [[ ${!v:-} =~ ^(0|[1-9][0-9]*)$ ]] || { echo "long_run.sh: set $v to a whole number; it has no default"; exit 2; }
done
STEPS=$LONG_STEPS SEED=$LONG_SEED EVAL_EVERY=$LONG_EVAL_EVERY CKPT_EVERY=$LONG_CKPT_EVERY HOURS=$LONG_SEGMENT_HOURS
MARGIN=${LONG_STOP_MARGIN_SEC:-1800}
if (( STEPS < 1 || EVAL_EVERY < 1 || CKPT_EVERY < 1 || EVAL_EVERY > STEPS || CKPT_EVERY > STEPS
      || HOURS < 1 || HOURS > 48 || MARGIN >= HOURS * 3600 )); then
    echo "long_run.sh: need 1 <= LONG_EVAL_EVERY, LONG_CKPT_EVERY <= LONG_STEPS, 1 <= LONG_SEGMENT_HOURS <= 48, stop margin below the limit"
    exit 2
fi
START=$(date +%s)
ROOT=${LONG_SHARED:-/ocean/projects/che260063p/shared}
REPO=${LONG_REPO:-$ROOT/repo/AI4Science-MLIPs}
PY=${LONG_PY:-$ROOT/envs/mlip/bin/python}
NAME=long_${VARIANT}_s${SEED}_${STEPS}
RUN=${LONG_RUNS_DIR:-$ROOT/runs}/$NAME
TRAIN=$RUN/train
CKDIR=$TRAIN/$NAME/checkpoints
LOGS=$RUN/logs
SUMMARY=$RUN/summary.txt
JOB=${SLURM_JOB_ID:-manual}
export OMOL_DATA_ROOT=$ROOT/data/omol25 OMOL_CACHE_DIR=$ROOT/cache
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
mkdir -p "$LOGS" "$TRAIN" || exit 1
cd "$REPO" || exit 1
now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(now) job $JOB: $*" | tee -a "$SUMMARY"; }
fail() { echo "$(now) job $JOB: $*" > "$RUN/FAILED"; note "RESULT: FAILED: $*. The later segments of this chain exit at once."; exit 1; }
last_complete() { ls -d "$CKDIR"/step_*/.metadata 2>/dev/null | sed -E 's#.*/step_([0-9]+)/\.metadata$#\1#' | sort -n | tail -1; }
# A segment killed together with this script cannot list its own W&B file, so every segment lists the ones still missing.
# The folder names carry the start time, so sorted order is segment order.
record_wandb() {
    find "$TRAIN" -name '*.wandb' | sort | sed "s#^$RUN/##" | while read -r f; do
        grep -qxF "wandb $f" "$RUN/segments.txt" 2>/dev/null || echo "wandb $f" >> "$RUN/segments.txt"
    done
}

if [ -f "$RUN/DONE" ]; then note "DONE marker present: the run is finished, nothing to do"; exit 0; fi
if [ -f "$RUN/FAILED" ]; then note "FAILED marker present ($(cat "$RUN/FAILED")); delete it after the fix to continue"; exit 1; fi

PILOT_TRAIN_LIST=${LONG_TRAIN_LIST:-$REPO/manifests/split_train_70.txt} PILOT_VAL_LIST=${LONG_VAL_LIST:-$REPO/manifests/split_val_monitor_5k.txt} \
    PILOT_VAL_SPLIT=${LONG_VAL_SPLIT:-train_4M} source "$REPO/slurm/pilot_data_overrides.sh"
DATA_NOTE="fixed lists: $(basename "$TRAIN_LIST") sha256 $(sha256sum "$TRAIN_LIST" | cut -c1-16), $(basename "$VAL_LIST") sha256 $(sha256sum "$VAL_LIST" | cut -c1-16) from $VAL_SPLIT"
SETTINGS="variant=$VARIANT steps=$STEPS seed=$SEED data=$DATA_NOTE eval_every=$EVAL_EVERY ckpt_every=$CKPT_EVERY"
if [ ! -f "$RUN/settings.txt" ]; then
    { echo "$SETTINGS"; echo "first_job=$JOB user=${USER:-unknown} repo=$REPO"; } > "$RUN/settings.txt"
elif [ "$(head -1 "$RUN/settings.txt")" != "$SETTINGS" ]; then
    note "RESULT: these settings differ from the first segment's ($RUN/settings.txt); nothing was started"; exit 2
fi
CODE=$(cd "$REPO" && cat slurm/long_run.sh configs/train.yaml configs/variant/*.yaml training/*.py local_routing/*.py | sha256sum | cut -c1-16)
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
note "segment start: $NAME, limit $HOURS h, GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ${GPU_MIB} MiB, host $(hostname), code sha256 $CODE"
[ "${GPU_MIB:-0}" -ge "$MIN_MIB" ] || fail "this GPU is too small for $VARIANT (need at least $MIN_MIB MiB)"

record_wandb
if [ ! -f "$CKDIR/final/.metadata" ]; then
    $PY -u -B scripts/check_pilot_data.py verify --samples manifests/split_samples.json --samples manifests/long_samples.json \
        --samples manifests/pilot_samples.json --train-rows "$TRAIN_LIST" --val-rows "$VAL_LIST" "${DATA_OVERRIDES[@]}" \
        > "$LOGS/verify_$JOB.log" 2>&1
    rc=$?
    grep -a -E 'PASS|FAIL|RESULT' "$LOGS/verify_$JOB.log" | tee -a "$SUMMARY"
    [ $rc -eq 0 ] || fail "the datasets do not match the fixed lists (log: $LOGS/verify_$JOB.log)"

    rm -f "$TRAIN/$NAME/.distributed-shared-file"     # a killed segment leaves it; only one segment of a run is alive at a time
    BEFORE=$(last_complete); BEFORE=${BEFORE:-0}
    LIMIT=$(( START + HOURS * 3600 - MARGIN - $(date +%s) ))
    (( LIMIT > 0 )) || fail "no time is left for training; raise LONG_SEGMENT_HOURS or lower LONG_STOP_MARGIN_SEC"
    LOG=$LOGS/train_$JOB.log
    echo "log logs/train_$JOB.log" >> "$RUN/segments.txt"     # before training, so a killed segment keeps its log
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$LOGS/gpu_memory_$JOB.txt"; sleep 5; done ) &
    POLL=$!
    note "START train from complete step $BEFORE, stop by time after $LIMIT s (log: $LOG)"
    T0=$(date +%s)
    # Stop by time, never with runner.max_steps (the final checkpoint would carry the next epoch). --foreground sends SIGTERM to
    # the training process only, as in the resume test; fairchem catches it.
    timeout --foreground --signal=TERM --kill-after=600 "$LIMIT" env OMOL_RUN_DIR="$TRAIN" $PY -u -B -m training.run --mode train \
        variant=$VARIANT job.run_name=$NAME +job.timestamp_id=$NAME job.seed=$SEED training.steps=$STEPS \
        training.eval_every=$EVAL_EVERY training.ckpt_every=$CKPT_EVERY training.max_saved_checkpoints=$(( STEPS / CKPT_EVERY + 2 )) \
        "${DATA_OVERRIDES[@]}" > "$LOG" 2>&1
    RC=$?
    kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
    ELAPSED=$(( $(date +%s) - T0 ))
    AFTER=$(last_complete); AFTER=${AFTER:-0}
    record_wandb
    # The stage status is 0 only when the final checkpoint exists: fairchem exits 1 on its own SIGTERM path (resume-test.md Section 7).
    # timeout returns 124 when it stopped the run, also after its KILL. 137 (killed) counts as a stop by time only within 120 s of
    # the limit; earlier it is a failure, for example the kernel's out-of-memory killer.
    if [ -f "$CKDIR/final/.metadata" ]; then STATUS=0
    elif [ $RC -eq 124 ] || { [ $RC -eq 137 ] && (( ELAPSED >= LIMIT - 120 )); }; then STATUS=124
    else STATUS=$(( RC ? RC : 1 )); fi
    note "END   train exit $STATUS (training process $RC; 0 = final checkpoint written, 124 = stopped by time as planned), complete checkpoints up to step $AFTER, $ELAPSED s, peak GPU memory $(sort -n "$LOGS/gpu_memory_$JOB.txt" 2>/dev/null | tail -1) MiB"
    grep -a 'OOM: ' "$LOG" | cut -c1-300 | head -5 | tee -a "$SUMMARY"
    grep -a -E 'OOM events' "$LOG" | tail -1 | sed 's/.*OOM events/OOM events/' | tee -a "$SUMMARY"
    if [ $STATUS -eq 124 ]; then
        (( AFTER > BEFORE )) || fail "no new checkpoint in this segment (last complete step $BEFORE); LONG_CKPT_EVERY is too large for the segment time"
        note "RESULT: stopped by time; complete checkpoints up to step $AFTER of $STEPS; the next segment resumes from there"
        exit 0
    fi
    [ $STATUS -eq 0 ] || [ $RC -ne 137 ] || fail "the training process was killed (exit 137) after $ELAPSED s, $(( LIMIT - ELAPSED )) s before the stop by time; out of memory or killed by the system (log: $LOG)"
    [ $STATUS -eq 0 ] || { grep -a -E 'Traceback|Error' "$LOG" | head -3 | tee -a "$SUMMARY"; fail "training exited $RC without the final checkpoint (log: $LOG)"; }
fi

# The final checkpoint exists: the checks run here, once, never in a segment that is still training.
FAILED=0
LAST_LOG=$RUN/$(grep '^log ' "$RUN/segments.txt" | tail -1 | cut -d' ' -f2)
grep -a -E 'val/loss|omol_energy,per_atom_mae|omol_forces,mae' "$LAST_LOG" | tail -3 | tee -a "$SUMMARY"
CKPT=$CKDIR/final/inference_ckpt.pt
note "final checkpoint: $CKPT"
if [ "$VARIANT" = global_frozen ]; then
    $PY -u -B scripts/check_frozen_router.py "$CKDIR/step_0/inference_ckpt.pt" "$CKPT" > "$LOGS/router_frozen_$JOB.log" 2>&1 || FAILED=1
    grep -a -E 'PASS|FAIL' "$LOGS/router_frozen_$JOB.log" | tee -a "$SUMMARY"
fi
note "START size_consistency"
$PY -u -B audit/check_checkpoint.py "$CKPT" --gaps 8 10 15 25 100 > "$LOGS/size_consistency_$JOB.log" 2>&1 || FAILED=1
note "END   size_consistency (log: $LOGS/size_consistency_$JOB.log)"
grep -a -E 'Delta|FAILED|Error' "$LOGS/size_consistency_$JOB.log" | sed 's#.*/checkpoints#checkpoints#' | tee -a "$SUMMARY"
if [ $FAILED -ne 0 ]; then
    note "RESULT: at least one stage failed; the final checkpoint exists, so a new segment runs only these checks"
    exit 1
fi
note "RESULT: all stages passed"
date '+%Y-%m-%d %H:%M:%S' > "$RUN/DONE"
exit 0
