#!/bin/bash
# Resume test (docs/resume-test.md): short runs on the fixed pilot lists. Some are stopped on purpose and resumed by a later job.
# Usage: resume_test.sh VARIANT SPEC...   SPEC = NAME[:term:STEP|:kill:STEP][:small]
#   NAME       run folder $RESUME_TEST_DIR/NAME and fairchem timestamp_id NAME. The same NAME in a later job finds the last complete
#              checkpoint of that folder by itself (fairchem TrainEvalRunner.load_state(None)) and resumes from it.
#   term:STEP  send SIGTERM once step STEP has run, as Slurm does at the time limit (fairchem then exits 0 without a new checkpoint)
#   kill:STEP  send SIGKILL once step STEP has run, then leave a half-written checkpoint folder (data file, no .metadata), as a kill
#              during a checkpoint write would
#   small      epoch test: train on every 100th row of the pilot training list (2,000 rows), validate on its first 200 rows
# A run with a DONE marker (final checkpoint and every step logged) is skipped. Probe files: NAME/probe/seg_<jobid>.csv.
# Settings: RESUME_TEST_DIR (shared/runs/resume_test), RESUME_REPO (a copy of the code), RESUME_STEPS (400), RESUME_CKPT_EVERY (100), RESUME_SEED (0).
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
VARIANT=${1:?usage: resume_test.sh dense|local6 SPEC...}; shift
case "$VARIANT" in dense) MIN_MIB=31000 ;; local6) MIN_MIB=44000 ;; *) echo "unknown variant $VARIANT"; exit 2 ;; esac
ROOT=/ocean/projects/che260063p/shared
TEST_DIR=${RESUME_TEST_DIR:-$ROOT/runs/resume_test}
REPO=${RESUME_REPO:-$TEST_DIR/code}
PY=$ROOT/envs/mlip/bin/python
STEPS=${RESUME_STEPS:-400}
CKPT=${RESUME_CKPT_EVERY:-100}
SEED=${RESUME_SEED:-0}
JOB=${SLURM_JOB_ID:-manual}
SUMMARY=$TEST_DIR/summary.txt
export OMOL_DATA_ROOT=$ROOT/data/omol25 OMOL_CACHE_DIR=$ROOT/cache
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
mkdir -p "$TEST_DIR/logs"
cd "$REPO" || exit 1
FAILED=0
now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(now) job $JOB: $*" | tee -a "$SUMMARY"; }

GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
note "variant $VARIANT, steps $STEPS, checkpoint every $CKPT, seed $SEED, host $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $GPU_MIB MiB, specs: $*"
if [ "${GPU_MIB:-0}" -lt "$MIN_MIB" ]; then note "RESULT: GPU too small for $VARIANT"; exit 1; fi

source "$REPO/slurm/pilot_data_overrides.sh"
PILOT_OVERRIDES=("${DATA_OVERRIDES[@]}")
$PY -u -B scripts/check_pilot_data.py verify --train-rows "$TRAIN_LIST" --val-rows "$VAL_LIST" "${PILOT_OVERRIDES[@]}" > "$TEST_DIR/logs/verify_$JOB.log" 2>&1
rc=$?; grep -a -E 'PASS|FAIL|RESULT' "$TEST_DIR/logs/verify_$JOB.log" | tee -a "$SUMMARY"
if [ $rc -ne 0 ]; then note "RESULT: the datasets do not match the fixed pilot lists; nothing was trained"; exit 1; fi

SMALL_TRAIN=$TEST_DIR/small_train_2k.txt SMALL_VAL=$TEST_DIR/small_val_200.txt
[ -s "$SMALL_TRAIN" ] || awk 'NR % 100 == 1' "$TRAIN_LIST" > "$SMALL_TRAIN"
[ -s "$SMALL_VAL" ] || head -200 "$VAL_LIST" > "$SMALL_VAL"
PILOT_TRAIN_LIST=$SMALL_TRAIN PILOT_VAL_LIST=$SMALL_VAL source "$REPO/slurm/pilot_data_overrides.sh"
SMALL_OVERRIDES=("${DATA_OVERRIDES[@]}")

last_step() { tail -n 1 "$1" 2>/dev/null | cut -d, -f1; }

run() {   # run SPEC: one segment of one run
    local name stop="" at="" small="" spec=$1
    IFS=: read -r name stop at small <<< "$spec"
    [ "$stop" = small ] && { small=small; stop=""; }
    local run=$TEST_DIR/$name probe=$TEST_DIR/$name/probe/seg_$JOB.csv log=$TEST_DIR/logs/${name}_$JOB.log
    local ckdir=$run/train/$name/checkpoints
    if [ -f "$run/DONE" ]; then note "$name: DONE marker present, skipped"; return; fi
    local data=("${PILOT_OVERRIDES[@]}"); [ -n "$small" ] && data=("${SMALL_OVERRIDES[@]}")
    mkdir -p "$run/probe"
    local store=$run/train/$name/.distributed-shared-file   # torch FileStore; a killed process leaves it behind
    note "$name: start ($(ls -d "$ckdir"/*/.metadata 2>/dev/null | wc -l) complete checkpoints on disk, store file $([ -e "$store" ] && echo present || echo absent)), stop ${stop:-none} ${at}, data ${small:-pilot lists}"
    env OMOL_RUN_DIR="$run/train" $PY -u -B -m training.run --mode train variant=$VARIANT job.run_name=resume_$name \
        +job.timestamp_id=$name job.seed=$SEED training.steps=$STEPS training.eval_every=1000000 training.ckpt_every=$CKPT \
        training.max_saved_checkpoints=$(( STEPS / CKPT + 4 )) \
        runner.train_eval_unit._target_=training.resume_probe.ProbeTrainEvalUnit "+runner.train_eval_unit.batch_log=$probe" \
        "${data[@]}" >> "$log" 2>&1 &
    local pid=$!
    if [ -n "$stop" ]; then
        local sig=TERM; [ "$stop" = kill ] && sig=KILL
        while kill -0 $pid 2>/dev/null; do
            local s; s=$(last_step "$probe")
            if [[ $s =~ ^[0-9]+$ ]] && (( s >= at )); then kill -$sig $pid; note "$name: sent SIG$sig after step $s"; break; fi
            sleep 1
        done
    fi
    wait $pid; local rc=$?
    note "$name: python exit $rc, last logged step $(last_step "$probe"), checkpoints: $(cd "$ckdir" 2>/dev/null && ls | tr '\n' ' ')"
    if [ "$stop" = kill ]; then
        local good; good=$(ls -d "$ckdir"/step_* | sort -t_ -k2 -n | tail -1)
        local half=$ckdir/step_$(( ($(basename "$good" | cut -d_ -f2) / CKPT + 1) * CKPT ))
        mkdir -p "$half" && head -c 1000000 "$good/__0_0.distcp" > "$half/__0_0.distcp" && cp "$good/resume.yaml" "$half/"
        note "$name: planted half-written checkpoint $(basename "$half") (data file cut to 1 MB, no .metadata)"
    elif [ -f "$ckdir/final/inference_ckpt.pt" ] && [ "$(last_step "$probe")" = $(( STEPS - 1 )) ]; then
        touch "$run/DONE"; note "$name: DONE"
    else
        [ "$stop" = term ] || FAILED=1
        note "$name: not finished"
    fi
}

for spec in "$@"; do run "$spec"; done
note "RESULT: $([ $FAILED -eq 0 ] && echo all runs behaved as planned || echo at least one run failed)"
exit $FAILED
