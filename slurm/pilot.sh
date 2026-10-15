#!/bin/bash
# One pilot run: train a variant on the real OMol25 data, then check the size consistency of its final checkpoint.
# Usage: pilot.sh dense|global|local6
# Settings come from environment variables; the job files set them (agreed by the team on 2026-10-11):
#   PILOT_RUN_PREFIX (pilot; use another prefix for a test)  PILOT_STEPS (50000)  PILOT_SEED (the Slurm array task id, 0 to 2, else 0)  PILOT_EVAL_EVERY, PILOT_CKPT_EVERY (steps/10)
# Data: the fixed lists manifests/pilot_train_200k.txt and pilot_val_5k.txt, the same for every variant and seed. The job
# stops before training if the datasets do not hold exactly those rows. Run slurm/prep_pilot_data.job once before the pilots.
# For a short test with a random sample instead, set PILOT_TRAIN_SAMPLES and PILOT_VAL_SAMPLES.
# Every stage writes its full output under $RUN/logs and a line to $RUN/summary.txt; the job exits 1 if a stage failed.
# GPU: Dense and Global on a V100 with 32 GB (v100-32) or larger, Local-6 on an H100 (h100-80, the team's choice for memory headroom)
# or an L40S (l40s-48).
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
VARIANT=${1:?usage: pilot.sh dense|global|local6}
case "$VARIANT" in dense|global) MIN_MIB=31000 ;; local6) MIN_MIB=44000 ;; *) echo "unknown variant $VARIANT"; exit 2 ;; esac
ROOT=/ocean/projects/che260063p/shared
REPO=${PILOT_REPO:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
STEPS=${PILOT_STEPS:-50000}
SEED=${PILOT_SEED:-${SLURM_ARRAY_TASK_ID:-0}}
TRAIN_N=${PILOT_TRAIN_SAMPLES:-}
VAL_N=${PILOT_VAL_SAMPLES:-}
EVAL_EVERY=${PILOT_EVAL_EVERY:-$(( STEPS / 10 > 0 ? STEPS / 10 : 1 ))}
CKPT_EVERY=${PILOT_CKPT_EVERY:-$EVAL_EVERY}
RUN=$ROOT/runs/${PILOT_RUN_PREFIX:-pilot}_${VARIANT}_s${SEED}_${SLURM_JOB_ID:-manual}
LOGS=$RUN/logs
SUMMARY=$RUN/summary.txt
export OMOL_DATA_ROOT=$ROOT/data/omol25 OMOL_CACHE_DIR=$ROOT/cache
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
mkdir -p "$LOGS" "$OMOL_CACHE_DIR" "$RUN/train"
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

if [ -n "$TRAIN_N" ]; then
    DATA_NOTE="random sample: train $TRAIN_N, val $VAL_N"
    DATA_OVERRIDES=("train_dataset.dataset_configs.omol.splits.train.sample_n=$TRAIN_N" "val_dataset.dataset_configs.omol.splits.val.sample_n=$VAL_N")
else
    source "$REPO/slurm/pilot_data_overrides.sh"
    DATA_NOTE="fixed lists: $(basename "$TRAIN_LIST") sha256 $(sha256sum "$TRAIN_LIST" | cut -c1-16), $(basename "$VAL_LIST") sha256 $(sha256sum "$VAL_LIST" | cut -c1-16)"
fi
note "pilot $VARIANT: steps $STEPS, seed $SEED, $DATA_NOTE, eval and checkpoint every $EVAL_EVERY and $CKPT_EVERY steps"
{ echo "variant=$VARIANT steps=$STEPS seed=$SEED data=$DATA_NOTE eval_every=$EVAL_EVERY ckpt_every=$CKPT_EVERY"
  echo "job=${SLURM_JOB_ID:-manual} array_task=${SLURM_ARRAY_TASK_ID:-none} user=$USER host=$(hostname) repo=$REPO"
  echo "config_files_sha256=$(cat "$REPO"/slurm/pilot.sh "$REPO"/configs/train.yaml "$REPO"/configs/variant/$VARIANT.yaml | sha256sum | cut -c1-16)"; } > "$RUN/settings.txt"

GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
note "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $GPU_MIB MiB"
if [ "${GPU_MIB:-0}" -lt "$MIN_MIB" ]; then
    note "RESULT: this GPU is too small for $VARIANT (need at least $MIN_MIB MiB: v100-32 for Dense and Global, h100-80 or l40s-48 for Local-6)"
    exit 1
fi

if [ -z "$TRAIN_N" ]; then
    stage verify_samples $PY -u -B scripts/check_pilot_data.py verify --train-rows "$TRAIN_LIST" --val-rows "$VAL_LIST" "${DATA_OVERRIDES[@]}"
    grep -a -E 'PASS|FAIL|RESULT' "$LOGS/verify_samples.log" | tee -a "$SUMMARY"
    if [ $FAILED -ne 0 ]; then note "RESULT: the datasets do not match the fixed sample lists; no training was started"; exit 1; fi
fi

( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$LOGS/gpu_memory.txt"; sleep 5; done ) &
POLL=$!
START=$(date +%s)
stage train env OMOL_RUN_DIR="$RUN/train" $PY -u -B -m training.run --mode train variant=$VARIANT job.run_name=pilot_$VARIANT \
    job.seed=$SEED training.steps=$STEPS training.eval_every=$EVAL_EVERY training.ckpt_every=$CKPT_EVERY \
    training.max_saved_checkpoints=$(( STEPS / CKPT_EVERY + 2 )) "${DATA_OVERRIDES[@]}"
kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
note "training wall time $(( $(date +%s) - START )) s, peak GPU memory $(sort -n "$LOGS/gpu_memory.txt" | tail -1) MiB"
grep -a -E "'train/loss'" "$LOGS/train.log" | tail -1 | sed -E "s/.*'train\/loss': ([0-9.]+).*'train\/step': ([0-9]+).*atoms_per_second\(approx\)': ([0-9.]+).*/last train step \2 loss \1 atoms per second \3/" | tee -a "$SUMMARY"
grep -a -E 'val/loss|omol_energy,per_atom_mae|omol_forces,mae' "$LOGS/train.log" | tail -3 | tee -a "$SUMMARY"
grep -a 'OOM: ' "$LOGS/train.log" | cut -c1-300 | head -5 | tee -a "$SUMMARY"
grep -a -E 'OOM events' "$LOGS/train.log" | tail -1 | sed 's/.*OOM events/OOM events/' | tee -a "$SUMMARY"
grep -a -E 'Traceback|Error' "$LOGS/train.log" | head -3 | tee -a "$SUMMARY"

CKPT=$(ls "$RUN"/train/*/checkpoints/final/inference_ckpt.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    note "final checkpoint: $CKPT"
    stage size_consistency $PY -u -B audit/check_checkpoint.py "$CKPT"
    grep -a -E 'Delta|FAILED|Error' "$LOGS/size_consistency.log" | sed 's#.*/checkpoints#checkpoints#' | tee -a "$SUMMARY"
else
    note "no final checkpoint found"; FAILED=1
fi
find "$RUN/train" -maxdepth 5 -name 'offline-run-*' | tee -a "$SUMMARY"
note "RESULT: $([ $FAILED -eq 0 ] && echo all stages passed || echo at least one stage failed)"
exit $FAILED
