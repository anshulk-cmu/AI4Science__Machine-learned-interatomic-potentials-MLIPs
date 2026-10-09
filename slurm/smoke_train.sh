#!/bin/bash
# Smoke test on one GPU: environment, unit tests, 20 training steps of each variant on a sample of the real OMol25 data,
# an out-of-memory drill (the same run with the GPU memory capped, which must skip batches and still finish), the
# size-consistency check on each checkpoint, and the W&B offline runs.
# Settings by environment variable: VARIANTS (default "dense global local6"), RUN_TESTS (default 1), DRILL_FRACTION
# (default 0.35 of the GPU memory). Dense and Global belong on a V100-32, Local-6 on an L40S (docs/TRAINING.md).
# Every stage writes its full output to its own file under $RUN/logs; $RUN/summary.txt gets one line per stage and the
# key numbers. The job exits 1 if any stage failed, so Slurm and the email report the true outcome.
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
VARIANTS=${VARIANTS:-dense global local6}
RUN_TESTS=${RUN_TESTS:-1}
DRILL_FRACTION=${DRILL_FRACTION:-0.35}
RUN=$ROOT/runs/smoke_${SLURM_JOB_ID:-manual}
export OMOL_DATA_ROOT=$ROOT/data/omol25 OMOL_CACHE_DIR=$ROOT/cache
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
LOGS=$RUN/logs
SUMMARY=$RUN/summary.txt
mkdir -p "$LOGS" "$OMOL_CACHE_DIR"
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

environment() {
    hostname; date; echo "repo $REPO"; nvidia-smi; $PY -V
    $PY -c "import torch, fairchem.core, wandb; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    $PY -m pip freeze | grep -i -E '^(torch|fairchem-core|numpy|ase|pymatgen|wandb|hydra-core)=='
    env | grep -E '^(SLURM_JOB|SLURM_CPUS|OMOL|WANDB|PYTHONNOUSERSITE)' | sort
    find "$REPO" -type f -not -name '*.pyc' -not -path '*/results/*' -exec sha256sum {} + | sort -k2 | sha256sum
}
stage environment environment

if [ "$RUN_TESTS" = 1 ]; then
    stage unit_tests $PY -u -B -m unittest discover -s tests -v
    grep -a -E '^(Ran |OK|FAILED)' "$LOGS/unit_tests.log" | tee -a "$SUMMARY"
fi

for V in $VARIANTS; do
    mkdir -p "$RUN/$V"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$LOGS/gpu_memory_$V.txt"; sleep 2; done ) &
    POLL=$!
    stage "train_$V" env OMOL_RUN_DIR="$RUN/$V" $PY -u -B -m training.run --mode train variant=$V job.run_name=smoke_$V \
        training.steps=20 train_dataset.dataset_configs.omol.splits.train.sample_n=256 \
        val_dataset.dataset_configs.omol.splits.val.sample_n=64
    kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
    note "peak GPU memory $V: $(sort -n "$LOGS/gpu_memory_$V.txt" | tail -1) MiB"
    grep -a -E "'train/loss'" "$LOGS/train_$V.log" | tail -1 | sed -E "s/.*'train\/loss': ([0-9.]+).*'train\/step': ([0-9]+).*/last train step \2 loss \1/" | tee -a "$SUMMARY"
    grep -a -E 'val/loss|omol_energy,per_atom_mae|omol_forces,mae' "$LOGS/train_$V.log" | head -3 | tee -a "$SUMMARY"
    grep -a -E 'Traceback|Error' "$LOGS/train_$V.log" | head -3 | tee -a "$SUMMARY"

    # OOM drill: cap the GPU memory so some batches cannot fit; the run must skip them and finish
    mkdir -p "$RUN/${V}_drill"
    stage "oom_drill_$V" env OMOL_RUN_DIR="$RUN/${V}_drill" OMOL_GPU_MEMORY_FRACTION=$DRILL_FRACTION         $PY -u -B -m training.run --mode train variant=$V job.run_name=drill_$V training.steps=30         train_dataset.dataset_configs.omol.splits.train.sample_n=256         val_dataset.dataset_configs.omol.splits.val.sample_n=64
    SKIPPED=$(grep -a -c 'OOM: skipped' "$LOGS/oom_drill_$V.log")
    note "OOM drill $V (memory capped to $DRILL_FRACTION): $SKIPPED skipped batches; $(grep -a 'OOM-skipped batches' "$LOGS/oom_drill_$V.log" | tail -1 | sed 's/.*OOM-skipped/OOM-skipped/')"
    [ "$SKIPPED" -eq 0 ] && note "DRILL INCONCLUSIVE for $V: no batch reached the cap, lower DRILL_FRACTION"
done

for V in $VARIANTS; do
    CKPT=$(ls "$RUN/$V"/*/checkpoints/final/inference_ckpt.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then note "no checkpoint for $V"; FAILED=1; continue; fi
    stage "size_consistency_$V" $PY -u -B audit/check_checkpoint.py "$CKPT"
    grep -a -E 'Delta|FAILED|Error' "$LOGS/size_consistency_$V.log" | tee -a "$SUMMARY"
    stage "wandb_$V" find "$RUN/$V" -maxdepth 5 -name 'offline-run-*'
    cat "$LOGS/wandb_$V.log" | tee -a "$SUMMARY"
done

note "RESULT: $([ $FAILED -eq 0 ] && echo all stages passed || echo at least one stage failed)"
echo "full logs: $LOGS  summary: $SUMMARY"
exit $FAILED
