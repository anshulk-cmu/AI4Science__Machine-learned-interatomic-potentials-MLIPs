#!/bin/bash
# Smoke test on one GPU: the unit tests, then 20 training steps of Dense, Global and Local-6 on a sample of the
# real OMol25 data, then the size-consistency check on each saved checkpoint. Run it inside a Slurm job.
set -uo pipefail
ROOT=/ocean/projects/che260063p/akumar45
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
export OMOL_DATA_ROOT=$ROOT/data/omol25 OMOL_CACHE_DIR=$ROOT/cache OMOL_RUN_DIR=$ROOT/runs/smoke_${SLURM_JOB_ID:-manual}
mkdir -p "$OMOL_RUN_DIR" "$OMOL_CACHE_DIR"
cd "$REPO"
echo "== unit tests"
$PY -B -m unittest discover -s tests 2>&1 | tail -4
for V in dense global local6; do
    echo "== train $V"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OMOL_RUN_DIR/mem_$V.txt"; sleep 2; done ) &
    POLL=$!
    $PY -B -m training.run --mode train variant=$V job.run_name=smoke_$V training.steps=20 \
        train_dataset.dataset_configs.omol.splits.train.sample_n=256 \
        val_dataset.dataset_configs.omol.splits.val.sample_n=64 > "$OMOL_RUN_DIR/train_$V.log" 2>&1
    echo "exit $?, peak MiB $(sort -n "$OMOL_RUN_DIR/mem_$V.txt" | tail -1)"
    kill $POLL 2>/dev/null
    grep -E "Traceback|Error" "$OMOL_RUN_DIR/train_$V.log" | head -3
done
echo "== size consistency of the checkpoints"
$PY -B audit/check_checkpoint.py "$OMOL_RUN_DIR"/*/checkpoints/final/inference_ckpt.pt
