#!/bin/bash
# One shard of scripts/eval_rows.py for the final checkpoint of one long run. Babel, preempt partition, 1 GPU.
# The array index is the shard; a preempted or requeued task resumes from its own CSV.
#   sbatch --array=0-$((N-1)) --job-name=mlip_eval_<variant>_<tag> \
#     --output=/data/user_data/anshulk/mlip/logs/mlip_eval_<variant>_<tag>_%A_%a.out \
#     --export=ALL,VARIANT=<variant>,LIST=manifests/split_test_eval_10k.txt,TAG=<tag>,NSHARDS=N slurm/eval_rows.sh
#SBATCH --account=bapoczos
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=24G
#SBATCH --time=3:30:00
#SBATCH --constraint=VRAM_48GB|VRAM_80GB|VRAM_96GB|VRAM_141GB
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
set -euo pipefail
# A preemption signal must not turn into a failure: Slurm requeues the task and the CSV resumes.
trap 'echo "signal at $(date)"; exit 0' USR1 TERM
SHARED=/data/user_data/anshulk/mlip
REPO=$SHARED/code-babel/AI4Science-MLIPs
SHARD=${SLURM_ARRAY_TASK_ID:-0}
CKPT=$(ls $SHARED/runs/long_${VARIANT}_s0_400000/train/*/checkpoints/final/inference_ckpt.pt)
OUT=$SHARED/eval/$TAG/${VARIANT}_s${SHARD}of${NSHARDS}.csv
cd $REPO
echo "$(date) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $VARIANT $LIST shard $SHARD/$NSHARDS restart ${SLURM_RESTART_COUNT:-0} $(git rev-parse --short HEAD)"
start=$(date +%s)
/home/anshulk/mlipEnv/bin/python scripts/eval_rows.py run --checkpoint "$CKPT" --data $SHARED/data/omol25/train_4M \
  --rows "$LIST" --shard $SHARD --num-shards $NSHARDS --batch-atoms ${BATCH_ATOMS:-2000} --dtype float32 --out "$OUT" &
wait $!
echo "$(date) done in $(( $(date +%s) - start )) s"
