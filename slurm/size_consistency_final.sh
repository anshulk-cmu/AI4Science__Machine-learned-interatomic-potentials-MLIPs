#!/bin/bash
# Size-consistency gap of the 8 final long-run checkpoints: ideal Al + Ni slabs plus the 20 rattled runs of the audit protocol.
#   sbatch --job-name=mlip_eval_sizecons --output=/data/user_data/anshulk/mlip/logs/mlip_eval_sizecons_%j.out slurm/size_consistency_final.sh
#SBATCH --account=bapoczos
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --constraint=VRAM_48GB|VRAM_80GB|VRAM_96GB|VRAM_141GB
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
set -uo pipefail
trap 'echo "signal at $(date)"; exit 0' USR1 TERM
SHARED=/data/user_data/anshulk/mlip
OUT=$SHARED/eval/sizecons
mkdir -p $OUT
cd $SHARED/code-babel/AI4Science-MLIPs/audit
echo "$(date) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(git rev-parse --short HEAD)"
for v in dense dense_scaled global_frozen global element local6 local_global local12; do
  [ -s $OUT/$v.txt ] && grep -q "run 20 " $OUT/$v.txt && continue      # done before a requeue
  ckpt=$(ls $SHARED/runs/long_${v}_s0_400000/train/*/checkpoints/final/inference_ckpt.pt)
  /home/anshulk/mlipEnv/bin/python check_checkpoint.py "$ckpt" --runs 20 --gaps 8 10 15 25 50 100 > $OUT/$v.txt.part 2> $OUT/$v.err &
  wait $!
  mv $OUT/$v.txt.part $OUT/$v.txt
  echo "$(date) $v done"
done
