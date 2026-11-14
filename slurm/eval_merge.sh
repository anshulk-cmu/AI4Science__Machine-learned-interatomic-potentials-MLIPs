#!/bin/bash
# Merge the shards of one evaluation tag for all 8 long runs (every row exactly once) and write the paired differences
# against Dense and Global. Run after the eval arrays, for example with --dependency=afterok:<array ids>.
#   sbatch --job-name=mlip_eval_merge_testfull --output=/data/user_data/anshulk/mlip/logs/mlip_eval_merge_testfull_%j.out \
#     --export=ALL,TAG=testfull,LIST=manifests/split_test_10.txt,SPLIT=test_full slurm/eval_merge.sh
#SBATCH --account=bapoczos
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --exclude=babel-l9-32
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
set -euo pipefail
E=/data/user_data/anshulk/mlip/eval
P=/home/anshulk/mlipEnv/bin/python
cd /data/user_data/anshulk/mlip/code-babel/AI4Science-MLIPs
VARIANTS="dense dense_scaled global_frozen global element local6 local_global local12"
rm -f $E/${TAG}_metrics.csv $E/paired_${TAG}.jsonl
for v in $VARIANTS; do
  $P scripts/eval_rows.py merge --rows $LIST --label $v --eval-split $SPLIT --table $E/${TAG}_metrics.csv $E/$TAG/${v}_s*of*.csv
done
for b in dense global; do
  for a in $VARIANTS; do
    [ $a = $b ] && continue
    r=$($P scripts/eval_rows.py paired --rows $LIST --a $E/$TAG/${a}_s*of*.csv --b $E/$TAG/${b}_s*of*.csv | tail -1)
    echo "{\"list\": \"$TAG\", \"a\": \"$a\", \"b\": \"$b\", \"result\": $r}" >> $E/paired_${TAG}.jsonl
  done
done
cat $E/${TAG}_metrics.csv
