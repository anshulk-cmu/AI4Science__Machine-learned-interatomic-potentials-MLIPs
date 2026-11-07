#!/bin/bash
# Submit one long run (one variant, one seed) as a chain of Slurm segments of slurm/long_run.job (docs/launch-checklist.md).
# Usage: submit_chain.sh VARIANT --steps N --seed S --segments K --hours H --eval-every E --ckpt-every C [--dry-run]
# Every option except --dry-run is required and has no default. Run it on CMU Babel, on a compute node where /data is mounted
# (ssh babel-<node> while one of your jobs runs there): the run folder and the job file live under /data.
# Segment k+1 waits for segment k with afterany (a segment that reaches its limit ends as TIMEOUT, which afterok treats as a
# failure). Each segment decides from the run folder what to do (slurm/long_run.sh).
# Babel: account bapoczos, partition preempt, QoS preempt_qos; a segment may be preempted at any time and is then requeued with
# the same job id (long_run.sh exits 0 on the stop signal and resumes from the last complete checkpoint).
# The GPU is any type with a memory feature that meets the guard of long_run.sh: VRAM_40GB or larger for dense, dense_scaled,
# global, global_frozen; VRAM_80GB or larger for element (relax with LONG_FEAT) and for local12 on one GPU; VRAM_48GB or larger
# for local6, local_global, and for every variant with LONG_RANKS=2. --dry-run prints the plan and runs sbatch --test-only for
# the first segment; it submits nothing.
# Required environment: LONG_SHARED (the project folder, for example /data/user_data/anshulk/mlip), LONG_PY (the python of the
# environment). Optional: LONG_FEAT (the --constraint feature list), LONG_EXCLUDE (nodes to avoid), LONG_RANKS=2 (two GPUs of one
# node, DDP, for local6, element, local_global, local12; docs/ddp.md), LONG_REPO (the code copy the jobs run),
# LONG_RUNS_DIR (where the run folder goes), LONG_TRAIN_LIST,
# LONG_VAL_LIST, LONG_VAL_SPLIT (the data of slurm/long_run.sh; unset, it uses split_train_70 and split_val_monitor_5k of train_4M).
umask 002                      # the run folder and chains.txt stay group-writable for the team
set -uo pipefail
usage() { echo "usage: submit_chain.sh VARIANT --steps N --seed S --segments K --hours H --eval-every E --ckpt-every C [--dry-run]"; exit 2; }
VARIANT=${1:-}
RANKS=${LONG_RANKS:-1}
case "$RANKS" in 1) ;; 2) [[ $VARIANT =~ ^(local6|element|local_global|local12)$ ]] || usage ;; *) usage ;; esac
# smallest card of each feature class against the guard of long_run.sh (31,000 / 44,000 / 79,000 MiB, 40,000 per GPU with 2 ranks)
case "$VARIANT:$RANKS" in
    dense:1|dense_scaled:1|global:1|global_frozen:1) FEAT='VRAM_40GB|VRAM_48GB|VRAM_80GB|VRAM_96GB|VRAM_141GB' ;;
    element:1|local12:1) FEAT='VRAM_80GB|VRAM_96GB|VRAM_141GB' ;;
    local6:1|local_global:1|*:2) FEAT='VRAM_48GB|VRAM_80GB|VRAM_96GB|VRAM_141GB' ;;
    *) usage ;;
esac
FEAT=${LONG_FEAT:-$FEAT}
shift
declare -A ARG=([steps]="" [seed]="" [segments]="" [hours]="" [eval-every]="" [ckpt-every]="")
DRY=0
while [ $# -gt 0 ]; do
    if [ "$1" = --dry-run ]; then DRY=1; shift; continue; fi
    KEY=${1#--}
    [[ $1 == --* && -v ARG[$KEY] && $# -ge 2 ]] || usage
    ARG[$KEY]=$2
    shift 2
done
for KEY in steps seed segments hours eval-every ckpt-every; do
    [[ ${ARG[$KEY]} =~ ^(0|[1-9][0-9]*)$ ]] || { echo "submit_chain.sh: --$KEY is required (a whole number, no default)"; exit 2; }
done
STEPS=${ARG[steps]} SEED=${ARG[seed]} SEGMENTS=${ARG[segments]} HOURS=${ARG[hours]} EVAL=${ARG[eval-every]} CKPT=${ARG[ckpt-every]}
if (( STEPS < 1 || SEGMENTS < 1 || HOURS < 1 || HOURS > 48 || EVAL < 1 || CKPT < 1 || EVAL > STEPS || CKPT > STEPS )); then
    echo "submit_chain.sh: need --steps >= 1, --segments >= 1, 1 <= --hours <= 48 (the partition limit), 1 <= --eval-every, --ckpt-every <= --steps"
    exit 2
fi
for v in LONG_SHARED LONG_PY; do [ -n "${!v:-}" ] || { echo "submit_chain.sh: set $v (no default)"; exit 2; }; done
SHARED=$LONG_SHARED
[ -d "$SHARED" ] || { echo "REFUSED: $SHARED is not visible here; run this on a compute node of one of your jobs (ssh babel-<node>)"; exit 1; }
REPO=${LONG_REPO:-$SHARED/repo/AI4Science-MLIPs}
NAME=long_${VARIANT}_s${SEED}_${STEPS}
RUN=${LONG_RUNS_DIR:-$SHARED/runs}/$NAME
EXPORT="LONG_VARIANT=$VARIANT,LONG_STEPS=$STEPS,LONG_SEED=$SEED,LONG_EVAL_EVERY=$EVAL,LONG_CKPT_EVERY=$CKPT,LONG_SEGMENT_HOURS=$HOURS,LONG_REPO=$REPO,LONG_SHARED=$SHARED,LONG_PY=$LONG_PY"
for v in LONG_RUNS_DIR LONG_TRAIN_LIST LONG_VAL_LIST LONG_VAL_SPLIT; do [ -n "${!v:-}" ] && EXPORT+=",$v=${!v}"; done
[ "$RANKS" = 1 ] || EXPORT+=",LONG_RANKS=$RANKS"
OPTS=(-A bapoczos -p preempt --qos=preempt_qos -N 1 "--gres=gpu:$RANKS" "--constraint=$FEAT" --cpus-per-gpu=4 --mem-per-gpu=32G
      --requeue --signal=B:USR1@90 --open-mode=append -t "$HOURS:00:00" "--job-name=$NAME" "--output=$SHARED/logs/slurm-%x-%j.out"
      --mail-type=BEGIN,END,FAIL --mail-user=anshulk@andrew.cmu.edu ${LONG_EXCLUDE:+"--exclude=$LONG_EXCLUDE"} "--export=$EXPORT")
JOBFILE=$REPO/slurm/long_run.job

echo "run $NAME: $STEPS steps, seed $SEED, eval every $EVAL, checkpoint every $CKPT"
echo "folder $RUN, code $REPO"
echo "data: train ${LONG_TRAIN_LIST:-manifests/split_train_70.txt}, validation ${LONG_VAL_LIST:-manifests/split_val_monitor_5k.txt} from ${LONG_VAL_SPLIT:-train_4M}"
echo "$SEGMENTS segments of at most $HOURS h on $RANKS GPU(s) of one node, feature $FEAT; Babel charges no SU (fair-share only)"
if [ -f "$RUN/DONE" ]; then echo "DONE marker present: this run is finished; nothing to submit"; exit 0; fi
if [ -f "$RUN/FAILED" ]; then echo "REFUSED: $RUN/FAILED exists: $(cat "$RUN/FAILED"). Fix the cause and delete it first."; exit 1; fi
QUEUED=$(squeue --me -h -n "$NAME" -o %i | tr '\n' ' ')
if [ -n "${QUEUED// /}" ]; then echo "REFUSED: jobs of $NAME are already queued or running: $QUEUED"; exit 1; fi
[ -f "$JOBFILE" ] || { echo "REFUSED: no job file $JOBFILE"; exit 1; }
for k in $(seq 1 "$SEGMENTS"); do
    echo "segment $k: sbatch ${OPTS[*]}$([ "$k" -gt 1 ] && echo " --dependency=afterany:<segment $((k - 1))>") $JOBFILE"
done
if [ $DRY -eq 1 ]; then
    sbatch --test-only "${OPTS[@]}" "$JOBFILE"
    exit $?
fi
mkdir -p "$RUN" "$SHARED/logs"
PREV="" IDS=""
for k in $(seq 1 "$SEGMENTS"); do
    DEP=()
    [ -n "$PREV" ] && DEP=("--dependency=afterany:$PREV")
    ID=$(sbatch --parsable "${OPTS[@]}" ${DEP[@]+"${DEP[@]}"} "$JOBFILE") || { echo "sbatch failed at segment $k; submitted so far: $IDS"; exit 1; }
    ID=${ID%%;*}
    IDS+="$ID " PREV=$ID
    echo "segment $k of $SEGMENTS: job $ID"
done
echo "$(date '+%Y-%m-%d %H:%M:%S') ${USER:-unknown} chain $IDS($SEGMENTS x $HOURS h, $RANKS x $FEAT, Babel)" >> "$RUN/chains.txt"
echo "submitted: $IDS"
