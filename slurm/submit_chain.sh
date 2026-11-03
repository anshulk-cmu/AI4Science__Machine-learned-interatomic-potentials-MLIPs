#!/bin/bash
# Submit one long run (one variant, one seed) as a chain of Slurm segments of slurm/long_run.job (docs/launch-checklist.md).
# Usage: submit_chain.sh VARIANT --steps N --seed S --segments K --hours H --eval-every E --ckpt-every C [--dry-run]
# Every option except --dry-run is required and has no default. Run it on a Bridges-2 login node.
# Segment k+1 waits for segment k with afterany (a segment that reaches its limit ends as TIMEOUT, which afterok treats as a
# failure). Each segment decides from the run folder what to do (slurm/long_run.sh).
# The GPU follows the variant: v100-32 for dense, dense_scaled, global, global_frozen; h100-80 for local6, element,
# local_global, local12. --dry-run prints the plan and runs sbatch --test-only for the first segment; it submits nothing.
# Optional environment: LONG_GPU=l40s-48 (local6, element, local_global only), LONG_REPO (the code copy the jobs run),
# LONG_RUNS_DIR (where the run folder goes), LONG_TRAIN_LIST,
# LONG_VAL_LIST, LONG_VAL_SPLIT (the data of slurm/long_run.sh; unset, it uses split_train_70 and split_val_monitor_5k of train_4M).
umask 002                      # the run folder and chains.txt stay group-writable for the team (guide B9)
set -uo pipefail
usage() { echo "usage: submit_chain.sh VARIANT --steps N --seed S --segments K --hours H --eval-every E --ckpt-every C [--dry-run]"; exit 2; }
VARIANT=${1:-}
case "$VARIANT" in
    dense|dense_scaled|global|global_frozen) GPU=v100-32 RATE=1 ;;
    local6|element|local_global|local12) GPU=h100-80 RATE=2 ;;
    *) usage ;;
esac
# LONG_GPU=l40s-48 (1 SU/h) is allowed only for variants whose memory minimum it meets; long_run.sh still checks the size at start
if [ -n "${LONG_GPU:-}" ] && [ "$LONG_GPU" != "$GPU" ]; then
    [[ $LONG_GPU == l40s-48 && $VARIANT =~ ^(local6|element|local_global)$ ]] || usage
    GPU=l40s-48 RATE=1
fi
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
SHARED=/ocean/projects/che260063p/shared
REPO=${LONG_REPO:-$SHARED/repo/AI4Science-MLIPs}
NAME=long_${VARIANT}_s${SEED}_${STEPS}
RUN=${LONG_RUNS_DIR:-$SHARED/runs}/$NAME
EXPORT="LONG_VARIANT=$VARIANT,LONG_STEPS=$STEPS,LONG_SEED=$SEED,LONG_EVAL_EVERY=$EVAL,LONG_CKPT_EVERY=$CKPT,LONG_SEGMENT_HOURS=$HOURS,LONG_REPO=$REPO"
for v in LONG_RUNS_DIR LONG_TRAIN_LIST LONG_VAL_LIST LONG_VAL_SPLIT; do [ -n "${!v:-}" ] && EXPORT+=",$v=${!v}"; done
OPTS=(-A che260063p -p GPU-shared "--gpus=$GPU:1" -t "$HOURS:00:00" "--job-name=$NAME" "--export=$EXPORT")
JOBFILE=$REPO/slurm/long_run.job

echo "run $NAME: $STEPS steps, seed $SEED, eval every $EVAL, checkpoint every $CKPT"
echo "folder $RUN, code $REPO"
echo "data: train ${LONG_TRAIN_LIST:-manifests/split_train_70.txt}, validation ${LONG_VAL_LIST:-manifests/split_val_monitor_5k.txt} from ${LONG_VAL_SPLIT:-train_4M}"
echo "$SEGMENTS segments of at most $HOURS h on $GPU: at most $(( SEGMENTS * HOURS * RATE )) SU (only the hours used are charged)"
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
mkdir -p "$RUN"
PREV="" IDS=""
for k in $(seq 1 "$SEGMENTS"); do
    DEP=()
    [ -n "$PREV" ] && DEP=("--dependency=afterany:$PREV")
    ID=$(sbatch --parsable "${OPTS[@]}" ${DEP[@]+"${DEP[@]}"} "$JOBFILE") || { echo "sbatch failed at segment $k; submitted so far: $IDS"; exit 1; }
    ID=${ID%%;*}
    IDS+="$ID " PREV=$ID
    echo "segment $k of $SEGMENTS: job $ID"
done
echo "$(date '+%Y-%m-%d %H:%M:%S') ${USER:-unknown} chain $IDS($SEGMENTS x $HOURS h, $GPU)" >> "$RUN/chains.txt"
echo "submitted: $IDS"
