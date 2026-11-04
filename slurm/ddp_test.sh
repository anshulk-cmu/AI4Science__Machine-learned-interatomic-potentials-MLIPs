#!/bin/bash
# Tests of the two-rank mode (docs/ddp.md). Usage: ddp_test.sh cpu|v100|l40s|loss, from a code copy (never shared/repo while chains run).
#   cpu   the unit tests on Linux (with the two-process gloo tests) and the batch statistics of the real lists; writes worst_rows.txt
#   v100  on 2 x V100-32: Dense one-GPU against two-rank runs, a two-rank run stopped by SIGTERM and resumed, Element one-GPU
#         against two ranks, Local+Global and Local-12 on two ranks, long_run.sh with LONG_RANKS=2 stopped by time and resumed,
#         the collector on that run, and the memory test on worst_rows.txt when it exists
#   l40s  on 2 x L40S: long_run.sh with LONG_RANKS=2 for Local+Global and Local-12 on the real lists, and the memory test
#   loss  on 2 x V100-32: train/loss in the log of a two-rank run against a one-GPU run, and a two-rank run stopped and
#         resumed on one GPU (the rank change of docs/ddp.md Section 6)
# Everything goes to DDP_DIR (shared/ddp_test): resume/ (probe runs), runs/ (long_run.sh), results/, logs/.
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
STAGE=${1:?usage: ddp_test.sh cpu|v100|l40s|loss}
SHARED=/ocean/projects/che260063p/shared
REPO=$(cd "$(dirname "$0")/.." && pwd)
DIR=${DDP_DIR:-$SHARED/ddp_test}
PY=$SHARED/envs/mlip/bin/python
DATA=$SHARED/data/omol25
JOB=${SLURM_JOB_ID:-manual}
OUT=$DIR/results
mkdir -p "$OUT" "$DIR/logs" || exit 1
export OMOL_DATA_ROOT=$DATA OMOL_CACHE_DIR=$SHARED/cache FAIRCHEM_CACHE_DIR=$SHARED/cache/fairchem WARP_CACHE_PATH=$SHARED/cache/warp
cd "$REPO" || exit 1
FAILED=0
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') job $JOB: $*" | tee -a "$OUT/summary.txt"; }
step() { local what=$1; shift; note "START $what"; "$@"; local rc=$?; note "END   $what: exit $rc"; [ $rc -eq 0 ] || FAILED=1; }
note "stage $STAGE, host $(hostname), $(nproc) cores, $(free -g | awk '/Mem/ {print $2}') GB, code $REPO, GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';')"
POLL=""
if [ "$STAGE" != cpu ]; then
    ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$DIR/logs/gpu_${STAGE}_$JOB.txt"; sleep 5; done ) &
    POLL=$!
fi

stats() { $PY -u -B scripts/ddp_batch_stats.py --index "$DATA/pilot/$1_index.npz" --rows "manifests/$2.txt" --seed 0 --out "$OUT/batches_$2_s0.json" "${@:3}"; }
compare() { $PY -u -B scripts/resume_compare.py "$DIR/resume" "$@"; }
ranks_agree() {   # the weights hash of every rank at the end of training, per run
    $PY - "$DIR/resume" "$@" <<'EOF'
import json, sys
from pathlib import Path
ok = True
for name in sys.argv[2:]:
    hashes = {}
    for f in sorted(Path(sys.argv[1], name, "probe").glob("rank*/seg_*.events.jsonl")):
        ends = [e for e in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if e["event"] == "train_end"]
        hashes.setdefault(f.parent.name, []).extend(e["weights_sha256"] for e in ends)
    same = len(hashes) == 2 and hashes["rank0"] and hashes["rank0"] == hashes["rank1"]
    ok &= bool(same)
    print(f"{'PASS' if same else 'FAIL'}  {name}: weights sha256 at each train end, rank0 {hashes.get('rank0')} rank1 {hashes.get('rank1')}")
sys.exit(0 if ok else 1)
EOF
}
memory_test() {   # two ranks on the structures with the most atom pairs, one epoch, a tiny validation
    local variant=$1 name=mem_$1_$STAGE
    head -100 manifests/split_val_monitor_5k.txt > "$DIR/val_100.txt"
    PILOT_TRAIN_LIST=$DIR/worst_rows.txt PILOT_VAL_LIST=$DIR/val_100.txt PILOT_VAL_SPLIT=train_4M REPO=$REPO source slurm/pilot_data_overrides.sh
    local S=training.split_sampler.SplitBatchSampler steps=$(( $(wc -l < "$DIR/worst_rows.txt") ))
    env OMOL_RUN_DIR="$DIR/resume/$name/train" PYTHONUNBUFFERED=1 $PY -u -B -m training.run --mode train variant=$variant \
        job.run_name=$name +job.timestamp_id=$name job.seed=0 training.steps=$steps training.eval_every=1000000 training.ckpt_every=1000000 \
        runner.train_eval_unit._target_=training.resume_probe.ProbeTrainEvalUnit "+runner.train_eval_unit.batch_log=$DIR/resume/$name/probe/seg_$JOB.csv" \
        job.scheduler.ranks_per_node=2 train_dataloader.batch_sampler_fn._target_=$S eval_dataloader.batch_sampler_fn._target_=$S \
        "${DATA_OVERRIDES[@]}" > "$DIR/logs/${name}_$JOB.log" 2>&1
    local rc=$?
    grep -a -E "OOM|Error" "$DIR/logs/${name}_$JOB.log" | head -5 | tee -a "$OUT/summary.txt"
    for f in "$DIR/resume/$name"/probe/rank*/mem_seg_*.csv; do
        note "$name $(basename "$(dirname "$f")"): steps $(( $(wc -l < "$f") - 1 )), largest allocated / reserved MiB: $(tail -n +2 "$f" | sort -t, -k5 -n | tail -1)"
    done
    return $rc
}

case "$STAGE" in
cpu)
    step "unit tests on Linux" $PY -m unittest discover -s tests > "$OUT/unittest_$JOB.log" 2>&1
    tail -3 "$OUT/unittest_$JOB.log" | tee -a "$OUT/summary.txt"
    grep -a -E "^(FAIL|ERROR):" "$OUT/unittest_$JOB.log" | tee -a "$OUT/summary.txt"
    step "batches of split_train_70 and its worst rows" stats train_4M split_train_70 --worst 20000 --data "$DATA/train_4M" --worst-out "$DIR/worst_rows.txt"
    step "batches of split_val_monitor_5k" stats train_4M split_val_monitor_5k --eval
    step "batches of pilot_train_200k" stats train_4M pilot_train_200k
    step "batches of pilot_val_5k" stats val pilot_val_5k --eval
    ;;
v100)
    export RESUME_TEST_DIR=$DIR/resume RESUME_REPO=$REPO RESUME_SEED=0 RESUME_CKPT_EVERY=100
    R="bash slurm/resume_test.sh"
    note "START Dense, one GPU each (S1 on GPU 0, S2 on GPU 1)"
    # wait for these two runs only: a bare wait would also wait for the endless GPU poll
    RESUME_STEPS=400 CUDA_VISIBLE_DEVICES=0 $R dense S1 & A=$!
    RESUME_STEPS=400 RESUME_SKIP_VERIFY=1 CUDA_VISIBLE_DEVICES=1 $R dense S2 & B=$!     # S1 checks the lists
    export RESUME_SKIP_VERIFY=1
    wait $A $B
    step "Dense, two ranks: D1, D2, DB stopped by SIGTERM after step 250, DB resumed" env RESUME_STEPS=400 RESUME_RANKS=2 $R dense D1 D2 DB:term:250 DB
    step "Dense, two ranks against one GPU" compare --refs S1 S2 --tests D1 D2 --steps 400 --probe-dir probe/global --uninterrupted --out "$OUT/dense_two_ranks_vs_one"
    step "Dense, two-rank noise" compare --refs D1 D2 --tests S1 S2 --steps 400 --probe-dir probe/global --uninterrupted --out "$OUT/dense_one_vs_two_ranks"
    for p in rank0 rank1 global; do
        step "Dense, two-rank resume, probe/$p" compare --refs D1 D2 --tests DB --steps 400 --probe-dir probe/$p --out "$OUT/dense_resume_$p"
    done
    note "START Element, one GPU each (E1, E1b), 100 steps"
    RESUME_STEPS=100 CUDA_VISIBLE_DEVICES=0 $R element E1:smallval & A=$!
    RESUME_STEPS=100 CUDA_VISIBLE_DEVICES=1 $R element E1b:smallval & B=$!
    wait $A $B
    for spec in "element E2:smallval" "local_global LG2:smallval" "local12 L12:smallval"; do
        step "two ranks, 100 steps: $spec" env RESUME_STEPS=100 RESUME_RANKS=2 $R $spec
    done
    step "Element, two ranks against one GPU" compare --refs E1 E1b --tests E2 --steps 100 --probe-dir probe/global --uninterrupted --out "$OUT/element_two_ranks_vs_one"
    step "ranks hold the same weights" ranks_agree D1 D2 DB E2 LG2 L12
    LR=(env LONG_REPO="$REPO" LONG_RUNS_DIR="$DIR/runs" LONG_STEPS=3000 LONG_SEED=0 LONG_EVAL_EVERY=3000 LONG_CKPT_EVERY=100
        LONG_RANKS=2 LONG_TRAIN_LIST="$REPO/manifests/pilot_train_200k.txt" LONG_VAL_LIST="$REPO/manifests/pilot_val_5k.txt"
        LONG_VAL_SPLIT=val LONG_SEGMENT_HOURS=1)
    step "long_run.sh dense, two ranks, segment 1 (stops by time)" "${LR[@]}" SLURM_JOB_ID=${JOB}1 LONG_STOP_MARGIN_SEC=2700 bash slurm/long_run.sh dense
    step "long_run.sh dense, two ranks, segment 2 (resumes and finishes)" "${LR[@]}" SLURM_JOB_ID=${JOB}2 LONG_STOP_MARGIN_SEC=300 bash slurm/long_run.sh dense
    RUN=$DIR/runs/long_dense_s0_3000
    step "collector, W&B and log" $PY -u -B scripts/collect_final_metrics.py "$RUN" --variants dense --expected-seeds 1 --steps 3000 --output "$OUT/collect_long_dense.csv"
    step "collector, log only" $PY -u -B scripts/collect_final_metrics.py "$RUN" --variants dense --expected-seeds 1 --steps 3000 --source log --output "$OUT/collect_long_dense_log.csv"
    step "checkpoint discovery" $PY -u -B -c "from training.checkpoints import discover; print([s for s, _ in discover('$RUN', 3000)])"
    cp "$RUN/summary.txt" "$OUT/long_dense_summary.txt"; cp "$RUN/settings.txt" "$OUT/long_dense_settings.txt"; cp "$RUN/segments.txt" "$OUT/long_dense_segments.txt"
    if [ -s "$DIR/worst_rows.txt" ]; then
        step "memory test, Local-12, two V100" memory_test local12
        step "memory test, Local+Global, two V100" memory_test local_global
    fi
    ;;
l40s)
    LR=(env LONG_REPO="$REPO" LONG_RUNS_DIR="$DIR/runs" LONG_SEED=0 LONG_RANKS=2 LONG_SEGMENT_HOURS=2 LONG_STOP_MARGIN_SEC=600 SLURM_JOB_ID=$JOB)
    step "long_run.sh local_global, two L40S, 1000 steps" "${LR[@]}" LONG_STEPS=1000 LONG_EVAL_EVERY=1000 LONG_CKPT_EVERY=500 bash slurm/long_run.sh local_global
    step "long_run.sh local12, two L40S, 600 steps" "${LR[@]}" LONG_STEPS=600 LONG_EVAL_EVERY=600 LONG_CKPT_EVERY=300 bash slurm/long_run.sh local12
    for RUN in "$DIR"/runs/long_local_global_s0_1000 "$DIR"/runs/long_local12_s0_600; do cp "$RUN/summary.txt" "$OUT/$(basename "$RUN")_summary.txt"; done
    if [ -s "$DIR/worst_rows.txt" ]; then
        step "memory test, Local-12, two L40S" memory_test local12
        step "memory test, Local+Global, two L40S" memory_test local_global
    fi
    ;;
loss)
    export RESUME_TEST_DIR=$DIR/resume RESUME_REPO=$REPO RESUME_SEED=0 RESUME_CKPT_EVERY=100 RESUME_STEPS=300
    R="bash slurm/resume_test.sh"
    step "Dense, one GPU: S1" env CUDA_VISIBLE_DEVICES=0 $R dense S1:smallval
    export RESUME_SKIP_VERIFY=1
    step "Dense, two ranks: D1, and DX stopped by SIGTERM after step 150" env RESUME_RANKS=2 $R dense D1:smallval DX:term:150:smallval
    rm -f "$DIR/resume/DX/train/DX/.distributed-shared-file"      # as long_run.sh does before every segment
    step "DX resumed on one GPU" env CUDA_VISIBLE_DEVICES=0 $R dense DX:smallval
    step "train/loss of the log, D1 against S1" $PY - "$DIR/resume/logs" "$JOB" <<'EOF'
import ast, re, sys
def losses(name):
    out = {}
    for line in open(f"{sys.argv[1]}/{name}_{sys.argv[2]}.log", encoding="utf-8", errors="replace"):
        m = re.search(r"INFO:root:(\{'train/loss'.*\})\s*$", line)
        if m:
            d = ast.literal_eval(m.group(1))
            out[d["train/step"]] = d["train/loss"]
    return out
one, two = losses("S1"), losses("D1")
rel = [abs(two[s] - one[s]) / abs(one[s]) for s in sorted(one)]
ok = sorted(one) == sorted(two) == list(range(300)) and max(rel) <= 2.5e-6
print(f"{'PASS' if ok else 'FAIL'}  logged train/loss, two ranks against one GPU, {len(rel)} steps: step 0 {rel[0]:.2e}, largest {max(rel):.2e} (limit 2.5e-6, 3 x the one-GPU noise of docs/ddp.md 4.2)")
sys.exit(0 if ok else 1)
EOF
    step "DX (two ranks, then one GPU) against S1 and D1" compare --refs S1 D1 --tests DX --steps 300 --probe-dir probe/global --out "$OUT/rank_change"
    ;;
*) echo "usage: ddp_test.sh cpu|v100|l40s|loss"; exit 2 ;;
esac
[ -n "$POLL" ] && kill $POLL 2>/dev/null
note "RESULT: $([ $FAILED -eq 0 ] && echo every step exited 0 || echo at least one step failed)"
exit $FAILED
