#!/bin/bash
# Molecular evaluation of the final checkpoints of the nine pilots with audit/molecular.py: additivity and force
# change for three far-apart neutral singlet pairs, and finite-difference force checks. Inference only, one GPU.
# Results go to $OUT/work/results, outside the repository, so a later syncrepo cannot overwrite them (guide B9 row 16).
umask 002
export PYTHONNOUSERSITE=1
set -uo pipefail
ROOT=/ocean/projects/che260063p/shared
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
PY=$ROOT/envs/mlip/bin/python
OUT=$ROOT/runs/molecular_eval_${SLURM_JOB_ID:-manual}
export FAIRCHEM_CACHE_DIR=$ROOT/cache/fairchem WARP_CACHE_PATH=$ROOT/cache/warp
mkdir -p "$OUT/logs"
now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(now) $*" | tee -a "$OUT/summary.txt"; }

# work in a copy (the loader imports training/ and local_routing/ to rebuild the Local-6 model)
WORK=$OUT/work
RES=$WORK/results
mkdir -p "$RES"
cp -r "$REPO/audit" "$REPO/training" "$REPO/local_routing" "$WORK/"
cd "$WORK"
note "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1); repo $REPO; results $RES"

# seeded pilots only: aborted_pilot_*, pilottest_* and the unseeded pilot_<variant>_<job> runs do not match
RUNS=$(ls -d $ROOT/runs/pilot_{dense,global,local6}_s[0-9]_* 2>/dev/null)
N=$(echo "$RUNS" | grep -c .)
note "$N runs: $(echo $RUNS | xargs -n1 basename | tr '\n' ' ')"
[ "$N" -eq 9 ] || { note "RESULT: failed, expected 9 runs"; exit 1; }

FAILED=0
for RUN in $RUNS; do
    NAME=$(basename "$RUN")
    CKPT=$(ls "$RUN"/train/*/checkpoints/final/inference_ckpt.pt)
    CFG=$(ls "$RUN"/train/*/canonical_config.yaml)
    STEPS=$(grep -o 'steps=[0-9]*' "$RUN/settings.txt" | head -1 | cut -d= -f2)
    LABEL=$(echo "$NAME" | sed -E 's/^pilot_(.*)_[0-9]+$/\1/')_step$STEPS     # e.g. local6_s0_step50000
    note "START $NAME label $LABEL"
    $PY -u -B -m audit.molecular "$CKPT" --run-config "$CFG" --label "$LABEL" --output "$RES/$NAME" \
        >> "$OUT/logs/molecular_eval.log" 2>&1
    RC=$?
    note "END   $NAME exit $RC"
    [ $RC -eq 0 ] || FAILED=$((FAILED + 1))
done

# one CSV per kind for all runs, the comparison figure, and the metadata without the copied config text
for KIND in locality finite_difference; do
    FILES=$(ls "$RES"/pilot_*/$KIND.csv 2>/dev/null)
    [ -n "$FILES" ] || continue
    head -1 $(echo "$FILES" | head -1) > "$RES/molecular_$KIND.csv"
    for F in $FILES; do tail -n +2 "$F"; done >> "$RES/molecular_$KIND.csv"
    note "wrote $RES/molecular_$KIND.csv, $(($(wc -l < "$RES/molecular_$KIND.csv") - 1)) rows"
done
$PY -B -m audit.plot_molecular "$RES"/pilot_*/locality.csv --output "$RES/molecular_locality.png" \
    >> "$OUT/logs/molecular_eval.log" 2>&1 || FAILED=$((FAILED + 1))
$PY -B - "$RES" >> "$OUT/logs/molecular_eval.log" 2>&1 <<'EOF' || FAILED=$((FAILED + 1))
import json, pathlib, sys
res = pathlib.Path(sys.argv[1])
meta = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(res.glob("pilot_*/metadata.json"))]
for m in meta:
    m.pop("resolved_config_text", None)
(res / "molecular_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
EOF
grep -a -E 'Wrote|Error|Traceback' "$OUT/logs/molecular_eval.log" | cut -c1-300 | tee -a "$OUT/summary.txt"
note "RESULT: $([ $FAILED -eq 0 ] && echo passed || echo "failed ($FAILED)")"
exit $FAILED
