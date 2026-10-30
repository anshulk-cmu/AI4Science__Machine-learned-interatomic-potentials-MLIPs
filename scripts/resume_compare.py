"""Compare resumed runs with uninterrupted runs of the resume test (docs/resume-test.md, slurm/resume_test.sh).

Each run folder holds probe/seg_<jobid>.csv (one row per step, from training/resume_probe.py), the matching
.events.jsonl, and train/<name>/checkpoints/final/inference_ckpt.pt. A resumed run has two or more segments (in
separate files, or in one file when one job ran both); a later segment replaces the rows of an earlier one from the
step where it resumed.

The noise floor N of a quantity is its largest difference between two uninterrupted reference runs. A resumed run
passes a numeric check when its largest difference to any reference run is at most FACTOR * N, or at most FLOOR
(for the case where the GPU happens to be deterministic and N is 0). Batch hashes, learning rates and step counters
must match exactly. The criteria were fixed before the runs (see the doc).

Usage (repository root): python scripts/resume_compare.py TEST_DIR --refs A1 A2 A3 --tests B C --steps 400 --out OUT_PREFIX
"""
import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

FACTOR = 3.0
FLOOR = 1e-6
COLUMNS = ["step", "epoch", "step_in_epoch", "structures", "atoms", "batch_sha256", "loss", "lr", "grad_norm"]


def read_segments(path):
    """Segments of one probe file, each as rows by step. Two segments of one job append to the same file, so a step that
    does not increase starts a new segment. A line cut short by a hard kill has too few fields and is dropped."""
    segments = []
    with open(path, encoding="utf-8", newline="") as f:
        for rec in csv.reader(f):
            if len(rec) != len(COLUMNS) or rec[0] == "step":
                continue
            row = dict(zip(COLUMNS, rec))
            for k in ("step", "epoch", "step_in_epoch", "structures", "atoms"):
                row[k] = int(row[k])
            for k in ("loss", "lr"):
                row[k] = float(row[k])
            if not segments or row["step"] <= max(segments[-1]):
                segments.append({})
            segments[-1][row["step"]] = row
    return segments


def merge_segments(segments):
    """Rows by step over segments in run order; a later segment overrides from its first step on."""
    merged, starts = {}, []
    for rows in segments:
        if not rows:
            continue
        first = min(rows)
        starts.append(first)
        merged = {s: r for s, r in merged.items() if s < first}
        merged.update(rows)
    return merged, starts


def load_run(run_dir):
    paths = sorted((run_dir / "probe").glob("seg_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    segments = [s for p in paths for s in read_segments(p)]
    rows, starts = merge_segments(segments)
    evals = []
    for p in paths:
        events_file = p.with_suffix(".events.jsonl")
        if events_file.exists():
            evals += [e for e in map(json.loads, events_file.read_text(encoding="utf-8").splitlines()) if e["event"] == "eval"]
    return {"name": run_dir.name, "rows": rows, "segments": segments, "starts": starts,
            "eval": evals[-1] if evals else None, "final": run_dir / "train" / run_dir.name / "checkpoints" / "final" / "inference_ckpt.pt"}


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def max_rel(x, y, steps):
    return max(rel(x["rows"][s]["loss"], y["rows"][s]["loss"]) for s in steps)


def weight_rel_diff(path_a, path_b, key):
    """||W_a - W_b|| / ||W_b|| over all floating-point tensors of a state dict in an inference checkpoint."""
    import torch
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    sa, sb = getattr(a, key), getattr(b, key)
    num = den = 0.0
    for name, tb in sb.items():
        if torch.is_floating_point(tb):
            ta = sa[name].double()
            num += float(((ta - tb.double()) ** 2).sum())
            den += float((tb.double() ** 2).sum())
    return (num / den) ** 0.5


def numeric_check(name, test_value, noise):
    limit = max(FACTOR * noise, FLOOR)
    return {"check": name, "value": test_value, "noise": noise, "limit": limit, "pass": test_value <= limit}


def compare(refs, test, steps, resume_steps):
    """All checks of one resumed run against the reference runs. resume_steps: the steps compared numerically."""
    out = []
    all_steps = range(steps)
    complete = all(s in test["rows"] for s in all_steps) and len(test["rows"]) == steps
    out.append({"check": "steps 0 to N-1 all present once, after merging segments", "pass": complete,
                "value": f"{len(test['rows'])} steps, segments start at {test['starts']}"})
    out.append({"check": "run was resumed (two or more segments)", "pass": len(test["starts"]) >= 2, "value": test["starts"]})
    if not complete:
        return out
    for key in ("batch_sha256", "epoch", "step_in_epoch", "structures", "atoms"):
        bad = [s for s in all_steps for r in refs if test["rows"][s][key] != r["rows"][s][key]]
        out.append({"check": f"{key} equal to every reference at every step", "pass": not bad,
                    "value": f"{len(bad)} mismatches, first at step {bad[0] if bad else None}"})
    lr_diff = max(rel(test["rows"][s]["lr"], r["rows"][s]["lr"]) for s in all_steps for r in refs)
    out.append({"check": "learning rate equal at every step (relative difference <= 1e-12)", "pass": lr_diff <= 1e-12, "value": lr_diff})

    pairs = list(itertools.combinations(refs, 2))
    noise = max(max_rel(a, b, resume_steps) for a, b in pairs)
    out.append(numeric_check(f"train loss, steps {resume_steps[0]} to {resume_steps[-1]}, max relative difference",
                             max(max_rel(test, r, resume_steps) for r in refs), noise))

    # in memory against from disk: the first segment ran on past the checkpoint; the next one restarted from it
    first, last = test["segments"][0], test["segments"][-1]
    overlap = [s for s in sorted(first) if s >= test["starts"][-1] and s in last]
    if overlap:
        same_state = max(rel(last[s]["loss"], first[s]["loss"]) for s in overlap)
        noise_overlap = max(max_rel(a, b, overlap) for a, b in pairs)
        out.append(numeric_check(f"train loss, continued in memory against resumed from disk, steps {overlap[0]} to {overlap[-1]}",
                                 same_state, noise_overlap))

    for key in ("model_state_dict", "ema_state_dict"):
        if all(r["final"].exists() for r in refs + [test]):
            noise_w = max(weight_rel_diff(a["final"], b["final"], key) for a, b in pairs)
            out.append(numeric_check(f"final {key}, relative L2 difference",
                                     max(weight_rel_diff(test["final"], r["final"], key) for r in refs), noise_w))
        else:
            out.append({"check": f"final {key}", "pass": False, "value": "a final checkpoint is missing"})

    if all(r["eval"] for r in refs + [test]):
        for key in sorted(k for k in test["eval"] if k.startswith("val/") and ("mae" in k or k == "val/loss")):
            noise_v = max(rel(a["eval"][key], b["eval"][key]) for a, b in pairs)
            out.append(numeric_check(f"{key}, relative difference", max(rel(test["eval"][key], r["eval"][key]) for r in refs), noise_v))
    else:
        out.append({"check": "validation metrics", "pass": False, "value": "an eval event is missing"})
    return out


def epoch_check(run):
    """Every batch of epoch 0 is distinct, and the batches seen in epoch 1 are a reordering of epoch 0 batches."""
    by_epoch = {}
    for s in sorted(run["rows"]):
        by_epoch.setdefault(run["rows"][s]["epoch"], []).append(run["rows"][s]["batch_sha256"])
    if 1 not in by_epoch:
        return [{"check": f"{run['name']}: reaches epoch 1", "pass": False, "value": sorted(by_epoch)}]
    e0, e1 = by_epoch[0], by_epoch[1]
    return [
        {"check": f"{run['name']}: epoch 0 batches all distinct", "pass": len(set(e0)) == len(e0), "value": f"{len(e0)} batches"},
        {"check": f"{run['name']}: epoch 1 batches are epoch 0 batches", "pass": not (Counter(e1) - Counter(e0)), "value": f"{len(e1)} batches"},
        {"check": f"{run['name']}: epoch 1 order differs from epoch 0", "pass": e1 != e0[:len(e1)], "value": ""},
    ]


def write_steps_csv(path, runs, steps):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"{r['name']}_{k}" for r in runs for k in ("epoch", "step_in_epoch", "batch_sha256", "loss", "lr", "grad_norm")])
        for s in range(steps):
            line = [s]
            for r in runs:
                row = r["rows"].get(s, {})
                line += [row.get(k, "") for k in ("epoch", "step_in_epoch", "batch_sha256", "loss", "lr", "grad_norm")]
            w.writerow(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test_dir", type=Path)
    ap.add_argument("--refs", nargs="+", required=True, help="uninterrupted runs, at least two")
    ap.add_argument("--tests", nargs="+", required=True, help="resumed runs")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--epoch-check", action="store_true", help="also check the batches of epoch 0 and epoch 1")
    ap.add_argument("--out", type=Path, required=True, help="prefix of the output files (_steps.csv, _checks.json)")
    args = ap.parse_args()
    if len(args.refs) < 2:
        ap.error("need at least two reference runs for a noise floor")

    refs = [load_run(args.test_dir / n) for n in args.refs]
    tests = [load_run(args.test_dir / n) for n in args.tests]
    result = {"factor": FACTOR, "floor": FLOOR, "runs": {}}
    for t in tests:
        start = t["starts"][-1] if len(t["starts"]) >= 2 else 0
        checks = compare(refs, t, args.steps, list(range(start, args.steps)))
        if args.epoch_check:
            checks += epoch_check(t)
        result["runs"][t["name"]] = checks
    if args.epoch_check:
        result["runs"]["references"] = [c for r in refs for c in epoch_check(r)]
    ok = all(c["pass"] for checks in result["runs"].values() for c in checks)
    result["verdict"] = "PASS" if ok else "FAIL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_steps_csv(f"{args.out}_steps.csv", refs + tests, args.steps)
    Path(f"{args.out}_checks.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    for name, checks in result["runs"].items():
        for c in checks:
            extra = f" (value {c['value']:.3e}, noise {c['noise']:.3e}, limit {c['limit']:.3e})" if "limit" in c else f" ({c['value']})"
            print(f"{'PASS' if c['pass'] else 'FAIL'}  {name}: {c['check']}{extra}")
    print("RESULT:", result["verdict"])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
