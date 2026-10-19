"""Validation curves of pilot-style runs side by side, from their logs/train.log (4 decimals, as fairchem prints them).

  python scripts/compare_val_curves.py RUNS/pilot_dense_s0_49041937 RUNS/pilot_global_s0_49041939 \
      RUNS/pilot_local6_s0_49041936 RUNS/frozen_global_s0_<jobid>

Each validation is labelled with the last training step logged before it. torchtnt also validates at the end of each
epoch and twice at the end (docs/TRAINING.md), so a step can appear twice; the last value wins. With one Dense, one
Global and one frozen run given, it also prints the share of the Dense -> Global force and energy gain that the frozen run
reaches: (Dense - frozen) / (Dense - Global). 1 means frozen equals Global, 0 means frozen equals Dense.
"""
import re
import sys
from pathlib import Path

STEP = re.compile(r"'train/step': (\d+)")
METRIC = {"force": re.compile(r"val/omol\.val,omol_forces,mae: ([0-9.]+)"),
          "energy": re.compile(r"val/omol\.val,omol_energy,per_atom_mae: ([0-9.]+)")}


def curve(log):
    """{step: {"force": x, "energy": y}} from one train.log."""
    out, step = {}, 0
    with open(log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP.search(line)
            if m:
                step = int(m.group(1)) + 1          # train/step counts from 0; the checkpoint after it is step + 1
                continue
            for name, rx in METRIC.items():
                m = rx.search(line)
                if m:
                    out.setdefault(step, {})[name] = float(m.group(1))
    return out


def main(run_dirs):
    runs = {Path(d).name: curve(Path(d) / "logs" / "train.log") for d in run_dirs}
    steps = sorted({s for c in runs.values() for s in c})
    names = list(runs)
    print("step," + ",".join(f"{n} force,{n} energy" for n in names))
    for s in steps:
        cells = [f"{runs[n].get(s, {}).get(k, float('nan')):.4f}" for n in names for k in ("force", "energy")]
        print(f"{s}," + ",".join(cells))
    kinds = {k: [n for n in names if n.startswith(k)] for k in ("pilot_dense", "pilot_global", "frozen_global")}
    if all(len(v) == 1 for v in kinds.values()):
        d, g, f = (runs[v[0]] for v in kinds.values())
        print("\nshare of the Dense -> Global gain reached by the frozen run")
        for s in sorted(set(d) & set(g) & set(f)):
            share = {k: (d[s][k] - f[s][k]) / (d[s][k] - g[s][k]) if d[s][k] != g[s][k] else float("nan")
                     for k in ("force", "energy")}
            print(f"step {s}: force {share['force']:.2f}, energy {share['energy']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
