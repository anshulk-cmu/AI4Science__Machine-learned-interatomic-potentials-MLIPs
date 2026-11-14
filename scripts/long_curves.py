"""Validation curves of the long runs from their segment logs (segments.txt order; a later segment wins at a repeated step).

  python scripts/long_curves.py finalRuns/long_*_s0_400000 --out results/long_validation_curves.csv
Writes the columns of results/pilot_validation_curves.csv. Each validation is labelled with the last training step logged
before it plus one (compare_val_curves.py); only labels at multiples of eval_every in settings.txt are kept, others are
printed. The training loss is the mean of the logged train/loss over the 500 steps before the label.
"""
import argparse
import csv
import re
from pathlib import Path

from compare_val_curves import METRIC, STEP

LOSS = re.compile(r"'train/loss': ([0-9.eE+-]+).*'train/step': (\d+)")
VAL_LOSS = re.compile(r"^\s*val/loss: ([0-9.eE+-]+)")
RUN = re.compile(r"^long_(.+)_s(\d+)_(\d+)$")


def run_curve(run_dir):
    settings = (run_dir / "settings.txt").read_text()
    every = int(re.search(r"eval_every=(\d+)", settings).group(1))
    variant, seed, steps = RUN.match(run_dir.name).groups()
    logs = [run_dir / line.split(None, 1)[1].strip() for line in (run_dir / "segments.txt").read_text().splitlines()
            if line.startswith("log ")]
    train, val, step = {}, {}, 0
    for log in logs:
        event = {}
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = LOSS.search(line)
                if m:
                    train[int(m.group(2))] = float(m.group(1))
                    step = int(m.group(2)) + 1
                    continue
                m = STEP.search(line)
                if m:
                    step = int(m.group(1)) + 1
                    continue
                m = VAL_LOSS.search(line)
                if m:
                    event = {"val_loss": float(m.group(1))}
                for name, rx in METRIC.items():
                    m = rx.search(line)
                    if m:
                        event[name] = float(m.group(1))
                if {"val_loss", "force", "energy"} <= event.keys():
                    if step % every:
                        print(f"{run_dir.name}: validation at step {step} in {log.name} is off the {every} grid; dropped")
                    else:
                        val[step] = event
                    event = {}
    rows = []
    for s in sorted(val):
        window = [train[t] for t in range(s - 500, s) if t in train]
        rows.append(dict(variant=variant, seed=int(seed), step=s, val_force_mae_eV_per_A=val[s]["force"],
                         val_energy_mae_per_atom_eV=val[s]["energy"], val_loss=val[s]["val_loss"],
                         train_loss_mean_preceding_500_steps=sum(window) / len(window) if window else float("nan")))
    expected = set(range(0, int(steps) + 1, every))
    if set(val) - expected or expected - set(val) - {0}:
        print(f"{run_dir.name}: steps missing {sorted(expected - set(val))}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", type=Path, help="also draw validation energy and force against step")
    a = ap.parse_args()
    rows = [r for d in a.runs for r in run_curve(d)]
    with open(a.out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")
    if a.plot:
        plot(rows, a.plot)


def plot(rows, out):
    from figure_common import STYLE, plt
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for v, (name, color, marker, line) in STYLE.items():
        own = [r for r in rows if r["variant"].replace("global_frozen", "frozen") == v]
        for ax, key in zip(axes, ("val_energy_mae_per_atom_eV", "val_force_mae_eV_per_A")):
            if own:
                ax.plot([r["step"] for r in own], [r[key] for r in own], color=color, ls=line, marker=marker, ms=2.5,
                        lw=1, label=name)
    for ax, label in zip(axes, ("energy MAE per atom (eV)", r"force MAE (eV/$\AA$)")):
        ax.set(xlabel="training step", ylabel=label, yscale="log")
    axes[1].legend(ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Long runs, seed 0: validation on split_val_monitor_5k (logged, 4 decimals)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
