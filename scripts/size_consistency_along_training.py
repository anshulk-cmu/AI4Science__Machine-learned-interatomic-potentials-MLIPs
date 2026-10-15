"""Size consistency along training: Delta = E(Al+Ni) - E(Al) - E(Ni) for every saved checkpoint of the nine pilots.

Same slabs, gaps, charge and spin as audit/check_checkpoint.py (the final-checkpoint stage of slurm/pilot.sh).
  run:   python scripts/size_consistency_along_training.py --out-dir DIR
  plot:  python scripts/size_consistency_along_training.py --plot results/size_consistency_training.csv
Writes size_consistency_training.csv, size_consistency_training_router.csv (Global: router mixing weights of the
union and of each slab) and size_consistency_training_loss.csv (train and validation loss at each checkpoint).
"""
import argparse
import csv
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^pilot_(dense|global|local6)_s(\d+)_(\d+)$")
FLOOR_MEV = 1e-3                                     # Dense and Local-6 must stay below this at every checkpoint
TRAIN_WINDOW = 500                                   # train loss is the mean of the last 500 logged steps


def checkpoints(run_dir):
    """(step, path) for every inference checkpoint of a run, 'final' labelled with the run's total steps."""
    steps = int(re.search(r"steps=(\d+)", (run_dir / "settings.txt").read_text()).group(1))
    out = []
    for p in run_dir.glob("train/*/checkpoints/*/inference_ckpt.pt"):
        name = p.parent.name
        out.append((steps if name == "final" else int(name.removeprefix("step_")), p))
    return sorted(out)


def final_delta_from_summary(run_dir):
    """The Delta (meV) the pilot job printed for its final checkpoint, by gap."""
    m = re.findall(r"(\d+) A: (\S+)", next(l for l in (run_dir / "summary.txt").read_text().splitlines() if "Delta meV" in l))
    return {int(g): float(v) for g, v in m}


def losses(run_dir, steps):
    """Per checkpoint step: mean train loss of the last TRAIN_WINDOW steps before it, and the last validation loss
    logged before its 'Saved dcp checkpoint' line (pilot.sh evaluates, then saves)."""
    log = run_dir / "logs" / "train.log"
    if not log.exists():
        return {}
    train, val, out = [], None, {}
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "'train/loss'" in line:
                m = re.search(r"'train/loss': ([0-9.eE+-]+).*'train/step': (\d+)", line)
                train.append((int(m.group(2)), float(m.group(1))))
            elif line.strip().startswith("val/loss:"):
                val = float(line.split()[1])
            elif "Saved dcp checkpoint to" in line:
                name = line.strip().rsplit("/", 1)[1]
                step = steps if name == "final" else int(name.removeprefix("step_"))
                window = [v for s, v in train if step - TRAIN_WINDOW <= s < step]
                out[step] = (sum(window) / len(window) if window else float("nan"), val if step else float("nan"))
    return out


def evaluate(runs, a):
    sys.path.insert(0, str(ROOT / "audit"))
    import torch
    import geometry as g
    import models

    al, ni = g.ideal_slab("Al"), g.ideal_slab("Ni")
    rows, router_rows, loss_rows, failed = [], [], [], 0
    for run_dir, variant, seed in runs:
        ckpts = checkpoints(run_dir)
        for step, (tr, va) in sorted(losses(run_dir, ckpts[-1][0]).items()):
            loss_rows.append(dict(run=run_dir.name, variant=variant, seed=seed, step=step, train_loss_mean500=tr, val_loss=va))
        for step, path in ckpts:
            try:
                energy, cutoff = models.load("uma", str(path), a.task, a.dtype)
                backbone = next((m for m in energy.unit.model.modules() if hasattr(m, "global_mole_tensors")), None)

                def mix():
                    return backbone.global_mole_tensors.expert_mixing_coefficients[0].tolist()

                e_al = energy(g.isolated(al))
                w = {"Al": mix() if variant == "global" else None}
                e_ni = energy(g.isolated(ni))
                w["Ni"] = mix() if variant == "global" else None
                for gap in a.gaps:
                    pair = g.stacked(al, ni, gap)
                    g.assert_disconnected(pair, cutoff)
                    e_pair = energy(pair)
                    delta = (e_pair - e_al - e_ni) * 1000.0
                    rows.append(dict(run=run_dir.name, variant=variant, seed=seed, step=step, gap=gap,
                                     E_union=repr(e_pair), E_a=repr(e_al), E_b=repr(e_ni), Delta_meV=repr(delta)))
                    if variant == "global":
                        w[f"union_{gap}"] = mix()
                    print(f"{run_dir.name} step {step} gap {gap}: Delta {delta:.6g} meV", flush=True)
                if variant == "global":
                    for which, coeffs in w.items():
                        router_rows.append(dict(run=run_dir.name, seed=seed, step=step, structure=which,
                                                **{f"w{k}": repr(c) for k, c in enumerate(coeffs)}))
                del energy, backbone
                torch.cuda.empty_cache()
            except Exception:                            # one bad checkpoint must not hide the others
                failed += 1
                print(f"{path} FAILED", flush=True)
                traceback.print_exc()
    return rows, router_rows, loss_rows, failed


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def selfcheck(rows, runs):
    """Dense and Local-6 below FLOOR_MEV at every checkpoint, and the final Delta equal to what the pilot job printed."""
    bad = [r for r in rows if r["variant"] != "global" and abs(float(r["Delta_meV"])) >= FLOOR_MEV]
    assert not bad, f"{len(bad)} Dense or Local-6 rows at or above {FLOOR_MEV} meV, first {bad[0]}"
    for run_dir, _, _ in runs:
        last = max(int(r["step"]) for r in rows if r["run"] == run_dir.name)
        for gap, printed in final_delta_from_summary(run_dir).items():
            ours = next(float(r["Delta_meV"]) for r in rows if r["run"] == run_dir.name and int(r["step"]) == last and int(r["gap"]) == gap)
            # the job printed 6 significant digits; the floor values are noise, so compare those on the floor scale
            assert abs(ours - printed) <= max(1e-5 * abs(printed), FLOOR_MEV), f"{run_dir.name} gap {gap}: {ours} vs printed {printed}"
    print(f"self-check passed: Dense and Local-6 below {FLOOR_MEV} meV at every checkpoint; final Delta equals the pilot summaries")


def plot(csv_path, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["abs"] = df.Delta_meV.abs()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {"dense": "#4C72B0", "global": "#C44E52", "local6": "#55A868"}
    names = {"dense": "Dense", "global": "Global", "local6": "Local-6"}
    for ax, gap in zip(axes, sorted(df.gap.unique())):
        d = df[df.gap == gap]
        for v, c in colors.items():
            dv = d[d.variant == v]
            for _, s in dv.groupby("seed"):
                ax.plot(s.step, s["abs"], color=c, lw=0.8, alpha=0.5)
            m = dv.groupby("step")["abs"].mean()
            ax.plot(m.index, m.values, color=c, lw=2.5, marker="o", ms=4, label=f"{names[v]} (mean of 3 seeds)")
        ax.axhline(FLOOR_MEV, color="gray", ls="--", lw=1, label="1e-3 meV")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("training step")
        ax.set_title(f"Al + Ni slabs, gap {gap} A")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|Delta| (meV), symlog, linear below 1e-6")
    axes[0].legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/ocean/projects/che260063p/shared/runs")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--task", default="omol")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--gaps", type=int, nargs="+", default=[8, 100])
    ap.add_argument("--plot", metavar="CSV", help="only draw the figure from an existing CSV")
    a = ap.parse_args()
    if a.plot:
        plot(a.plot, ROOT / "results" / "figures" / "size_consistency_training.png")
        return
    runs = sorted((p, m.group(1), int(m.group(2))) for p in Path(a.runs_root).iterdir() if (m := RUN_RE.match(p.name)))
    print(f"{len(runs)} runs: " + ", ".join(p.name for p, _, _ in runs), flush=True)
    rows, router_rows, loss_rows, failed = evaluate(runs, a)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "size_consistency_training.csv", rows)
    if router_rows:
        write(out / "size_consistency_training_router.csv", router_rows)
    if loss_rows:
        write(out / "size_consistency_training_loss.csv", loss_rows)
    selfcheck(rows, runs)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
