"""Run the size-consistency audit for one (model, task) and write all energies to a CSV."""
import argparse
from pathlib import Path

import pandas as pd

import geometry as g
import models

ROOT = Path(__file__).resolve().parents[1]
FRAGMENTS = {"Al_a": "Al", "Al_b": "Al", "Ni_a": "Ni", "Ni_b": "Ni"}
CASES = {"AlNi": ("Al_a", "Ni_a"), "AlAl": ("Al_a", "Al_b"), "NiNi": ("Ni_a", "Ni_b")}


def finished_runs(out, per_run):
    """Runs already saved in full, for resuming after a crash. Rows of an unfinished run are removed from the file."""
    if not out.exists():
        return set()
    df = pd.read_csv(out)
    counts = df.run.value_counts()
    done = set(counts[counts == per_run].index)
    if len(done) * per_run != len(df):
        df[df.run.isin(done)].to_csv(out, index=False)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["uma", "orb", "mace"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="notask")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--model-version", type=float, default=None)   # router version override, UMA only
    ap.add_argument("--label", default=None)                       # name for the output files, needed for checkpoint paths
    ap.add_argument("--runs", type=int, default=20)       # rattled runs, plus run 0 (ideal)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    energy, cutoff = models.load(a.family, a.model, a.task, a.dtype, a.model_version)
    label = a.label or a.model + (f"-v{a.model_version:g}" if a.model_version else "")
    ideal = {s: g.ideal_slab(s) for s in ("Al", "Ni")}
    out = ROOT / "results" / f"energies_{label}_{a.task}.csv"
    out.parent.mkdir(exist_ok=True)
    done = finished_runs(out, len(CASES) * len(g.DISTANCES))

    for run in range(a.runs + 1):
        if run in done:
            continue
        frag = {n: g.rattled(ideal[s], g.seed_for(run, n)) for n, s in FRAGMENTS.items()}
        e_iso = {n: energy(g.isolated(f)) for n, f in frag.items()}
        rows = []
        for case, (lo, up) in CASES.items():
            for d in g.DISTANCES:
                pair = g.stacked(frag[lo], frag[up], d)
                g.assert_disconnected(pair, cutoff)
                rows.append(dict(model=label, task=a.task, dtype=a.dtype, case=case, distance=d,
                                 run=run, e_a=e_iso[lo], e_b=e_iso[up], e_union=energy(pair)))
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{label} {a.task} run {run} done (cutoff {cutoff:g} A)")

    if a.wandb:
        import wandb
        run = wandb.init(project="mlip-audit", name=f"{label}-{a.task}", config={**vars(a), "cutoff": cutoff})
        art = wandb.Artifact(f"energies-{label}-{a.task}", type="results")
        art.add_file(str(out))
        run.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()
