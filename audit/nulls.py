"""Null experiments for one model: changes that cannot alter the energy, used to set the tolerance."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import geometry as g
import models

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["uma", "orb", "mace"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="notask")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--model-version", type=float, default=None)   # router version override, UMA only
    a = ap.parse_args()
    energy, _ = models.load(a.family, a.model, a.task, a.dtype, a.model_version)
    label = a.model + (f"-v{a.model_version:g}" if a.model_version else "")
    al, ni = g.ideal_slab("Al"), g.ideal_slab("Ni")
    rng = np.random.default_rng(7)
    rows = []

    def add(test, name, value, ref):
        rows.append(dict(model=label, task=a.task, test=test, structure=name, energy=value, ref=ref))

    for name, atoms in (("Al_slab", g.isolated(al)), ("AlNi_15", g.stacked(al, ni, 15))):
        ref = energy(atoms)
        add("N1", name, energy(atoms), ref)                        # same input again
        for _ in range(5):
            add("N2", name, energy(atoms[rng.permutation(len(atoms))]), ref)   # shuffled atom order
        for _ in range(5):
            moved = atoms.copy()
            moved.translate(rng.uniform(0.0, 5.0, 3))
            moved.wrap()
            add("N3", name, energy(moved), ref)                    # shifted origin
    ref = energy(g.isolated(al, 20.0))
    for pad in (12.0, 20.0, 40.0):
        add("N4", f"Al_pad{pad:g}", energy(g.isolated(al, pad)), ref)   # empty space above the slab

    out = ROOT / "results" / f"nulls_{label}_{a.task}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    dev = np.array([r["energy"] - r["ref"] for r in rows])
    print(f"sigma_E = {np.sqrt((dev ** 2).mean()) * 1000:.6f} meV over {len(dev)} repeats")


if __name__ == "__main__":
    main()
