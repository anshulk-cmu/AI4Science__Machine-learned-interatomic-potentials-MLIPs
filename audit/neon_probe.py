"""Neon probe: one lone Ne atom far above a slab. A local model leaves the forces on the slab unchanged, a global one does not."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms

import geometry as g
import models
from run_audit import finished_runs

ROOT = Path(__file__).resolve().parents[1]
DISTANCES = (10, 50)        # surface to probe, angstrom; both beyond the 6 A cutoff


def probe_cell(slab, d):
    """Slab in a cell with the same empty space d above and below it, and the Ne atom d above the top plane."""
    base = g.isolated(slab, 2 * d)
    xy = base.cell.array[:2].sum(axis=0) / 2
    ne = Atoms("Ne", positions=[[xy[0], xy[1], slab.positions[:, 2].max() + d]], cell=base.cell, pbc=True)
    both = base + ne
    both.info = {"n_lower": len(slab)}
    return base, ne, both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["uma", "orb", "mace"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="notask")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--model-version", type=float, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()

    energy, cutoff = models.load(a.family, a.model, a.task, a.dtype, a.model_version)
    label = a.label or a.model + (f"-v{a.model_version:g}" if a.model_version else "")
    ideal = g.ideal_slab("Al")
    out = ROOT / "results" / f"neon_{label}_{a.task}.csv"
    done = finished_runs(out, len(DISTANCES))

    for run in range(a.runs + 1):
        if run in done:
            continue
        slab = g.rattled(ideal, g.seed_for(run, "Al_a"))
        rows = []
        for d in DISTANCES:
            base, ne, both = probe_cell(slab, d)
            g.assert_disconnected(both, cutoff)
            f0 = energy.forces(base)
            n = len(slab)
            dforce = np.linalg.norm(energy.forces(both)[:n] - f0, axis=1) * 1000.0     # meV/A per slab atom
            perm = np.random.default_rng(run).permutation(n)
            f_perm = np.empty_like(f0)
            f_perm[perm] = energy.forces(base[perm])
            # noise floor in meV/A: largest force change on a slab atom when only the atom order of the slab alone changes
            noise = np.linalg.norm(f_perm - f0, axis=1).max() * 1000.0
            rows.append(dict(model=label, task=a.task, dtype=a.dtype, distance=d, run=run,
                             d_energy_mev=(energy(both) - energy(base) - energy(ne)) * 1000.0,
                             max_dforce=dforce.max(), mean_dforce=dforce.mean(), noise_dforce=noise))
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{label} {a.task} run {run} done: max dF {rows[-1]['max_dforce']:.4g} meV/A at {DISTANCES[-1]} A")


if __name__ == "__main__":
    main()
