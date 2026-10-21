"""Statistics of the final validation errors of the pilots (3 seeds per model).

Input: results/pilot_final_metrics.csv (variant, seed, energy MAE per atom in eV, force MAE in eV/A, validation loss),
the last validation values of each run read from its offline W&B file. Usage:

    python scripts/pilot_statistics.py [results/pilot_final_metrics.csv] [--draws 1000000]

Prints, per metric: mean, sd and 95 percent Student-t interval of each model; reductions against Dense; Welch tests of
every pair; the retention share r = (e_Dense - e_X) / (e_Dense - e_Global) as a ratio of means and per seed; and
one-sided 95 percent lower bounds of r from (a) normal draws around each seed mean with the standard error s / sqrt(3)
(the main method of the report), (b) the exact resample of all 3^9 = 19,683 seed combinations, (c) draws with a
Student-t distribution of 2 degrees of freedom. All bounds come from only 3 seeds per model.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
METRICS = {"energy_mae_per_atom_eV": "energy MAE per atom (eV)", "force_mae_eV_per_A": "force MAE (eV/A)"}
MODELS = ["dense", "global", "local6", "frozen", "dense_scaled"]


def values(df, model, metric):
    return df[df.variant == model].sort_values("seed")[metric].to_numpy()


def interval(x):
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def retention(d, g, x):
    return (d - x) / (d - g)


def bounds(df, metric, model, draws, seed=0):
    """Lower bounds (5th percentile of r) by the three methods; Dense must be above Global in a draw."""
    rng = np.random.default_rng(seed)
    v = {m: values(df, m, metric) for m in ("dense", "global", model)}
    out = {}
    for name, draw in (("normal", lambda x: rng.normal(x.mean(), x.std(ddof=1) / np.sqrt(len(x)), draws)),
                       ("t2", lambda x: x.mean() + x.std(ddof=1) / np.sqrt(len(x)) * rng.standard_t(2, draws))):
        s = {m: draw(x) for m, x in v.items()}
        ok = s["dense"] > s["global"]
        out[name] = np.percentile(retention(s["dense"], s["global"], s[model])[ok], 5)
    means = {m: np.array([x[list(c)].mean() for c in itertools.product(range(3), repeat=3)]) for m, x in v.items()}
    a, b, c = np.meshgrid(means["dense"], means["global"], means[model], indexing="ij")
    ok = a > b
    out["exact"] = np.percentile(retention(a, b, c)[ok], 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(ROOT / "results" / "pilot_final_metrics.csv"))
    ap.add_argument("--draws", type=int, default=1_000_000)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    models = [m for m in MODELS if m in set(df.variant)]
    for metric, label in METRICS.items():
        print(f"== {label}")
        v = {m: values(df, m, metric) for m in models}
        for m in models:
            lo, hi = interval(v[m])
            print(f"  {m:13s} mean {v[m].mean():.6f}  sd {v[m].std(ddof=1):.6f}  95% t interval [{lo:.6f}, {hi:.6f}]")
        for m in models[1:]:
            print(f"  {m:13s} error is {100 * (1 - v[m].mean() / v['dense'].mean()):.2f} percent below Dense")
        for a, b in itertools.combinations(models, 2):
            p = stats.ttest_ind(v[a], v[b], equal_var=False).pvalue
            print(f"  Welch {a} vs {b}: difference {v[a].mean() - v[b].mean():+.6f}, p = {p:.4f}")
        for m in [m for m in models if m not in ("dense", "global")]:
            d, g, x = v["dense"], v["global"], v[m]
            per = ", ".join(f"{r:.3f}" for r in retention(d, g, x))
            b = bounds(df, metric, m, args.draws)
            print(f"  retention of the Global gain by {m}: ratio of means {retention(d.mean(), g.mean(), x.mean()):.4f}; "
                  f"per seed {per}")
            print(f"    lower bound (5th percentile): normal {b['normal']:.3f}, exact resample {b['exact']:.3f}, t(2) {b['t2']:.3f}")


if __name__ == "__main__":
    main()
