"""Statistics of final validation metrics, with seed counts inferred from the CSV.

Usage: python scripts/pilot_statistics.py [metrics.csv] [--require-final]
See docs/final-statistics.md for methods, assumptions and interpretation.
"""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
METRICS = {"energy_mae_per_atom_eV": "energy MAE per atom (eV)", "force_mae_eV_per_A": "force MAE (eV/A)"}
MODELS = ["dense", "global", "element", "local6", "local12", "local_global", "frozen", "dense_scaled"]
MAX_EXACT_COMBINATIONS = 5_000_000


def validate_data(df, require_final=False):
    """Return normalized data; each present model must have the same unique seeds."""
    missing = {"variant", "seed", *METRICS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.copy()
    if df.empty or df.variant.isna().any():
        raise ValueError("Input must contain named variants and metric rows")
    df["variant"] = df.variant.replace({"global_frozen": "frozen"})
    unknown = set(df.variant) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown, key=str)}")
    if not {"dense", "global"}.issubset(set(df.variant)):
        raise ValueError("Both dense and global are required for retention")
    for column in ["seed", *METRICS]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all() or (df[column] < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative numbers")
    if (df.seed != np.floor(df.seed)).any():
        raise ValueError("Seeds must be integers")
    if df.duplicated(["variant", "seed"]).any():
        raise ValueError("Duplicate (variant, seed) rows")
    seeds = set(df.loc[df.variant == "dense", "seed"])
    if len(seeds) < 2:
        raise ValueError("At least two seeds per variant are required")
    for model, rows in df.groupby("variant"):
        if set(rows.seed) != seeds:
            raise ValueError(f"Seed set for {model} differs from dense: expected {sorted(seeds)}")
    if require_final and (set(df.variant) != set(MODELS) or seeds != set(range(5))):
        raise ValueError("Final analysis requires all 8 variants and seeds 0, 1, 2, 3, 4")
    return df


def values(df, model, metric):
    return df[df.variant == model].sort_values("seed")[metric].to_numpy(dtype=float)


def interval(x):
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def retention(d, g, x):
    return (d - x) / (d - g)


def check_exact_size(n):
    combinations = math.comb(2 * n - 1, n) ** 3
    if combinations > MAX_EXACT_COMBINATIONS:
        raise ValueError(f"Exact resampling for {n} seeds requires {combinations:,} compressed combinations; "
                         f"limit is {MAX_EXACT_COMBINATIONS:,}. Use --skip-exact explicitly.")


def bootstrap_means(x):
    """Bootstrap means and multiplicities, grouped by unordered seed selections."""
    n = len(x)
    means, weights = [], []
    for selection in itertools.combinations_with_replacement(range(n), n):
        counts = np.bincount(selection, minlength=n)
        means.append(np.dot(counts, x) / n)
        weights.append(math.factorial(n) // math.prod(math.factorial(int(c)) for c in counts))
    return np.asarray(means), np.asarray(weights, dtype=np.int64)


def weighted_percentile(x, weights, percentile=5):
    """Match np.percentile(method='linear') on integer-weight-expanded samples."""
    order = np.argsort(x)
    x, weights = x[order], weights[order]
    cumulative = np.cumsum(weights, dtype=np.int64)
    rank = (int(cumulative[-1]) - 1) * percentile / 100
    left = np.searchsorted(cumulative, math.floor(rank), side="right")
    right = np.searchsorted(cumulative, math.ceil(rank), side="right")
    return float(x[left] + (rank - math.floor(rank)) * (x[right] - x[left]))


def exact_bound(d, g, x):
    """Exact independent seed bootstrap, conditional on Dense error > Global error."""
    n = len(d)
    if len(g) != n or len(x) != n:
        raise ValueError("Exact resampling requires equal seed counts")
    check_exact_size(n)
    dm, dw = bootstrap_means(d)
    gm, gw = bootstrap_means(g)
    xm, xw = bootstrap_means(x)
    # Filter Dense/Global pairs before dividing; no zero-denominator warnings.
    di, gi = np.nonzero(dm[:, None] > gm[None, :])
    if not len(di):
        return float("nan"), 0.0
    ratios = retention(dm[di, None], gm[gi, None], xm[None, :]).ravel()
    weights = (dw[di, None] * gw[gi, None] * xw[None, :]).ravel()
    accepted = int(weights.sum()) / n ** (3 * n)
    return weighted_percentile(ratios, weights), accepted


def bounds(df, metric, model, draws, seed=0, skip_exact=False):
    """Conditional fifth percentiles and accepted fractions for all three methods.

    Draw each model independently, retaining the legacy RNG order. Student-t
    degrees of freedom are n-1. The t2 alias preserves the old three-seed API.
    """
    df = validate_data(df)
    if model in ("dense", "global") or model not in set(df.variant):
        raise ValueError("Retention bounds require a present non-baseline variant")
    if draws < 1 or int(draws) != draws:
        raise ValueError("draws must be a positive integer")
    v = {m: values(df, m, metric) for m in ("dense", "global", model)}
    n = len(v[model])
    if not skip_exact:
        check_exact_size(n)
    rng = np.random.default_rng(seed)
    out = {"df": n - 1, "accepted": {}}
    for name, draw in (("normal", lambda x: rng.normal(x.mean(), x.std(ddof=1) / np.sqrt(len(x)), draws)),
                       ("t", lambda x: x.mean() + x.std(ddof=1) / np.sqrt(len(x)) * rng.standard_t(n - 1, draws))):
        s = {m: draw(x) for m, x in v.items()}
        ok = s["dense"] > s["global"]
        out["accepted"][name] = float(ok.mean())
        out[name] = float(np.percentile(retention(s["dense"][ok], s["global"][ok], s[model][ok]), 5)) if ok.any() else float("nan")
    if n == 3:
        out["t2"] = out["t"]
    if not skip_exact:
        out["exact"], out["accepted"]["exact"] = exact_bound(v["dense"], v["global"], v[model])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", default=str(ROOT / "results" / "pilot_final_metrics.csv"))
    ap.add_argument("--draws", type=int, default=1_000_000)
    ap.add_argument("--rng-seed", type=int, default=0, help="Monte Carlo seed (default: 0)")
    ap.add_argument("--require-final", action="store_true", help="Require all 8 variants and seeds 0 through 4")
    ap.add_argument("--skip-exact", action="store_true", help="Explicitly omit exact bootstrap for larger seed counts")
    args = ap.parse_args(argv)
    try:
        if args.draws < 1 or args.rng_seed < 0:
            raise ValueError("draws must be positive and rng-seed must be nonnegative")
        df = validate_data(pd.read_csv(args.csv), args.require_final)
        n = len(values(df, "dense", next(iter(METRICS))))
        if not args.skip_exact:
            check_exact_size(n)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    models = [m for m in MODELS if m in set(df.variant)]
    seeds = sorted(df.seed.unique())
    print(f"Variants: {len(models)}; seeds per variant: {n}; seeds: {', '.join(str(int(s)) for s in seeds)}")
    print(f"Monte Carlo draws: {args.draws}; RNG seed: {args.rng_seed}; Student-t df: {n - 1}")
    print("Retention bounds condition on Dense > Global; model draws are independent. Welch p-values are unadjusted.")
    for metric, label in METRICS.items():
        print(f"== {label}")
        v = {m: values(df, m, metric) for m in models}
        for m in models:
            lo, hi = interval(v[m])
            print(f"  {m:13s} n {n}  mean {v[m].mean():.6f}  sd {v[m].std(ddof=1):.6f}  95% t interval [{lo:.6f}, {hi:.6f}]")
        for m in models[1:]:
            reduction = 100 * (1 - v[m].mean() / v['dense'].mean()) if v['dense'].mean() > 0 else float('nan')
            print(f"  {m:13s} error is {reduction:.2f} percent below Dense")
        for a, b in itertools.combinations(models, 2):
            # Welch is undefined when both sample variances vanish.
            p = stats.ttest_ind(v[a], v[b], equal_var=False).pvalue if np.ptp(v[a]) or np.ptp(v[b]) else float('nan')
            print(f"  Welch {a} vs {b}: difference {v[a].mean() - v[b].mean():+.6f}, p = {p:.4f}")
        aligned = df.pivot(index="seed", columns="variant", values=metric).reindex(seeds)
        for m in [m for m in models if m not in ("dense", "global")]:
            # Reindex by seed explicitly for the descriptive paired ratios.
            d, g, x = (aligned[k].to_numpy() for k in ("dense", "global", m))
            point = retention(d.mean(), g.mean(), x.mean()) if d.mean() > g.mean() else float("nan")
            per = ", ".join(f"{int(s)}:{retention(a, b, c):.3f}" if a > b else f"{int(s)}:undefined"
                            for s, a, b, c in zip(seeds, d, g, x))
            b = bounds(df, metric, m, args.draws, args.rng_seed, args.skip_exact)
            print(f"  retention of the Global gain by {m}: ratio of means {point:.4f}; per seed {per}")
            exact = "skipped" if args.skip_exact else f"{b['exact']:.3f}"
            print(f"    lower bound (5th percentile): normal {b['normal']:.3f}, exact resample {exact}, t({b['df']}) {b['t']:.3f}")
            print("    accepted Dense > Global: " + ", ".join(f"{k} {f:.2%}" for k, f in b["accepted"].items()))
            if m == "local6":
                decision = ("PASS" if b["normal"] >= 0.5 else "FAIL") if np.isfinite(point) and np.isfinite(b["normal"]) else "UNAVAILABLE"
                print(f"    Local-6 decision for {label}: {decision} (normal lower bound >= 0.5)")


if __name__ == "__main__":
    main()
