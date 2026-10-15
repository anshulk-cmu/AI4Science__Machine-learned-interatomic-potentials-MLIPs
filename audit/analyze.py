"""Merge the energy CSVs, apply the decision rule, and write the summary tables and figures."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = {  # printed in arXiv 2510.25380 Table XII, meV (the uma-m omat row is internally inconsistent)
    ("uma-s-1p1", "omat"): -453.8, ("uma-m-1p1", "omat"): 1436.9, ("orb-v3", "notask"): -709.7,
    ("uma-s-1p1", "omol"): 88670.6, ("uma-m-1p1", "omol"): -385824.5,
}


def load():
    df = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("energies_*.csv"))], ignore_index=True)
    df["delta_mev"] = (df.e_union.astype("float64") - df.e_a - df.e_b) * 1000.0
    return df


def check_dtypes(df):
    """One dtype per (model, task) in the energy files, and the same one in its null tests when they record it."""
    dtypes = df.groupby(["model", "task"])["dtype"].unique()
    for key, v in dtypes.items():
        if len(v) != 1:
            raise ValueError(f"{key}: energy files mix dtypes {sorted(v)}")
    for p in sorted(RES.glob("nulls_*.csv")):
        n = pd.read_csv(p)
        key = (n.model[0], n.task[0])
        if "dtype" not in n:
            print(f"warning: {p.name} has no dtype column, so its precision is not checked")
        elif key in dtypes and set(n["dtype"]) != {dtypes[key][0]}:
            raise ValueError(f"{key}: null tests ran in {sorted(set(n['dtype']))}, energies in {dtypes[key][0]}")


def tolerances():
    """tau = 5 sigma_E per (model, task), from the null experiments, with a floor of 1e-3 meV."""
    out = {}
    for p in sorted(RES.glob("nulls_*.csv")):
        n = pd.read_csv(p)
        sigma_mev = np.sqrt(((n.energy - n.ref) ** 2).mean()) * 1000.0
        out[(n.model[0], n.task[0])] = max(5.0 * sigma_mev, 1e-3)
    return out


def interval(x):
    """Mean and 95 percent t-interval of a sample."""
    n = len(x)
    half = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), x.mean() - half, x.mean() + half


def classify(x, tau):
    x = np.asarray(x, dtype="float64")
    if len(x) < 3 or np.isnan(x).any():
        raise ValueError(f"classify needs at least 3 runs and no NaN, got {len(x)} values")
    mean, lo, hi = interval(x)
    same_sign = (np.sign(x) == np.sign(mean)).sum() >= -(-95 * len(x) // 100)   # ceil(0.95 n) in integers
    if (lo > tau or hi < -tau) and same_sign:
        return "VIOLATION"
    return "CONSISTENT" if (lo >= -tau and hi <= tau) else "INCONCLUSIVE"


def summarize(df, tau):
    rows = []
    for (m, t, c, d), grp in df[df.run > 0].groupby(["model", "task", "case", "distance"]):
        x = grp.delta_mev.to_numpy()
        mean, lo, hi = interval(x)
        ideal = df[(df.run == 0) & (df.model == m) & (df.task == t) & (df.case == c) & (df.distance == d)]
        rows.append(dict(model=m, task=t, case=c, distance=d, n=len(x), mean_mev=mean, sd_mev=x.std(ddof=1),
                         ci_lo=lo, ci_hi=hi, ideal_mev=float(ideal.delta_mev.iloc[0]),
                         tau_mev=tau[(m, t)], verdict=classify(x, tau[(m, t)])))
    return pd.DataFrame(rows)


def flatness(df, tau):
    """P2: is Delta at 100 A equal to Delta at 8 A, run by run, within tau?"""
    rows = []
    for (m, t, c), grp in df[df.run > 0].groupby(["model", "task", "case"]):
        wide = grp.pivot(index="run", columns="distance", values="delta_mev")
        mean, lo, hi = interval((wide[100] - wide[8]).to_numpy())
        rows.append(dict(model=m, task=t, case=c, diff_mev=mean, ci_lo=lo, ci_hi=hi,
                         flat=bool(lo >= -tau[(m, t)] and hi <= tau[(m, t)])))
    return pd.DataFrame(rows)


def compare(summary):
    s = summary[(summary.case == "AlNi") & (summary.distance == 100)]
    rows = [dict(model=r.model, task=r.task, ours_mean_mev=r.mean_mev, ours_ideal_mev=r.ideal_mev,
                 paper_mev=PAPER.get((r.model, r.task), np.nan)) for r in s.itertuples()]
    return pd.DataFrame(rows)


def plot(summary):
    fig_dir = RES / "figures"
    fig_dir.mkdir(exist_ok=True)
    pairs = summary[["model", "task"]].drop_duplicates().itertuples(index=False)
    for m, t in pairs:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sub = summary[(summary.model == m) & (summary.task == t)]
        for c, grp in sub.groupby("case"):
            ax.errorbar(grp.distance, grp.mean_mev, yerr=[grp.mean_mev - grp.ci_lo, grp.ci_hi - grp.mean_mev],
                        marker="o", capsize=2, label=c)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=max(float(sub.tau_mev.iloc[0]), 1e-3))
        ax.set(xlabel="gap (angstrom)", ylabel="Delta (meV)", title=f"{m} {t}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"delta_{m}_{t}.png", dpi=150)
        plt.close(fig)


def main():
    df, tau = load(), tolerances()
    check_dtypes(df)
    summary = summarize(df, tau)
    summary.to_csv(RES / "summary.csv", index=False)
    flatness(df, tau).to_csv(RES / "flatness.csv", index=False)
    compare(summary).to_csv(RES / "comparison.csv", index=False)
    plot(summary)
    print(summary.groupby(["model", "task", "case"]).verdict.agg(lambda v: ",".join(sorted(set(v)))))


if __name__ == "__main__":
    main()
