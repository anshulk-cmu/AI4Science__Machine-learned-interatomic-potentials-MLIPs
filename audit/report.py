"""Write the results tables (markdown) and one overview figure from the files made by analyze.py."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def num(x):
    return f"{x:,.3f}" if abs(x) < 10 else f"{x:,.1f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def tables(s, flat, comp):
    d = s[s.distance == 100]
    main = [(r.model, r.task, num(r.ideal_mev), num(r.mean_mev), num(r.sd_mev),
             f"[{num(r.ci_lo)}, {num(r.ci_hi)}]", r.verdict) for r in d[d.case == "AlNi"].itertuples()]
    ctrl = [(r.model, r.task, r.case, num(r.mean_mev), num(r.sd_mev), r.verdict)
            for r in d[d.case != "AlNi"].itertuples()]
    flat_rows = [(r.model, r.task, r.case, num(r.diff_mev), "yes" if r.flat else "NO") for r in flat.itertuples()]
    paper = [(r.model, r.task, num(r.ours_mean_mev), "-" if np.isnan(r.paper_mev) else num(r.paper_mev))
             for r in comp.itertuples()]
    return "\n\n".join([
        "## Al + Ni, gap 100 A (meV; ideal geometry, then the mean, sd and 95 percent CI over the shaken runs)",
        table(["model", "task", "ideal", "mean", "sd", "95% CI", "verdict"], main),
        "## Same-element controls, gap 100 A (meV)",
        table(["model", "task", "case", "mean", "sd", "verdict"], ctrl),
        "## Is Delta at 100 A equal to Delta at 8 A, within the tolerance?",
        table(["model", "task", "case", "mean difference", "flat"], flat_rows),
        "## Ours against the paper's printed value (meV, Al + Ni)",
        table(["model", "task", "ours (mean)", "paper"], paper)])


def overview(s):
    d = s[(s.distance == 100)]
    keys = list(d[["model", "task"]].drop_duplicates().itertuples(index=False))
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(keys)), 4.2))
    width = 0.27
    for k, case in enumerate(("AlNi", "AlAl", "NiNi")):
        vals = []
        for m, t in keys:
            r = d[(d.model == m) & (d.task == t) & (d.case == case)].iloc[0]
            vals.append(max(abs(r.mean_mev), r.tau_mev))          # values under tau are drawn at tau
        ax.bar(np.arange(len(keys)) + (k - 1) * width, vals, width, label=case)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f"{m}\n{t}" for m, t in keys], rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="|Delta| at gap 100 A (meV, floor = tolerance)", title="Size-consistency error by model")
    ax.legend()
    fig.tight_layout()
    (RES / "figures").mkdir(exist_ok=True)
    fig.savefig(RES / "figures" / "overview.png", dpi=150)


def main():
    s = pd.read_csv(RES / "summary.csv")
    text = tables(s, pd.read_csv(RES / "flatness.csv"), pd.read_csv(RES / "comparison.csv"))
    (RES / "results.md").write_text("# Audit results\n\n" + text + "\n", encoding="utf-8")
    overview(s)
    print("wrote results.md and figures/overview.png")


if __name__ == "__main__":
    main()
