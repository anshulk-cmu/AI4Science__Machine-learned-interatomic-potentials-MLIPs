"""Regenerate the three main-text figures of the midterm report from the result CSVs.

Usage: python make_midterm_figures.py <results_dir> <out_dir>
Writes audit_combined.png, size_consistency_training.png, router_check.png into out_dir.
Okabe-Ito colours (colour-blind safe); figure width 5.5 in = NeurIPS text width, so the
font sizes below are the printed sizes when the figure is included at \\linewidth.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
WIDTH = 5.5


def audit(res, out):
    s = pd.read_csv(res / "summary.csv")
    labels = [("medium", "notask", "MACE-MP-0", OI["black"], "o"),
              ("orb-v3", "notask", "ORB-v3", OI["orange"], "s"),
              ("uma-s-1p1", "omat", "UMA-S-1.1 omat", OI["sky"], "^"),
              ("uma-s-1p1", "omol", "UMA-S-1.1 omol", OI["sky"], "v"),
              ("uma-s-1p2", "omat", "UMA-S-1.2 omat", OI["blue"], "^"),
              ("uma-s-1p2", "omol", "UMA-S-1.2 omol", OI["blue"], "v"),
              ("uma-m-1p1", "omat", "UMA-M-1.1 omat", OI["vermillion"], "^"),
              ("uma-m-1p1", "omol", "UMA-M-1.1 omol", OI["vermillion"], "v")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.3), gridspec_kw={"width_ratios": [1.6, 1]})
    for model, task, name, col, mk in labels:
        d = s[(s.model == model) & (s.task == task) & (s.case == "AlNi")].sort_values("distance")
        assert len(d) == 6, (model, task)
        y = d.mean_mev.abs().to_numpy()
        hollow = model == "medium"
        a.plot(d.distance, y, marker=mk, color=col, lw=1, ms=4,
               mfc="none" if hollow else col, label=name)
    a.axhline(1e-3, ls="--", lw=0.8, color="grey")
    a.text(8.3, 1e-5, "float64 tolerance 0.001 meV", fontsize=7, color="grey")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks([8, 10, 15, 25, 50, 100]); a.set_xticklabels(["8", "10", "15", "25", "50", "100"])
    a.set_xlabel("gap between the slabs (Å)"); a.set_ylabel(r"$|\Delta|$, Al+Ni (meV)")
    a.set_ylim(1e-10, 3e6); a.set_yticks([1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6])
    fig.legend(*a.get_legend_handles_labels(), ncol=4, loc="upper center", frameon=False, fontsize=7, columnspacing=0.8, handlelength=1.6)
    a.set_title("(a) Al+Ni against the gap", fontsize=8)
    # (b) same-metal pairs at 100 A: UMA-S-1.1 (N divisor) against UMA-S-1.2 (N+1 divisor)
    cells = [("uma-s-1p1", "omat"), ("uma-s-1p1", "omol"), ("uma-s-1p2", "omat"), ("uma-s-1p2", "omol")]
    xt = []
    for i, (model, task) in enumerate(cells):
        for j, (case, mk) in enumerate([("AlAl", "o"), ("NiNi", "D")]):
            r = s[(s.model == model) & (s.task == task) & (s.case == case) & (s.distance == 100)]
            assert len(r) == 1
            v = abs(float(r.mean_mev.iloc[0]))
            col = OI["sky"] if model == "uma-s-1p1" else OI["blue"]
            b.plot(i + (j - 0.5) * 0.25, v, marker=mk, color=col, ms=5, ls="none",
                   label=("Al+Al" if case == "AlAl" else "Ni+Ni") if i == 0 else None,
                   mfc=col if case == "AlAl" else "none")
        xt.append(("S-1.1" if model == "uma-s-1p1" else "S-1.2") + "\n" + task)
    b.axhline(1e-3, ls="--", lw=0.8, color="grey")
    b.set_yscale("log"); b.set_ylim(1e-11, 1e4)
    b.set_xticks(range(4)); b.set_xticklabels(xt)
    b.set_ylabel("$|\\Delta|$ at 100 Å (meV)")
    b.legend(frameon=False, loc="upper left")
    b.set_title("(b) same-metal pairs, 100 \u00c5", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out / "audit_combined.png")
    plt.close(fig)


def sizecons(res, out):
    d = pd.read_csv(res / "size_consistency_training.csv")
    d = d[d.gap == 8]
    g100 = pd.read_csv(res / "size_consistency_training.csv")
    g100 = g100[g100.gap == 100].set_index(["variant", "seed", "step"]).Delta_meV
    assert np.array_equal(d.set_index(["variant", "seed", "step"]).Delta_meV.loc[g100.index].to_numpy(),
                          g100.to_numpy()), "Delta at 8 and 100 A must be identical"
    fig, ax = plt.subplots(figsize=(WIDTH, 1.75))
    style = {"dense": ("Dense", OI["blue"], "o"), "global": ("Global", OI["vermillion"], "s"),
             "local6": ("Local-6", OI["green"], "^")}
    for v, (name, col, mk) in style.items():
        dv = d[d.variant == v]
        for seed, ds in dv.groupby("seed"):
            ds = ds.sort_values("step")
            ax.plot(ds.step, ds.Delta_meV.abs(), color=col, lw=0.6, alpha=0.5)
        m = dv.assign(a=dv.Delta_meV.abs()).groupby("step").a.mean()
        ax.plot(m.index, m.values, color=col, lw=1.8, marker=mk, ms=3.5, label=f"{name}, mean of 3 seeds")
    ax.axhline(1e-3, ls="--", lw=0.8, color="grey")
    ax.text(30500, 1.6e-3, "0.001 meV (audit tolerance)", fontsize=7, color="grey")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1e5)
    ax.set_xlabel("training step"); ax.set_ylabel(r"$|\Delta|$ (meV)")
    ax.set_xticks(range(0, 50001, 10000))
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
    fig.tight_layout()
    fig.savefig(out / "size_consistency_training.png")
    plt.close(fig)


def router(res, out):
    d = pd.read_csv(res / "router_check_ablation.csv")
    tr = d[d.condition == "trained"].set_index(["variant", "seed"])
    conds = ["uniform", "eighth", "mean"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4))
    for ax, (col_name, ylab) in zip(axes, [("force_mae", "force MAE change (percent)"),
                                           ("energy_mae_per_atom", "energy MAE change (percent)")]):
        for k, (v, name, c, mk) in enumerate([("global", "Global", OI["vermillion"], "s"),
                                               ("local6", "Local-6", OI["green"], "^")]):
            for i, cond in enumerate(conds):
                a = d[(d.variant == v) & (d.condition == cond)].set_index("seed")[col_name]
                base = tr.loc[v][col_name].loc[a.index]
                pct = 100 * (a / base - 1)
                assert (pct > 0).all()
                x = i + (k - 0.5) * 0.28 + np.array([-0.05, 0, 0.05])[: len(pct)]
                ax.plot(x, pct.values, ls="none", marker=mk, color=c, ms=4,
                        label=name if i == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(range(3)); ax.set_xticklabels(["uniform", "eighth", "mean"])
        ax.set_ylabel(ylab); ax.set_xlim(-0.6, 2.6)
        ax.grid(axis="y", which="major", lw=0.3, alpha=0.5)
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].set_title("(a) force", fontsize=8); axes[1].set_title("(b) energy", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "router_check.png")
    plt.close(fig)


if __name__ == "__main__":
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    audit(res, out); sizecons(res, out); router(res, out)
    print("ok")
