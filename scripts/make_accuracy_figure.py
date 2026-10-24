"""Accuracy figure of the midterm report (Section 6.1): training loss and validation force error of the five models.

Reads results/pilot_validation_curves.csv (validation values at the periodic evaluations of every run, parsed from the
train.log of each run; the value at the step is the log value printed to 4 decimals; training loss is the mean of the
500 steps before the evaluation) and results/pilot_final_metrics.csv (full-precision final values).
Usage: python scripts/make_accuracy_figure.py <out.pdf>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00", "purple": "#CC79A7"}
STYLE = {"dense": ("Dense", OI["blue"], "o", "-"), "global": ("Global", OI["vermillion"], "s", "-"),
         "local6": ("Local-6", OI["green"], "^", "-"), "frozen": ("Frozen", OI["orange"], "D", "--"),
         "dense_scaled": ("Scaled Dense", OI["purple"], "v", "--")}
plt.rcParams.update({"font.size": 8.3, "axes.labelsize": 8.3, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
                     "legend.fontsize": 7.8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def band(ax, df, col, scale=1.0, start=0, ref=None):
    for v, (name, c, mk, ls) in STYLE.items():
        g = df[df.variant == v].pivot(index="step", columns="seed", values=col).sort_index()
        g = g[g.index >= start]
        y = g.to_numpy() * scale
        if ref is not None:
            y = (ref.loc[g.index].to_numpy()[:, None] - g.to_numpy()) * scale
        ax.fill_between(g.index, y.min(1), y.max(1), color=c, alpha=0.18, lw=0)
        ax.plot(g.index, y.mean(1), color=c, ls=ls, marker=mk, ms=3, lw=1.1, label=name)


def main(out):
    cur = pd.read_csv(ROOT / "results" / "pilot_validation_curves.csv")
    fin = pd.read_csv(ROOT / "results" / "pilot_final_metrics.csv")
    dense = cur[cur.variant == "dense"].pivot(index="step", columns="seed", values="val_force_mae_eV_per_A").mean(axis=1)
    fig, ax = plt.subplots(2, 2, figsize=(5.5, 3.55))
    band(ax[0, 0], cur, "train_loss_mean_preceding_500_steps")
    ax[0, 0].set(ylabel="training loss", title="(a) training loss")
    band(ax[0, 1], cur, "val_force_mae_eV_per_A")
    ax[0, 1].set(ylabel=r"force MAE (eV/$\AA$)", yscale="log", title="(b) validation force error")
    band(ax[1, 0], cur, "val_force_mae_eV_per_A", scale=1e3, start=15000, ref=dense)
    ax[1, 0].axhline(0, color="grey", lw=0.6)
    ax[1, 0].set(ylabel="Dense minus model\n" r"($10^{-3}$ eV/$\AA$)", title="(c) gain over Dense")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("training step"); a.set_xticks(range(0, 50001, 10000)); a.set_xticklabels(["0", "10k", "20k", "30k", "40k", "50k"])
    t = stats.t.ppf(0.975, 2)
    for i, (v, (name, c, mk, ls)) in enumerate(STYLE.items()):
        y = fin[fin.variant == v].sort_values("seed").force_mae_eV_per_A.to_numpy()
        h = t * y.std(ddof=1) / np.sqrt(3)
        ax[1, 1].errorbar(i, y.mean(), yerr=h, color=c, capsize=2.5, lw=1.1, zorder=2)
        ax[1, 1].plot(np.full(3, i) + np.array([-0.12, 0, 0.12]), y, ls="none", marker=mk, ms=3, color=c, alpha=0.7, zorder=3)
    ax[1, 1].set(ylabel=r"force MAE (eV/$\AA$)", title="(d) step 50,000, 95% interval")
    ax[1, 1].set_xticks(range(5)); ax[1, 1].set_xticklabels(["Dense", "Global", "Local\n-6", "Frozen", "Scaled\nDense"], fontsize=7.2)
    ax[1, 1].set_xlim(-0.6, 4.6)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", frameon=False, columnspacing=1.2, handlelength=2.2, bbox_to_anchor=(0.5, -0.005))
    for a in ax.flat:
        a.title.set_fontsize(8.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.9, w_pad=0.8)
    fig.savefig(out)


if __name__ == "__main__":
    main(sys.argv[1])
