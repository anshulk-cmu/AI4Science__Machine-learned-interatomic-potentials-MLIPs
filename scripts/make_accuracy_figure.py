"""Accuracy figures for pilots, the 8-variant final table, and long runs.

Legacy usage: python scripts/make_accuracy_figure.py out.pdf
See docs/final-figures.md for explicit CSV inputs and metrics-only figures.
"""
import argparse
from pathlib import Path

import matplotlib.ticker as ticker
import pandas as pd

from figure_common import (STYLE, OI, plt, numeric, validate_metrics, seed_points,
                           model_ticks, save_figure)

ROOT = Path(__file__).resolve().parents[1]
FORCE = "force_mae_eV_per_A"
CURVE_FORCE = "val_force_mae_eV_per_A"
TRAIN = "train_loss_mean_preceding_500_steps"


def validate_curves(cur, fin):
    missing = {"variant", "seed", "step", CURVE_FORCE, TRAIN} - set(cur.columns)
    if missing:
        raise ValueError(f"Missing curve columns: {sorted(missing)}")
    cur = cur.copy()
    cur["variant"] = cur.variant.replace({"global_frozen": "frozen"})
    numeric(cur, ["seed", "step", CURVE_FORCE, TRAIN])
    if (cur.seed % 1 != 0).any() or (cur.step % 1 != 0).any():
        raise ValueError("Curve seeds and steps must be integers")
    if cur.duplicated(["variant", "seed", "step"]).any():
        raise ValueError("Duplicate (variant, seed, step) curve rows")
    expected = set(map(tuple, fin[["variant", "seed"]].to_numpy()))
    actual = set(map(tuple, cur[["variant", "seed"]].to_numpy()))
    if expected != actual:
        raise ValueError("Curve variant/seed pairs must match final metrics")
    if "dense" not in set(fin.variant):
        raise ValueError("Curves require Dense for the gain panel; use --metrics-only for subsets")
    grids = cur.groupby(["variant", "seed"]).step.apply(lambda s: frozenset(s))
    if len(set(grids)) != 1:
        raise ValueError("Curves require the same evaluation steps for every run; use --metrics-only for unequal budgets")
    if (cur[CURVE_FORCE] <= 0).any():
        raise ValueError("Curve force MAEs must be positive for the log axis")
    if "step" in fin and not (fin.step == cur.step.max()).all():
        raise ValueError("Final metric steps must match the last curve evaluation")
    return cur.sort_values(["variant", "seed", "step"])


def band(ax, df, col, scale=1.0, start=0, ref=None):
    for v, (name, color, marker, line) in STYLE.items():
        if v not in set(df.variant):
            continue
        g = df[df.variant == v].pivot(index="step", columns="seed", values=col).sort_index()
        g = g[g.index >= start]
        y = g.to_numpy() * scale
        if ref is not None:
            y = (ref.loc[g.index].to_numpy()[:, None] - g.to_numpy()) * scale
        ax.fill_between(g.index, y.min(1), y.max(1), color=color, alpha=.18, lw=0)
        ax.plot(g.index, y.mean(1), color=color, ls=line, marker=marker, ms=3,
                markeredgecolor="#444444", markeredgewidth=.35, lw=1.1, label=name)


def endpoint_panel(ax, fin, metric, models, descriptive=False):
    for i, model in enumerate(models):
        y = fin.loc[fin.variant == model].sort_values("seed")[metric].to_numpy()
        seed_points(ax, y, i, model, descriptive)
    steps = None
    if "step" in fin:
        steps = {m: "/".join(f"{int(s):,}" for s in sorted(fin.loc[fin.variant == m, "step"].unique()))
                 for m in models}
    model_ticks(ax, models, steps)


def build_figure(fin, cur=None, long_run=False, title=None):
    fin = validate_metrics(fin)
    if cur is not None:
        cur = validate_curves(cur, fin)
    models = [m for m in STYLE if m in set(fin.variant)]
    n = fin.seed.nunique()
    descriptive = long_run or n < 3
    summary = "points: seeds; bars: means" if descriptive else "points: seeds; bars: means and 95% t intervals"
    heading = title or ("Long training (descriptive)" if long_run else "Validation accuracy")
    if cur is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
        for ax, metric, label in zip(axes, ("energy_mae_per_atom_eV", FORCE),
                                     ("energy MAE per atom (eV)", r"force MAE (eV/$\AA$)")):
            endpoint_panel(ax, fin, metric, models, descriptive)
            ax.set_ylabel(label)
        fig.suptitle(f"{heading} | {n} seeds per variant", fontsize=10)
        fig.text(.5, .01, summary, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, .93))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.6))
        dense = cur[cur.variant == "dense"].pivot(index="step", columns="seed", values=CURVE_FORCE).mean(axis=1)
        band(axes[0, 0], cur, TRAIN)
        axes[0, 0].set(ylabel="training loss", title="(a) training loss")
        band(axes[0, 1], cur, CURVE_FORCE)
        axes[0, 1].set(ylabel=r"force MAE (eV/$\AA$)", yscale="log", title="(b) validation force error")
        start = float(cur.step.max()) * .3
        band(axes[1, 0], cur, CURVE_FORCE, scale=1e3, start=start, ref=dense)
        axes[1, 0].axhline(0, color="grey", lw=.6)
        axes[1, 0].set(ylabel="Dense mean minus model\n" r"($10^{-3}$ eV/$\AA$)", title="(c) gain over Dense")
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xlabel("training step")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, pos: f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"))
        endpoint_panel(axes[1, 1], fin, FORCE, models, descriptive)
        axes[1, 1].set(ylabel=r"force MAE (eV/$\AA$)", title=f"(d) final evaluation, step {int(cur.step.max()):,}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, .035), frameon=False)
        fig.suptitle(f"{heading} | {n} seeds per variant", fontsize=10)
        fig.text(.5, .012, "Curves: mean and seed range; final " + summary, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .14, 1, .94), h_pad=1.3)
    return fig


def main(out, metrics=None, curves=None, metrics_only=False, long_run=False, require_final=False, title=None):
    if curves is not None and metrics is None:
        raise ValueError("Custom --curves requires an explicit --metrics CSV")
    if (metrics is not None or long_run) and curves is None and not metrics_only:
        raise ValueError("Specify --curves with custom metrics/long runs, or use --metrics-only")
    if metrics_only and curves is not None:
        raise ValueError("--metrics-only cannot be combined with --curves")
    if long_run and metrics is None:
        raise ValueError("--long-run requires an explicit --metrics CSV")
    metrics = Path(metrics) if metrics is not None else ROOT / "results/pilot_final_metrics.csv"
    fin = validate_metrics(pd.read_csv(metrics), require_final)
    inputs = [metrics]
    cur = None
    if not metrics_only:
        curves = Path(curves) if curves is not None else ROOT / "results/pilot_validation_curves.csv"
        cur = pd.read_csv(curves)
        inputs.append(curves)
    fig = build_figure(fin, cur, long_run, title)
    save_figure(fig, out, inputs)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="Output .pdf, .png or .svg")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--curves", type=Path)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--long-run", action="store_true", help="Descriptive means and seeds; no confidence intervals")
    parser.add_argument("--require-final", action="store_true", help="Require all 8 variants and seeds 0 through 4")
    parser.add_argument("--title")
    args = parser.parse_args()
    try:
        main(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    cli()
