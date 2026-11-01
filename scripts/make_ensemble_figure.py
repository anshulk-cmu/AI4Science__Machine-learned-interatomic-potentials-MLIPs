"""Compare final, selected best and ensemble models on the report set.

Usage: python scripts/make_ensemble_figure.py metrics.csv out.pdf
The input contract is documented in docs/final-figures.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figure_common import STYLE, METRICS, plt, validate_metrics, seed_points, model_ticks, save_figure

METHODS = {"final": ("Final checkpoint", "o"), "best": ("Selected best", "s"),
           "weight_average": ("Weight average", "^"), "prediction_average": ("Prediction average", "D")}


def validate_ensemble(df):
    if not {"method", "eval_split"}.issubset(df.columns):
        raise ValueError("Ensemble CSV requires method and eval_split columns")
    if df.empty or not df.eval_split.eq("report").all():
        raise ValueError("Ensemble figure requires eval_split=report for every row")
    methods = set(df.method)
    if not methods.issubset(METHODS) or not {"final", "best"}.issubset(methods):
        raise ValueError("Methods must include final and best, plus weight_average and/or prediction_average")
    if not methods.intersection({"weight_average", "prediction_average"}):
        raise ValueError("At least one ensemble method is required")
    tables = [validate_metrics(df[df.method == m]) for m in METHODS if m in methods]
    pairs = [set(map(tuple, t[["variant", "seed"]].to_numpy())) for t in tables]
    if any(p != pairs[0] for p in pairs[1:]):
        raise ValueError("Every method must contain the same variant/seed pairs")
    return pd.concat(tables, ignore_index=True)


def build_figure(df, descriptive=False, title=None):
    df = validate_ensemble(df)
    models = [m for m in STYLE if m in set(df.variant)]
    methods = [m for m in METHODS if m in set(df.method)]
    n = df.seed.nunique()
    descriptive = descriptive or n < 3
    positions = np.linspace(-.3, .3, len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, metric, label in zip(axes, METRICS, ("energy MAE per atom (eV)", r"force MAE (eV/$\AA$)")):
        for i, model in enumerate(models):
            for position, method in zip(positions, methods):
                y = df[(df.variant == model) & (df.method == method)].sort_values("seed")[metric].to_numpy()
                seed_points(ax, y, i + position, model, descriptive, width=.035, marker=METHODS[method][1])
        model_ticks(ax, models)
        ax.set_ylabel(label)
    handles = [Line2D([], [], color="#444444", marker=METHODS[m][1], ls="none", ms=5, label=METHODS[m][0]) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .04), ncol=len(methods), frameon=False)
    fig.suptitle(f"{title or 'Report-set checkpoint comparison'} | {n} seeds per variant", fontsize=10)
    summary = "points: seeds; bars: means (descriptive)" if descriptive else "points: seeds; bars: means and 95% t intervals"
    fig.text(.5, .01, summary, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .16, 1, .94))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--descriptive", action="store_true", help="Show means and seeds without confidence intervals")
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    try:
        fig = build_figure(pd.read_csv(args.csv), args.descriptive, args.title)
        save_figure(fig, args.out, [args.csv])
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
