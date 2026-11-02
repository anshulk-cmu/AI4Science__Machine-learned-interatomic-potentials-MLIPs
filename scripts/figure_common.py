"""Shared styles, validated metric tables and seed summaries for report figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pilot_statistics import MODELS, METRICS

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00",
      "purple": "#CC79A7", "black": "#000000", "sky": "#56B4E9", "yellow": "#F0E442"}
STYLE = {
    "dense": ("Dense", OI["blue"], "o", "-"),
    "global": ("Global", OI["vermillion"], "s", "-"),
    "element": ("Element", OI["black"], "P", ":"),
    "local6": ("Local-6", OI["green"], "^", "-"),
    "local12": ("Local-12", OI["sky"], "X", "-."),
    "local_global": ("Local+Global", OI["yellow"], "h", "-."),
    "frozen": ("Frozen", OI["orange"], "D", "--"),
    "dense_scaled": ("Scaled Dense", OI["purple"], "v", "--"),
}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"{col} must contain finite nonnegative values")


def validate_metrics(df, require_final=False):
    """Validate one row per variant/seed, including tables of selected variants."""
    missing = {"variant", "seed", *METRICS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    df = df.copy()
    df["variant"] = df.variant.replace({"global_frozen": "frozen"})
    if df.empty or df.variant.isna().any() or not set(df.variant).issubset(STYLE):
        raise ValueError("Metrics must contain known, nonempty variants")
    numeric(df, ["seed", *METRICS])
    if (df.seed % 1 != 0).any():
        raise ValueError("Seeds must be integers")
    if df.duplicated(["variant", "seed"]).any():
        raise ValueError("Duplicate (variant, seed) metrics")
    sets = df.groupby("variant").seed.apply(lambda s: frozenset(s))
    if len(set(sets)) != 1:
        raise ValueError("Every variant must have the same seed IDs")
    if require_final and (set(df.variant) != set(MODELS) or set(df.seed) != set(range(5))):
        raise ValueError("Final figure requires 8 variants and seeds 0 through 4")
    if "step" in df:
        numeric(df, ["step"])
        if (df.step % 1 != 0).any():
            raise ValueError("Steps must be integers")
    return df.sort_values(["variant", "seed"])


def offsets(n, width=.12):
    return np.linspace(-width, width, n) if n > 1 else np.zeros(1)


def mean_interval(y, descriptive=False):
    """Two-seed follow-ups are descriptive under the final training plan."""
    mean = float(np.mean(y))
    half = None if descriptive or len(y) < 3 else float(stats.t.ppf(.975, len(y) - 1) * np.std(y, ddof=1) / np.sqrt(len(y)))
    return mean, half


def seed_points(ax, y, position, variant, descriptive=False, width=.12, marker=None):
    _, color, default_marker, _ = STYLE[variant]
    mean, half = mean_interval(y, descriptive)
    ax.errorbar(position, mean, yerr=half, fmt="_", color=color, ms=10, capsize=3, lw=1.2, zorder=3)
    ax.scatter(position + offsets(len(y), width), y, marker=marker or default_marker,
               s=22, color=color, edgecolors="#444444", linewidths=.4, alpha=.85, zorder=4)


def model_ticks(ax, models, steps=None):
    labels = [STYLE[m][0].replace("+", "+\n").replace("Scaled ", "Scaled\n") for m in models]
    if steps is not None:
        labels = [f"{label}\n{steps[m]}" for m, label in zip(models, labels)]
    ax.set_xticks(range(len(models)), labels, rotation=25, ha="right")
    ax.set_xlim(-.6, len(models) - .4)
    ax.grid(axis="y", alpha=.2, lw=.5)


def save_figure(fig, output, inputs=()):
    output = Path(output)
    try:
        if output.resolve() in {Path(p).resolve() for p in inputs}:
            raise ValueError("Output must not overwrite an input CSV")
        if output.suffix.lower() not in {".pdf", ".png", ".svg"}:
            raise ValueError("Figure output must end in .pdf, .png or .svg")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
