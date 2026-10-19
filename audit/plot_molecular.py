"""Plot molecular locality CSVs from one or more checkpoint runs."""
import argparse
import csv
from pathlib import Path

FLOOR = 1e-16       # symlog linear range: the float64 floor (about 1e-13 eV, 1e-15 eV/Å) must stay visible


def styles(labels):
    """Colour by variant (the label text before the first '_'), line style by run within the variant."""
    variants = sorted({l.split("_")[0] for l in labels})
    out = {}
    for label in sorted(labels):
        variant = label.split("_")[0]
        runs = sorted(l for l in labels if l.split("_")[0] == variant)
        out[label] = (f"C{variants.index(variant)}", ("-", "--", ":", "-.")[runs.index(label) % 4])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for path in args.csvs:
        with path.open(encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    pairs = sorted({r["pair"] for r in rows})
    if not pairs:
        ap.error("no measurements")
    style = styles({r["label"] for r in rows})
    fig, axes = plt.subplots(len(pairs), 2, squeeze=False, sharey="col", figsize=(10, 3.3*len(pairs)))
    for index, pair in enumerate(pairs):
        for label in sorted({r["label"] for r in rows if r["pair"] == pair}):
            data = sorted((r for r in rows if r["pair"] == pair and r["label"] == label),
                          key=lambda r: float(r["gap_A"]))
            x = [float(r["gap_A"]) for r in data]
            color, line = style[label]
            axes[index, 0].plot(x, [abs(float(r["delta_eV"])) for r in data], marker="o", color=color,
                                ls=line, label=label)
            axes[index, 1].plot(x, [max(float(r["max_dforce_a_eV_A"]), float(r["max_dforce_b_eV_A"]))
                                  for r in data], marker="o", color=color, ls=line, label=label)
        for ax, unit in zip(axes[index], ("|additivity residual| (eV)", "max force change (eV/Å)")):
            ax.set(title=pair, xlabel="Bounding-plane gap (Å)", ylabel=unit)
            ax.set_yscale("symlog", linthresh=FLOOR)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[0]:
        ax.set_ylim(0, max(ax.get_ylim()[1], 1e-8))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
