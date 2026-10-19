"""Plot molecular locality CSVs from one or more checkpoint runs."""
import argparse
import csv
from pathlib import Path


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
    fig, axes = plt.subplots(len(pairs), 2, squeeze=False, sharey="col", figsize=(10, 3.3*len(pairs)))
    for index, pair in enumerate(pairs):
        for label in sorted({r["label"] for r in rows if r["pair"] == pair}):
            data = sorted((r for r in rows if r["pair"] == pair and r["label"] == label),
                          key=lambda r: float(r["gap_A"]))
            x = [float(r["gap_A"]) for r in data]
            axes[index, 0].plot(x, [abs(float(r["delta_eV"])) for r in data], "o-", label=label)
            axes[index, 1].plot(x, [max(float(r["max_dforce_a_eV_A"]), float(r["max_dforce_b_eV_A"]))
                                  for r in data], "o-", label=label)
        for ax, unit in zip(axes[index], ("|additivity residual| (eV)", "max force change (eV/Å)")):
            ax.set(title=pair, xlabel="Bounding-plane gap (Å)", ylabel=unit)
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.legend(fontsize=7)
    for ax in axes[0]:
        ax.set_ylim(0, max(ax.get_ylim()[1], 1e-8))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
