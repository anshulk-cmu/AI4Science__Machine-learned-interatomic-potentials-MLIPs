"""Size-consistency check of trained checkpoints: Delta for the Al and Ni slabs (ideal geometry) at each gap."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as g
import models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--task", default="omol")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--gaps", type=int, nargs="+", default=[8, 100])
    a = ap.parse_args()
    al, ni = g.ideal_slab("Al"), g.ideal_slab("Ni")
    for path in a.checkpoints:
        energy, cutoff = models.load("uma", path, a.task, a.dtype)
        e_al, e_ni = energy(g.isolated(al)), energy(g.isolated(ni))
        deltas = []
        for gap in a.gaps:
            pair = g.stacked(al, ni, gap)
            g.assert_disconnected(pair, cutoff)
            deltas.append(f"{gap} A: {(energy(pair) - e_al - e_ni) * 1000:.6g}")
        print(f"{path}  Delta meV  " + "  ".join(deltas), flush=True)


if __name__ == "__main__":
    main()
