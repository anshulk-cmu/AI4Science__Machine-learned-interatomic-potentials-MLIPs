"""Size-consistency check of trained checkpoints: Delta for the Al and Ni slabs (ideal geometry) at each gap."""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as g
import models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--prediction-average", action="store_true", help="evaluate one equal-weight ensemble of all supplied checkpoints")
    ap.add_argument("--device", default="cuda", help="device for prediction averaging")
    ap.add_argument("--task", default="omol")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--gaps", type=int, nargs="+", default=[8, 10, 15, 25, 100])
    a = ap.parse_args()
    al, ni = g.ideal_slab("Al"), g.ideal_slab("Ni")
    failed = 0
    for path in ([a.checkpoints] if a.prediction_average else a.checkpoints):
        try:
            energy, cutoff = (models.load_prediction_average(path, a.task, a.dtype, a.device)
                              if a.prediction_average else models.load("uma", path, a.task, a.dtype))
            radius = models.interaction_radius(energy, cutoff)
            e_al, e_ni = energy(g.isolated(al)), energy(g.isolated(ni))
            deltas = []
            for gap in a.gaps:
                pair = g.stacked(al, ni, gap)
                g.assert_disconnected(pair, cutoff)
                status = "disconnected" if g.disconnected(pair, radius) else "routing-connected"
                deltas.append(f"{gap} A: {(energy(pair) - e_al - e_ni) * 1000:.6g} [{status}]")
            print(f"{path}  Delta meV  " + "  ".join(deltas), flush=True)
        except Exception:                      # one bad checkpoint must not hide the others
            failed += 1
            print(f"{path}  FAILED", flush=True)
            traceback.print_exc()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
