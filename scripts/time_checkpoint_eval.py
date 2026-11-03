"""Seconds per structure of the checkpoint tools' evaluator (training.checkpoints.evaluate), to size the evaluation lists.

The same predictor and loop that scripts/checkpoint_tools.py uses (one structure at a time), on a uniform sample of a list.
  python scripts/time_checkpoint_eval.py --checkpoint CKPT --data <OMol25 train_4M> --rows manifests/split_test_10.txt \
      --n 300 --device cuda --out timing.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import checkpoints as ck  # noqa: E402
from checkpoint_tools import dataset_at, release  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    data = dataset_at(args.data)
    rows = np.sort(np.random.default_rng(0).choice(ck.read_rows(args.rows), args.n + args.warmup, replace=False))
    warm, timed = rows[:args.warmup].tolist(), rows[args.warmup:].tolist()
    result = {"checkpoint": args.checkpoint, "rows": args.rows, "n": args.n, "device": args.device,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None, "torch": torch.__version__,
              "atoms_mean": float(np.mean([len(data.get_atoms(i)) for i in timed])), "runs": []}
    for dtype in args.dtypes:
        t0 = time.perf_counter()
        unit = ck.predictor(args.checkpoint, args.device, dtype)
        load = time.perf_counter() - t0
        ck.evaluate(unit, data, warm)
        t0 = time.perf_counter()
        metrics = ck.evaluate(unit, data, timed)
        seconds = time.perf_counter() - t0
        result["runs"].append({"dtype": dtype, "load_s": load, "eval_s": seconds, "s_per_structure": seconds / len(timed),
                               **metrics})
        print(json.dumps(result["runs"][-1]), flush=True)
        del unit
        release()
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
