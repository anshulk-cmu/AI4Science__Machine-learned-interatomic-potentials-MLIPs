"""Write the fixed pilot sample lists: row numbers of train_4M and val, drawn once, with their hashes.

The lists are the source of truth. The draw below only documents how they were made, and is not rerun by the jobs,
because a random generator is not guaranteed to give the same numbers in every library version.
Run from the repository root: python scripts/make_pilot_manifests.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests"
SEED = 11785                      # the seed of manifests/omol25_data_manifest.json
DRAWS = {"train_4M": ("pilot_train_200k.txt", 200_000, SEED), "val": ("pilot_val_5k.txt", 5_000, SEED + 1)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads((OUT / "omol25_data_manifest.json").read_text(encoding="utf-8"))
    rows = {s["split"]: s["row_count"] for s in data["splits"]}
    info = {"method": "numpy.random.default_rng(seed).choice(rows, n, replace=False), then sorted", "lists": {}}
    for split, (name, n, seed) in DRAWS.items():
        picked = np.sort(np.random.default_rng(seed).choice(rows[split], n, replace=False))
        path = OUT / name
        path.write_text("".join(f"{i}\n" for i in picked), encoding="utf-8", newline="\n")
        info["lists"][name] = {"split": split, "rows_in_split": rows[split], "count": n, "seed": seed,
                               "first": int(picked[0]), "last": int(picked[-1]), "sha256": sha256(path)}
        print(name, n, info["lists"][name]["sha256"])
    (OUT / "pilot_samples.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
