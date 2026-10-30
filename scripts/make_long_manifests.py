"""Write the long-run training list and the checkpoint-selection list (plan item B4, docs/data-lists.md), and check them.

  long_train_1200k.txt  the 200,000 rows of pilot_train_200k.txt plus 1,000,000 rows of train_4M drawn uniformly without
                        replacement from the rows that are not in the pilot list, sorted
  select_val_5k.txt     5,000 rows of val drawn uniformly without replacement from the rows that are not in
                        pilot_val_5k.txt (the report set), sorted

Same generator, format and conventions as scripts/make_pilot_manifests.py: numpy.random.default_rng(seed).choice(...,
replace=False), one row number per line, sorted. The written lists are the source of truth; manifests/long_samples.json
holds their hashes and is what `check_pilot_data.py verify --samples` reads for them.

  python scripts/make_long_manifests.py write [--out-dir DIR]     the lists and long_samples.json (no data needed)
  python scripts/make_long_manifests.py compare --data-root R --out F.json [--train-sources S] [--workers N]
        cluster: atoms per structure and data_ids of each list (from the index files), and shared compositions and
        source systems between the selection set and the report set, and between them and the pilot training rows
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
SEED_LONG, SEED_SELECT = 11787, 11788      # the pilot draws used 11785 (train) and 11786 (val)
LONG, SELECT = "long_train_1200k.txt", "select_val_5k.txt"
PILOT_TRAIN, REPORT = "pilot_train_200k.txt", "pilot_val_5k.txt"


def read_rows(path):
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def write_rows(path, rows):
    Path(path).write_text("".join(f"{i}\n" for i in rows), encoding="utf-8", newline="\n")


def draw_outside(rows_in_split, taken, n, seed):
    """n distinct rows of range(rows_in_split) that are not in `taken`, uniform without replacement, sorted."""
    pool = np.setdiff1d(np.arange(rows_in_split, dtype=np.int64), taken, assume_unique=True)
    return np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))


def build(rows_in_split, pilot_train, report, n_extra=1_000_000, n_select=5_000):
    long_rows = np.sort(np.concatenate([pilot_train, draw_outside(rows_in_split["train_4M"], pilot_train, n_extra, SEED_LONG)]))
    select = draw_outside(rows_in_split["val"], report, n_select, SEED_SELECT)
    return long_rows, select


def list_checks(long_rows, select, pilot_train, report, rows_in_split, n_long=1_200_000, n_select=5_000):
    return {
        "long: count": len(long_rows) == n_long,
        "long: no duplicates": len(np.unique(long_rows)) == len(long_rows),
        "long: rows in range of train_4M": bool(long_rows.min() >= 0 and long_rows.max() < rows_in_split["train_4M"]),
        "long: contains every pilot training row": bool(np.isin(pilot_train, long_rows).all()),
        "select: count": len(select) == n_select,
        "select: no duplicates": len(np.unique(select)) == len(select),
        "select: rows in range of val": bool(select.min() >= 0 and select.max() < rows_in_split["val"]),
        "select: no row of the report set": not np.isin(select, report).any(),
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(args):
    data = json.loads((MANIFESTS / "omol25_data_manifest.json").read_text(encoding="utf-8"))
    rows_in_split = {s["split"]: s["row_count"] for s in data["splits"]}
    pilot_train, report = read_rows(MANIFESTS / PILOT_TRAIN), read_rows(MANIFESTS / REPORT)
    long_rows, select = build(rows_in_split, pilot_train, report)
    checks = list_checks(long_rows, select, pilot_train, report, rows_in_split)
    for text, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    if not all(checks.values()):
        raise SystemExit("a list check FAILED; nothing was written")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {"method": "numpy.random.default_rng(seed).choice(rows not in the base list, n, replace=False), then sorted; "
                      "the long list is that draw joined with the base list", "lists": {}}
    for name, rows, split, seed, base, relation in ((LONG, long_rows, "train_4M", SEED_LONG, PILOT_TRAIN, "contains"),
                                                   (SELECT, select, "val", SEED_SELECT, REPORT, "disjoint_from")):
        write_rows(out / name, rows)
        info["lists"][name] = {"split": split, "rows_in_split": rows_in_split[split], "count": len(rows), "seed": seed,
                               relation: base, "first": int(rows[0]), "last": int(rows[-1]), "sha256": sha256(out / name)}
        print(name, len(rows), info["lists"][name]["sha256"])
    (out / "long_samples.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")


def system(source):
    """The calculation folder of a structure: its source path without the last file and without a trailing stepN."""
    return re.sub(r"/step\d+$", "", source.rsplit("/", 1)[0])


def read_info(task):
    from fairchem.core.datasets import AseDBDataset
    src, rows = task
    data = AseDBDataset({"src": src})
    return [(str(a.info.get("composition")), str(a.info.get("source"))) for a in map(data.get_atoms, map(int, rows))]


def describe(natoms, data_ids):
    q = np.percentile(natoms, [5, 50, 95])
    return {"structures": int(len(natoms)), "atoms_mean": float(natoms.mean()), "atoms_p5_p50_p95": q.tolist(),
            "atoms_max": int(natoms.max()), "data_ids": {k: v / len(data_ids) for k, v in sorted(Counter(data_ids).items())}}


def largest_share_gap(a, b):
    return max(abs(a["data_ids"].get(k, 0) - b["data_ids"].get(k, 0)) for k in set(a["data_ids"]) | set(b["data_ids"]))


def compare(args):
    from scipy.stats import ks_2samp
    root = Path(args.data_root)
    index = {s: np.load(root / "pilot" / f"{s}_index.npz", allow_pickle=True) for s in ("train_4M", "val")}
    lists = {"pilot_train": ("train_4M", read_rows(MANIFESTS / PILOT_TRAIN)), "long": ("train_4M", read_rows(MANIFESTS / LONG)),
             "report": ("val", read_rows(MANIFESTS / REPORT)), "select": ("val", read_rows(MANIFESTS / SELECT))}
    lists["long_new"] = ("train_4M", np.setdiff1d(lists["long"][1], lists["pilot_train"][1]))
    natoms = {k: index[s]["natoms"][r] for k, (s, r) in lists.items()}
    result = {"lists": {k: describe(natoms[k], index[s]["data_ids"][r].astype(str)) for k, (s, r) in lists.items()}}
    for a, b in (("long_new", "pilot_train"), ("long", "pilot_train"), ("select", "report")):
        result[f"{a} vs {b}"] = {"ks_atoms_per_structure": ks_2samp(natoms[a], natoms[b]).statistic,
                                 "largest_data_ids_share_gap": largest_share_gap(result["lists"][a], result["lists"][b])}

    with mp.Pool(args.workers) as pool:
        info = {k: [x for part in pool.map(read_info, [(str(root / "val"), c) for c in np.array_split(lists[k][1], args.workers * 4)])
                    for x in part] for k in ("report", "select")}
    comp = {k: Counter(c for c, _ in v) for k, v in info.items()}
    systems = {k: {system(s) for _, s in v} for k, v in info.items()}
    shared = set(comp["select"]) & set(comp["report"])
    overlap = {"select rows whose composition is in the report set": sum(comp["select"][c] for c in shared),
               "report rows whose composition is in the selection set": sum(comp["report"][c] for c in shared),
               "distinct compositions": {"select": len(comp["select"]), "report": len(comp["report"]), "shared": len(shared)},
               "distinct source systems": {"select": len(systems["select"]), "report": len(systems["report"]),
                                           "shared": len(systems["select"] & systems["report"])}}
    if args.train_sources:
        train_systems = {system(s) for s in Path(args.train_sources).read_text(encoding="utf-8").split()}
        overlap[f"source systems shared with the training rows of {Path(args.train_sources).name}"] = {
            k: len(v & train_systems) for k, v in systems.items()}
    result["select vs report, overlap"] = overlap
    Path(args.out).write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "lists"}, indent=2, default=float))
    for k, v in result["lists"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "data_ids"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("write")
    p.add_argument("--out-dir", default=str(MANIFESTS))
    p.set_defaults(fn=write)
    p = sub.add_parser("compare")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-sources", help="source path of each training row, one per line (the --sids output of check_pilot_data.py stats)")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
