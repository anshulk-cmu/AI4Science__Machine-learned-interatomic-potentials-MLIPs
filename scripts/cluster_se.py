"""Cluster-robust standard errors of the full-test MAEs and paired differences, with the composition as the cluster.

The test list was split by composition (docs/data-split.md), so rows of one composition are not independent draws. Both
MAEs are ratio estimators, R = sum(num) / sum(den): energy num = |error| per atom of a structure, den = 1; force
num = sum of |force error| components of a structure, den = its number of components. Cluster sandwich variance:
  u_c = sum over rows i of cluster c of (num_i - R den_i),   Var(R) = G / (G - 1) * sum_c u_c^2 / (sum den)^2
With one row per cluster this is exactly the row SE of eval_rows.py. A cluster bootstrap (resample whole compositions)
checks the formula.

  compositions: python scripts/cluster_se.py compositions --assignment split_assignment.npz --out test_row_composition.csv
  tables:       python scripts/cluster_se.py tables --compositions test_row_composition.csv --dir finalRuns/_eval/eval/testfull
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_rows as er  # noqa: E402

MODELS = ("dense", "dense_scaled", "global_frozen", "global", "element", "local6", "local_global", "local12")
TEST = ROOT / "manifests" / "split_test_10.txt"
PAIRED = ROOT / "results" / "final_test_paired_full.jsonl"
SUBSETS = {"all": lambda n: n > 0, "natoms_2": lambda n: n == 2, "natoms_above_2": lambda n: n > 2,
           "natoms_3_to_20": lambda n: (n >= 3) & (n <= 20), "natoms_above_20": lambda n: n > 20}


def ratio_se(num, den, cluster):
    """R = sum(num) / sum(den) and its cluster sandwich SE. `cluster` holds codes 0..G-1."""
    r = num.sum() / den.sum()
    u = np.bincount(cluster, num - r * den)
    g = np.count_nonzero(np.bincount(cluster))
    return r, float(np.sqrt(g / (g - 1) * (u ** 2).sum()) / den.sum())


def bootstrap_se(nums, dens, cluster, draws, seed, chunk=50):
    """SE of each R_q = sum(nums[q]) / sum(dens[q]) over `draws` resamples of whole clusters (with replacement).
    One resample serves every quantity, so the SEs of all models and pairs come from the same draws."""
    n_c = np.stack([np.bincount(cluster, x) for x in nums])
    d_c = np.stack([np.bincount(cluster, x) for x in dens])
    g = n_c.shape[1]
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, draws, chunk):
        w = np.stack([np.bincount(rng.integers(0, g, g), minlength=g) for _ in range(min(chunk, draws - start))])
        values.append((w @ n_c.T) / (w @ d_c.T))
    return np.concatenate(values).std(axis=0, ddof=1)


def compositions(a):
    """row,composition of every row of split_test_10.txt, from the split's row -> composition -> set table."""
    with np.load(a.assignment) as f:
        split, group, keys, names = f["split"], f["group"], f["group_keys"], f["set_names"]
    rows = np.flatnonzero(split == list(names).index("test"))
    if not np.array_equal(rows, np.array(er.ck.read_rows(TEST))):
        raise SystemExit("the test rows of the table differ from manifests/split_test_10.txt")
    with open(a.out, "w", newline="\n", encoding="utf-8") as out:
        out.write("row,composition\n")
        out.writelines(f"{r},{keys[group[r]]}\n" for r in rows)
    print(f"{a.out}: {len(rows)} rows, {len(np.unique(group[rows]))} compositions")


def read_compositions(path):
    with open(path, newline="", encoding="utf-8") as f:
        pairs = [(int(line["row"]), line["composition"]) for line in csv.DictReader(f)]
    return dict(pairs)


def stats(num, den, cluster):
    r, se_c = ratio_se(num, den, cluster)
    _, se_r = ratio_se(num, den, np.arange(len(num)))
    return dict(value=r * 1e3, se_row=se_r * 1e3, se_cluster=se_c * 1e3, z_row=r / se_r, z_cluster=r / se_c)


def table_rows(data, pairs, cluster, natoms, draws, seed, subsets):
    """One dict per (subset, model or pair). data[k] = (energy error per atom, force abs sum, force components) per
    row. Values in meV/atom (energy) and meV/A (force); a pair is A minus B on the same rows."""
    out = []
    for subset in subsets:
        m = SUBSETS[subset](natoms)
        _, cl = np.unique(cluster[m], return_inverse=True)
        sizes = np.bincount(cl)
        rows, nums, dens = [], [], []
        for a, b in [(k, "") for k in data] + pairs:
            e = data[a][0][m] - (data[b][0][m] if b else 0)
            f = data[a][1][m] - (data[b][1][m] if b else 0)
            c = data[a][2][m]
            row = dict(subset=subset, a=a, b=b, n_structures=int(m.sum()), n_clusters=len(sizes),
                       largest_cluster=int(sizes.max()))
            for name, num, den in (("energy", e, np.ones(len(e))), ("force", f, c)):
                row.update({f"{name}_{k}": v for k, v in stats(num, den, cl).items()})
                nums.append(num)
                dens.append(den)
            rows.append(row)
        if draws:
            se = iter(bootstrap_se(nums, dens, cl, draws, seed) * 1e3)
            for row in rows:
                row["energy_se_bootstrap"], row["force_se_bootstrap"] = next(se), next(se)
        out += rows
    return out


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def tables(a):
    comp = read_compositions(a.compositions)
    data = {}
    for k in MODELS:
        records, _ = er.load(TEST, sorted(Path(a.dir).glob(f"{k}_s?of4.csv")))
        rows = sorted(records)
        data[k] = tuple(np.array([records[r][i] for r in rows], np.float64) for i in range(3))
    if sorted(comp) != rows:
        raise SystemExit("the composition table does not cover exactly the test rows")
    for k in MODELS:
        if not np.array_equal(data[k][2], data[MODELS[0]][2]):
            raise SystemExit(f"{k}: force component counts differ from {MODELS[0]}")
    cluster = np.unique([comp[r] for r in rows], return_inverse=True)[1]
    natoms = (data[MODELS[0]][2] // 3).astype(int)
    pairs = [(d["a"], d["b"]) for d in map(json.loads, PAIRED.read_text(encoding="utf-8").splitlines())]
    out = table_rows(data, pairs, cluster, natoms, a.draws, a.seed, SUBSETS)
    full = [r for r in out if r["subset"] == "all"]
    # the row SEs must reproduce the committed tables
    old = {(d["a"], d["b"]): d["result"] for d in map(json.loads, PAIRED.read_text(encoding="utf-8").splitlines())}
    gap = max(abs(r["energy_se_row"] / 1e3 - old[r["a"], r["b"]]["energy_diff_se"]) +
              abs(r["force_se_row"] / 1e3 - old[r["a"], r["b"]]["force_diff_se"]) for r in full if r["b"])
    print(f"largest gap to final_test_paired_full.jsonl row SEs: {gap:.2e} eV")
    out_dir = Path(a.out_dir)
    write(out_dir / "final_test_cluster_se.csv", [r for r in full if not r["b"]])
    write(out_dir / "final_test_paired_cluster.csv", [r for r in full if r["b"]])
    write(out_dir / "final_test_by_size.csv", [r for r in out if r["subset"] != "all"])
    for r in out:
        if r["subset"] in ("all", "natoms_above_2"):
            print(f"{r['subset']:15s} {r['a']}{'-' + r['b'] if r['b'] else '':22s} "
                  f"E {r['energy_value']:+.4f} row {r['energy_se_row']:.4f} cl {r['energy_se_cluster']:.4f} "
                  f"z {r['energy_z_cluster']:+.1f} | F {r['force_value']:+.4f} row {r['force_se_row']:.4f} "
                  f"cl {r['force_se_cluster']:.4f} z {r['force_z_cluster']:+.1f} | G {r['n_clusters']} max {r['largest_cluster']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compositions")
    p.add_argument("--assignment", type=Path, required=True, help="split_assignment.npz of make_split_manifests.py write")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=compositions)
    p = sub.add_parser("tables")
    p.add_argument("--compositions", type=Path, required=True)
    p.add_argument("--dir", type=Path, required=True, help="the 32 per-row shard CSVs of the full test list")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    p.add_argument("--draws", type=int, default=1000, help="cluster bootstrap resamples (0: skip)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=tables)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
