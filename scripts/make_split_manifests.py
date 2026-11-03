"""The 70/20/10 split of train_4M by composition (docs/data-split.md): read the group keys, write the lists, check them.

  extract  cluster: read every row of one split (composition, calculation folder, charge, spin, elements) with workers,
           add natoms and data_ids from metadata.npz, write one table (npz)
  write    the lists split_train_70, split_val_20, split_test_10 and their fixed subsets, split_samples.json and the
           row -> group -> set table (no data needed beyond the train_4M table)
  check    counts, shares, disjoint rows, compositions and calculation folders, distributions against a null of other
           seeds, the relation to the official val split and to the earlier lists, and the rows a run of --steps steps
           reads (fairchem's own batch sampler); writes a JSON report

Whole compositions go to one set, separately within each data source (the data_ids value of most of a composition's rows):
shuffle the compositions of the source with a seeded generator, take their rows in that order, and give each composition to
the set in which the middle of its rows falls (70 and 90 percent of the source's rows are the borders). The written lists
are the source of truth; split_samples.json holds their hashes for `check_pilot_data.py verify --samples`.

  python scripts/make_split_manifests.py extract --data-root R --split train_4M --out keys_train_4M.npz --workers 16
  python scripts/make_split_manifests.py write --table keys_train_4M.npz [--out-dir manifests] [--assignment F.npz]
  python scripts/make_split_manifests.py check --table keys_train_4M.npz [--val-table keys_val.npz] --out checks.json
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from make_long_manifests import read_rows, sha256, system, write_rows

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
SETS = ("train", "val", "test")
SHARES = (0.7, 0.2, 0.1)
TOLERANCE = 0.005                     # largest allowed gap between a set's row share and its target
SEED = 11790                          # the group order; the earlier lists used 11785 to 11788
LISTS = ("split_train_70.txt", "split_val_20.txt", "split_test_10.txt")
# name, set it is drawn from, size, seed, list it must not share a row with
SUBSETS = (("split_val_monitor_5k.txt", 1, 5_000, SEED + 1, None),
           ("split_val_select_20k.txt", 1, 20_000, SEED + 2, "split_val_monitor_5k.txt"),
           ("split_test_eval_20k.txt", 2, 20_000, SEED + 3, None))
EARLIER = {"train_4M": ("pilot_train_200k.txt", "long_train_1200k.txt"), "val": ("pilot_val_5k.txt", "select_val_5k.txt")}


def read_keys(task):
    """Composition, calculation folder, charge, spin and the set of elements of each row of one chunk."""
    from fairchem.core.datasets import AseDBDataset
    src, rows = task
    data = AseDBDataset({"src": src})
    out = []
    for row in rows:
        atoms = data.get_atoms(int(row))
        mask = sum(1 << int(z) for z in set(atoms.numbers.tolist()))
        out.append((str(atoms.info["composition"]), system(str(atoms.info["source"])), int(atoms.info["charge"]),
                    int(atoms.info["spin"]), mask & (2**64 - 1), mask >> 64, len(atoms)))
    return out


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def factorize(values):
    """Integer codes into the sorted unique values, so codes do not depend on the reading order."""
    keys, codes = np.unique(np.asarray(values), return_inverse=True)
    return codes.astype(np.int32), keys


def extract(args):
    t0 = time.time()
    meta = np.load(Path(args.data_root) / args.split / "metadata.npz", allow_pickle=True)
    n = len(meta["natoms"])
    chunks = [(str(Path(args.data_root) / args.split), part) for part in np.array_split(np.arange(n), args.workers * 8)]
    parts = []
    with mp.Pool(args.workers) as pool:
        for i, part in enumerate(pool.imap(read_keys, chunks), 1):
            parts.append(part)
            print(f"{i}/{len(chunks)} chunks, {time.time() - t0:.0f} s", flush=True)
    rows = [r for part in parts for r in part]
    composition, system_, charge, spin, lo, hi, natoms = zip(*rows)
    if len(rows) != n or not np.array_equal(np.array(natoms), meta["natoms"]):
        raise SystemExit("the rows read do not match metadata.npz (count or atoms per structure)")
    table = {"composition": factorize(composition), "system": factorize(system_), "data_ids": factorize(meta["data_ids"])}
    fields = {k: v for name, (codes, keys) in table.items() for k, v in ((name, codes), (name + "_keys", keys))}
    np.savez_compressed(args.out, natoms=meta["natoms"].astype(np.int16), charge=np.array(charge, np.int16),
                        spin=np.array(spin, np.int16), elements_lo=np.array(lo, np.uint64),
                        elements_hi=np.array(hi, np.uint64), **fields)
    print(f"{args.out}: {n} rows, {len(fields['composition_keys'])} compositions, {len(fields['system_keys'])} "
          f"calculation folders, natoms equal metadata.npz for every row, {time.time() - t0:.0f} s", flush=True)


def strata_of(group, labels):
    """Stratum of each composition: the label (data_ids code) of most of its rows; a tie goes to the lower code."""
    n = labels.max() + 1
    return np.bincount(group.astype(np.int64) * n + labels, minlength=(group.max() + 1) * n).reshape(-1, n).argmax(axis=1)


def assign(group, strata=None, seed=SEED, shares=SHARES):
    """Set code of every row (0 train, 1 val, 2 test). Codes in `group` run 0..G-1; `strata` holds one code per group
    (None: one stratum). One generator serves the strata in code order."""
    sizes = np.bincount(group)
    strata = np.zeros(len(sizes), np.int64) if strata is None else strata
    rng = np.random.default_rng(seed)
    of_group = np.empty(len(sizes), np.int8)
    for s in np.unique(strata):
        ids = rng.permutation(np.flatnonzero(strata == s))
        middle = np.cumsum(sizes[ids]) - sizes[ids] / 2
        of_group[ids] = np.searchsorted(np.cumsum(shares)[:-1] * sizes[ids].sum(), middle, side="right")
    return of_group[group]


def draw(rows, n, seed, exclude=None):
    pool = rows if exclude is None else np.setdiff1d(rows, exclude, assume_unique=True)
    return np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))


def build(group, strata=None):
    """All lists as {name: sorted rows}, and the set code of every row."""
    split = assign(group, strata)
    lists = {name: np.flatnonzero(split == k) for k, name in enumerate(LISTS)}
    for name, k, n, seed, exclude in SUBSETS:
        lists[name] = draw(lists[LISTS[k]], n, seed, lists.get(exclude))
    return lists, split


def list_checks(lists, split, group, tolerance=TOLERANCE):
    n = len(split)
    rows = [lists[name] for name in LISTS]
    checks = {f"{name}: share within {tolerance * 100:g} points of {s:g}": abs(len(r) / n - s) <= tolerance
              for name, r, s in zip(LISTS, rows, SHARES)}
    checks["every row in exactly one set"] = bool((np.bincount(np.concatenate(rows), minlength=n) == 1).all())
    checks["no composition in two sets"] = len(np.unique(group)) == sum(len(np.unique(group[r])) for r in rows)
    for name, k, size, _, exclude in SUBSETS:
        r = lists[name]
        checks[f"{name}: {size} distinct rows of {LISTS[k]}"] = len(np.unique(r)) == size and bool((split[r] == k).all())
        if exclude:
            checks[f"{name}: no row of {exclude}"] = not np.isin(r, lists[exclude]).any()
    return checks


def write(args):
    table = load(args.table)
    group = table["composition"]
    strata = strata_of(group, table["data_ids"])
    lists, split = build(group, strata)
    checks = list_checks(lists, split, group)
    for text, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    if not all(checks.values()):
        raise SystemExit("a list check FAILED; nothing was written")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {"method": "within each data source (the data_ids of most rows of a composition, in code order), the compositions "
                      f"in numpy.random.default_rng({SEED}).permutation order (one generator); each composition goes to the "
                      "set where the middle of its rows falls on the cumulative row count of the source (borders at 70 and "
                      "90 percent); "
                      "subsets: default_rng(seed).choice(rows of the set not in the excluded list, n, replace=False), "
                      "sorted", "group_key": "composition (atoms.info['composition'])", "group_seed": SEED,
            "table_sha256": sha256(args.table), "lists": {}}
    for name, rows in lists.items():
        write_rows(out / name, rows)
        sub = next((s for s in SUBSETS if s[0] == name), None)
        entry = {"split": "train_4M", "rows_in_split": len(split), "count": len(rows),
                 "seed": SEED if sub is None else sub[3], "first": int(rows[0]), "last": int(rows[-1])}
        if sub:
            entry["subset_of"] = LISTS[sub[1]]
            if sub[4]:
                entry["disjoint_from"] = sub[4]
        info["lists"][name] = {**entry, "sha256": sha256(out / name)}
        print(name, len(rows), info["lists"][name]["sha256"])
    (out / "split_samples.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.assignment:
        np.savez_compressed(args.assignment, split=split, group=group, group_keys=table["composition_keys"],
                            group_stratum=strata, stratum_keys=table["data_ids_keys"], set_names=np.array(SETS))


def describe_groups(codes):
    sizes = np.bincount(codes)
    top = np.sort(sizes)[::-1]
    return {"groups": len(sizes), "rows_per_group_p50_p90_p99": np.percentile(sizes, [50, 90, 99]).tolist(),
            "largest_groups": top[:5].tolist(), "largest_group_share": float(top[0] / len(codes)),
            "rows_in_groups_of_1": int((sizes == 1).sum())}


def shares(codes, n_codes):
    return np.bincount(codes, minlength=n_codes) / len(codes)


def element_shares(lo, hi):
    return np.array([((lo >> np.uint64(z)) & np.uint64(1)).mean() if z < 64 else ((hi >> np.uint64(z - 64)) & np.uint64(1)).mean()
                     for z in range(100)])


def gaps(t, split):
    """KS statistic of atoms per structure and the largest share gaps, for each pair of sets."""
    from scipy.stats import ks_2samp
    labels = {"data_ids": t["data_ids_keys"], "elements": np.arange(100)}
    codes = {"data_ids": t["data_ids"]}
    for key in ("charge", "spin"):
        labels[key], codes[key] = np.unique(t[key], return_inverse=True)
    per_set = []
    for k in range(3):
        m = split == k
        per_set.append({"natoms": t["natoms"][m], "elements": element_shares(t["elements_lo"][m], t["elements_hi"][m]),
                        **{key: shares(c[m], len(labels[key])) for key, c in codes.items()}})
    out = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        x, y = per_set[a], per_set[b]
        row = {"ks_atoms_per_structure": float(ks_2samp(x["natoms"], y["natoms"]).statistic)}
        for key in ("data_ids", "elements", "charge", "spin"):
            d = np.abs(x[key] - y[key])
            row[f"largest_{key}_share_gap"] = float(d.max())
            row[f"largest_{key}_share_gap_at"] = labels[key][d.argmax()].item()
        out[f"{SETS[a]} vs {SETS[b]}"] = row
    return out


def sampler_rows(natoms, steps, seed, max_atoms=450):
    """Batches per epoch, and rows in the first `steps` batches of epoch 0, with fairchem's own batch builder and order
    (MaxAtomDistributedBatchSampler, one GPU, the run's seed and training.max_atoms)."""
    import torch
    from fairchem.core.datasets.samplers.max_atom_distributed_sampler import get_batches
    order = np.random.default_rng(seed).permutation(len(natoms))
    batches = get_batches(natoms[order].astype(np.int64), order, max_atoms, 0)[0]
    g = torch.Generator()
    g.manual_seed(seed)
    first = torch.randperm(len(batches), generator=g)[:steps].tolist()
    return len(batches), sum(len(batches[i]) for i in first)


def check(args):
    t = load(args.table)
    info = json.loads((Path(args.lists_dir) / "split_samples.json").read_text(encoding="utf-8"))["lists"]
    lists = {name: read_rows(Path(args.lists_dir) / name) for name in info}
    group, n = t["composition"], len(t["composition"])
    split = np.full(n, -1, np.int8)
    for k, name in enumerate(LISTS):
        split[lists[name]] = k
    strata = strata_of(group, t["data_ids"])
    rebuilt, _ = build(group, strata)
    report = {"rows": n, "checks": {f"{name}: sha256 equals split_samples.json": sha256(Path(args.lists_dir) / name) == e["sha256"]
                                    for name, e in info.items()}}
    report["checks"]["lists equal the assignment recomputed from the table"] = all(
        np.array_equal(rebuilt[name], lists[name]) for name in info)
    report["checks"].update(list_checks(lists, split, group))
    sys_sets = [np.unique(t["system"][lists[name]]) for name in LISTS]
    report["checks"]["no calculation folder in two sets"] = len(np.unique(t["system"])) == sum(map(len, sys_sets))
    report["sets"] = {name: {"rows": len(lists[name]), "share": len(lists[name]) / n,
                             "gap_points": 100 * (len(lists[name]) / n - s), "compositions": len(np.unique(group[lists[name]])),
                             "calculation_folders": len(sys_sets[k])} for k, (name, s) in enumerate(zip(LISTS, SHARES))}
    comp_per_system = np.bincount(np.unique(np.stack([t["system"], group]), axis=1)[0])
    systems_per_comp = np.bincount(np.unique(np.stack([group, t["system"]]), axis=1)[0])
    report["groups"] = {"composition": describe_groups(group), "calculation_folder": describe_groups(t["system"]),
                        "compositions_with_several_folders": int((systems_per_comp > 1).sum()),
                        "rows_of_compositions_with_several_folders": int(np.isin(group, np.flatnonzero(systems_per_comp > 1)).sum()),
                        "folders_with_several_compositions": int((comp_per_system > 1).sum())}
    gap = gaps(t, split)
    null = [gaps(t, assign(group, strata, seed)) for seed in range(args.null_seeds)]
    for pair, row in gap.items():
        for key in [k for k in row if not k.endswith("_at")]:
            values = np.array([s[pair][key] for s in null])
            row[key + "_null_p50_max"] = [float(np.median(values)), float(values.max())]
            row[key + "_rank_among_null"] = int((values < row[key]).sum())
    report["distributions"] = gap
    report["distributions_note"] = (f"null: the same statistics for {args.null_seeds} other group orders of the same rule (seeds 0 to "
                                    f"{args.null_seeds - 1}); rank = how many null values are below this split's value")
    earlier = {}
    pilot = json.loads((MANIFESTS / "pilot_samples.json").read_text(encoding="utf-8"))["lists"]
    real = n == pilot["pilot_train_200k.txt"]["rows_in_split"]       # the earlier lists index the real train_4M only
    for name in EARLIER["train_4M"] if real else ():
        rows = read_rows(MANIFESTS / name)
        earlier[name] = {s: int((split[rows] == k).sum()) for k, s in enumerate(SETS)}
    if args.val_table:
        v = load(args.val_table)
        # membership per val composition, then per row through the codes
        in_set = [np.isin(v["composition_keys"], t["composition_keys"][np.unique(group[lists[name]])]) for name in LISTS]
        in_train_4M = np.isin(v["composition_keys"], t["composition_keys"])
        report["official_val"] = {
            "rows": len(v["composition"]), "compositions": len(v["composition_keys"]),
            "compositions_also_in_train_4M": int(in_train_4M.sum()),
            "rows_whose_composition_is_in_train_4M": int(in_train_4M[v["composition"]].sum()),
            "rows_whose_composition_is_in": {s: int(m[v["composition"]].sum()) for s, m in zip(SETS, in_set)},
            "calculation_folders_also_in_train_4M": int(np.isin(v["system_keys"], t["system_keys"]).sum())}
        for name in EARLIER["val"] if real else ():
            codes = v["composition"][read_rows(MANIFESTS / name)]
            earlier[name] = {f"rows whose composition is in {s}": int(m[codes].sum()) for s, m in zip(SETS, in_set)}
    report["earlier_lists"] = earlier
    if args.steps:
        train = {LISTS[0]: lists[LISTS[0]], **{name: read_rows(MANIFESTS / name) for name in EARLIER["train_4M"] if real}}
        report["sampler"] = {name: {seed: dict(zip(("batches_per_epoch", f"rows_in_first_{args.steps}_batches"),
                                                   sampler_rows(t["natoms"][rows], args.steps, seed))) for seed in range(5)}
                             for name, rows in train.items()}
        if real:      # the pilot logs: epoch-end validation at steps 27,375, 27,402 and 27,401 for seeds 0, 1, 2
            got = [report["sampler"]["pilot_train_200k.txt"][s]["batches_per_epoch"] for s in range(3)]
            report["checks"]["sampler gives the measured pilot epochs 27375, 27402, 27401"] = got == [27375, 27402, 27401]
    report["passed"] = all(report["checks"].values())
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for text, ok in report["checks"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    print("RESULT:", "all checks passed" if report["passed"] else "a check FAILED")
    sys.exit(0 if report["passed"] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", required=True, choices=("train_4M", "val"))
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=extract)
    p = sub.add_parser("write")
    p.add_argument("--table", required=True, help="the extract output of train_4M")
    p.add_argument("--out-dir", default=str(MANIFESTS))
    p.add_argument("--assignment", help="also write the row -> composition -> set table here (npz)")
    p.set_defaults(fn=write)
    p = sub.add_parser("check")
    p.add_argument("--table", required=True)
    p.add_argument("--val-table", help="the extract output of val, for the relation to the official val split")
    p.add_argument("--lists-dir", default=str(MANIFESTS))
    p.add_argument("--null-seeds", type=int, default=20)
    p.add_argument("--steps", type=int, default=150_000, help="run length for the rows-seen count (0: skip)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=check)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
