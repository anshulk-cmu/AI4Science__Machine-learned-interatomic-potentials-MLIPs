"""Pilot data checks on CPU: build the index files, verify the fixed sample lists, measure the training data.

  prep    write <data-root>/pilot/{train_4M,val}_index.npz, a copy of metadata.npz plus an integer field `index`,
          which fairchem's subset_to hook needs to select rows by number
  verify  compose configs/train.yaml with the given Hydra overrides, build the train and val datasets, and check that
          they hold exactly the rows of the sample lists, and that the lists match their hashes
  stats   for the rows of one list: neighbours within the cutoff, force and energy scales with the official element
          references, elements, charges and spins, and the structure ids

Run from the repository root. Examples:
  python scripts/check_pilot_data.py prep --data-root $OMOL_DATA_ROOT
  python scripts/check_pilot_data.py verify -- 'train_dataset...no_shuffle=true' ...
  python scripts/check_pilot_data.py stats --src $OMOL_DATA_ROOT/train_4M --rows manifests/pilot_train_200k.txt --out stats.json
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
SPLITS = {"train_4M": "train", "val": "val"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def prep(args):
    out_dir = Path(args.data_root) / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        meta = np.load(Path(args.data_root) / split / "metadata.npz", allow_pickle=True)
        fields = {k: meta[k] for k in meta.files}
        fields["index"] = np.arange(len(fields["natoms"]), dtype=np.int64)
        out = out_dir / f"{split}_index.npz"
        np.savez(out, **fields)
        print(f"{out}: {len(fields['natoms'])} rows, fields {sorted(fields)}", flush=True)


def stub_fcntl():
    try:
        import fcntl  # noqa: F401
    except ImportError:          # native Windows only
        stub = types.ModuleType("fcntl")
        for name in ("LOCK_EX", "LOCK_SH", "LOCK_UN", "LOCK_NB"):
            setattr(stub, name, 0)
        stub.flock = lambda *a, **k: None
        sys.modules["fcntl"] = stub


def verify(args):
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    stub_fcntl()
    from fairchem.core._cli import get_hydra_config_from_yaml

    samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))["lists"]
    GlobalHydra.instance().clear()
    cfg = get_hydra_config_from_yaml(str(ROOT / "configs" / "train.yaml"), list(args.overrides))
    ok = True
    for loader, rows_file in (("train_dataloader", args.train_rows), ("eval_dataloader", args.val_rows)):
        name = Path(rows_file).name
        want = read_rows(rows_file)
        info = samples[name]
        sub = hydra.utils.instantiate(cfg.runner[loader].dataset).datasets[0]
        got = np.sort(np.asarray(sub.indices))
        checks = {
            "list sha256 equals manifest": sha256(rows_file) == info["sha256"],
            "list size equals manifest": len(want) == info["count"],
            "dataset rows equal list": got.shape == want.shape and bool((got == np.sort(want)).all()),
            "split size equals manifest": sub.dataset.num_samples == info["rows_in_split"],
        }
        for text, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'}  {name}: {text}", flush=True)
            ok &= passed
    print("RESULT:", "all checks passed" if ok else "a check FAILED", flush=True)
    sys.exit(0 if ok else 1)


def official_refs():
    import yaml
    text = (ROOT / "configs" / "element_refs" / "omol.yaml").read_text(encoding="utf-8")
    return np.array(yaml.safe_load(text)["omol_elem_refs"], dtype=np.float64)


def measure(task):
    """Statistics of one chunk of rows, as sums that can be added across chunks."""
    from scipy.spatial import cKDTree
    src, rows, cutoff, refs = task
    from fairchem.core.datasets import AseDBDataset
    data = AseDBDataset({"src": src})
    caps = (30, 50, 100, 300)
    out = {"structures": 0, "atoms": 0, "edges": 0, "max_neighbors": 0, "over_cap": dict.fromkeys(caps, 0),
           "force_sq": 0.0, "force_max": 0.0, "res_sum": 0.0, "res_sq": 0.0, "res_abs_max": 0.0,
           "res_structs": 0, "max_natoms": 0, "elements": {}, "charge": {}, "spin": {}, "sids": []}
    for row in rows:
        atoms = data.get_atoms(int(row))
        z, pos, n = atoms.get_atomic_numbers(), atoms.get_positions(), len(atoms)
        degree = cKDTree(pos).query_ball_point(pos, r=cutoff, return_length=True) - 1
        force = atoms.get_forces()
        residual = (atoms.get_potential_energy() - refs[z].sum()) / n
        out["structures"] += 1
        out["atoms"] += n
        out["edges"] += int(degree.sum())
        out["max_neighbors"] = max(out["max_neighbors"], int(degree.max()))
        for cap in caps:
            out["over_cap"][cap] += int((degree > cap).sum())
        out["force_sq"] += float((force ** 2).sum())
        out["force_max"] = max(out["force_max"], float(np.linalg.norm(force, axis=1).max()))
        out["res_sum"] += float(residual)
        out["res_sq"] += float(residual) ** 2
        out["res_abs_max"] = max(out["res_abs_max"], abs(float(residual)))
        out["res_structs"] += 1
        out["max_natoms"] = max(out["max_natoms"], n)
        for e in np.unique(z):
            out["elements"][int(e)] = out["elements"].get(int(e), 0) + 1
        for key in ("charge", "spin"):
            value = str(atoms.info.get(key))
            out[key][value] = out[key].get(value, 0) + 1
        out["sids"].append(str(atoms.info.get("source", "")))
    return out


def merge(parts):
    total = parts[0]
    for part in parts[1:]:
        for k, v in part.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    total[k][kk] = total[k].get(kk, 0) + vv
            elif k == "sids":
                total[k] += v
            elif k.endswith("max") or k in ("max_neighbors", "max_natoms"):
                total[k] = max(total[k], v)
            else:
                total[k] += v
    return total


def stats(args):
    rows = read_rows(args.rows)
    refs = official_refs()
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    chunks = [(args.src, part, args.cutoff, refs) for part in np.array_split(rows, max(workers * 8, 1))]
    with mp.Pool(workers) as pool:
        total = merge(pool.map(measure, chunks))
    atoms, structs = total["atoms"], total["structures"]
    sids = total.pop("sids")
    result = {
        "rows_file": str(args.rows), "rows_sha256": sha256(args.rows), "structures": structs, "atoms": atoms,
        "mean_atoms_per_structure": atoms / structs, "max_atoms_per_structure": total["max_natoms"],
        "cutoff": args.cutoff, "mean_neighbors_per_atom": total["edges"] / atoms,
        "max_neighbors_per_atom": total["max_neighbors"],
        "atoms_over_cap": {f"{c}": n for c, n in total["over_cap"].items()},
        "fraction_atoms_over_cap": {f"{c}": n / atoms for c, n in total["over_cap"].items()},
        "force_norm_rms_per_atom": float(np.sqrt(total["force_sq"] / atoms)),
        "force_component_rms": float(np.sqrt(total["force_sq"] / (3 * atoms))),
        "force_norm_max": total["force_max"],
        "energy_residual_per_atom_mean": total["res_sum"] / structs,
        "energy_residual_per_atom_rms": float(np.sqrt(total["res_sq"] / structs)),
        "energy_residual_per_atom_abs_max": total["res_abs_max"],
        "elements_structures": dict(sorted(total["elements"].items())),
        "charge_structures": total["charge"], "spin_structures": total["spin"],
        "sids_sha256": hashlib.sha256("\n".join(sids).encode("utf-8")).hexdigest(),
    }
    if args.sids:
        Path(args.sids).write_text("\n".join(sids) + "\n", encoding="utf-8", newline="\n")
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("elements")}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data-root", default=os.environ.get("OMOL_DATA_ROOT"))
    p.set_defaults(fn=prep)
    p = sub.add_parser("verify")
    p.add_argument("--samples", default=str(MANIFESTS / "pilot_samples.json"))
    p.add_argument("--train-rows", default=str(MANIFESTS / "pilot_train_200k.txt"))
    p.add_argument("--val-rows", default=str(MANIFESTS / "pilot_val_5k.txt"))
    p.add_argument("overrides", nargs="*", help="Hydra overrides, the same as the training run uses")
    p.set_defaults(fn=verify)
    p = sub.add_parser("stats")
    p.add_argument("--src", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sids")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--workers", type=int)
    p.set_defaults(fn=stats)
    args = ap.parse_args()
    if args.cmd == "prep" and not args.data_root:
        sys.exit("set --data-root or OMOL_DATA_ROOT")
    args.fn(args)


if __name__ == "__main__":
    main()
