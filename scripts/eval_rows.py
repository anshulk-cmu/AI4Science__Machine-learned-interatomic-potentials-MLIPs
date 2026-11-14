"""Per-row energy and force errors of one checkpoint on a row list: sharded, resumable, several structures per forward pass.

Same predictor, conversion and metrics as training.checkpoints.evaluate, but one CSV line per structure, so a preempted job
resumes where it stopped, shards merge with a check that every row appears exactly once, and the standard error is known.
  run:    python scripts/eval_rows.py run --checkpoint CKPT --data <OMol25 train_4M> --rows manifests/split_test_eval_10k.txt \
              --shard 0 --num-shards 1 --batch-atoms 2000 --dtype float32 --out DIR/dense_test10k_s0of1.csv
  merge:  python scripts/eval_rows.py merge --rows manifests/split_test_eval_10k.txt --label dense --eval-split test_eval_10k \
              --table results.csv DIR/dense_test10k_s*.csv
--batch-atoms 0 evaluates one structure at a time (the evaluator of checkpoint_tools.py).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training import checkpoints as ck  # noqa: E402

FIELDS = ["row", "natoms", "energy_pred", "energy_true", "energy_abs_err_per_atom", "force_abs_sum", "force_components"]


def shard_rows(rows_path, shard, num_shards):
    if not 0 <= shard < num_shards:
        raise ValueError("need 0 <= shard < num_shards")
    return ck.read_rows(rows_path)[shard::num_shards]


def done_rows(out):
    """Rows already written. A preemption can leave one cut last line; it is removed. Any other bad line is an error."""
    if not out.exists():
        return set()
    lines = out.read_text().splitlines(keepends=True)
    if len(lines) <= 1 and not "".join(lines).endswith("\n"):     # cut while writing the header
        out.unlink()
        return set()
    good = [lines[0]] if lines and lines[0].rstrip("\n").split(",") == FIELDS else []
    if not good:
        raise ValueError(f"{out}: missing header")
    for i, line in enumerate(lines[1:], 1):
        ok = line.endswith("\n") and len(line.split(",")) == len(FIELDS)
        if not ok and i != len(lines) - 1:
            raise ValueError(f"{out}: malformed line {i + 1}")
        if ok:
            good.append(line)
    if len(good) != len(lines):
        out.write_text("".join(good))
    rows = [int(line.split(",")[0]) for line in good[1:]]
    if len(set(rows)) != len(rows):
        raise ValueError(f"{out}: duplicate rows")
    return set(rows)


def run(a):
    import torch
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    from checkpoint_tools import dataset_at, check_split
    data = dataset_at(a.data)
    check_split(a.rows, data)
    rows = shard_rows(a.rows, a.shard, a.num_shards)
    if max(rows) >= len(data):
        raise ValueError("row index outside dataset")
    meta = dict(checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_sha256=ck.sha256(a.checkpoint),
                rows=str(Path(a.rows).resolve()), rows_sha256=ck.sha256(a.rows), data_rows=len(data),
                shard=a.shard, num_shards=a.num_shards, dtype=a.dtype, batch_atoms=a.batch_atoms)
    sidecar = a.out.with_suffix(".json")
    if sidecar.exists():
        old = json.loads(sidecar.read_text())
        if old != meta:
            raise ValueError(f"{sidecar} records another setup: {old}")
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    done = done_rows(a.out)
    if not done <= set(rows):
        raise ValueError(f"{a.out} holds rows outside this shard")
    todo = [r for r in rows if r not in done]
    print(f"shard {a.shard}/{a.num_shards}: {len(rows)} rows, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    unit = ck.predictor(a.checkpoint, a.device, a.dtype)
    new = not a.out.exists()
    with open(a.out, "a", newline="") as stream:
        if new:
            stream.write(",".join(FIELDS) + "\n")
        batch, atoms_in_batch = [], 0

        def flush():
            items = [AtomicData.from_ase(at, task_name="omol", r_edges=False, r_data_keys=["charge", "spin"],
                                         r_forces=False, r_stress=False,
                                         molecule_cell_size=120.0 if at.cell.volume == 0 else None,
                                         target_dtype=torch.float64) for _, at in batch]
            pred = unit.predict(atomicdata_list_to_batch(items))
            energy = pred["energy"].detach().cpu().double().reshape(-1).numpy()
            forces = pred["forces"].detach().cpu().double().numpy()
            if len(energy) != len(batch) or len(forces) != sum(len(at) for _, at in batch):
                raise ValueError("prediction does not match the batch")
            start = 0
            for (index, at), e in zip(batch, energy):
                f, target_f, target_e = forces[start:start + len(at)], at.get_forces(), at.get_potential_energy()
                start += len(at)
                if f.shape != target_f.shape or not np.isfinite(f).all() or not np.isfinite(target_f).all() \
                        or not math.isfinite(e + target_e):
                    raise ValueError(f"nonfinite or malformed prediction/target at row {index}")
                stream.write(f"{index},{len(at)},{e!r},{target_e!r},{abs(e - target_e) / len(at)!r},"
                             f"{float(np.abs(f - target_f).sum())!r},{target_f.size}\n")
            stream.flush()
            batch.clear()

        for index in todo:
            at = data.get_atoms(index)
            if not len(at) or not {"charge", "spin"} <= at.info.keys():
                raise ValueError(f"row {index} needs atoms, charge and spin")
            # fairchem answers a lone non-periodic atom from a lookup table only when it is alone in its batch
            alone = len(at) == 1 and not at.pbc.any()
            if batch and (alone or atoms_in_batch + len(at) > a.batch_atoms):
                flush()
                atoms_in_batch = 0
            batch.append((index, at))
            atoms_in_batch += len(at)
            if alone or a.batch_atoms <= 0:
                flush()
                atoms_in_batch = 0
        if batch:
            flush()
    print(f"shard {a.shard}/{a.num_shards} complete", flush=True)


def summarize(records):
    """records: (energy_abs_err_per_atom, force_abs_sum, force_components) per structure.

    Energy MAE is a mean over structures; its standard error is s / sqrt(n). Force MAE is a ratio of sums over all
    components; its standard error is the ratio estimator's, with the structure as the sampling unit."""
    e, fsum, fcount = (np.array(c, dtype=np.float64) for c in zip(*records))
    n = len(e)
    force = fsum.sum() / fcount.sum()
    resid = fsum - force * fcount
    return dict(n_structures=n, n_force_components=int(fcount.sum()),
                energy_mae_per_atom_eV=float(e.mean()), energy_se=float(e.std(ddof=1) / math.sqrt(n)),
                force_mae_eV_per_A=float(force),
                force_se=float(math.sqrt((resid ** 2).sum() / (n * (n - 1))) / fcount.mean()))


def merge(a):
    expected = ck.read_rows(a.rows)
    rows_sha = ck.sha256(a.rows)
    seen, records, metas = {}, [], []
    for path in a.shards:
        meta = json.loads(path.with_suffix(".json").read_text())
        metas.append(meta)
        with open(path, newline="") as stream:
            for line in csv.DictReader(stream):
                row = int(line["row"])
                if row in seen:
                    raise ValueError(f"row {row} in {seen[row]} and {path}")
                seen[row] = path
                records.append((float(line["energy_abs_err_per_atom"]), float(line["force_abs_sum"]),
                                int(line["force_components"])))
    for key in ("checkpoint_sha256", "rows_sha256", "dtype", "num_shards", "data_rows"):
        if len({m[key] for m in metas}) != 1:
            raise ValueError(f"shards disagree on {key}")
    if metas[0]["rows_sha256"] != rows_sha:
        raise ValueError("shards were run on another row list")
    if sorted({m["shard"] for m in metas}) != list(range(metas[0]["num_shards"])):
        raise ValueError("a shard file is missing")
    missing, extra = set(expected) - seen.keys(), seen.keys() - set(expected)
    if missing or extra:
        raise ValueError(f"{len(missing)} rows missing, {len(extra)} rows not in the list")
    result = dict(variant=a.label, eval_split=a.eval_split, dtype=metas[0]["dtype"], **summarize(records),
                  checkpoint_sha256=metas[0]["checkpoint_sha256"], rows_sha256=rows_sha)
    print(json.dumps(result), flush=True)
    if a.table:
        new = not a.table.exists()
        with open(a.table, "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            if new:
                writer.writeheader()
            writer.writerow(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--rows", type=Path, required=True)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--batch-atoms", type=int, default=2000, help="atoms per forward pass; 0 = one structure at a time")
    r.add_argument("--device", default="cuda")
    r.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    r.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--rows", type=Path, required=True)
    m.add_argument("--label", required=True)
    m.add_argument("--eval-split", required=True)
    m.add_argument("--table", type=Path, help="append the result to this CSV")
    m.add_argument("shards", type=Path, nargs="+")
    a = ap.parse_args()
    run(a) if a.command == "run" else merge(a)


if __name__ == "__main__":
    main()
