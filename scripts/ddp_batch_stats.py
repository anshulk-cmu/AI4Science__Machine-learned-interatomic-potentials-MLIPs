"""Batches of the two-rank sampler against one GPU, on the real atom counts of a row list (docs/ddp.md). CPU only.

One epoch of: (a) fairchem's sampler on one GPU at 450 atoms; (b) fairchem's sampler on 2 ranks at 225 atoms each, the
plain alternative; (c) training/split_sampler.py on 2 ranks. The check: (c) gives at every step exactly the structures
of (a), each once (a batch that every rank holds counts once), also after a resume inside the epoch. --worst N also
counts the atom pairs closer than 6 A (message edges) and 12 A (the Local-12 routing graph) for the N largest structures
and writes the rows with the most pairs, for the memory test.

  python scripts/ddp_batch_stats.py --index $OMOL_DATA_ROOT/pilot/train_4M_index.npz --rows manifests/split_train_70.txt \
      --seed 0 --out results/ddp/batches_split_train_70.json [--eval] [--worst 20000 --data $OMOL_DATA_ROOT/train_4M]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from check_pilot_data import read_rows, stub_fcntl  # noqa: E402

stub_fcntl()
from fairchem.core.datasets.samplers.max_atom_distributed_sampler import MaxAtomDistributedBatchSampler  # noqa: E402

from training.split_sampler import SplitBatchSampler  # noqa: E402


class Counts:
    """The two things the samplers read from a dataset: its length and the atom count of each row."""

    def __init__(self, natoms):
        self.natoms = natoms

    def __len__(self):
        return len(self.natoms)

    def get_metadata(self, attr, idx):
        return self.natoms[idx]


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "p1": float(np.percentile(a, 1)), "p50": float(np.median(a)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def split_check(ds, seed, shuffle, epoch=0, start=0):
    one = MaxAtomDistributedBatchSampler(ds, 450, 1, 0, seed, shuffle=shuffle)
    parts = [SplitBatchSampler(ds, 450, 2, r, seed, shuffle=shuffle) for r in (0, 1)]
    for s in (one, *parts):
        s.set_epoch_and_start_iteration(epoch, start)
    bad, copies, part_atoms, larger = 0, 0, [], []
    for whole, p0, p1 in zip(one, *parts):
        if p0 == p1:
            copies += 1
            bad += p0 != whole or len(whole) > 1
        else:
            bad += sorted(p0 + p1) != sorted(whole) or bool(set(p0) & set(p1))
        a = [int(ds.natoms[p].sum()) for p in (p0, p1)]
        part_atoms += a
        larger.append(max(a))
    steps = len(parts[0])
    return dict(steps=steps, mismatched_steps=bad, steps_compared=len(larger), batches_held_by_both_ranks=copies,
                atoms_per_rank=summary(part_atoms), larger_part_atoms=summary(larger),
                steps_with_a_part_above_225=int(np.sum(np.array(larger) > 225)), same_length_as_one_gpu=steps == len(one))


def stats(natoms, seed, shuffle):
    ds = Counts(natoms)
    one = MaxAtomDistributedBatchSampler(ds, 450, 1, 0, seed, shuffle=shuffle)
    out = {"rows": len(natoms), "atoms_per_structure": summary(natoms)}
    batches = list(one)
    out["one_gpu_450"] = dict(steps=len(one), atoms_per_step=summary([natoms[b].sum() for b in batches]),
                              structures_per_step=summary([len(b) for b in batches]),
                              dropped_structures=int((natoms > 450).sum()))
    ranks = [MaxAtomDistributedBatchSampler(ds, 225, 2, r, seed, shuffle=shuffle) for r in (0, 1)]
    steps = list(zip(*ranks))
    big = natoms > 225
    out["two_ranks_225_each"] = dict(
        steps=len(ranks[0]), atoms_per_step=summary([natoms[a].sum() + natoms[b].sum() for a, b in steps]),
        structures_per_step=summary([len(a) + len(b) for a, b in steps]),
        dropped_structures=int(big.sum()), dropped_structure_share=float(big.mean()),
        dropped_atom_share=float(natoms[big].sum() / natoms.sum()),
        padded_batches=ranks[0].total_size - len(ranks[0].all_batches))
    out["split_two_ranks"] = split_check(ds, seed, shuffle)
    out["split_two_ranks_resumed_in_epoch1"] = {"start_step": len(one) // 3, **split_check(ds, seed, shuffle, 1, len(one) // 3)}
    return out


def pair_counts(data, rows):
    from scipy.spatial.distance import pdist
    from fairchem.core.datasets.ase_datasets import AseDBDataset

    ds = AseDBDataset({"src": str(data), "a2g_args": {"r_edges": False}})
    out = []
    for row in rows:
        d = pdist(ds.get_atoms(int(row)).get_positions())
        out.append((int(row), 2 * int((d < 6.0).sum()), 2 * int((d < 12.0).sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, required=True, help="<data root>/pilot/<split>_index.npz")
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eval", action="store_true", help="the validation sampler (shuffle false)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worst", type=int, default=0, help="pair counts for this many of the largest structures")
    ap.add_argument("--data", type=Path, help="the split folder, for --worst")
    ap.add_argument("--worst-out", type=Path, help="row list of the structures with the most pairs, for --worst")
    args = ap.parse_args()

    rows = read_rows(args.rows)                    # the dataset holds the listed rows in ascending order
    natoms = np.load(args.index)["natoms"][np.sort(rows)].astype(np.int64)
    result = {"rows_file": args.rows.name, "seed": args.seed, "shuffle": not args.eval, **stats(natoms, args.seed, not args.eval)}
    if args.worst:
        largest = np.sort(rows)[np.argsort(-natoms, kind="stable")[:args.worst]]
        pairs = pair_counts(args.data, largest)
        by6 = sorted(pairs, key=lambda p: -p[1])[:48]
        by12 = sorted(pairs, key=lambda p: -p[2])[:48]
        chosen = sorted({p[0] for p in by6 + by12})
        index_natoms = np.load(args.index)["natoms"]
        result["worst"] = dict(candidates=len(pairs), smallest_candidate_atoms=int(natoms[np.argsort(-natoms, kind="stable")[args.worst - 1]]),
                               edges_6A=summary([p[1] for p in pairs]), edges_12A=summary([p[2] for p in pairs]),
                               top_by_6A=[dict(row=r, atoms=int(index_natoms[r]), edges_6A=e6, edges_12A=e12) for r, e6, e12 in by6[:10]],
                               top_by_12A=[dict(row=r, atoms=int(index_natoms[r]), edges_6A=e6, edges_12A=e12) for r, e6, e12 in by12[:10]],
                               written=len(chosen))
        args.worst_out.write_text("".join(f"{r}\n" for r in chosen), encoding="utf-8", newline="\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    s, t = result["split_two_ranks"], result["two_ranks_225_each"]
    print(f"{args.rows.name} seed {args.seed}: one GPU {result['one_gpu_450']['steps']} steps; split {s['steps']} steps, "
          f"{s['mismatched_steps']} mismatched of {s['steps_compared']}, {s['batches_held_by_both_ranks']} held by both; "
          f"225 per rank {t['steps']} steps, {t['dropped_structures']} structures dropped ({100 * t['dropped_structure_share']:.3f} percent)")


if __name__ == "__main__":
    main()
