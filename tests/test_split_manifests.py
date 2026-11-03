"""The 70/20/10 composition split (docs/data-split.md): the assignment and the checks on synthetic tables, and the written lists."""

import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_split_manifests as msm  # noqa: E402


def synthetic_groups(n_groups=30_000, seed=1):
    """Composition codes of about 600,000 rows, 1 to about 200 rows per composition, rows in random order."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.repeat(np.arange(n_groups), rng.geometric(0.05, n_groups))).astype(np.int32)


def synthetic_table(path, group, seed=2):
    rng = np.random.default_rng(seed)
    n = len(group)
    np.savez_compressed(path, natoms=rng.integers(2, 300, n).astype(np.int16), charge=rng.integers(-2, 3, n).astype(np.int16),
                        spin=rng.integers(1, 4, n).astype(np.int16), elements_lo=rng.integers(0, 2**63, n, dtype=np.uint64),
                        elements_hi=np.zeros(n, np.uint64), composition=group,
                        composition_keys=np.array([f"C{i}" for i in range(group.max() + 1)]),
                        system=group, system_keys=np.array([f"s{i}" for i in range(group.max() + 1)]),
                        data_ids=rng.integers(0, 3, n).astype(np.int32), data_ids_keys=np.array(["a", "b", "c"]))


class AssignTests(unittest.TestCase):
    def test_whole_compositions_near_the_target_shares_and_deterministic(self):
        group = synthetic_groups()
        split = msm.assign(group)
        sets_per_group = np.zeros((group.max() + 1, 3), np.int64)
        np.add.at(sets_per_group, (group, split), 1)
        self.assertTrue(((sets_per_group > 0).sum(axis=1) == 1).all())
        shares = np.bincount(split, minlength=3) / len(group)
        # the border rule bounds the gap of each set by the largest composition
        self.assertLessEqual(np.abs(shares - msm.SHARES).max(), np.bincount(group).max() / len(group))
        self.assertTrue(np.array_equal(split, msm.assign(group)))
        self.assertFalse(np.array_equal(split, msm.assign(group, seed=1)))

    def test_strata_get_the_shares_each(self):
        group = synthetic_groups()
        rng = np.random.default_rng(3)
        label_of_group = rng.integers(0, 4, group.max() + 1)
        labels = np.where(rng.random(len(group)) < 0.1, rng.integers(0, 4, len(group)), label_of_group[group])
        strata = msm.strata_of(group, labels)
        self.assertEqual(msm.strata_of(np.array([0, 0, 0, 1]), np.array([2, 1, 2, 0])).tolist(), [2, 0])
        split = msm.assign(group, strata)
        sizes = np.bincount(group)
        for s in range(4):
            rows = strata[group] == s
            shares = np.bincount(split[rows], minlength=3) / rows.sum()
            self.assertLessEqual(np.abs(shares - msm.SHARES).max(), sizes[strata == s].max() / rows.sum())
        self.assertTrue(np.array_equal(msm.assign(group, np.zeros(len(sizes), np.int64)), msm.assign(group)))

    def test_lists_and_their_checks(self):
        group = synthetic_groups()
        lists, split = msm.build(group)
        checks = msm.list_checks(lists, split, group)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual([len(lists[name]) for name, *_ in msm.SUBSETS], [5_000, 20_000, 20_000])
        again, _ = msm.build(group)
        self.assertTrue(all(np.array_equal(lists[k], again[k]) for k in lists))

        bad = dict(lists)
        moved = lists["split_val_20.txt"][:1]                        # one row of a val composition put into train as well
        bad["split_train_70.txt"] = np.sort(np.concatenate([lists["split_train_70.txt"], moved]))
        checks = msm.list_checks(bad, split, group)
        self.assertFalse(checks["every row in exactly one set"])
        self.assertFalse(checks["no composition in two sets"])
        bad = dict(lists, **{"split_val_select_20k.txt": lists["split_val_monitor_5k.txt"]})
        checks = msm.list_checks(bad, split, group)
        self.assertFalse(checks["split_val_select_20k.txt: no row of split_val_monitor_5k.txt"])
        self.assertFalse(checks["split_val_select_20k.txt: 20000 distinct rows of split_val_20.txt"])
        lopsided = np.where(split == 2, 1, split).astype(np.int8)     # no test set
        checks = msm.list_checks({**lists, "split_test_10.txt": np.flatnonzero(lopsided == 2)}, lopsided, group)
        self.assertFalse(checks["split_test_10.txt: share within 0.5 points of 0.1"])

    def test_write_and_check_on_a_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            table, out = Path(tmp) / "keys.npz", Path(tmp) / "lists"
            synthetic_table(table, synthetic_groups())
            msm.write(Namespace(table=str(table), out_dir=str(out), assignment=str(Path(tmp) / "assign.npz")))
            info = json.loads((out / "split_samples.json").read_text(encoding="utf-8"))
            self.assertEqual(info["table_sha256"], hashlib.sha256(table.read_bytes()).hexdigest())
            self.assertEqual(info["lists"]["split_val_select_20k.txt"]["disjoint_from"], "split_val_monitor_5k.txt")
            with self.assertRaises(SystemExit) as done:
                msm.check(Namespace(table=str(table), val_table=str(table), lists_dir=str(out), null_seeds=2, steps=200,
                                    out=str(Path(tmp) / "checks.json")))
            self.assertEqual(done.exception.code, 0)
            report = json.loads((Path(tmp) / "checks.json").read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(set(report["distributions"]), {"train vs val", "train vs test", "val vs test"})
            self.assertEqual(sum(report["official_val"]["rows_whose_composition_is_in"].values()), report["rows"])
            first = report["sampler"]["split_train_70.txt"]["0"]
            self.assertLess(200, first["rows_in_first_200_batches"])
            self.assertLess(first["rows_in_first_200_batches"], report["sets"]["split_train_70.txt"]["rows"])
            assignment = msm.load(Path(tmp) / "assign.npz")
            self.assertTrue(np.array_equal(np.flatnonzero(assignment["split"] == 2),
                                           np.loadtxt(out / "split_test_10.txt", dtype=np.int64)))


class ListFileTests(unittest.TestCase):
    def test_lists_match_their_hashes_and_partition_train_4M(self):
        info = json.loads((ROOT / "manifests" / "split_samples.json").read_text(encoding="utf-8"))["lists"]
        self.assertEqual(set(info), set(msm.LISTS) | {s[0] for s in msm.SUBSETS})
        rows = {}
        for name, entry in info.items():
            data = (ROOT / "manifests" / name).read_bytes()
            self.assertNotIn(b"\r", data)
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            r = np.array(data.split(), dtype=np.int64)
            self.assertEqual((len(r), entry["split"], entry["rows_in_split"]), (entry["count"], "train_4M", 3_986_754))
            self.assertTrue((np.diff(r) > 0).all())                 # sorted and no duplicates
            rows[name] = r
        everything = np.concatenate([rows[name] for name in msm.LISTS])
        self.assertTrue(np.array_equal(np.sort(everything), np.arange(3_986_754)))
        for name, k, size, _, exclude in msm.SUBSETS:
            self.assertEqual(len(rows[name]), size)
            self.assertTrue(np.isin(rows[name], rows[msm.LISTS[k]]).all())
            if exclude:
                self.assertFalse(np.isin(rows[name], rows[exclude]).any())


if __name__ == "__main__":
    unittest.main()
