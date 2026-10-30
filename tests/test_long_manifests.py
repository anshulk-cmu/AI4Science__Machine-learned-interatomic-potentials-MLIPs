"""Long-run and selection lists (docs/data-lists.md): the draw on a small synthetic split, and the written list files."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_long_manifests as mlm  # noqa: E402


class DrawTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.rows_in_split = {"train_4M": 1_000, "val": 500}
        self.base = np.sort(rng.choice(1_000, 100, replace=False))
        self.report = np.sort(rng.choice(500, 50, replace=False))

    def test_superset_disjoint_and_deterministic(self):
        long_rows, select = mlm.build(self.rows_in_split, self.base, self.report, n_extra=300, n_select=50)
        checks = mlm.list_checks(long_rows, select, self.base, self.report, self.rows_in_split, n_long=400, n_select=50)
        self.assertTrue(all(checks.values()), checks)
        again = mlm.build(self.rows_in_split, self.base, self.report, n_extra=300, n_select=50)
        self.assertTrue(np.array_equal(long_rows, again[0]) and np.array_equal(select, again[1]))

    def test_checks_catch_a_missing_base_row_and_a_shared_row(self):
        long_rows, select = mlm.build(self.rows_in_split, self.base, self.report, n_extra=300, n_select=50)
        select = select.copy()
        select[0] = self.report[0]
        checks = mlm.list_checks(long_rows[1:], select, self.base, self.report, self.rows_in_split, n_long=400, n_select=50)
        self.assertFalse(checks["select: no row of the report set"])
        self.assertFalse(checks["long: count"])

    def test_cannot_draw_more_than_the_pool(self):
        with self.assertRaises(ValueError):
            mlm.draw_outside(10, np.arange(5), 6, 1)

    def test_system_of_a_source_path(self):
        self.assertEqual(mlm.system("omol/electrolytes/x/azanide_mol328_solv3_-1_1/step3/orca.tar.zst"),
                         "omol/electrolytes/x/azanide_mol328_solv3_-1_1")
        self.assertEqual(mlm.system("ani1xbb/aniBB_021_391079_0_1/orca.tar.zst"), "ani1xbb/aniBB_021_391079_0_1")


class ListFileTests(unittest.TestCase):
    def test_lists_match_their_hashes_and_relations(self):
        info = json.loads((ROOT / "manifests" / "long_samples.json").read_text(encoding="utf-8"))["lists"]
        self.assertEqual({k: (v["split"], v["count"]) for k, v in info.items()},
                         {"long_train_1200k.txt": ("train_4M", 1_200_000), "select_val_5k.txt": ("val", 5_000)})
        rows = {}
        for name, entry in info.items():
            data = (ROOT / "manifests" / name).read_bytes()
            self.assertNotIn(b"\r", data)
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            r = np.array(data.split(), dtype=np.int64)
            self.assertEqual(len(r), entry["count"])
            self.assertTrue((np.diff(r) > 0).all())                 # sorted and no duplicates
            self.assertGreaterEqual(r[0], 0)
            self.assertLess(r[-1], entry["rows_in_split"])
            rows[name] = r
        pilot = np.loadtxt(ROOT / "manifests" / "pilot_train_200k.txt", dtype=np.int64)
        report = np.loadtxt(ROOT / "manifests" / "pilot_val_5k.txt", dtype=np.int64)
        self.assertTrue(np.isin(pilot, rows["long_train_1200k.txt"]).all())
        self.assertFalse(np.isin(rows["select_val_5k.txt"], report).any())


if __name__ == "__main__":
    unittest.main()
