"""CPU checks of scripts/cluster_se.py: the cluster sandwich SE against the row SE of eval_rows.py, against duplicated
rows, and against the cluster bootstrap."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cluster_se as cs  # noqa: E402
import eval_rows as er  # noqa: E402


class ClusterSETest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.n = 4000
        self.e = rng.exponential(0.005, self.n)
        self.c = 3.0 * rng.integers(3, 60, self.n)
        self.f = self.c * rng.exponential(0.03, self.n)

    def test_one_row_per_cluster_is_the_row_se(self):
        s = er.summarize(list(zip(self.e, self.f, self.c)))
        ident = np.arange(self.n)
        r, se = cs.ratio_se(self.e, np.ones(self.n), ident)
        self.assertAlmostEqual(r, s["energy_mae_per_atom_eV"], places=15)
        self.assertAlmostEqual(se, s["energy_se"], places=15)
        r, se = cs.ratio_se(self.f, self.c, ident)
        self.assertAlmostEqual(r, s["force_mae_eV_per_A"], places=15)
        self.assertAlmostEqual(se, s["force_se"], places=15)

    def test_duplicated_rows(self):
        # k copies of each row in one cluster: the cluster SE equals the row SE of the original rows,
        # while the row SE of the copies is about sqrt(k) too small
        k = 5
        num, den = np.repeat(self.f, k), np.repeat(self.c, k)
        _, se_orig = cs.ratio_se(self.f, self.c, np.arange(self.n))
        _, se_cl = cs.ratio_se(num, den, np.repeat(np.arange(self.n), k))
        _, se_row = cs.ratio_se(num, den, np.arange(self.n * k))
        self.assertAlmostEqual(se_cl, se_orig, places=15)
        self.assertAlmostEqual(se_orig / se_row, math.sqrt(k), delta=0.01)

    def test_bootstrap_agrees_with_sandwich(self):
        cluster = np.random.default_rng(2).integers(0, 800, self.n)
        _, cluster = np.unique(cluster, return_inverse=True)
        boot = cs.bootstrap_se([self.e, self.f], [np.ones(self.n), self.c], cluster, 1000, 0)
        for b, (num, den) in zip(boot, ((self.e, np.ones(self.n)), (self.f, self.c))):
            self.assertAlmostEqual(b / cs.ratio_se(num, den, cluster)[1], 1.0, delta=0.1)

    def test_table_rows_pairs_and_subsets(self):
        natoms = (self.c // 3).astype(int)
        natoms[:50] = 2
        c = 3.0 * natoms
        data = {"a": (self.e, self.f, c), "b": (self.e * 0.9, self.f * 1.1, c)}
        cluster = np.arange(self.n) // 4
        out = cs.table_rows(data, [("a", "b")], cluster, natoms, 0, 0, ["all", "natoms_2", "natoms_above_2"])
        get = {(r["subset"], r["a"], r["b"]): r for r in out}
        for subset in ("all", "natoms_2"):
            for q in ("energy", "force"):
                diff = get[subset, "a", ""][f"{q}_value"] - get[subset, "b", ""][f"{q}_value"]
                self.assertAlmostEqual(get[subset, "a", "b"][f"{q}_value"], diff, places=12)
        self.assertEqual(get["natoms_2", "a", ""]["n_structures"], 50)
        self.assertEqual(get["natoms_above_2", "a", ""]["n_structures"], self.n - 50)
        self.assertEqual(get["all", "a", ""]["largest_cluster"], 4)


if __name__ == "__main__":
    unittest.main()
