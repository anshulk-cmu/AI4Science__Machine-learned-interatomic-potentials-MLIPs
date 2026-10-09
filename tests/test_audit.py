"""Tests of the audit code. CPU only: every model is replaced by a stub, so nothing is downloaded or loaded."""
import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list
from scipy import stats

AUDIT = Path(__file__).resolve().parents[1] / "audit"
sys.path.insert(0, str(AUDIT))

import analyze  # noqa: E402
import f1_forcing  # noqa: E402
import geometry as g  # noqa: E402
import models  # noqa: E402
import neon_probe  # noqa: E402
import report  # noqa: E402
import run_audit  # noqa: E402


class CountingCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results = {"energy": float(atoms.positions.sum()), "forces": -0.01 * atoms.positions}


def stub_load(*args, **kwargs):
    energy = models._energy(CountingCalculator())
    return energy, 6.0


class Statistics(unittest.TestCase):
    tau = 1e-3

    def test_classify(self):
        self.assertEqual(analyze.classify(np.full(20, 1.0), self.tau), "VIOLATION")
        self.assertEqual(analyze.classify(np.zeros(20), self.tau), "CONSISTENT")
        x = np.r_[np.full(18, 1.0), -0.1, -0.1]
        self.assertGreater(analyze.interval(x)[1], self.tau)       # outside tau, so only the sign rule decides
        self.assertEqual(analyze.classify(x, self.tau), "INCONCLUSIVE")
        self.assertEqual(analyze.classify(np.r_[np.full(19, 1.0), -0.1], self.tau), "VIOLATION")
        self.assertEqual(analyze.classify(np.r_[np.full(4, 1.0), -0.01], self.tau), "INCONCLUSIVE")   # n = 5 needs 5 of 5
        with self.assertRaises(ValueError):
            analyze.classify(np.array([1.0]), self.tau)
        with self.assertRaises(ValueError):
            analyze.classify(np.r_[np.full(19, 1.0), np.nan], self.tau)

    def test_interval(self):
        x = np.random.default_rng(1).normal(3, 1, 20)
        half = stats.t.ppf(0.975, 19) * x.std(ddof=1) / math.sqrt(20)
        np.testing.assert_allclose(analyze.interval(x), (x.mean(), x.mean() - half, x.mean() + half), rtol=0, atol=1e-12)
        rng = np.random.default_rng(2026)
        hits = 0
        for _ in range(2000):
            _, lo, hi = analyze.interval(rng.normal(3, 1, 20))
            hits += lo <= 3 <= hi
        self.assertTrue(0.93 <= hits / 2000 <= 0.97, hits / 2000)

    def write_nulls(self, folder, dev):
        n = len(dev)
        pd.DataFrame(dict(model="m", task="t", test="N2", structure="s", energy=-10.0 + np.asarray(dev),
                          ref=np.full(n, -10.0))).to_csv(folder / "nulls_m_t.csv", index=False)

    def test_tolerances(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(analyze, "RES", Path(tmp)):
            self.write_nulls(Path(tmp), np.full(25, 0.001))
            self.assertAlmostEqual(analyze.tolerances()[("m", "t")], 5.0, places=9)
            self.write_nulls(Path(tmp), np.zeros(25))
            self.assertEqual(analyze.tolerances()[("m", "t")], 1e-3)

    def test_load(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(analyze, "RES", Path(tmp)):
            pd.DataFrame(dict(model=["m"], task=["t"], dtype=["float64"], case=["AlNi"], distance=[8], run=[1],
                              e_a=[-1.5], e_b=[-2.25], e_union=[-3.5])).to_csv(Path(tmp) / "energies_m_t.csv", index=False)
            self.assertAlmostEqual(analyze.load().delta_mev[0], 250.0, places=9)

    def test_flatness(self):
        rng = np.random.default_rng(3)
        base = rng.normal(100.0, 5.0, 20)
        rows = [dict(model="m", task="t", case="AlNi", distance=d, run=r + 1, delta_mev=base[r])
                for d in (8, 100) for r in range(20)]
        df = pd.DataFrame(rows)
        tau = {("m", "t"): 0.5}
        self.assertTrue(analyze.flatness(df, tau).flat[0])
        df.loc[df.distance == 100, "delta_mev"] += 2 * 0.5
        self.assertFalse(analyze.flatness(df, tau).flat[0])

    def test_dtype_checks(self):
        def energies(dtypes):
            return pd.DataFrame(dict(model="m", task="t", dtype=dtypes))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(analyze, "RES", Path(tmp)):
            with self.assertRaises(ValueError):
                analyze.check_dtypes(energies(["float64", "float32"]))
            self.write_nulls(Path(tmp), np.zeros(3))
            with redirect_stdout(io.StringIO()) as out:
                analyze.check_dtypes(energies(["float64"]))
            self.assertIn("warning", out.getvalue())
            n = pd.read_csv(Path(tmp) / "nulls_m_t.csv").assign(dtype="float32")
            n.to_csv(Path(tmp) / "nulls_m_t.csv", index=False)
            with self.assertRaises(ValueError):
                analyze.check_dtypes(energies(["float64"]))
            analyze.check_dtypes(energies(["float32"]))

    def test_report_reproduces_results_md(self):
        res = analyze.RES
        text = report.tables(pd.read_csv(res / "summary.csv"), pd.read_csv(res / "flatness.csv"),
                             pd.read_csv(res / "comparison.csv"))
        self.assertEqual("# Audit results\n\n" + text + "\n", (res / "results.md").read_text(encoding="utf-8"))


class Geometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.al, cls.ni = g.ideal_slab("Al"), g.ideal_slab("Ni")

    def test_guard(self):
        pair = g.stacked(self.al, self.ni, 100)
        cell = pair.cell.array.copy()
        cell[2, 2] = self.al.info["thickness"] + 100 + self.ni.info["thickness"] + 3.0    # periodic image 3 A away
        pair.set_cell(cell)
        with self.assertRaises(RuntimeError):
            g.assert_disconnected(pair, 6.0)
        with self.assertRaises(RuntimeError):
            g.assert_disconnected(g.stacked(self.al, self.ni, 6.5), 6.0)
        for run in (0, 1):
            for d in g.DISTANCES:
                g.assert_disconnected(g.stacked(g.rattled(self.al, g.seed_for(run, "Al_a")),
                                                g.rattled(self.ni, g.seed_for(run, "Ni_a")), d), 6.0)

    def test_rattle_reused_across_distances(self):
        lo, up = g.rattled(self.al, g.seed_for(3, "Al_a")), g.rattled(self.ni, g.seed_for(3, "Ni_a"))
        for d in g.DISTANCES:
            pair = g.stacked(lo, up, d)
            n = len(lo)
            np.testing.assert_array_equal(pair.positions[:n], lo.positions)
            np.testing.assert_array_equal(pair.positions[n:, :2], up.positions[:, :2])
            np.testing.assert_allclose(pair.positions[n:, 2] - up.positions[:, 2], lo.info["thickness"] + d, atol=1e-12)

    def test_isolated_slab_does_not_see_its_image(self):
        for run in range(21):
            for name, slab in (("Al_a", self.al), ("Ni_a", self.ni)):
                iso = g.isolated(g.rattled(slab, g.seed_for(run, name)))
                _, _, shift = neighbor_list("ijS", iso, 7.0)
                self.assertFalse((shift[:, 2] != 0).any())

    def test_seeds_and_rattle_size(self):
        seeds = [g.seed_for(r, n) for r in range(1, 21) for n in g.FRAGMENT_ID]
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(s > 0 for s in seeds))
        self.assertEqual({g.seed_for(0, n) for n in g.FRAGMENT_ID}, {0})
        dev = np.concatenate([(g.rattled(self.al, g.seed_for(r, "Al_a")).positions - self.al.positions).ravel()
                              for r in range(1, 21)])
        self.assertLess(abs(dev.std() - g.SIGMA), 0.05 * g.SIGMA)

    def test_neon_probe_cell(self):
        slab = g.rattled(self.al, g.seed_for(1, "Al_a"))
        for d in neon_probe.DISTANCES:
            base, ne, both = neon_probe.probe_cell(slab, d)
            self.assertEqual(len(both), 129)
            self.assertAlmostEqual(both.cell[2, 2], self.al.info["thickness"] + 2 * d, places=9)
            self.assertAlmostEqual(ne.positions[0, 2] - slab.positions[:, 2].max(), d, places=9)
            np.testing.assert_array_equal(ne.cell.array, base.cell.array)


class Models(unittest.TestCase):
    def test_energy_recomputes(self):
        calc = CountingCalculator()
        energy = models._energy(calc)
        atoms = g.isolated(g.ideal_slab("Al"))
        energy(atoms)
        energy(atoms)
        self.assertEqual(calc.calls, 2)
        energy.forces(atoms)
        energy.forces(atoms)
        self.assertEqual(calc.calls, 4)

    def test_find_ignores_bool(self):
        outer, inner = torch.nn.Module(), torch.nn.Module()
        outer.cutoff = True
        inner.cutoff = 6.0
        outer.add_module("inner", inner)
        self.assertEqual(models._find(outer, "cutoff"), 6.0)


class Resume(unittest.TestCase):
    def run_main(self, module, tmp, *args):
        argv = [module.__name__, "--family", "mace", "--model", "stub", "--runs", "1", *args]
        with mock.patch.object(models, "load", stub_load), mock.patch.object(module, "ROOT", Path(tmp)), \
                mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            module.main()

    def check(self, module, prefix, per_run):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(module, tmp)
            out = Path(tmp) / "results" / f"{prefix}_stub_notask.csv"
            full = pd.read_csv(out)
            self.assertEqual(len(full), 2 * per_run)
            full[full.run == 0].to_csv(out, index=False)
            full[full.run == 1].head(per_run - 1).to_csv(out, mode="a", header=False, index=False)
            self.run_main(module, tmp)
            again = pd.read_csv(out)
            self.assertEqual(again.run.value_counts().to_dict(), {0: per_run, 1: per_run})
            pd.testing.assert_frame_equal(again.sort_values(list(again.columns)).reset_index(drop=True),
                                          full.sort_values(list(full.columns)).reset_index(drop=True))
            return again

    def test_audit_resume(self):
        self.check(run_audit, "energies", 18)

    def test_neon_resume_and_noise(self):
        again = self.check(neon_probe, "neon", 2)
        self.assertTrue((again.noise_dforce < 1e-9).all())


class StubBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb):
        self.seen.append((atomic_numbers_full, batch_full, csd_mixed_emb))


def composition(z, batch, table, include_self):
    """The composition mean as fairchem 2.21.0 escn_moe.set_MOLE_coefficients computes it, for one system."""
    return table[z].new_zeros(1, table.shape[1]).index_reduce_(0, batch, table[z], reduce="mean",
                                                               include_self=include_self)


class F1Hook(unittest.TestCase):
    def setUp(self):
        self.backbone = StubBackbone()
        unit = mock.Mock()
        unit.model = torch.nn.Sequential(torch.nn.Identity(), self.backbone)
        self.state = f1_forcing.hook(unit)
        al, ni = g.ideal_slab("Al"), g.ideal_slab("Ni")
        self.union = torch.as_tensor(g.stacked(al, ni, 100).numbers)
        self.fragment = torch.as_tensor(g.isolated(al).numbers)
        self.csd = torch.zeros(1, 4)

    def test_substitution(self):
        batch = torch.zeros_like(self.fragment)
        self.backbone.set_MOLE_coefficients(self.fragment, batch, self.csd)
        z, b, c = self.backbone.seen[-1]
        self.assertIs(z, self.fragment)
        self.assertIs(b, batch)
        self.assertEqual(self.state["calls"], 0)
        self.state["z"] = self.union.numpy()
        self.backbone.set_MOLE_coefficients(self.fragment, batch, self.csd)
        z, b, c = self.backbone.seen[-1]
        self.assertTrue(torch.equal(z, self.union))
        self.assertEqual(z.dtype, self.fragment.dtype)
        self.assertTrue(torch.equal(b, torch.zeros_like(self.union)))
        self.assertIs(c, self.csd)
        self.assertEqual(self.state["calls"], 1)

    def test_hook_input_gives_the_union_composition(self):
        self.state["z"] = self.union.numpy()
        self.backbone.set_MOLE_coefficients(self.fragment, torch.zeros_like(self.fragment), self.csd)
        z, b, _ = self.backbone.seen[-1]
        table = torch.randn(100, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        for include_self in (True, False):
            self.assertTrue(torch.equal(composition(z, b, table, include_self),
                                        composition(self.union, torch.zeros_like(self.union), table, include_self)))
        al_al = torch.cat([self.fragment, self.fragment])     # same composition, twice the atoms
        self.assertFalse(torch.allclose(composition(al_al, torch.zeros_like(al_al), table, True),
                                        composition(self.fragment, torch.zeros_like(self.fragment), table, True)))


if __name__ == "__main__":
    unittest.main()
