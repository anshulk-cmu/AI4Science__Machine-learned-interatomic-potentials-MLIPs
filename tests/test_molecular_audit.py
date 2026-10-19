"""Exercise measurement code with known local and deliberately nonlocal potentials."""
import csv
import unittest
import tempfile
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock
import numpy as np
from audit.molecular import fragments, locality, finite_difference, predict


class PairPotential:
    """Smooth short-range toy energy. A global multiplier deliberately breaks locality."""
    def __init__(self, global_scale=False, wrong_force=False):
        self.global_scale, self.wrong_force = global_scale, wrong_force

    def evaluate(self, atoms):
        pos = atoms.positions
        f = np.zeros_like(pos)
        e = 0.0
        scale = float(atoms.numbers.mean()) if self.global_scale else 1.0
        for i in range(len(atoms)):
            for j in range(i):
                d = pos[i] - pos[j]
                q = 1 - d @ d / 9
                if q > 0:
                    e += q**3
                    force = 6 * q**2 * d / 9
                    f[i] += force
                    f[j] -= force
        return scale * e, scale * f

    def __call__(self, atoms):
        return self.evaluate(atoms)[0]

    def forces(self, atoms):
        f = self.evaluate(atoms)[1]
        return -f if self.wrong_force else f


class MolecularAuditTests(unittest.TestCase):
    def test_local_model_preserves_energy_and_both_fragment_forces(self):
        for gap in (8, 40):
            a, b, pair = fragments("H2O", "CH4", gap)
            row = locality(PairPotential(), a, b, pair, 3)
            for key in ("delta_eV", "max_dforce_a_eV_A", "max_dforce_b_eV_A"):
                self.assertAlmostEqual(row[key], 0, places=12)
            self.assertGreaterEqual(row["min_distance_A"], gap)

    def test_global_composition_dependence_is_detected(self):
        row = locality(PairPotential(global_scale=True), *fragments("H2O", "CH4", 20), 3)
        self.assertGreater(abs(row["delta_eV"]), 0.01)
        self.assertGreater(row["max_dforce_a_eV_A"], 0.01)
        self.assertGreater(row["max_dforce_b_eV_A"], 0.01)

    def test_force_finite_differences_and_wrong_sign(self):
        a, _, _ = fragments("H2O", "CH4", 8)
        good = finite_difference(PairPotential(), a, 1e-4)
        bad = finite_difference(PairPotential(wrong_force=True), a, 1e-4)
        self.assertEqual(len(good), 3*len(a))
        self.assertLess(max(r["abs_error_eV_A"] for r in good), 1e-7)
        self.assertGreater(max(r["abs_error_eV_A"] for r in bad), 0.1)

    def test_reject_connected_fragments(self):
        with self.assertRaisesRegex(ValueError, "connected"):
            locality(PairPotential(), *fragments("H2O", "CH4", 1), 6)

    def test_preserve_coordinates_and_nonperiodic_scope(self):
        a, b, pair = fragments("H2O", "CH4", 8)
        pair.positions[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "preserve"):
            locality(PairPotential(), a, b, pair, 3)
        pair = a + b
        pair.pbc = True
        with self.assertRaisesRegex(ValueError, "nonperiodic"):
            locality(PairPotential(), a, b, pair, 3)

    def test_reject_invalid_inputs_and_predictions(self):
        for step in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                fragments("H2O", "CH4", step)
            with self.assertRaises(ValueError):
                finite_difference(PairPotential(), fragments("H2O", "CH4", 8)[0], step)
        model = PairPotential()
        model.forces = lambda atoms: np.full((len(atoms), 3), np.nan)
        with self.assertRaisesRegex(ValueError, "invalid"):
            predict(model, fragments("H2O", "CH4", 8)[0])

    def test_loader_recomputes_identical_atoms(self):
        """The repeat columns need a fresh calculation; plain ASE returns its cached result."""
        from ase.calculators.calculator import Calculator, all_changes
        from audit.models import _energy

        class Counter(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                self.calls = getattr(self, "calls", 0) + 1
                self.results = dict(energy=float(self.calls), forces=np.zeros((len(atoms), 3)))

        water = fragments("H2O", "CH4", 8)[0]
        water.calc = Counter()
        self.assertEqual(water.get_potential_energy(), water.get_potential_energy())
        energy = _energy(Counter(), {"charge": 0, "spin": 1})
        self.assertNotEqual(energy(water), energy(water))

    def test_cli_records_outputs_and_failure(self):
        from audit import molecular
        class ToyBackbone:
            pass
        class FakeModel:                     # like AveragedModel around HydraModel, which holds the backbone
            def modules(self):
                return [self, types.SimpleNamespace(backbone=ToyBackbone())]
        model = PairPotential()
        model.unit = types.SimpleNamespace(model=FakeModel())
        loader = types.ModuleType("audit.models")
        loader.load = lambda *args: (model, 3.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, config = root / "model.pt", root / "config.yaml"
            checkpoint.write_bytes(b"test checkpoint")
            config.write_text("variant: test-only\n", encoding="utf-8")
            args = ["molecular", str(checkpoint), "--run-config", str(config),
                    "--label", "toy", "--output", str(root / "output"), "--gaps", "8"]
            with mock.patch.object(sys, "argv", args), mock.patch.dict(sys.modules, {"audit.models": loader}), \
                 mock.patch.object(molecular, "version", return_value="2.21.0"):
                molecular.main()
                metadata = json.loads((root / "output/metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
                self.assertTrue(metadata["backbone_class"].endswith(".ToyBackbone"))
                self.assertTrue((root / "output/locality.csv").exists())
                with (root / "output/finite_difference.csv").open(encoding="utf-8") as stream:
                    fd = list(csv.DictReader(stream))
                # every distinct molecule once: 3 + 5 + 4 + 3 atoms, 3 axes, 2 default steps
                self.assertEqual(sorted({r["molecule"] for r in fd}), ["CH4", "CO2", "H2O", "NH3"])
                self.assertEqual(len(fd), 15 * 3 * 2)
                with self.assertRaises(FileExistsError):
                    molecular.main()
                args[args.index("--output")+1] = str(root / "failed")
                loader.load = mock.Mock(side_effect=RuntimeError("test load failure"))
                with self.assertRaisesRegex(RuntimeError, "test load failure"):
                    molecular.main()
                self.assertEqual(json.loads((root / "failed/metadata.json").read_text(encoding="utf-8"))["status"], "failed")
                # the Bridges-2 copy of the repository has no .git
                args[args.index("--output")+1] = str(root / "nogit")
                loader.load = lambda *args: (model, 3.0)
                with mock.patch.object(molecular.subprocess, "check_output",
                                       side_effect=subprocess.CalledProcessError(128, "git")):
                    molecular.main()
                metadata = json.loads((root / "nogit/metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "complete")
                self.assertIsNone(metadata["evaluation_commit"])

    def test_plot_colours_by_variant_and_writes_figure(self):
        from audit import plot_molecular
        style = plot_molecular.styles({"dense_s0", "dense_s1", "global_s0"})
        self.assertEqual(style["dense_s0"][0], style["dense_s1"][0])
        self.assertNotEqual(style["dense_s0"][1], style["dense_s1"][1])
        self.assertNotEqual(style["dense_s0"][0], style["global_s0"][0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "locality.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["label", "pair", "gap_A", "delta_eV", "max_dforce_a_eV_A", "max_dforce_b_eV_A"])
                for label, delta in (("dense_s0", 0.0), ("global_s0", 0.03)):
                    for gap in (8, 40):
                        writer.writerow([label, "NH3+CO2", gap, delta, 1e-15, delta])
            figure = Path(tmp) / "out" / "locality.png"
            with mock.patch.object(sys, "argv", ["plot", str(path), "--output", str(figure)]):
                plot_molecular.main()
            self.assertGreater(figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
