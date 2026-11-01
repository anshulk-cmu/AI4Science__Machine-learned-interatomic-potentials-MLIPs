"""Real fairchem checkpoint round trips and selection/report separation, on CPU."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

from training import checkpoints as ck
from test_local_backbone import SMALL, generic
from test_training_config import compose


class SelectionTests(unittest.TestCase):
    def test_config_comparison_accepts_saved_omegaconf(self):
        from omegaconf import OmegaConf
        self.assertTrue(ck.same_config({"tasks": OmegaConf.create([{"rmsd": 1.4}])},
                                       {"tasks": [{"rmsd": 1.4}]}))
        self.assertFalse(ck.same_config(OmegaConf.create([1,2]), [1,3]))

    def test_rule_excludes_early_steps_and_breaks_ties_by_step(self):
        rows = [dict(step=s, energy_mae_per_atom_eV=1., force_mae_eV_per_A=f)
                for s, f in [(0, 0.), (20, .2), (30, .1), (40, .1)]]
        self.assertEqual([r["step"] for r in ck.select(rows, 40, 2)], [30, 40])
        with self.assertRaises(ValueError):
            ck.select(rows, 40, 4)
        rows[-1]["force_mae_eV_per_A"] = float("nan")
        with self.assertRaises(ValueError):
            ck.select(rows, 40, 1)

    def test_disjoint_unique_row_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            a.write_text("0\n1\n"); b.write_text("2\n3\n")
            self.assertEqual(ck.split_rows(a,b), ([0,1], [2,3]))
            b.write_text("1\n3\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                ck.split_rows(a,b)
            b.write_text("2\n2\n")
            with self.assertRaises(ValueError):
                ck.read_rows(b)

    def test_metrics_weight_force_components_not_molecules(self):
        class Dataset:
            def get_atoms(self, index):
                atoms = generic(molecule("H2O" if index == 0 else "CH4"))
                atoms.info.update(charge=0, spin=1)
                atoms.calc = SinglePointCalculator(atoms, energy=0., forces=np.zeros((len(atoms),3)))
                return atoms
        class Unit:
            def predict(self, batch):
                n = len(batch.pos)
                return {"energy": torch.tensor([2.*n]), "forces": torch.full((n,3), float(n))}
        metrics = ck.evaluate(Unit(), Dataset(), [0,1])
        self.assertEqual(metrics["energy_mae_per_atom_eV"], 2.)
        self.assertEqual(metrics["force_mae_eV_per_A"], (3*3+5*5)/8)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hydra
        from omegaconf import OmegaConf
        from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.run_dir = cls.root / "run"
        cls.run_dir.mkdir()
        (cls.run_dir / "settings.txt").write_text("variant=local12 seed=0 steps=40\n")
        cfg = compose("local12", *SMALL)
        torch.manual_seed(19)
        model = hydra.utils.instantiate(cfg.runner.train_eval_unit.model).double()
        cls.paths = []
        for i, step in enumerate((20,30,40)):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.001)
            ema = torch.optim.swa_utils.AveragedModel(model)
            raw = {key: value.clone() for key, value in model.state_dict().items()}
            # Raw differs from EMA: averaging the wrong representation must fail the test.
            for key, _ in model.named_parameters():
                raw[key].add_(1.)
            checkpoint = MLIPInferenceCheckpoint(
                OmegaConf.to_container(cfg.runner.train_eval_unit.model, resolve=True), raw, ema.state_dict(),
                OmegaConf.to_container(cfg.runner.train_eval_unit.tasks, resolve=True))
            path = cls.run_dir / f"train/run/checkpoints/step_{step}/inference_ckpt.pt"
            path.parent.mkdir(parents=True)
            torch.save(checkpoint, path)
            cls.paths.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_average_ema_parameters_and_roundtrip(self):
        out = self.root / "averaged.pt"
        result = ck.average_weights(self.paths, out)
        inputs = [ck.load_checkpoint(path) for path in self.paths]
        key = next(key for key in result.ema_state_dict if key.startswith("module.backbone") and key.endswith("weight"))
        expected = sum(c.ema_state_dict[key].double() for c in inputs)/3
        torch.testing.assert_close(result.ema_state_dict[key], expected)
        unit = ck.predictor(out)
        loaded = unit.model.state_dict()
        torch.testing.assert_close(loaded[key], result.ema_state_dict[key])
        with self.assertRaises(FileExistsError):
            ck.average_weights(self.paths, out)

    def test_incompatible_task_config_is_rejected(self):
        checkpoint = ck.load_checkpoint(self.paths[0])
        checkpoint.tasks_config[0]["normalizer"]["rmsd"] = 9.
        path = self.root/"wrong.pt"
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "configurations"):
            ck.average_weights([self.paths[0], path], self.root/"bad.pt")
        self.assertFalse((self.root/"bad.pt").exists())

    def test_different_buffers_and_changed_checkpoint_hash_are_rejected(self):
        checkpoint = ck.load_checkpoint(self.paths[0])
        checkpoint.ema_state_dict["module.test_buffer"] = torch.tensor([1.])
        a = self.root/"buffer-a.pt"
        torch.save(checkpoint, a)
        checkpoint.ema_state_dict["module.test_buffer"] = torch.tensor([2.])
        b = self.root/"buffer-b.pt"
        torch.save(checkpoint, b)
        with self.assertRaisesRegex(ValueError, "buffer differs"):
            ck.average_weights([a,b], self.root/"buffer-mean.pt")
        record = {"run_dir": str(self.root), "selected": [{"path":str(a), "sha256":"wrong"}],
                  "final":{"path":str(b), "sha256":ck.sha256(b)}}
        path = self.root/"changed.json"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "changed"):
            ck.checked_selection(path)

    def test_discovery_rejects_incomplete_length(self):
        self.assertEqual([s for s, _ in ck.discover(self.run_dir, 40)], [20,30,40])
        with self.assertRaisesRegex(ValueError, "last saved step"):
            ck.discover(self.run_dir, 50)

    def test_prediction_average_matches_members_and_energy_gradient(self):
        from audit.models import load_prediction_average
        from audit.molecular import finite_difference, locality, fragments
        energy, cutoff = load_prediction_average(self.paths, device="cpu")
        atoms = generic(molecule("H2O"))
        singles = [load_prediction_average([p], device="cpu")[0] for p in self.paths]
        self.assertAlmostEqual(energy(atoms), np.mean([f(atoms) for f in singles]), places=10)
        np.testing.assert_allclose(energy.forces(atoms), np.mean([f.forces(atoms) for f in singles], axis=0), atol=1e-10)
        errors = finite_difference(energy, atoms, 1e-4)
        self.assertLess(max(r["abs_error_eV_A"] for r in errors), 1e-5)
        row = locality(energy, *fragments("H2O", "CH4", 20), cutoff)
        self.assertLess(abs(row["delta_eV"]), 1e-8)

    def test_select_average_report_end_to_end(self):
        import ase.db
        from scripts.checkpoint_tools import main
        dbpath = self.root / "tiny.db"
        with ase.db.connect(str(dbpath)) as db:
            for name in ("H2O", "CH4", "H2O", "CH4"):
                atoms = generic(molecule(name))
                atoms.calc = SinglePointCalculator(atoms, energy=0., forces=np.zeros((len(atoms),3)))
                db.write(atoms, data={"charge":0, "spin":1})
        selection_rows, report_rows = self.root/"selection.txt", self.root/"report.txt"
        selection_rows.write_text("0\n1\n"); report_rows.write_text("2\n3\n")
        output = self.root / "selected"
        def cli(*args):
            with patch("sys.argv", ["checkpoint_tools.py", *map(str,args)]):
                main()
        cli("select", "--run-dir", self.run_dir, "--total-steps", 40, "--variant", "local12", "--seed", 0,
            "--k", 3, "--selection-rows", selection_rows, "--report-rows", report_rows,
            "--data", dbpath, "--output", output)
        selection = output/"selection.json"
        record = ck.checked_selection(selection)
        self.assertEqual(len(record["selected"]), 3)
        avg = self.root/"cli-average.pt"
        cli("average", "--selection", selection, "--output", avg)
        report = self.root/"report"
        cli("report", "--selection", selection, "--averaged", avg, "--data", dbpath, "--output", report)
        import csv
        with (report/"ensemble_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({r["method"] for r in rows}, {"final", "best", "weight_average", "prediction_average"})
        self.assertTrue(all(r["eval_split"] == "report" and r["n_structures"] == "2" for r in rows))
        report_rows.write_text("0\n3\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            cli("report", "--selection", selection, "--averaged", avg, "--data", dbpath, "--output", self.root/"rejected")


if __name__ == "__main__":
    unittest.main()
