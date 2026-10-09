"""Training config contract on CPU: the composed Hydra config, model sizes, schedule, and run.py guards.

Composes configs/train.yaml through fairchem's own CLI path (get_hydra_config_from_yaml), instantiates
the model and the tasks, and never builds a dataset or a dataloader.
"""

import argparse
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]

try:
    import fcntl  # noqa: F401
except ImportError:
    # Native Windows only: clusterscope (imported by fairchem.core.launchers.api) needs POSIX fcntl.
    _stub = types.ModuleType("fcntl")
    for _name in ("LOCK_EX", "LOCK_SH", "LOCK_UN", "LOCK_NB"):
        setattr(_stub, _name, 0)
    _stub.flock = lambda *args, **kwargs: None
    sys.modules["fcntl"] = _stub

PARAMS = {
    "dense": {"total": 6_267_777, "active": 6_267_777},
    "global": {"total": 37_938_985, "active": 6_285_097, "bank": 36_179_968,
               "mole_layers": 24, "router": 17_320},
}


def compose(variant, *overrides):
    from hydra.core.global_hydra import GlobalHydra
    from fairchem.core._cli import get_hydra_config_from_yaml

    GlobalHydra.instance().clear()
    try:
        return get_hydra_config_from_yaml(str(ROOT / "configs" / "train.yaml"),
                                          [f"variant={variant}", *overrides])
    finally:
        GlobalHydra.instance().clear()


def count(model):
    from fairchem.core.models.uma.nn.mole import MOLE

    total = sum(p.numel() for p in model.parameters())
    moles = [m for m in model.modules() if isinstance(m, MOLE)]
    bank = sum(p.numel() for m in moles for p in m.parameters())
    active = total
    for m in moles:
        # Only the [E, O, I] expert weights are per expert; the MOLE bias is shared.
        experts = m.weights.shape[0]
        active -= m.weights.numel() * (experts - 1) // experts
    router = sum(p.numel() for name, p in model.named_parameters()
                 if name.startswith(("backbone.routing_mlp", "backbone.composition_embedding")))
    return {"total": total, "active": active, "bank": bank, "mole_layers": len(moles), "router": router}


class ComposedModelTests(unittest.TestCase):
    def test_parameter_counts(self):
        import hydra

        for variant, expected in PARAMS.items():
            with self.subTest(variant=variant):
                cfg = compose(variant)
                model = hydra.utils.instantiate(cfg.runner.train_eval_unit.model)
                measured = count(model)
                self.assertEqual({k: measured[k] for k in expected}, expected)


class ComposedConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = compose("dense")
        cls.unit = cls.cfg.runner.train_eval_unit

    def test_energy_refs_and_shared_rmsd(self):
        import hydra

        energy, forces = self.unit.tasks
        refs = energy.element_references.element_references
        self.assertEqual(refs._target_, "torch.DoubleTensor")
        self.assertEqual(len(refs._args_[0]), 100)
        self.assertEqual(refs._args_[0][1], -16.001316695716987)
        self.assertIsNone(forces.element_references)
        tasks = [hydra.utils.instantiate(t) for t in self.unit.tasks]
        buffer = tasks[0].element_references.element_references
        self.assertEqual((buffer.dtype, tuple(buffer.shape)), (torch.float64, (100,)))
        self.assertIsNone(tasks[1].element_references)
        for task in tasks:
            self.assertAlmostEqual(float(task.normalizer.rmsd), 1.433569, places=6)
            self.assertEqual(float(task.normalizer.mean), 0.0)

    def test_wandb_logger_on_by_default_and_switchable(self):
        logger = self.cfg.job.logger
        self.assertTrue(logger._target_.endswith("WandBSingletonLogger.init_wandb"))
        self.assertEqual(logger.project, "mlip-local-routing")
        self.assertIsNone(compose("dense", "logger=none").job.logger)

    def test_ema_and_inference_checkpoint(self):
        self.assertEqual(self.unit.ema_decay, 0.999)
        self.assertIs(self.unit.save_inference_ckpt, True)

    def test_eval_and_checkpoint_intervals(self):
        steps = self.cfg.runner.max_steps
        self.assertEqual(self.cfg.runner.evaluate_every_n_steps, steps)
        self.assertEqual(self.cfg.runner.callbacks[0].checkpoint_every_n_steps, steps)
        cfg = compose("dense", "training.steps=50", "training.eval_every=7", "training.ckpt_every=11")
        self.assertEqual(cfg.runner.max_steps, 50)
        self.assertEqual(cfg.runner.evaluate_every_n_steps, 7)
        self.assertEqual(cfg.runner.callbacks[0].checkpoint_every_n_steps, 11)

    def test_neighbors_and_data_root(self):
        for loader in (self.cfg.runner.train_dataloader, self.cfg.runner.eval_dataloader):
            self.assertEqual(loader.dataset.dataset_configs.omol.a2g_args.max_neigh, 300)
        self.assertEqual(self.cfg.runner.train_dataloader.batch_sampler_fn.max_atoms, 320)
        with mock.patch.dict(os.environ, {"OMOL_DATA_ROOT": "/data/omol"}):
            cfg = compose("dense")
        splits = (cfg.runner.train_dataloader.dataset.dataset_configs.omol.splits.train,
                  cfg.runner.eval_dataloader.dataset.dataset_configs.omol.splits.val)
        self.assertEqual([s.src for s in splits], ["/data/omol/train_4M", "/data/omol/val"])


class SchedulerTests(unittest.TestCase):
    def lrs(self, n_iters_per_epoch, steps=10_000):
        import hydra

        cfg = compose("dense", f"training.steps={steps}")
        factory = hydra.utils.instantiate(cfg.runner.train_eval_unit.cosine_lr_scheduler_fn)
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
        scheduler = factory(n_iters_per_epoch=n_iters_per_epoch, optimizer=optimizer)
        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(steps):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        return lrs

    def test_warmup_in_steps(self):
        lrs = self.lrs(n_iters_per_epoch=660_000)
        warmup = 500  # round(0.05 * 10000)
        self.assertAlmostEqual(lrs[0], 0.2 * 2e-4, delta=1e-12)
        self.assertLess(lrs[warmup - 1], 2e-4)
        self.assertAlmostEqual(lrs[warmup], 2e-4, delta=1e-12)
        self.assertLess(lrs[warmup + 1], 2e-4)
        self.assertAlmostEqual(lrs[-1], 0.01 * 2e-4, delta=1e-12)
        self.assertEqual(lrs, self.lrs(n_iters_per_epoch=1))

    def test_explicit_warmup_and_bounds(self):
        from training.schedulers import cosine_with_warmup

        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = cosine_with_warmup(optimizer, n_iters_per_epoch=3, steps=100, warmup_steps=10)
        self.assertEqual(scheduler.lr_lambdas[0](10), 1.0)
        with self.assertRaises(ValueError):
            cosine_with_warmup(optimizer, n_iters_per_epoch=3, steps=100, warmup_steps=101)


class RunGuardTests(unittest.TestCase):
    def setUp(self):
        from training import run

        self.run = run
        self.cwd = os.getcwd()
        os.chdir(ROOT)
        patches = [mock.patch.object(sys, "argv", ["run", "variant=dense"]),
                   mock.patch.dict(os.environ)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_wrong_version(self):
        with mock.patch.object(self.run, "version", return_value="2.23.0"):
            with self.assertRaisesRegex(RuntimeError, "2.21.0"):
                self.run.main()

    def test_wrong_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(self.run, "version", return_value="2.21.0"):
            os.chdir(tmp)
            try:
                with self.assertRaisesRegex(RuntimeError, "Run from"):
                    self.run.main()
            finally:
                os.chdir(ROOT)

    def test_shadowed_package(self):
        shadow = mock.Mock()
        shadow.locate_file.return_value = str(ROOT.parent / "fairchem" / "src" / "fairchem" / "core" / "__init__.py")
        with mock.patch.object(self.run, "version", return_value="2.21.0"), \
                mock.patch.object(self.run, "distribution", return_value=shadow), \
                mock.patch("fairchem.core._cli.main") as cli:
            with self.assertRaisesRegex(RuntimeError, "not the installed"):
                self.run.main()
            cli.assert_not_called()

    def test_valid_call_reaches_cli(self):
        os.environ.pop("FAIRCHEM_CACHE_DIR", None)
        os.environ["OMOL_CACHE_DIR"] = str(ROOT.parent / "cacheProbe")
        with mock.patch.object(self.run, "version", return_value="2.21.0"), \
                mock.patch("fairchem.core._cli.main") as cli:
            self.run.main()
        cli.assert_called_once_with(argparse.Namespace(config="configs/train.yaml"), ["variant=dense"])
        self.assertEqual(os.environ["FAIRCHEM_CACHE_DIR"], str((ROOT.parent / "cacheProbe" / "fairchem").resolve()))

    def test_wandb_defaults_to_offline_and_respects_the_environment(self):
        for key in ("WANDB_MODE", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
            os.environ.pop(key, None)
        os.environ["OMOL_CACHE_DIR"] = str(ROOT.parent / "cacheProbe")
        with mock.patch.object(self.run, "version", return_value="2.21.0"), mock.patch("fairchem.core._cli.main"):
            self.run.main()
        self.assertEqual(os.environ["WANDB_MODE"], "offline")
        self.assertEqual(os.environ["WANDB_CACHE_DIR"], str((ROOT.parent / "cacheProbe" / "wandb").resolve()))
        os.environ["WANDB_MODE"] = "online"
        with mock.patch.object(self.run, "version", return_value="2.21.0"), mock.patch("fairchem.core._cli.main"):
            self.run.main()
        self.assertEqual(os.environ["WANDB_MODE"], "online")


if __name__ == "__main__":
    unittest.main()
