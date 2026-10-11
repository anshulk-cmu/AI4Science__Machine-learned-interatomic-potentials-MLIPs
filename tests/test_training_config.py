"""Training config contract on CPU: the composed Hydra config, model sizes, schedule, and run.py guards.

Composes configs/train.yaml through fairchem's own CLI path (get_hydra_config_from_yaml), instantiates
the model and the tasks, and never builds a dataset or a dataloader.
"""

import argparse
import hashlib
import json
import os
import re
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
    "dense": {"total": 5_981_057, "active": 5_981_057},
    "global": {"total": 37_652_265, "active": 5_998_377, "bank": 36_179_968,
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
        self.assertEqual(self.cfg.runner.train_dataloader.batch_sampler_fn.max_atoms, 450)
        with mock.patch.dict(os.environ, {"OMOL_DATA_ROOT": "/data/omol"}):
            cfg = compose("dense")
        splits = (cfg.runner.train_dataloader.dataset.dataset_configs.omol.splits.train,
                  cfg.runner.eval_dataloader.dataset.dataset_configs.omol.splits.val)
        self.assertEqual([s.src for s in splits], ["/data/omol/train_4M", "/data/omol/val"])


class PilotSettingsTests(unittest.TestCase):
    def test_basis_and_neighbor_cap_are_pinned_in_every_variant(self):
        for variant in ("dense", "global", "local6"):
            with self.subTest(variant=variant):
                backbone = compose(variant).runner.train_eval_unit.model.backbone
                self.assertEqual(backbone.num_distance_basis, 64)
                self.assertEqual(backbone.max_neighbors, 300)
                cfg = compose(variant, "training.max_neighbors=77")
                graph = cfg.runner.train_dataloader.dataset.dataset_configs.omol.a2g_args.max_neigh
                self.assertEqual(cfg.runner.train_eval_unit.model.backbone.max_neighbors, graph)
                self.assertEqual(graph, 77)

    def test_checkpoints_kept(self):
        self.assertEqual(compose("dense").runner.callbacks[0].max_saved_checkpoints, 3)
        cfg = compose("dense", "training.max_saved_checkpoints=12")
        self.assertEqual(cfg.runner.callbacks[0].max_saved_checkpoints, 12)

    def test_fixed_lists_replace_the_random_draw(self):
        t = "train_dataset.dataset_configs.omol.splits.train"
        v = "val_dataset.dataset_configs.omol.splits.val"
        cfg = compose("dense", f"{t}.sample_n=null", f"{t}.no_shuffle=true", f"{t}.metadata_path=/d/pilot/train.npz",
                      f"{t}.subset_to=[{{metadata_key:index,op:in,rhv:/r/train.txt}}]",
                      f"{v}.sample_n=null", f"{v}.no_shuffle=true", f"{v}.metadata_path=/d/pilot/val.npz",
                      f"{v}.subset_to=[{{metadata_key:index,op:in,rhv:/r/val.txt}}]")
        for loader, split, path in ((cfg.runner.train_dataloader, "train", "train"), (cfg.runner.eval_dataloader, "val", "val")):
            s = loader.dataset.dataset_configs.omol.splits[split]
            self.assertIsNone(s.sample_n)
            self.assertIs(s.no_shuffle, True)
            self.assertEqual(s.metadata_path, f"/d/pilot/{path}.npz")
            self.assertEqual([dict(x) for x in s.subset_to],
                             [{"metadata_key": "index", "op": "in", "rhv": f"/r/{path}.txt"}])
        default = compose("dense").runner.train_dataloader.dataset.dataset_configs.omol.splits.train
        self.assertEqual((default.sample_n, default.no_shuffle, default.metadata_path, list(default.subset_to)),
                         (64, None, None, []))


class PilotFilesTests(unittest.TestCase):
    def test_sample_lists_match_their_hashes_and_are_valid(self):
        info = json.loads((ROOT / "manifests" / "pilot_samples.json").read_text(encoding="utf-8"))["lists"]
        self.assertEqual({k: (v["split"], v["count"]) for k, v in info.items()},
                         {"pilot_train_200k.txt": ("train_4M", 200_000), "pilot_val_5k.txt": ("val", 5_000)})
        for name, entry in info.items():
            data = (ROOT / "manifests" / name).read_bytes()
            self.assertNotIn(b"\r", data)
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            rows = [int(x) for x in data.decode("ascii").split()]
            self.assertEqual(len(rows), entry["count"])
            self.assertEqual(rows, sorted(set(rows)))
            self.assertGreaterEqual(rows[0], 0)
            self.assertLess(rows[-1], entry["rows_in_split"])

    def test_job_files_carry_the_agreed_settings(self):
        expected = {"pilot_dense": ("v100-32", "8:00:00"), "pilot_global": ("v100-32", "8:00:00"),
                    "pilot_local6": ("l40s-48", "14:00:00")}
        for name, (gpu, limit) in expected.items():
            text = (ROOT / "slurm" / f"{name}.job").read_text(encoding="utf-8")
            self.assertIn("#SBATCH -A che260063p", text)
            self.assertIn(f"#SBATCH --gpus={gpu}:1", text)
            self.assertIn(f"#SBATCH -t {limit}", text)
            self.assertIn("#SBATCH --array=0-2", text)
            self.assertIn("export PILOT_STEPS=20000", text)
            self.assertNotIn("PILOT_SEED", text)
            self.assertNotIn("PILOT_TRAIN_SAMPLES", text)
            self.assertIn("--mail-user=anshulk@andrew.cmu.edu", text)
            self.assertTrue(text.rstrip().endswith(f"pilot.sh {name.split('_')[1]}"))

    def test_pilot_script_checks_the_lists_before_training(self):
        text = (ROOT / "slurm" / "pilot.sh").read_text(encoding="utf-8")
        self.assertLess(text.index("stage verify_samples"), text.index("stage train "))
        self.assertIn("SLURM_ARRAY_TASK_ID", text)
        self.assertIn("PILOT_STEPS:-20000", text)
        overrides = (ROOT / "slurm" / "pilot_data_overrides.sh").read_text(encoding="utf-8")
        for key in ("sample_n=null", "no_shuffle=true", "metadata_path", "subset_to"):
            self.assertEqual(len(re.findall(re.escape(key), overrides)), 2, key)


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

    def test_small_gpu_gets_a_warning(self):
        import io
        from contextlib import redirect_stdout

        for gib, warned in ((11.9, True), (31.7, False), (79.6, False)):
            props = mock.Mock(total_memory=int(gib * 2**30))
            with mock.patch("torch.cuda.is_available", return_value=True),                     mock.patch("torch.cuda.get_device_properties", return_value=props):
                out = io.StringIO()
                with redirect_stdout(out):
                    self.run.warn_if_small_gpu()
            self.assertEqual("WARNING" in out.getvalue(), warned, gib)

    def test_allocator_setting_and_memory_cap(self):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["OMOL_GPU_MEMORY_FRACTION"] = "0.3"
        with mock.patch.object(self.run, "version", return_value="2.21.0"), mock.patch("fairchem.core._cli.main"),                 mock.patch("torch.cuda.set_per_process_memory_fraction") as cap:
            self.run.main()
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        cap.assert_called_once_with(0.3)

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
