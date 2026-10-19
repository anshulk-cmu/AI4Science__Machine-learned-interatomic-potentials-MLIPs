"""global_frozen: Global with a constant router (alpha = 0.13 per expert), on tiny float64 models and the composed configs.

The router parameters must get no gradient and stay fixed under AdamW with weight decay, the output must equal the
'uniform' router of scripts/router_check.py on the same weights, the inference checkpoint must rebuild the same model,
and the three existing variants must compose exactly as on main.
"""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "tests", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_local_backbone import DEVICE, SMALL, build, fragments, generic, run  # noqa: E402
from test_training_config import compose, count  # noqa: E402
import check_frozen_router  # noqa: E402
import router_check as rc  # noqa: E402

UNIFORM = np.full(rc.K, 1.0 / rc.K + 0.005)
ROUTER = ("routing_mlp.", "composition_embedding.")
# sha256 of the composed config without the job sections (machine paths, timestamps), measured on main at 9817f01
MAIN_CONFIG_SHA256 = {"dense": "cb391fef53c065382cda065e68d38ca7fdd6cb2fd2566571bab1924e1144e7f4",
                      "global": "2ba3e5fc3046ca795e0c7253128fbd165ae0f6c315e003ece3e1bb18505d071a",
                      "local6": "ddccb3156477c4184db3c61f4b41bce6b2256c529c14cd714402a7100b97eeb8"}


def router(model):
    return {n: p for n, p in model.backbone.named_parameters() if n.startswith(ROUTER)}


def config_sha256(variant):
    from omegaconf import OmegaConf
    with mock.patch.dict(os.environ, {"OMOL_DATA_ROOT": "/data/omol", "WANDB_ENTITY": "anshulk-cmu"}):
        cfg = compose(variant)
    del cfg.job
    del cfg.runner.train_eval_unit.job_config
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=False).encode()).hexdigest()


def batch_of(*structures):
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    items = []
    for atoms in structures:
        atoms = atoms.copy()
        atoms.info.update(charge=0, spin=1)
        items.append(AtomicData.from_ase(atoms, task_name="omol", r_edges=False, r_data_keys=["spin", "charge"],
                                         r_forces=False, r_stress=False, molecule_cell_size=120.0,
                                         target_dtype=torch.float64))
    return atomicdata_list_to_batch(items).to(DEVICE)


class FrozenRouterTests(unittest.TestCase):
    def test_existing_variants_compose_as_on_main(self):
        import hydra
        for variant, expected in MAIN_CONFIG_SHA256.items():
            self.assertEqual(config_sha256(variant), expected, variant)
        sizes = {v: sum(p.numel() for p in hydra.utils.instantiate(compose(v).runner.train_eval_unit.model).parameters())
                 for v in ("dense", "global", "local6", "global_frozen")}
        self.assertEqual(sizes, {"dense": 5_981_057, "global": 37_652_265, "local6": 37_652_265, "global_frozen": 37_652_265})

    def test_frozen_variant_is_global_with_a_constant_router(self):
        import hydra
        from omegaconf import OmegaConf
        cfg = {v: compose(v).runner.train_eval_unit.model for v in ("global", "global_frozen")}
        g, f = (OmegaConf.to_container(cfg[v], resolve=True) for v in ("global", "global_frozen"))
        self.assertEqual(f["backbone"].pop("model"), "training.frozen_moe.eSCNMDFrozenMoeBackbone")
        self.assertEqual(f["backbone"].pop("fixed_alpha"), 0.13)
        g["backbone"].pop("model")
        self.assertEqual(f, g)
        models = {v: hydra.utils.instantiate(c) for v, c in cfg.items()}
        self.assertEqual(count(models["global_frozen"]), count(models["global"]))
        frozen = {v: {n for n, p in m.named_parameters() if not p.requires_grad} for v, m in models.items()}
        new = frozen["global_frozen"] - frozen["global"]
        self.assertEqual(new, {"backbone." + n for n in router(models["global_frozen"])})
        self.assertEqual(sum(p.numel() for n, p in models["global_frozen"].named_parameters() if n in new), 17_320)
        # same seed -> the same initial weights as Global
        small = {v: build(v).state_dict() for v in ("global", "global_frozen")}
        self.assertEqual(list(small["global"]), list(small["global_frozen"]))
        for k, v in small["global"].items():
            self.assertTrue(torch.equal(v, small["global_frozen"][k]), k)

    def test_job_file_carries_the_pilot_recipe(self):
        text = (ROOT / "slurm" / "frozen_global.job").read_text(encoding="utf-8")
        for line in ("#SBATCH -A che260063p", "#SBATCH -p GPU-shared", "#SBATCH --gpus=v100-32:1", "#SBATCH -t 48:00:00",
                     "#SBATCH --mail-type=ALL,TIME_LIMIT_80", "#SBATCH --mail-user=anshulk@andrew.cmu.edu",
                     "#SBATCH --output=/ocean/", "PILOT_STEPS:-50000", "PILOT_SEED:-0", "PILOT_EVAL_EVERY:-5000",
                     "PILOT_CKPT_EVERY:-5000", "PILOT_RUN_PREFIX:-frozen", "shared/repo_frozen/AI4Science-MLIPs"):
            self.assertIn(line, text)
        self.assertNotIn("--array", text)
        self.assertTrue(text.rstrip().endswith("pilot.sh\" global_frozen"))
        pilot = (ROOT / "slurm" / "pilot.sh").read_text(encoding="utf-8")
        self.assertIn("dense|global|global_frozen) MIN_MIB=31000", pilot)
        self.assertLess(pilot.index("stage router_frozen"), pilot.index("stage size_consistency"))

    def test_val_curve_parser_labels_each_validation_with_its_step(self):
        import compare_val_curves
        log = Path(tempfile.mkdtemp()) / "train.log"
        lines = ["{'train/loss': 0.5, 'train/step': 4999, 'train/lr': 1e-4}",
                 "  val/loss: 0.1863", "  val/omol.val,omol_energy,per_atom_mae: 0.0503", "  val/omol.val,omol_forces,mae: 0.1099",
                 "{'train/loss': 0.4, 'train/step': 9999}",
                 "  val/omol.val,omol_energy,per_atom_mae: 0.0305", "  val/omol.val,omol_forces,mae: 0.0745",
                 "  val/omol.val,omol_energy,per_atom_mae: 0.0300", "  val/omol.val,omol_forces,mae: 0.0740"]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(compare_val_curves.curve(log), {5000: {"energy": 0.0503, "force": 0.1099},
                                                         10000: {"energy": 0.0300, "force": 0.0740}})

    def test_router_parameters_are_frozen_flags(self):
        # requires_grad False is what keeps DDP from waiting for gradients that never come
        params = router(build("global_frozen"))
        self.assertEqual(len(params), 7)
        self.assertFalse(any(p.requires_grad for p in params.values()))

    def test_router_gets_no_gradient_and_stays_fixed_over_optimizer_steps(self):
        from ase.build import molecule
        from fairchem.core.models.uma.nn.mole import MOLE

        model = build("global_frozen").train()        # training mode keeps the graph of the forces for a second derivative
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
        moles = [m for m in model.modules() if isinstance(m, MOLE)]
        for name in ("CH3OH", "H2O", "CH4", "CH3OH"):
            e, f = run(model, generic(molecule(name)))
            optimizer.zero_grad(set_to_none=True)
            (e.abs().sum() + f.square().sum()).backward()
            self.assertTrue(all(p.grad is None for p in router(model).values()))
            for m in moles:                           # alpha is the same for every expert, so is the gradient
                g = m.weights.grad
                torch.testing.assert_close(g, g[:1].expand_as(g), rtol=1e-12, atol=1e-15)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            coefficients = model.backbone.global_mole_tensors.expert_mixing_coefficients
            self.assertTrue(torch.equal(coefficients, torch.full_like(coefficients, 0.13)))
        for n, p in model.backbone.named_parameters():
            if n.startswith(ROUTER):
                self.assertTrue(torch.equal(p, before["backbone." + n]), n)
        moved = [n for n, p in model.named_parameters() if p.requires_grad and not torch.equal(p, before[n])]
        for mn, m in model.named_modules():
            if isinstance(m, MOLE):
                self.assertIn(f"{mn}.weights", moved)

    def test_output_equals_the_uniform_router_of_router_check(self):
        a, b, union = fragments()
        glob, frozen = build("global"), build("global_frozen")
        with rc.fixed_alpha(glob.backbone, UNIFORM):
            e1, f1 = run(glob, generic(union))
        e2, f2 = run(frozen, generic(union))
        torch.testing.assert_close(e2, e1, rtol=0, atol=1e-12)
        torch.testing.assert_close(f2, f1, rtol=0, atol=1e-12)
        self.assertGreater(abs(run(glob, generic(union))[0].item() - e1.item()), 1e-8)   # the trained router differs
        with torch.no_grad():
            routed = rc.route(frozen.backbone, batch_of(a, union, b))
        torch.testing.assert_close(routed, torch.full((3, rc.K), 0.13, dtype=routed.dtype, device=routed.device),
                                   rtol=0, atol=0)

    def test_inference_checkpoint_rebuilds_the_frozen_model(self):
        import hydra
        from omegaconf import OmegaConf
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, MLIPInferenceCheckpoint
        from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

        tmp = Path(tempfile.mkdtemp())
        cfg = compose("global_frozen", f"job.run_dir={tmp}", *SMALL)
        unit_cfg = cfg.runner.train_eval_unit
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)        # as the predict unit builds it, so the grid buffers match
        try:
            model = hydra.utils.instantiate(unit_cfg.model)
        finally:
            torch.set_default_dtype(torch.float32)

        def save(path):
            ema = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
            torch.save(MLIPInferenceCheckpoint(model_config=OmegaConf.to_container(unit_cfg.model, resolve=True),
                                               model_state_dict=model.state_dict(), ema_state_dict=ema.state_dict(),
                                               tasks_config=OmegaConf.to_container(unit_cfg.tasks, resolve=True)), path)

        save(tmp / "step_0.pt")
        settings = InferenceSettings(base_precision_dtype=torch.float64, use_quaternion_wigner=None,
                                     execution_mode="general", auto_add_default_untrained_tasks=False)
        unit = MLIPPredictUnit(str(tmp / "step_0.pt"), device=DEVICE, inference_settings=settings)
        loaded = unit.model.module
        self.assertEqual(type(loaded.backbone).__name__, "eSCNMDFrozenMoeBackbone")
        self.assertEqual(loaded.backbone.fixed_alpha, 0.13)
        self.assertFalse(any(p.requires_grad for p in router(loaded).values()))
        _, _, union = fragments()
        record = []
        with rc.fixed_alpha(loaded.backbone, None, record):
            unit.predict(batch_of(generic(union)))
        self.assertTrue(torch.equal(record[0], torch.full_like(record[0], 0.13)))
        model.to(DEVICE).eval()
        expected = model(batch_of(generic(union)))["omol_energy"]["energy"]
        got = loaded(batch_of(generic(union)))["omol_energy"]["energy"]
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)

        # check_frozen_router.py: PASS after training the other weights, FAIL once a router weight moves
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.add_(0.01)
        save(tmp / "final.pt")
        self.assertTrue(all(ok for _, ok in check_frozen_router.checks(tmp / "step_0.pt", tmp / "final.pt")))
        with torch.no_grad():
            model.backbone.routing_mlp[0].bias.add_(1e-6)
        save(tmp / "broken.pt")
        failed = [text for text, ok in check_frozen_router.checks(tmp / "step_0.pt", tmp / "broken.pt") if not ok]
        self.assertEqual(len(failed), 2, failed)       # the model and the EMA state dicts


if __name__ == "__main__":
    unittest.main()
