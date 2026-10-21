"""dense_scaled: Dense whose SO(2) linear weights start and train like W_eff = 0.13 * sum_k W_k of global_frozen.

Tiny float64 models built from the real configs. The scaled weights must start at 0.13 * sqrt(8) times U(-k, k),
every other parameter must equal Dense for the same seed, the scaled AdamW group must have 1.04 times the learning
rate, the model must stay a Dense function with Dense's parameter count, its inference checkpoint must load, it must
follow the frozen model for a few AdamW steps, and the four existing variants must compose exactly as on main.
"""

import functools
import hashlib
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "tests", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_local_backbone import DEVICE, SMALL, build, fragments, generic, run  # noqa: E402
from test_training_config import compose  # noqa: E402
from test_frozen_router import batch_of  # noqa: E402
from training.dense_scaled import lr_scaled  # noqa: E402

ALPHA, EXPERTS = 0.13, 8
INIT_SCALE, LR_SCALE = ALPHA * math.sqrt(EXPERTS), EXPERTS * ALPHA
# sha256 of the composed config without the job sections, measured on main at 4cbd05f
MAIN_CONFIG_SHA256 = {"dense": "cb391fef53c065382cda065e68d38ca7fdd6cb2fd2566571bab1924e1144e7f4",
                      "global": "2ba3e5fc3046ca795e0c7253128fbd165ae0f6c315e003ece3e1bb18505d071a",
                      "local6": "ddccb3156477c4184db3c61f4b41bce6b2256c529c14cd714402a7100b97eeb8",
                      "global_frozen": "294b9c0c3c6d098aace94da65dc005d3c700146939b0eb2f5e3ea478a4ba8426"}
MOLECULES = ("CH3OH", "H2O", "CH4", "CH3OH", "H2O")


def config_sha256(variant):
    from omegaconf import OmegaConf
    with mock.patch.dict(os.environ, {"OMOL_DATA_ROOT": "/data/omol", "WANDB_ENTITY": "anshulk-cmu"}):
        cfg = compose(variant)
    del cfg.job
    del cfg.runner.train_eval_unit.job_config
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=False).encode()).hexdigest()


def optimizer_fn(**changes):
    import hydra
    fn = hydra.utils.instantiate(compose("dense").runner.train_eval_unit.optimizer_fn)
    return functools.partial(fn.func, **{**fn.keywords, **changes})


def mole_weight_names(model):
    """The Dense names of the weights that MoLE replaces in Global: '<layer>.weights' becomes '<layer>.weight'."""
    from fairchem.core.models.uma.nn.mole import MOLE
    return {f"{n}.weight" for n, m in model.named_modules() if isinstance(m, MOLE)}


def loss_of(model, name):
    from ase.build import molecule
    e, f = run(model, generic(molecule(name)))
    return e.abs().sum() + f.square().sum()


def copy_frozen_into_scaled(frozen, scaled):
    """W = 0.13 * sum_k W_k for the MoLE layers, every other shared tensor copied: the same function, two parameterisations."""
    weights = dict(frozen.named_parameters())
    with torch.no_grad():
        for n, p in scaled.named_parameters():
            p.copy_(ALPHA * weights[n + "s"].sum(0) if hasattr(p, "lr_scale") else weights[n])


def trajectory_gap(steps=len(MOLECULES), clip=10.0, dense_eps=False, decay_match=True, lr=2e-4, wd=1e-3):
    """Run the same AdamW steps on global_frozen and on dense_scaled from the same function.

    Returns the largest absolute differences after the steps: of the scaled weights against 0.13 * sum_k W_k, of the
    other parameters, and of energy and forces on a held-out molecule. dense_eps gives the scaled group Dense's eps
    instead of eps / 0.13 (the eps that each expert sees in global_frozen); decay_match False keeps weight decay
    undivided by 1.04.
    """
    from ase.build import molecule
    frozen, scaled = build("global_frozen").train(), build("dense_scaled").train()
    copy_frozen_into_scaled(frozen, scaled)
    opt_f = torch.optim.AdamW(frozen.parameters(), lr=lr, weight_decay=wd)
    opt_s = lr_scaled(functools.partial(torch.optim.AdamW, lr=lr, weight_decay=wd))(params=scaled.parameters())
    if dense_eps:
        opt_s.param_groups[1]["eps"] = opt_s.defaults["eps"]
    if not decay_match:
        opt_s.param_groups[1]["weight_decay"] = wd
    for name in MOLECULES[:steps]:
        for model, opt in ((frozen, opt_f), (scaled, opt_s)):
            opt.zero_grad(set_to_none=True)
            loss_of(model, name).backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    weights = dict(frozen.named_parameters())
    gap = {"scaled": 0.0, "other": 0.0, "moved": 0.0}
    with torch.no_grad():
        for n, p in scaled.named_parameters():
            if hasattr(p, "lr_scale"):
                gap["scaled"] = max(gap["scaled"], (p - ALPHA * weights[n + "s"].sum(0)).abs().max().item())
            else:
                gap["other"] = max(gap["other"], (p - weights[n]).abs().max().item())
    fresh = build("dense_scaled")
    copy_frozen_into_scaled(build("global_frozen"), fresh)
    gap["moved"] = max((p - q).abs().max().item() for (n, p), q in
                       zip(scaled.named_parameters(), fresh.parameters()) if hasattr(p, "lr_scale"))
    frozen.eval(), scaled.eval()
    held_out = generic(molecule("CH3CH2OH"))
    (e1, f1), (e2, f2) = run(frozen, held_out), run(scaled, held_out)
    gap["energy"], gap["forces"] = (e1 - e2).abs().max().item(), (f1 - f2).abs().max().item()
    return gap


class DenseScaledTests(unittest.TestCase):
    def test_existing_variants_compose_as_on_main(self):
        import hydra
        for variant, expected in MAIN_CONFIG_SHA256.items():
            self.assertEqual(config_sha256(variant), expected, variant)
        sizes = {v: sum(p.numel() for p in hydra.utils.instantiate(compose(v).runner.train_eval_unit.model).parameters())
                 for v in ("dense", "global", "local6", "global_frozen", "dense_scaled")}
        self.assertEqual(sizes, {"dense": 5_981_057, "global": 37_652_265, "local6": 37_652_265,
                                 "global_frozen": 37_652_265, "dense_scaled": 5_981_057})

    def test_variant_is_dense_with_scaled_mole_layers(self):
        import hydra
        from omegaconf import OmegaConf
        cfg = {v: compose(v).runner.train_eval_unit.model for v in ("dense", "dense_scaled", "global")}
        d, s = (OmegaConf.to_container(cfg[v], resolve=True) for v in ("dense", "dense_scaled"))
        self.assertEqual(s["backbone"].pop("model"), "training.dense_scaled.eSCNMDScaledBackbone")
        self.assertEqual((s["backbone"].pop("frozen_alpha"), s["backbone"].pop("frozen_experts")), (ALPHA, EXPERTS))
        d["backbone"].pop("model")
        self.assertEqual(s, d)
        scaled, glob = (hydra.utils.instantiate(cfg[v]) for v in ("dense_scaled", "global"))
        self.assertEqual((scaled.backbone.init_scale, scaled.backbone.lr_scale), (INIT_SCALE, LR_SCALE))
        self.assertAlmostEqual(INIT_SCALE, 0.3677, places=4)
        self.assertAlmostEqual(LR_SCALE, 1.04, places=15)
        names = {"backbone." + n for n in scaled.backbone.scaled_parameters()}
        self.assertEqual(len(names), 24)
        self.assertEqual(names, mole_weight_names(glob))       # exactly the weights MoLE replaces, no bias

    def test_initial_scale_and_other_parameters_bit_identical(self):
        dense, scaled, frozen = build("dense"), build("dense_scaled"), build("global_frozen")
        tagged = {"backbone." + n for n in scaled.backbone.scaled_parameters()}
        frozen_w = dict(frozen.named_parameters())
        self.assertEqual(len(tagged), 8)                       # 2 blocks x 2 SO2 convolutions x (m = 0, m = 1)
        for (n, d), s in zip(dense.named_parameters(), scaled.parameters()):
            if n not in tagged:
                self.assertTrue(torch.equal(d, s), n)
                continue
            dense_factor = 1.0 if ".fc_m0." in n else 1 / math.sqrt(2)
            torch.testing.assert_close(s, d * INIT_SCALE / dense_factor, rtol=1e-6, atol=0)
            # in expectation ||W|| = sqrt(numel / 3) * k * scale for U(-k, k), k = 1 / sqrt(fan_in)
            expected = math.sqrt(s.numel() / 3) / math.sqrt(s.shape[1]) * INIT_SCALE
            w_eff = ALPHA * frozen_w[n + "s"].sum(0)
            for w in (s, w_eff):
                self.assertLess(abs(w.norm().item() / expected - 1), 0.1, n)

    def test_optimizer_group_has_the_lr_multiplier(self):
        from training.schedulers import cosine_with_warmup
        model = build("dense_scaled")
        fn = lr_scaled(optimizer_fn())
        self.assertEqual(fn.keywords, optimizer_fn().keywords)   # fairchem reads weight_decay from the keywords
        opt = fn(params=model.parameters())
        base, group = opt.param_groups
        self.assertEqual({id(p) for p in group["params"]}, {id(p) for p in model.backbone.scaled_parameters().values()})
        self.assertEqual(len(base["params"]) + len(group["params"]), len(list(model.parameters())))
        self.assertEqual((base["lr"], base["weight_decay"]), (2e-4, 1e-3))
        self.assertAlmostEqual(group["lr"] / base["lr"], LR_SCALE, places=12)
        self.assertAlmostEqual(group["lr"] * group["weight_decay"], base["lr"] * base["weight_decay"], places=18)
        self.assertAlmostEqual(group["eps"] * ALPHA, base["eps"], places=20)
        scheduler = cosine_with_warmup(opt, n_iters_per_epoch=1, steps=100)
        for _ in range(30):
            opt.step(), scheduler.step()
            self.assertAlmostEqual(opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"], LR_SCALE, places=12)
        for variant in ("dense", "global", "local6", "global_frozen"):    # untagged: one group, as before
            m = build(variant)
            opt = lr_scaled(optimizer_fn())(params=m.parameters())
            self.assertEqual(len(opt.param_groups), 1, variant)
            self.assertEqual([id(p) for p in opt.param_groups[0]["params"]], [id(p) for p in m.parameters()])

    def test_training_unit_wraps_the_optimizer(self):
        from training import guarded_unit as gu
        seen = {}
        with mock.patch.object(gu.MLIPTrainEvalUnit, "__init__", lambda self, **kw: seen.update(kw)):
            gu.GuardedTrainEvalUnit(optimizer_fn=optimizer_fn(), model=None)
        self.assertEqual(seen["optimizer_fn"].func.__name__, "_grouped")

    def test_forward_is_a_dense_output(self):
        dense, scaled = build("dense"), build("dense_scaled")
        _, _, union = fragments()
        e, f = run(scaled, generic(union))
        self.assertTrue(torch.isfinite(e).all() and torch.isfinite(f).all())
        self.assertEqual((e.shape, f.shape), (torch.Size([1]), torch.Size([len(union), 3])))
        dense.load_state_dict(scaled.state_dict(), strict=True)            # the same tensors fit plain Dense
        e2, f2 = run(dense, generic(union))
        torch.testing.assert_close((e2, f2), (e, f), rtol=0, atol=1e-12)

    def test_inference_checkpoint_loads(self):
        import hydra
        from omegaconf import OmegaConf
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, MLIPInferenceCheckpoint
        from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

        tmp = Path(tempfile.mkdtemp())
        unit_cfg = compose("dense_scaled", f"job.run_dir={tmp}", *SMALL).runner.train_eval_unit
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)
        try:
            model = hydra.utils.instantiate(unit_cfg.model)
        finally:
            torch.set_default_dtype(torch.float32)
        ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        torch.save(MLIPInferenceCheckpoint(model_config=OmegaConf.to_container(unit_cfg.model, resolve=True),
                                           model_state_dict=model.state_dict(), ema_state_dict=ema.state_dict(),
                                           tasks_config=OmegaConf.to_container(unit_cfg.tasks, resolve=True)),
                   tmp / "final.pt")
        settings = InferenceSettings(base_precision_dtype=torch.float64, use_quaternion_wigner=None,
                                     execution_mode="general", auto_add_default_untrained_tasks=False)
        loaded = MLIPPredictUnit(str(tmp / "final.pt"), device=DEVICE, inference_settings=settings).model.module
        self.assertEqual(type(loaded.backbone).__name__, "eSCNMDScaledBackbone")
        self.assertAlmostEqual(loaded.backbone.lr_scale, LR_SCALE, places=15)
        _, _, union = fragments()
        model.to(DEVICE).eval()
        expected = model(batch_of(generic(union)))["omol_energy"]["energy"]
        got = loaded(batch_of(generic(union)))["omol_energy"]["energy"]
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)

    def test_job_file_carries_the_pilot_recipe(self):
        text = (ROOT / "slurm" / "dense_scaled.job").read_text(encoding="utf-8")
        for line in ("#SBATCH -A che260063p", "#SBATCH -p GPU-shared", "#SBATCH --gpus=v100-32:1", "#SBATCH -t 48:00:00",
                     "#SBATCH --mail-type=ALL,TIME_LIMIT_80", "#SBATCH --mail-user=anshulk@andrew.cmu.edu",
                     "#SBATCH --output=/ocean/", "PILOT_STEPS:-50000", "PILOT_SEED:-0", "PILOT_EVAL_EVERY:-5000",
                     "PILOT_CKPT_EVERY:-5000", "PILOT_RUN_PREFIX:-dense", "PILOT_RUN_TAG:-scaled",
                     "shared/repo_dscaled/AI4Science-MLIPs"):
            self.assertIn(line, text)
        self.assertNotIn("--array", text)
        self.assertTrue(text.rstrip().endswith("pilot.sh\" dense_scaled"))
        self.assertIn("dense_scaled|dense|global|global_frozen) MIN_MIB=31000",
                      (ROOT / "slurm" / "pilot.sh").read_text(encoding="utf-8"))

    def test_follows_the_frozen_model_over_adamw_steps(self):
        # the pilot recipe (lr 2e-4, weight decay 1e-3, clipping at 10): equal up to rounding
        gap = trajectory_gap()
        self.assertGreater(gap["moved"], 1e-4)                   # the steps really moved the weights
        for key in ("scaled", "other", "energy", "forces"):
            self.assertLess(gap[key], 1e-12, (key, gap))
        # each of the two matched details matters: Dense's eps or undivided weight decay leave a gap
        self.assertGreater(trajectory_gap(dense_eps=True)["scaled"], 1e-6)
        self.assertGreater(trajectory_gap(decay_match=False)["scaled"], 1e-10)


if __name__ == "__main__":
    unittest.main()
