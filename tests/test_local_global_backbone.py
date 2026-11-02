"""Local+Global integration: composition, batch isolation, gradients and checkpoints."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from ase.build import molecule

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from test_local_backbone import DEVICE, SMALL, build, generic, run  # noqa: E402
from test_element_router import batch_of, coefficients  # noqa: E402
from test_training_config import compose  # noqa: E402


def responsive_model(*overrides):
    model = build("local_global", 0, *overrides)
    # Keep the test router responsive rather than saturated at random initialization.
    with torch.no_grad():
        for p in model.backbone.routing_mlp.parameters():
            p.mul_(0.2)
    return model


def capture_input(model, *structures):
    captured = []
    handle = model.backbone.routing_mlp.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    try:
        model(batch_of(*structures))
    finally:
        handle.remove()
    return captured[-1]


class LocalGlobalBackboneTests(unittest.TestCase):
    def test_config_and_first_layer_dimensions(self):
        from omegaconf import OmegaConf

        local = OmegaConf.to_container(compose("local6").runner.train_eval_unit.model, resolve=True)
        joined = OmegaConf.to_container(compose("local_global").runner.train_eval_unit.model, resolve=True)
        self.assertEqual(joined["backbone"].pop("model"),
                         "training.local_global_moe.eSCNMDLocalGlobalMoeBackbone")
        local["backbone"].pop("model")
        self.assertEqual(joined, local)
        model = responsive_model()
        self.assertEqual(model.backbone.routing_mlp[0].in_features, 3 * model.backbone.sphere_channels)
        self.assertEqual(model.backbone.routing_radius, 6.0)
        self.assertEqual(model.backbone.cutoff, 6.0)

    def test_compositions_context_and_receiver_mapping_match_reference(self):
        model = responsive_model()
        a, b = generic(molecule("H2O")), generic(molecule("CH4"))
        data = batch_of(a, b)
        features = capture_input(model, a, b)
        backbone = model.backbone
        emb = backbone.composition_embedding(data.atomic_numbers).detach()
        d = emb.shape[-1]
        self.assertEqual(tuple(features.shape), (len(a) + len(b), 3 * d))
        # Independent dense reference, including each central atom with weight one.
        distance = torch.cdist(data.pos, data.pos)
        weights = (1 - distance.square() / 6.0**2).clamp_min(0).pow(3)
        weights *= (data.batch[:, None] == data.batch[None, :])
        expected_local = (weights @ emb) / weights.sum(-1, keepdim=True)
        expected_global = torch.cat((emb[:len(a)].mean(0).expand(len(a), -1),
                                     emb[len(a):].mean(0).expand(len(b), -1)))
        context = backbone.csd_embedding(charge=data.charge, spin=data.spin, dataset=data.dataset).detach()
        torch.testing.assert_close(features[:, :d], expected_local, rtol=0, atol=1e-12)
        torch.testing.assert_close(features[:, d:2*d], expected_global, rtol=0, atol=1e-12)
        torch.testing.assert_close(features[:, 2*d:], context[data.batch], rtol=0, atol=1e-12)
        g = backbone.global_mole_tensors
        self.assertEqual(tuple(g.expert_mixing_coefficients.shape), (len(a) + len(b), 8))
        torch.testing.assert_close(g.expert_mixing_coefficients.sum(-1),
                                   emb.new_full((len(a) + len(b),), 1.04))
        # Capture the actual graph as it is built to verify receiver indexing.
        graphs = []
        original = backbone._generate_graph
        from unittest import mock
        def record_graph(data_dict):
            graph = original(data_dict)
            graphs.append(graph)
            return graph
        with mock.patch.object(backbone, "_generate_graph", side_effect=record_graph):
            model(batch_of(a, b))
        torch.testing.assert_close(g.edge_coefficients,
                                   g.expert_mixing_coefficients[graphs[0]["edge_index"][1]])
        self.assertIsNone(backbone._routing_inputs)
        self.assertIsNone(backbone._routing_graph)

    def test_batch_does_not_mix_global_compositions(self):
        model = responsive_model()
        a, b = generic(molecule("H2O")), generic(molecule("CH4"))
        expected = torch.cat((coefficients(model, a), coefficients(model, b)))
        torch.testing.assert_close(coefficients(model, a, b), expected, rtol=0, atol=1e-12)
        ea, fa = run(model, a)
        eb, fb = run(model, b)
        together = model(batch_of(a, b))
        torch.testing.assert_close(together["omol_energy"]["energy"], torch.cat((ea, eb)), rtol=0, atol=1e-10)
        torch.testing.assert_close(together["omol_forces"]["forces"], torch.cat((fa, fb)), rtol=0, atol=1e-10)

    def test_far_fragment_changes_global_but_not_local_features_and_breaks_additivity(self):
        model = responsive_model()
        a, b = generic(molecule("H2O")), generic(molecule("CH4"))
        separate = capture_input(model, a)
        alpha = coefficients(model, a)
        d = model.backbone.sphere_channels
        deltas = []
        for gap in (40.0, 100.0):
            shifted = b.copy()
            shifted.translate([gap, 0.0, 0.0])
            union = a + shifted
            joined = capture_input(model, union)
            torch.testing.assert_close(joined[:len(a), :d], separate[:, :d], rtol=0, atol=1e-12)
            self.assertGreater(float((joined[:len(a), d:2*d] - separate[:, d:2*d]).abs().max()), 1e-6)
            self.assertGreater(float((coefficients(model, union)[:len(a)] - alpha).abs().max()), 1e-8)
            delta = run(model, union)[0].item() - run(model, a)[0].item() - run(model, b)[0].item()
            self.assertGreater(abs(delta), 1e-8)
            deltas.append(delta)
        self.assertAlmostEqual(deltas[0], deltas[1], delta=1e-10)

    def test_zero_global_weights_recover_local6_energy_and_forces(self):
        local = build("local6", 0, "variant.backbone.checkpoint_chunk_size=5")
        joined = responsive_model("variant.backbone.checkpoint_chunk_size=5")
        state = local.state_dict()
        key = "backbone.routing_mlp.0.weight"
        d = local.backbone.sphere_channels
        weight = state[key]
        state[key] = torch.cat((weight[:, :d], torch.zeros_like(weight[:, :d]), weight[:, d:]), dim=-1)
        joined.load_state_dict(state, strict=True)
        for got, want in zip(run(joined, generic(molecule("CH3OH"))), run(local, generic(molecule("CH3OH")))):
            torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_forces_match_finite_differences(self):
        model = responsive_model()
        atoms = generic(molecule("CH3OH"))
        force = run(model, atoms)[1].detach()
        h = 1e-4
        for atom, axis in ((0, 0), (1, 2), (3, 1)):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += h
            minus.positions[atom, axis] -= h
            numeric = -(run(model, plus)[0].item() - run(model, minus)[0].item()) / (2*h)
            self.assertAlmostEqual(numeric, force[atom, axis].item(), delta=1e-7)

    def test_energy_force_loss_trains_both_composition_inputs(self):
        model = responsive_model("variant.backbone.checkpoint_chunk_size=5").train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        e, f = run(model, generic(molecule("CH3OH")))
        loss = e.square().mean() + f.square().mean()
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for name, p in model.backbone.named_parameters():
            if name.startswith(("routing_mlp.", "composition_embedding.")):
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
                self.assertGreater(p.grad.abs().sum().item(), 0.0, name)
        weight = model.backbone.routing_mlp[0].weight
        d = model.backbone.sphere_channels
        for start in (0, d, 2*d):
            self.assertGreater(weight.grad[:, start:start+d].abs().sum().item(), 0.0)
        before = weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(weight, before))

    def test_charge_spin_task_and_symmetry(self):
        model = responsive_model("+variant.backbone.dataset_mapping.omat=omat")
        atoms = generic(molecule("CH3OH"))
        expected = coefficients(model, atoms)
        for context in ({"charge": 1}, {"spin": 2}, {"task": "omat"}):
            self.assertGreater(float((coefficients(model, atoms, **context) - expected).abs().max()), 1e-8)
        moved = atoms.copy()
        moved.rotate(37.0, (1, 2, 3))
        moved.translate([5, -3, 1])
        torch.testing.assert_close(coefficients(model, moved), expected, rtol=0, atol=1e-12)
        order = list(reversed(range(len(atoms))))
        torch.testing.assert_close(coefficients(model, atoms[order]), expected[order], rtol=0, atol=1e-12)

    def test_inference_checkpoint_rebuilds_new_class(self):
        import hydra
        from omegaconf import OmegaConf
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, MLIPInferenceCheckpoint
        from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

        with tempfile.TemporaryDirectory() as tmp:
            cfg = compose("local_global", *SMALL).runner.train_eval_unit
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            try:
                model = hydra.utils.instantiate(cfg.model).to(DEVICE).eval()
            finally:
                torch.set_default_dtype(previous_dtype)
            ema = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
            path = Path(tmp) / "inference_ckpt.pt"
            torch.save(MLIPInferenceCheckpoint(
                model_config=OmegaConf.to_container(cfg.model, resolve=True),
                model_state_dict=model.state_dict(), ema_state_dict=ema.state_dict(),
                tasks_config=OmegaConf.to_container(cfg.tasks, resolve=True)), path)
            settings = InferenceSettings(base_precision_dtype=torch.float64, use_quaternion_wigner=None,
                                         execution_mode="general", auto_add_default_untrained_tasks=False)
            unit = MLIPPredictUnit(str(path), device=DEVICE, inference_settings=settings)
            loaded = unit.model.module
            self.assertEqual(type(loaded.backbone).__name__, "eSCNMDLocalGlobalMoeBackbone")
            atoms = generic(molecule("CH3OH"))
            unit.predict(batch_of(atoms))
            for got, want in zip(run(loaded, atoms), run(model, atoms)):
                torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
