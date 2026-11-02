"""Element routing in the real fairchem backbone, using small float64 CPU models.

Set ROUTER_TEST_DEVICE=cuda to run the same checks on a GPU. No dataset is needed.
"""
import io
import sys
import unittest
from pathlib import Path

import torch
from ase.build import molecule

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from test_local_backbone import DEVICE, build, generic, run  # noqa: E402
from test_training_config import compose  # noqa: E402


def batch_of(*structures, charge=0, spin=1, task="omol"):
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    items = []
    for atoms in structures:
        atoms = atoms.copy()
        atoms.info.update(charge=charge, spin=spin)
        items.append(AtomicData.from_ase(
            atoms, task_name=task, r_edges=False, r_data_keys=["spin", "charge"],
            r_forces=False, r_stress=False, molecule_cell_size=120.0,
            target_dtype=torch.float64,
        ))
    return atomicdata_list_to_batch(items).to(DEVICE)


def coefficients(model, *structures, **context):
    model(batch_of(*structures, **context))
    return model.backbone.global_mole_tensors.expert_mixing_coefficients.detach().clone()


class ElementRouterTests(unittest.TestCase):
    def test_config_only_changes_routing_radius(self):
        from omegaconf import OmegaConf

        local = OmegaConf.to_container(compose("local6").runner.train_eval_unit.model, resolve=True)
        element = OmegaConf.to_container(compose("element").runner.train_eval_unit.model, resolve=True)
        self.assertEqual(element["backbone"]["routing_radius"], 0.0)
        self.assertEqual(element["backbone"]["cutoff"], 6.0)
        element["backbone"]["routing_radius"] = local["backbone"]["routing_radius"]
        self.assertEqual(element, local)

    def test_same_element_has_same_coefficients_and_native_normalization(self):
        model = build("element")
        atoms = generic(molecule("CH3OH"))
        alpha = coefficients(model, atoms)
        self.assertEqual(tuple(alpha.shape), (len(atoms), 8))
        # fairchem 2.21 uses softmax + 0.005, so eight coefficients sum to 1.04.
        torch.testing.assert_close(alpha.sum(-1), torch.full_like(alpha[:, 0], 1.04))
        self.assertTrue(bool(torch.isfinite(alpha).all()))
        for z in set(atoms.numbers):
            indexes = [i for i, number in enumerate(atoms.numbers) if number == z]
            group = alpha[indexes]
            torch.testing.assert_close(group, group[:1].expand_as(group), rtol=0, atol=1e-12)

    def test_geometry_and_neighbor_elements_do_not_change_target_routing(self):
        model = build("element")
        atoms = generic(molecule("CH3OH"))
        expected = coefficients(model, atoms)
        moved = atoms.copy()
        moved.positions[1] += [1.1, -0.3, 0.2]
        torch.testing.assert_close(coefficients(model, moved), expected, rtol=0, atol=1e-12)
        changed = atoms.copy()
        changed.numbers[1] = 7
        actual = coefficients(model, changed)
        keep = [i for i in range(len(atoms)) if i != 1]
        torch.testing.assert_close(actual[keep], expected[keep], rtol=0, atol=1e-12)
        self.assertGreater(float((actual[1] - expected[1]).abs().max()), 1e-8)

    def test_charge_spin_and_task_still_condition_the_router(self):
        # The production config trains only omol. Add a second task only for this test.
        model = build("element", 0, "+variant.backbone.dataset_mapping.omat=omat")
        atoms = generic(molecule("H2O"))
        reference = coefficients(model, atoms)
        for context in ({"charge": 1}, {"spin": 2}, {"task": "omat"}):
            with self.subTest(context=context):
                actual = coefficients(model, atoms, **context)
                self.assertGreater(float((actual - reference).abs().max()), 1e-8)
                torch.testing.assert_close(actual[1], actual[2], rtol=0, atol=1e-12)

    def test_batch_isolation_and_far_fragment_additivity(self):
        model = build("element")
        a, b = generic(molecule("H2O")), generic(molecule("CH4"))
        shifted = b.copy()
        shifted.translate([100.0, 0.0, 0.0])
        union = a + shifted
        separate = torch.cat([coefficients(model, a), coefficients(model, b)])
        torch.testing.assert_close(coefficients(model, a, b), separate, rtol=0, atol=1e-12)
        torch.testing.assert_close(coefficients(model, union), separate, rtol=0, atol=1e-12)
        e_union = model(batch_of(union))["omol_energy"]["energy"].sum()
        e_parts = model(batch_of(a, b))["omol_energy"]["energy"].sum()
        self.assertLess(abs(float((e_union - e_parts).detach())), 1e-10)

    def test_forces_match_energy_finite_differences(self):
        model = build("element")
        atoms = generic(molecule("CH3OH"))
        _, forces = run(model, atoms)
        forces = forces.detach()
        step = 1e-4
        for atom, axis in ((0, 0), (1, 2), (3, 1)):
            with self.subTest(atom=atom, axis=axis):
                plus, minus = atoms.copy(), atoms.copy()
                plus.positions[atom, axis] += step
                minus.positions[atom, axis] -= step
                numeric = -(run(model, plus)[0].item() - run(model, minus)[0].item()) / (2 * step)
                self.assertAlmostEqual(numeric, forces[atom, axis].item(), delta=1e-7)

    def test_energy_force_loss_updates_router_parameters(self):
        model = build("element").train()
        params = {name: p for name, p in model.backbone.named_parameters()
                  if name.startswith(("routing_mlp.", "composition_embedding."))}
        before = {name: p.detach().clone() for name, p in params.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        e, f = run(model, generic(molecule("CH3OH")))
        loss = e.square().mean() + f.square().mean()
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for name, p in params.items():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        optimizer.step()
        for prefix in ("routing_mlp.", "composition_embedding."):
            self.assertTrue(any(not torch.equal(p, before[name]) for name, p in params.items()
                                if name.startswith(prefix)), prefix)

    def test_state_dict_roundtrip_preserves_element_routing(self):
        model = build("element")
        atoms = generic(molecule("CH3OH"))
        expected = run(model, atoms)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = build("element", seed=7)
        restored.load_state_dict(torch.load(buffer, map_location=DEVICE, weights_only=True), strict=True)
        self.assertEqual(restored.backbone.routing_radius, 0.0)
        for got, want in zip(run(restored, atoms), expected):
            torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
