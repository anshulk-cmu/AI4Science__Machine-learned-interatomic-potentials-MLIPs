"""Local routing inside the real fairchem backbone: locality, exactness against Global, forces, symmetry.

Small random-weight models (float64), built from configs/variant/*.yaml with the model dimensions
overridden, run on tiny molecules through fairchem's own AtomicData and batching. No data files.
Set ROUTER_TEST_DEVICE=cuda to run the same suite on the GPU.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("ROUTER_TEST_DEVICE", "cpu")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import fcntl  # noqa: F401
except ImportError:
    _stub = types.ModuleType("fcntl")
    for _name in ("LOCK_EX", "LOCK_SH", "LOCK_UN", "LOCK_NB"):
        setattr(_stub, _name, 0)
    _stub.flock = lambda *args, **kwargs: None
    sys.modules["fcntl"] = _stub

SMALL = ["variant.backbone.otf_graph=true", "variant.backbone.sphere_channels=16",
         "variant.backbone.hidden_channels=16", "variant.backbone.edge_channels=16",
         "variant.backbone.num_layers=2", "variant.backbone.lmax=1", "variant.backbone.mmax=1",
         "variant.backbone.num_distance_basis=16"]


def build(variant, seed=0, *overrides):
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from fairchem.core._cli import get_hydra_config_from_yaml

    GlobalHydra.instance().clear()
    try:
        cfg = get_hydra_config_from_yaml(str(ROOT / "configs" / "train.yaml"),
                                         [f"variant={variant}", f"job.run_dir={tempfile.mkdtemp()}",
                                          *SMALL, *overrides])
    finally:
        GlobalHydra.instance().clear()
    torch.manual_seed(seed)
    model = hydra.utils.instantiate(cfg.runner.train_eval_unit.model).double()
    backbone = model.backbone
    if getattr(backbone, "routing_mlp", None) is not None:
        # a responsive but unsaturated router, so a leak between fragments is visible
        for p in backbone.routing_mlp.parameters():
            torch.nn.init.normal_(p, std=1.0)
        backbone.composition_embedding.weight.data.copy_(
            torch.randn_like(backbone.composition_embedding.weight) * 0.25)
    return model.to(DEVICE).eval()


def run(model, atoms, charge=0, spin=1):
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    atoms = atoms.copy()
    atoms.info.update(charge=charge, spin=spin)
    data = AtomicData.from_ase(atoms, task_name="omol", r_edges=False, r_data_keys=["spin", "charge"],
                               r_forces=False, r_stress=False, molecule_cell_size=120.0,
                               target_dtype=torch.float64)
    batch = atomicdata_list_to_batch([data]).to(DEVICE)
    out = model(batch)
    return out["omol_energy"]["energy"], out["omol_forces"]["forces"]


def energy(model, atoms, **kwargs):
    return run(model, atoms, **kwargs)[0].item()


def generic(atoms):
    """Rotate off the coordinate axes: fairchem builds each edge frame from Euler angles, which are
    singular for an edge exactly along y, and its forces then differ from the energy gradient (Dense too)."""
    atoms = atoms.copy()
    atoms.rotate(23.0, (0.3, 0.8, 0.5))
    atoms.rotate(41.0, (1.0, -0.2, 0.6))
    return atoms


def fragments():
    from ase.build import molecule

    a, b = molecule("H2O"), molecule("CH4")
    b.translate([40.0, 0.0, 0.0])
    return a, molecule("CH4"), a + b


class LocalBackboneTests(unittest.TestCase):
    def test_same_parameters_and_state_dict_as_global(self):
        local, glob = build("local6"), build("global")
        self.assertEqual(sum(p.numel() for p in local.parameters()), sum(p.numel() for p in glob.parameters()))
        self.assertEqual(list(local.state_dict()), list(glob.state_dict()))
        result = glob.load_state_dict(local.state_dict(), strict=True)
        self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))

    def test_full_size_parameter_count_equals_global(self):
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        from fairchem.core._cli import get_hydra_config_from_yaml

        counts = {}
        for variant in ("global", "local6"):
            GlobalHydra.instance().clear()
            cfg = get_hydra_config_from_yaml(str(ROOT / "configs" / "train.yaml"),
                                             [f"variant={variant}", f"job.run_dir={tempfile.mkdtemp()}"])
            GlobalHydra.instance().clear()
            counts[variant] = sum(p.numel() for p in hydra.utils.instantiate(cfg.runner.train_eval_unit.model).parameters())
        self.assertEqual(counts["local6"], counts["global"])
        self.assertEqual(counts["local6"], 37_938_985)

    def test_far_fragments_are_additive_for_local_and_dense_but_not_global(self):
        a, b, union = fragments()
        delta = {}
        for variant in ("local6", "dense", "global"):
            m = build(variant)
            delta[variant] = energy(m, union) - energy(m, a) - energy(m, b)
        self.assertLess(abs(delta["local6"]), 1e-10)
        self.assertLess(abs(delta["dense"]), 1e-10)
        self.assertGreater(abs(delta["global"]), 1e-6)

    def test_local_coefficients_differ_between_atoms_and_match_isolated_fragment(self):
        a, b, union = fragments()
        m = build("local6")
        energy(m, union)
        coefficients = m.backbone.global_mole_tensors.expert_mixing_coefficients.detach().clone()
        self.assertEqual(coefficients.shape, (len(union), 8))
        self.assertGreater(float(coefficients.max(-1).values.max() - coefficients.max(-1).values.min()), 1e-4)
        energy(m, a)
        alone = m.backbone.global_mole_tensors.expert_mixing_coefficients.detach()
        torch.testing.assert_close(coefficients[: len(a)], alone, rtol=0, atol=1e-12)

    def test_local_equals_global_when_every_atom_gets_the_structure_mean(self):
        # Replaces only the averaging by the structure mean: the per-edge expert mixing must then
        # reproduce fairchem's merged-weight layers exactly (energies and forces), also when the
        # edges are processed in activation-checkpoint chunks.
        a, b, union = fragments()
        import training.local_moe as lm

        def mean_composition(embeddings, batch, neighbors, radius, power, **kwargs):
            totals = torch.zeros(int(batch.max()) + 1, embeddings.shape[1], dtype=embeddings.dtype, device=embeddings.device)
            totals = totals.index_add(0, batch, embeddings)
            counts = torch.bincount(batch).to(embeddings.dtype)
            return (totals / counts[:, None])[batch]

        glob, local = build("global"), build("local6")
        for chunk in (None, 5):
            if chunk is not None:
                for model in (glob, local):
                    for module in model.modules():
                        if getattr(module, "activation_checkpoint_chunk_size", None) is not None:
                            module.activation_checkpoint_chunk_size = chunk
            with mock.patch.object(lm, "local_composition", mean_composition):
                e_local, f_local = run(local, union)
            e_glob, f_glob = run(glob, union)
            torch.testing.assert_close(e_local, e_glob, rtol=0, atol=1e-11)
            torch.testing.assert_close(f_local, f_glob, rtol=0, atol=1e-10)

    def test_forces_match_finite_differences_of_the_energy(self):
        # Close atoms, so the router itself depends on positions and its gradient is part of the force.
        from ase.build import molecule

        atoms = generic(molecule("CH3OH"))
        m = build("local6")
        _, forces = run(m, atoms)
        forces = forces.detach().cpu().numpy()
        step = 1e-4
        for atom, axis in ((0, 0), (1, 2), (3, 1), (5, 0)):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            numeric = -(energy(m, plus) - energy(m, minus)) / (2 * step)
            self.assertAlmostEqual(numeric, forces[atom, axis], delta=1e-7)

    def test_periodic_slabs_run_and_far_slabs_are_additive(self):
        # The benchmark geometry: periodic Al and Ni slabs in a 10.7 A cell. fairchem's graph holds
        # several images of one atom pair at the same distance there, which once failed the duplicate check.
        sys.path.insert(0, str(ROOT / "audit"))
        import geometry as g
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

        def periodic_energy(model, atoms):
            atoms = atoms.copy()
            atoms.info.update(charge=0, spin=1)
            data = AtomicData.from_ase(atoms, task_name="omol", r_edges=False, r_data_keys=["spin", "charge"],
                                       r_forces=False, r_stress=False, target_dtype=torch.float64)
            return model(atomic_list_to_batch([data]).to(DEVICE))["omol_energy"]["energy"].item()

        atomic_list_to_batch = atomicdata_list_to_batch
        al, ni = g.ideal_slab("Al"), g.ideal_slab("Ni")
        for variant in ("local6", "dense"):
            m = build(variant)
            delta = (periodic_energy(m, g.stacked(al, ni, 10)) - periodic_energy(m, g.isolated(al))
                     - periodic_energy(m, g.isolated(ni)))
            self.assertLess(abs(delta), 1e-9, variant)

    def test_energy_is_invariant_to_rotation_translation_and_atom_order(self):
        from ase.build import molecule

        atoms = generic(molecule("CH3OH"))
        m = build("local6")
        reference = energy(m, atoms)
        rotated = atoms.copy()
        rotated.rotate(37.0, (1.0, 2.0, 3.0))
        rotated.translate([5.0, -3.0, 1.0])
        # the base model is equivariant only to about 1e-7 (Dense shows the same), so the bound is 1e-6
        self.assertAlmostEqual(energy(m, rotated), reference, delta=1e-6)
        order = np.random.default_rng(0).permutation(len(atoms))
        self.assertAlmostEqual(energy(m, atoms[order]), reference, delta=1e-10)

    def test_gradient_reaches_the_router_parameters(self):
        from ase.build import molecule

        m = build("local6").train()          # training mode keeps the graph of the forces for a second derivative
        e, f = run(m, generic(molecule("CH3OH")))
        (e.sum() + f.square().sum()).backward()
        for name in ("routing_mlp", "composition_embedding"):
            grads = [p.grad for n, p in m.backbone.named_parameters() if n.startswith(name)]
            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(float(sum(g.abs().sum() for g in grads)), 0.0, name)

    def test_charge_and_spin_context_still_reach_the_router(self):
        from ase.build import molecule

        atoms = molecule("CH3OH")
        m = build("local6")
        self.assertGreater(abs(energy(m, atoms, charge=0, spin=1) - energy(m, atoms, charge=1, spin=2)), 1e-6)

    def test_edges_are_checkpointed_in_small_chunks(self):
        for variant, expected in (("local6", 4096), ("global", 128 * 1024)):
            m = build(variant)
            sizes = {x.activation_checkpoint_chunk_size for x in m.modules()
                     if getattr(x, "activation_checkpoint_chunk_size", None) is not None}
            self.assertEqual(sizes, {expected})

    def test_routing_state_is_cleared_after_a_forward_pass(self):
        from ase.build import molecule

        m = build("local6")
        energy(m, molecule("H2O"))
        self.assertIsNone(m.backbone._routing_inputs)
        self.assertIsNone(m.backbone._routing_graph)

    def test_unsupported_settings_raise(self):
        # Hydra wraps the constructor error, so match on the message
        for override, message in (("routing_radius=7.0", "routing radius above the message cutoff"),
                                  ("use_composition_embedding=false", "needs experts and use_composition_embedding"),
                                  ("composition_dropout=0.1", "composition_dropout is not defined")):
            with self.assertRaisesRegex(Exception, message):
                build("local6", 0, f"+variant.backbone.{override}" if override.startswith("composition_dropout") else f"variant.backbone.{override}")
        with self.assertRaises(NotImplementedError):
            build("local6").backbone.merge_MOLE_model({})

    def test_a_smaller_routing_radius_changes_the_result(self):
        from ase.build import molecule

        atoms = generic(molecule("CH3OH"))
        full, short = build("local6"), build("local6", 0, "variant.backbone.routing_radius=1.0")
        self.assertGreater(abs(energy(full, atoms) - energy(short, atoms)), 1e-8)


if __name__ == "__main__":
    unittest.main()
