"""Local-12 graph separation, periodic images and conservative forces."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule

from test_local_backbone import build, run, energy, generic, DEVICE


def data_for(atoms, edges=False):
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    atoms = atoms.copy()
    atoms.info.update(charge=0, spin=1)
    data = AtomicData.from_ase(
        atoms, task_name="omol", r_edges=edges, r_data_keys=["spin", "charge"],
        r_forces=False, r_stress=False, target_dtype=torch.float64,
    )
    return atomicdata_list_to_batch([data]).to(DEVICE)


class Local12Tests(unittest.TestCase):
    def test_shell_changes_router_but_not_message_edges(self):
        model = build("local12")
        a = generic(molecule("H2O"))
        energy(model, a)
        alone = model.backbone.global_mole_tensors.expert_mixing_coefficients.detach().clone()
        for gap, expected_same in ((10, False), (15, True)):
            b = generic(molecule("CH4"))
            b.translate([a.positions[:, 0].max() - b.positions[:, 0].min() + gap, 0, 0])
            with patch.object(model.backbone, "_generate_graph", wraps=model.backbone._generate_graph) as graph:
                energy(model, a + b)
            coeff = model.backbone.global_mole_tensors.expert_mixing_coefficients[:len(a)]
            self.assertEqual(torch.allclose(coeff, alone, atol=1e-12, rtol=0), expected_same)
            self.assertEqual(model.backbone.cutoff, 6)
            self.assertIsNone(model.backbone._routing_graph)
            self.assertEqual(graph.call_count, 1)
            # Edge coefficients belong to message edges, not the larger routing graph.
            message = data_for(a + b)
            result = model.backbone._generate_graph(message)
            self.assertLessEqual(float(result["edge_distance"].max()), 6)
            self.assertEqual(model.backbone.global_mole_tensors.edge_coefficients.shape[0], result["edge_index"].shape[1])
            model.backbone._routing_graph = None

    def test_far_fragments_additive(self):
        model = build("local12")
        a, b = generic(molecule("H2O")), generic(molecule("CH4"))
        b.translate([20, 0, 0])
        self.assertAlmostEqual(energy(model, a+b), energy(model, a)+energy(model, b), delta=1e-10)

    def test_precomputed_message_edges_match_on_the_fly(self):
        model = build("local12")
        atoms = generic(molecule("CH3OH"))
        batch = data_for(atoms)
        graph = model.backbone._generate_graph(batch)
        batch["edge_index"] = graph["edge_index"]
        batch["cell_offsets"] = graph["cell_offsets"]
        batch["nedges"] = graph["neighbors"]
        expected = model(batch)
        model.backbone.otf_graph = False
        actual = model(batch)
        for head, prop in (("omol_energy", "energy"), ("omol_forces", "forces")):
            torch.testing.assert_close(actual[head][prop], expected[head][prop], atol=1e-11, rtol=0)

    def test_router_graph_includes_periodic_images_without_message_cap(self):
        model = build("local12")
        atoms = Atoms("Al", positions=[[0,0,0]], cell=[4,4,4], pbc=True)
        batch = data_for(atoms)
        model.backbone.max_neighbors = 2
        message = model.backbone._generate_graph(batch)
        route = model.backbone._routing_graph
        self.assertGreater(route["edge_index"].shape[1], message["edge_index"].shape[1])
        self.assertGreater(float(route["edge_distance"].max()), 6)
        from ase.neighborlist import neighbor_list
        expected = neighbor_list("d", atoms, 12.0 - 1e-8)
        actual = route["edge_distance"].detach().cpu().numpy()
        actual = actual[actual < 12.0 - 1e-8]
        np.testing.assert_allclose(np.sort(actual), np.sort(expected), atol=1e-12)

    def test_batched_molecules_do_not_share_routing_edges(self):
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
        model = build("local12")
        a, b = generic(molecule("H2O")), generic(molecule("CH4"))
        # Both structures occupy the origin. Only their batch labels separate them.
        batch = atomicdata_list_to_batch([data_for(a), data_for(b)])
        model.backbone._generate_graph(batch)
        edge = model.backbone._routing_graph["edge_index"]
        self.assertTrue(torch.equal(batch.batch[edge[0]], batch.batch[edge[1]]))

    def test_router_distances_contribute_to_forces_and_training(self):
        model = build("local12")
        a, b = generic(molecule("H2O")), generic(molecule("CH4"))
        b.translate([10, 0, 0])
        atoms = a+b
        _, force = run(model, atoms)
        h = 1e-4
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[len(a):, 0] += h
        minus.positions[len(a):, 0] -= h
        numerical = -(energy(model, plus) - energy(model, minus)) / (2*h)
        self.assertGreater(abs(numerical), 1e-9)
        self.assertAlmostEqual(float(force[len(a):, 0].sum()), numerical, delta=1e-7)
        model.train()
        e, f = run(model, atoms)
        (e.sum() + f.square().sum()).backward()
        for p in model.backbone.routing_mlp.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
