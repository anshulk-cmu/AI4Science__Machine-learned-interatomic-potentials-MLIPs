import os
import unittest
import torch
from local_routing import (CompositionRouter, RoutingInputs, RoutingNeighbors,
                           RoutingResult, LinearExperts, edge_coefficients,
                           local_composition, reference_neighbors)


def setUpModule():
    torch.set_default_device(os.environ.get("ROUTER_TEST_DEVICE", "cpu"))


def tearDownModule():
    torch.set_default_device(None)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.z = torch.tensor([8, 1, 6, 1])
        self.pos = torch.tensor([[0., 0., 0.], [1., .2, .1],
                                 [2., .3, 0.], [3., .1, .3]], dtype=torch.float64)
        self.batch = torch.zeros(4, dtype=torch.long)

    def inputs(self, pos=None, batch=None, z=None, radius=6.):
        pos = self.pos if pos is None else pos
        batch = self.batch if batch is None else batch
        z = self.z if z is None else z
        b = int(batch.max()) + 1
        return RoutingInputs(z, pos, batch, torch.zeros(b, dtype=torch.long),
                             torch.zeros(b, dtype=torch.long), torch.ones(b, dtype=torch.long),
                             reference_neighbors(pos, batch, radius))

    def router(self, variant="local", radius=6.):
        return CompositionRouter(variant, embedding_dim=4, hidden_dim=8,
                                 radius=radius, allow_unconditioned=True).double()

    def test_hand_calculated_composition(self):
        pos = torch.tensor([[0., 0., 0.], [3., 0., 0.]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        emb = torch.eye(2, dtype=pos.dtype)
        w = (1 - .25)**3
        expected = torch.tensor([[1., w], [w, 1.]], dtype=pos.dtype) / (1+w)
        torch.testing.assert_close(local_composition(emb, batch, reference_neighbors(pos, batch, 6)), expected)

    def test_scopes_and_normalization(self):
        inputs = self.inputs(batch=torch.tensor([0, 0, 1, 1]))
        for mode, shape, scope in [("global", (2, 8), "structure"), ("local", (4, 8), "atom")]:
            result = self.router(mode)(inputs)
            self.assertEqual(result.scope, scope)
            self.assertEqual(result.coefficients.shape, shape)
            torch.testing.assert_close(result.coefficients.sum(-1), torch.ones(shape[0], dtype=self.pos.dtype))

    def test_element_limit_and_isolated_atom(self):
        emb = torch.randn(4, 3, dtype=self.pos.dtype)
        neighbors = reference_neighbors(self.pos, self.batch, 6)
        torch.testing.assert_close(local_composition(emb, self.batch, neighbors, 0), emb)
        single = reference_neighbors(self.pos[:1], self.batch[:1], 6)
        torch.testing.assert_close(local_composition(emb[:1], self.batch[:1], single), emb[:1])

    def test_sparse_matches_all_pairs_reference(self):
        emb = torch.randn(4, 3, dtype=self.pos.dtype)
        d2 = (self.pos[:, None] - self.pos[None, :]).square().sum(-1)
        weights = (1 - d2 / 36).clamp_min(0).pow(3)
        expected = weights @ emb / weights.sum(-1, keepdim=True)
        actual = local_composition(emb, self.batch, self.inputs().neighbors)
        torch.testing.assert_close(actual, expected)

    def test_batch_isolation(self):
        batch = torch.tensor([0, 0, 1, 1])
        for mode in ["local", "global"]:
            router = self.router(mode)
            together = router(self.inputs(batch=batch)).coefficients
            parts = [router(self.inputs(self.pos[s:s+2], torch.zeros(2, dtype=torch.long),
                                        self.z[s:s+2])).coefficients for s in [0, 2]]
            torch.testing.assert_close(together, torch.cat(parts))

    def test_disconnected_local6_and_local12(self):
        pos = self.pos.clone()
        pos[2:] += 40
        for radius in [6., 12.]:
            router = self.router(radius=radius)
            together = router(self.inputs(pos=pos, radius=radius)).coefficients
            parts = [router(self.inputs(pos[s:s+2], torch.zeros(2, dtype=torch.long),
                                        self.z[s:s+2], radius)).coefficients for s in [0, 2]]
            torch.testing.assert_close(together, torch.cat(parts))
            torch.testing.assert_close(together.square().sum(), sum(p.square().sum() for p in parts))

    def test_global_mean_unchanged_for_duplicate_structure(self):
        router = self.router("global")
        expected = router(self.inputs()).coefficients
        duplicate = self.inputs(torch.cat((self.pos, self.pos + 40)),
                                torch.zeros(8, dtype=torch.long), torch.cat((self.z, self.z)))
        torch.testing.assert_close(router(duplicate).coefficients, expected)

    def test_rigid_motion_and_permutation(self):
        router = self.router()
        expected = router(self.inputs()).coefficients
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=self.pos.dtype))
        torch.testing.assert_close(router(self.inputs(pos=self.pos @ q + 100)).coefficients, expected)
        order = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(router(self.inputs(self.pos[order], self.batch[order], self.z[order])).coefficients,
                                   expected[order])

    def test_position_derivatives(self):
        router = self.router()
        pos = self.pos.clone().requires_grad_()
        f = lambda p: router(self.inputs(pos=p)).coefficients
        self.assertTrue(torch.autograd.gradcheck(f, (pos,)))
        self.assertTrue(torch.autograd.gradgradcheck(f, (pos,)))

    def test_cutoff_smoothness(self):
        emb = torch.eye(2, dtype=self.pos.dtype)
        batch = torch.zeros(2, dtype=torch.long)
        for distance in [6.-1e-5, 6., 6.+1e-5]:
            r = torch.tensor([distance, distance], dtype=self.pos.dtype, requires_grad=True)
            neighbors = RoutingNeighbors(torch.tensor([[0, 1], [1, 0]]), r)
            value = local_composition(emb, batch, neighbors)[0, 1]
            first, = torch.autograd.grad(value, r, create_graph=True)
            second, = torch.autograd.grad(first.sum(), r)
            self.assertLess(max(value.abs().item(), first.abs().max().item(), second.abs().max().item()), 1e-5)

    def test_receiver_adapter_both_scopes(self):
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        batch = torch.tensor([0, 0, 1, 1])
        atom = torch.arange(8.).reshape(4, 2)
        structure = torch.arange(4.).reshape(2, 2)
        torch.testing.assert_close(edge_coefficients(RoutingResult(atom, "atom"), edges, batch), atom[edges[1]])
        torch.testing.assert_close(edge_coefficients(RoutingResult(structure, "structure"), edges, batch),
                                   structure[batch[edges[1]]])

    def test_expert_modes_outputs_and_gradients(self):
        for variant in ["local", "global"]:
            router = self.router(variant)
            layer = LinearExperts(2, 3).double()
            inputs = self.inputs(pos=self.pos.clone().requires_grad_())
            result = router(inputs)
            edges = torch.tensor([[0, 2, 3], [1, 1, 2]])
            x = torch.randn(3, 2, dtype=self.pos.dtype, requires_grad=True)
            merge_mode = "atom_merge" if variant == "local" else "structure_merge"
            a = layer(x, result, edges, self.batch, merge_mode)
            b = layer(x, result, edges, self.batch, "expert_output_mix")
            torch.testing.assert_close(a, b)
            variables = (x, *router.parameters(), *layer.parameters())
            if variant == "local":
                variables += (inputs.positions,)
            ga = torch.autograd.grad(a.square().sum(), variables, retain_graph=True)
            gb = torch.autograd.grad(b.square().sum(), variables)
            for left, right in zip(ga, gb):
                torch.testing.assert_close(left, right)

    def test_merge_mode_requires_matching_scope(self):
        edges = torch.tensor([[0], [1]])
        layer = LinearExperts(1, 1).double()
        x = torch.ones(1, 1, dtype=self.pos.dtype)
        for variant, wrong_mode in [("local", "structure_merge"), ("global", "atom_merge")]:
            with self.assertRaises(ValueError):
                layer(x, self.router(variant)(self.inputs()), edges, self.batch, wrong_mode)

    def test_force_loss_training_step(self):
        router = self.router()
        layer = LinearExperts(1, 1).double()
        inputs = self.inputs(pos=self.pos.clone().requires_grad_())
        edges, distances = inputs.neighbors.edge_index, inputs.neighbors.distances
        output = layer(torch.exp(-distances.square())[:, None], router(inputs), edges, inputs.batch)
        energy = output.square().sum()  # Toy scalar, not an MLIP.
        force = -torch.autograd.grad(energy, inputs.positions, create_graph=True)[0]
        torch.testing.assert_close(force.sum(0), torch.zeros(3, dtype=self.pos.dtype), atol=1e-12, rtol=0)
        loss = energy.square() + force.square().mean()
        optimizer = torch.optim.SGD([*router.parameters(), *layer.parameters()], lr=.01)
        before = router.gate[-1].weight.detach().clone()
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in router.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, router.gate[-1].weight))

    def test_conditioning_required_and_used(self):
        with self.assertRaises(ValueError):
            CompositionRouter("local")
        class SharedConditioner(torch.nn.Module):
            def forward(self, task, charge, spin):
                return torch.stack((charge, spin), dim=-1).double()
        router = CompositionRouter("local", context_dim=2, conditioner=SharedConditioner()).double()
        inputs = self.inputs()
        before = router(inputs).coefficients
        inputs.charge = inputs.charge + 1
        self.assertFalse(torch.allclose(before, router(inputs).coefficients))

    def test_reject_cross_batch_neighbors_and_edges(self):
        batch = torch.tensor([0, 0, 1, 1])
        edges = torch.tensor([[0], [2]])
        with self.assertRaises(ValueError):
            local_composition(torch.randn(4, 3), batch, RoutingNeighbors(edges, torch.ones(1)))
        with self.assertRaises(ValueError):
            edge_coefficients(RoutingResult(torch.randn(4, 8), "atom"), edges, batch)

    def test_local12_has_separate_neighbor_radius(self):
        pos = torch.tensor([[0., 0., 0.], [8., 0., 0.]], dtype=self.pos.dtype)
        batch = torch.zeros(2, dtype=torch.long)
        self.assertEqual(reference_neighbors(pos, batch, 6).distances.numel(), 0)
        neighbors = reference_neighbors(pos, batch, 12)
        self.assertEqual(neighbors.distances.numel(), 2)
        self.assertGreater(local_composition(torch.eye(2, dtype=pos.dtype), batch, neighbors, 12)[0, 1], 0)


if __name__ == "__main__":
    unittest.main()
