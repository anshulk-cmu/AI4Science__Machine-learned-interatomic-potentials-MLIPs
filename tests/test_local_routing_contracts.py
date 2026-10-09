import os
import unittest
import torch
from local_routing import (CompositionRouter, RoutingInputs, RoutingNeighbors,
                           RoutingResult, LinearExperts, edge_coefficients,
                           local_composition, reference_neighbors)

F64 = torch.float64


def setUpModule():
    torch.set_default_device(os.environ.get("ROUTER_TEST_DEVICE", "cpu"))


def tearDownModule():
    torch.set_default_device(None)


class ChargeSpinConditioner(torch.nn.Module):
    def forward(self, task, charge, spin):
        return torch.stack((charge, spin), dim=-1).to(F64)


def make_inputs(z, pos, batch, radius=6., charge=None, neighbors=None):
    b = int(batch.max()) + 1
    charge = torch.zeros(b, dtype=torch.long) if charge is None else charge
    neighbors = reference_neighbors(pos, batch, radius) if neighbors is None else neighbors
    return RoutingInputs(z, pos, batch, torch.zeros(b, dtype=torch.long), charge,
                         torch.ones(b, dtype=torch.long), neighbors)


def make_router(variant="local", radius=6., power=3, conditioned=False):
    if conditioned:
        router = CompositionRouter(variant, embedding_dim=4, hidden_dim=8, radius=radius,
                                   power=power, context_dim=2, conditioner=ChargeSpinConditioner())
    else:
        router = CompositionRouter(variant, embedding_dim=4, hidden_dim=8, radius=radius,
                                   power=power, allow_unconditioned=True)
    return router.double()


class BehaviourTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.z = torch.tensor([8, 1, 6, 1])
        self.pos = torch.tensor([[0., 0., 0.], [1., .2, .1],
                                 [2., .3, 0.], [3., .1, .3]], dtype=F64)
        self.batch = torch.zeros(4, dtype=torch.long)

    def test_global_unequal_sizes_is_per_structure_mean(self):
        z = torch.tensor([8, 1, 1, 6, 1, 1, 1, 7])
        pos = torch.randn(8, 3, dtype=F64)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        router = make_router("global")
        together = router(make_inputs(z, pos, batch)).coefficients
        emb = router.element_embedding(z)
        mean = torch.stack((emb[:3].mean(0), emb[3:].mean(0)))
        torch.testing.assert_close(together, router.gate(mean).softmax(-1))
        parts = [router(make_inputs(z[s:e], pos[s:e], torch.zeros(e - s, dtype=torch.long))).coefficients
                 for s, e in ((0, 3), (3, 8))]
        torch.testing.assert_close(together, torch.cat(parts))

    def test_distant_fragment_changes_global_not_local(self):
        a_z = torch.tensor([8, 1, 1])
        a_pos = torch.tensor([[0., 0., 0.], [.96, 0., 0.], [-.24, .93, 0.]], dtype=F64)
        b_pos = a_pos + 40
        batch = torch.zeros(6, dtype=torch.long)
        pos = torch.cat((a_pos, b_pos))
        fragments = [torch.tensor([6, 1, 1]), torch.tensor([7, 9, 9])]
        coefficients = {}
        for variant in ("global", "local"):
            router = make_router(variant)
            coefficients[variant] = [router(make_inputs(torch.cat((a_z, b_z)), pos, batch)).coefficients
                                     for b_z in fragments]
        self.assertFalse(torch.allclose(*coefficients["global"]))
        left, right = (c[:3] for c in coefficients["local"])
        self.assertTrue(torch.equal(left, right))

    def test_structure_merge_with_several_structures(self):
        batch = torch.tensor([0, 0, 1, 1])
        router = make_router("global")
        result = router(make_inputs(self.z, self.pos, batch))
        self.assertFalse(torch.allclose(result.coefficients[0], result.coefficients[1]))
        layer = LinearExperts(2, 3).double()
        edges = torch.tensor([[0, 1, 2, 3, 3], [1, 0, 3, 2, 2]])
        x = torch.randn(5, 2, dtype=F64, requires_grad=True)
        merged = layer(x, result, edges, batch, "structure_merge")
        mixed = layer(x, result, edges, batch, "expert_output_mix")
        torch.testing.assert_close(merged, mixed)
        gm, = torch.autograd.grad(merged.square().sum(), x)
        gx, = torch.autograd.grad(mixed.square().sum(), x)
        torch.testing.assert_close(gm, gx)

    def test_global_with_conditioner(self):
        batch = torch.tensor([0, 0, 1, 1])
        charge = torch.tensor([0, 1])
        router = make_router("global", conditioned=True)
        inputs = make_inputs(self.z, self.pos, batch, charge=charge)
        emb = router.element_embedding(self.z)
        mean = torch.stack((emb[:2].mean(0), emb[2:].mean(0)))
        context = torch.stack((charge, torch.ones(2, dtype=torch.long)), dim=-1).to(F64)
        expected = router.gate(torch.cat((mean, context), dim=-1)).softmax(-1)
        torch.testing.assert_close(router(inputs).coefficients, expected)
        inputs.charge = inputs.charge + 1
        self.assertFalse(torch.allclose(router(inputs).coefficients, expected))

    def test_local_fragment_independence_with_conditioner(self):
        router = make_router("local", conditioned=True)
        pos = self.pos.clone()
        pos[2:] += 40
        alone = router(make_inputs(self.z[:2], pos[:2], torch.zeros(2, dtype=torch.long))).coefficients
        together = router(make_inputs(self.z, pos, self.batch)).coefficients
        torch.testing.assert_close(together[:2], alone)
        # Documented exception: the structure context reaches every atom, so a
        # different total charge changes the coefficients of a distant fragment.
        charged = router(make_inputs(self.z, pos, self.batch, charge=torch.tensor([1]))).coefficients
        self.assertFalse(torch.allclose(charged[:2], alone))

    def test_permuted_edge_order(self):
        router = make_router()
        inputs = make_inputs(self.z, self.pos, self.batch)
        expected = router(inputs).coefficients
        order = torch.randperm(inputs.neighbors.distances.numel())
        inputs.neighbors = RoutingNeighbors(inputs.neighbors.edge_index[:, order],
                                            inputs.neighbors.distances[order])
        torch.testing.assert_close(router(inputs).coefficients, expected)

    def test_interleaved_batch_ids(self):
        batch = torch.tensor([0, 1, 0, 1])
        for variant in ("local", "global"):
            router = make_router(variant)
            together = router(make_inputs(self.z, self.pos, batch)).coefficients
            parts = [router(make_inputs(self.z[i], self.pos[i], torch.zeros(2, dtype=torch.long))).coefficients
                     for i in (torch.tensor([0, 2]), torch.tensor([1, 3]))]
            if variant == "local":
                torch.testing.assert_close(together[torch.tensor([0, 2])], parts[0])
                torch.testing.assert_close(together[torch.tensor([1, 3])], parts[1])
            else:
                torch.testing.assert_close(together, torch.cat(parts))

    def test_all_isolated_atoms_through_router(self):
        pos = (torch.arange(4, dtype=F64)[:, None] * 50).expand(4, 3).clone().requires_grad_()
        router = make_router()
        inputs = make_inputs(self.z, pos, self.batch)
        self.assertEqual(inputs.neighbors.distances.numel(), 0)
        expected = router.gate(router.element_embedding(self.z)).softmax(-1)
        torch.testing.assert_close(router(inputs).coefficients, expected)

    def test_zero_edges_in_linear_experts(self):
        layer = LinearExperts(2, 3).double()
        edges = torch.zeros(2, 0, dtype=torch.long)
        x = torch.zeros(0, 2, dtype=F64)
        for variant, merge in (("local", "atom_merge"), ("global", "structure_merge")):
            result = make_router(variant)(make_inputs(self.z, self.pos, self.batch))
            for mode in ("expert_output_mix", merge):
                self.assertEqual(layer(x, result, edges, self.batch, mode).shape, (0, 3))

    def test_linear_experts_without_bias(self):
        layer = LinearExperts(2, 3, bias=False).double()
        self.assertIsNone(layer.bias)
        self.assertEqual([name for name, _ in layer.named_parameters()], ["weight"])
        result = make_router()(make_inputs(self.z, self.pos, self.batch))
        edges = torch.tensor([[0, 2, 3], [1, 1, 2]])
        x = torch.randn(3, 2, dtype=F64)
        alpha = result.coefficients[edges[1]]
        expected = torch.stack([sum(alpha[e, k] * layer.weight[k] @ x[e] for k in range(8))
                                for e in range(3)])
        for mode in ("expert_output_mix", "atom_merge"):
            torch.testing.assert_close(layer(x, result, edges, self.batch, mode), expected)

    def test_powers_four_and_five_against_dense_reference(self):
        emb = torch.randn(4, 3, dtype=F64)
        neighbors = reference_neighbors(self.pos, self.batch, 6.)
        d2 = (self.pos[:, None] - self.pos[None, :]).square().sum(-1)
        for power in (4, 5):
            weights = (1 - d2 / 36).clamp_min(0).pow(power)
            expected = weights @ emb / weights.sum(-1, keepdim=True)
            torch.testing.assert_close(local_composition(emb, self.batch, neighbors, 6., power), expected)

    def test_radius_zero_through_router(self):
        router = make_router(radius=0.)
        inputs = make_inputs(self.z, self.pos, self.batch, radius=0.)
        self.assertEqual(inputs.neighbors.distances.numel(), 0)
        expected = router.gate(router.element_embedding(self.z)).softmax(-1)
        torch.testing.assert_close(router(inputs).coefficients, expected)

    def test_pair_at_exactly_the_radius_has_zero_weight(self):
        emb = torch.eye(2, dtype=F64)
        batch = torch.zeros(2, dtype=torch.long)
        neighbors = RoutingNeighbors(torch.tensor([[0, 1], [1, 0]]), torch.full((2,), 6., dtype=F64))
        self.assertTrue(torch.equal(local_composition(emb, batch, neighbors, 6.), emb))
        pos = torch.tensor([[0., 0., 0.], [6., 0., 0.]], dtype=F64)
        self.assertEqual(reference_neighbors(pos, batch, 6.).distances.numel(), 0)

    def test_float32_paths(self):
        pos = self.pos.float().requires_grad_()
        layer = LinearExperts(2, 3)
        edges = torch.tensor([[0, 2, 3], [1, 1, 2]])
        x = torch.randn(3, 2)
        for variant, merge in (("local", "atom_merge"), ("global", "structure_merge")):
            router = CompositionRouter(variant, embedding_dim=4, hidden_dim=8, allow_unconditioned=True)
            result = router(make_inputs(self.z, pos, self.batch))
            self.assertEqual(result.coefficients.dtype, torch.float32)
            self.assertEqual(edge_coefficients(result, edges, self.batch).dtype, torch.float32)
            mixed = layer(x, result, edges, self.batch, "expert_output_mix")
            torch.testing.assert_close(layer(x, result, edges, self.batch, merge), mixed)
            self.assertEqual(mixed.dtype, torch.float32)
            if variant == "local":
                grad, = torch.autograd.grad(mixed.square().sum(), pos)
                self.assertEqual(grad.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(grad).all()))
                self.assertGreater(grad.abs().sum().item(), 0)


class ErrorPathTests(unittest.TestCase):
    def setUp(self):
        self.emb = torch.randn(4, 3, dtype=F64)
        self.batch = torch.zeros(4, dtype=torch.long)

    def composition(self, edges, distances):
        return local_composition(self.emb, self.batch,
                                 RoutingNeighbors(torch.tensor(edges), torch.tensor(distances, dtype=F64)))

    def test_wrong_distance_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.composition([[0, 1], [1, 0]], [1., 1., 1.])

    def test_out_of_range_index(self):
        for edges in ([[0], [4]], [[-1], [0]]):
            with self.assertRaisesRegex(ValueError, "outside atom range"):
                self.composition(edges, [1.])
            with self.assertRaisesRegex(ValueError, "outside atom range"):
                edge_coefficients(RoutingResult(torch.ones(4, 8), "atom"), torch.tensor(edges), self.batch)

    def test_zero_distance_self_edge(self):
        with self.assertRaisesRegex(ValueError, "self edges"):
            self.composition([[1], [1]], [0.])

    def test_negative_and_nan_distance(self):
        for value in (-1., float("nan")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                self.composition([[0], [1]], [value])

    def test_bad_scope(self):
        with self.assertRaisesRegex(ValueError, "unknown routing scope"):
            edge_coefficients(RoutingResult(torch.ones(4, 8), "edge"), torch.tensor([[0], [1]]), self.batch)

    def test_bad_mode_and_wrong_x_shape(self):
        layer = LinearExperts(2, 3).double()
        result = RoutingResult(torch.full((4, 8), 1 / 8, dtype=F64), "atom")
        edges = torch.tensor([[0, 2], [1, 3]])
        with self.assertRaisesRegex(ValueError, "mode must be"):
            layer(torch.ones(2, 2, dtype=F64), result, edges, self.batch, "edge_merge")
        for x in (torch.ones(2, 3, dtype=F64), torch.ones(3, 2, dtype=F64), torch.ones(2, dtype=F64)):
            with self.assertRaisesRegex(ValueError, "x must have shape"):
                layer(x, result, edges, self.batch)


class ContractTests(unittest.TestCase):
    """R1 to R5 of the neighbor-list and coefficient contracts."""
    def setUp(self):
        torch.manual_seed(5)
        self.emb = torch.randn(3, 2, dtype=F64)
        self.batch = torch.zeros(3, dtype=torch.long)

    def composition(self, edges, distances, emb=None):
        emb = self.emb if emb is None else emb
        return local_composition(emb, self.batch, RoutingNeighbors(torch.tensor(edges), distances))

    def test_r1_duplicate_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate neighbor edges"):
            self.composition([[0, 1, 0], [1, 0, 1]], torch.tensor([2., 2., 2.], dtype=F64))

    def test_r1_same_pair_at_different_distances_allowed(self):
        w1, w2 = (1 - 4 / 36)**3, (1 - 9 / 36)**3
        out = self.composition([[0, 0], [1, 1]], torch.tensor([2., 3.], dtype=F64))
        expected = (self.emb[1] + (w1 + w2) * self.emb[0]) / (1 + w1 + w2)
        torch.testing.assert_close(out[1], expected)

    def test_r2_detached_distances_rejected(self):
        z = torch.tensor([8, 1, 1])
        pos = torch.tensor([[0., 0., 0.], [.96, 0., 0.], [-.24, .93, 0.]], dtype=F64)
        router = make_router()
        detached = reference_neighbors(pos, self.batch, 6.)
        tracked = pos.clone().requires_grad_()
        with self.assertRaisesRegex(ValueError, "computed from positions with autograd"):
            router(make_inputs(z, tracked, self.batch, neighbors=detached))
        with torch.no_grad():
            router(make_inputs(z, tracked, self.batch, neighbors=detached))
        router(make_inputs(z, pos, self.batch, neighbors=detached))
        router(make_inputs(z, tracked, self.batch))

    def test_r3_distance_dtype_and_device_must_match(self):
        with self.assertRaisesRegex(ValueError, "matching dtype/device"):
            self.composition([[0], [1]], torch.tensor([2.], dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "matching dtype/device"):
            self.composition([[0], [1]], torch.tensor([2.], dtype=F64, device="meta"))

    def test_r4_zero_distance_self_edge_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero-distance self edges"):
            self.composition([[0, 0], [0, 1]], torch.tensor([0., 2.], dtype=F64))

    def test_r4_periodic_self_image_allowed(self):
        w = (1 - 9 / 36)**3
        out = self.composition([[0, 1, 0], [0, 0, 1]], torch.tensor([3., 3., 3.], dtype=F64))
        torch.testing.assert_close(out[0], (self.emb[0] * (1 + w) + w * self.emb[1]) / (1 + 2 * w))
        z = torch.tensor([13])
        pos = torch.zeros(1, 3, dtype=F64)
        images = RoutingNeighbors(torch.tensor([[0, 0], [0, 0]]), torch.tensor([3., 3.5], dtype=F64))
        router = make_router()
        expected = router.gate(router.element_embedding(z)).softmax(-1)
        result = router(make_inputs(z, pos, torch.zeros(1, dtype=torch.long), neighbors=images))
        torch.testing.assert_close(result.coefficients, expected)

    def test_r5_coefficient_rows_must_match_scope(self):
        edges = torch.tensor([[0, 2], [1, 3]])
        batch = torch.tensor([0, 0, 1, 1])
        for coefficients, scope in ((torch.ones(3, 8), "atom"), (torch.ones(5, 8), "atom"),
                                    (torch.ones(1, 8), "structure"), (torch.ones(3, 8), "structure"),
                                    (torch.ones(8), "atom")):
            with self.assertRaisesRegex(ValueError, "one row per"):
                edge_coefficients(RoutingResult(coefficients, scope), edges, batch)
        edge_coefficients(RoutingResult(torch.ones(4, 8), "atom"), edges, batch)
        edge_coefficients(RoutingResult(torch.ones(2, 8), "structure"), edges, batch)
        layer = LinearExperts(2, 3)
        with self.assertRaisesRegex(ValueError, "one row per structure"):
            layer(torch.ones(2, 2), RoutingResult(torch.ones(3, 8), "structure"), edges, batch, "structure_merge")


if __name__ == "__main__":
    unittest.main()
