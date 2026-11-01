"""The locality guard uses both graphs, without discarding close-gap diagnostics."""
import unittest
from types import SimpleNamespace
from audit.geometry import ideal_slab, stacked, assert_disconnected, disconnected
from audit.models import interaction_radius
from audit.molecular import locality, fragments
from test_molecular_audit import PairPotential
from scripts.size_consistency_along_training import RUN_RE, selfcheck


class RoutingGuardTests(unittest.TestCase):
    def test_periodic_guard_uses_routing_radius(self):
        a, b = ideal_slab("Al"), ideal_slab("Ni")
        radius = interaction_radius(SimpleNamespace(routing_radius=12), 6)
        for gap in (8, 10, 15, 25, 100):
            pair = stacked(a, b, gap)
            assert_disconnected(pair, 6)
            self.assertEqual(disconnected(pair, radius), gap >= 15)
            if gap < 15:
                with self.assertRaises(RuntimeError):
                    assert_disconnected(pair, radius)

    def test_molecular_close_gaps_are_measured_and_labelled(self):
        model = PairPotential()
        model.routing_radius = 12
        for gap in (8, 15):
            row = locality(model, *fragments("H2O", "CH4", gap), 6)
            self.assertEqual(row["routing_disconnected"], gap == 15)
            self.assertIn("delta_eV", row)

    def test_all_variants_and_connected_rows(self):
        for variant in ("dense", "global", "local6", "local12", "element", "local_global", "global_frozen", "dense_scaled"):
            match = RUN_RE.match(f"pilot_{variant}_s4_123")
            self.assertEqual(match.group(1), variant)
        selfcheck([dict(variant="local12", expected_additive=False, Delta_meV=1)], [])
        with self.assertRaises(AssertionError):
            selfcheck([dict(variant="local12", expected_additive=True, Delta_meV=1)], [])


if __name__ == "__main__":
    unittest.main()
