"""The size-consistency predictions of docs/final-training-plan.md Section 3, on small float64 models (CPU or CUDA).

Delta = E(pair) - E(a) - E(b) for H2O and CH4 whose x bounding planes are `gap` angstrom apart.
Dense, Element, Local-6: zero at every gap. Local-12: not zero at 8 and 10 A, zero from 15 A.
Global, Local+Global: not zero and the same at every gap (the global composition does not depend on positions).
"""
import unittest

from audit.molecular import fragments
from test_local_backbone import build, energy

GAPS = (8, 10, 15, 25, 100)
ZERO = 1e-11          # eV; measured float64 noise of these small models is at most 3e-15
NONZERO = 1e-8        # eV; the smallest measured leak is 3e-7 (Local-12 at 10 A); random weights, not physical sizes


def deltas(variant):
    model = build(variant)
    out = {}
    for gap in GAPS:
        a, b, pair = fragments("H2O", "CH4", gap)
        for atoms in (a, b, pair):
            atoms.cell[:] = 0      # the test helper puts each structure in its own 120 A molecule cell
        out[gap] = energy(model, pair) - energy(model, a) - energy(model, b)
    return out


class SizeConsistencyPredictionTests(unittest.TestCase):
    def test_local_models_are_additive_beyond_the_message_cutoff(self):
        for variant in ("dense", "element", "local6"):
            with self.subTest(variant=variant):
                for gap, d in deltas(variant).items():
                    self.assertLess(abs(d), ZERO, f"gap {gap}")

    def test_local12_sees_the_other_fragment_until_its_routing_radius(self):
        for gap, d in deltas("local12").items():
            with self.subTest(gap=gap):
                if gap < 12:
                    self.assertGreater(abs(d), NONZERO)
                else:
                    self.assertLess(abs(d), ZERO)

    def test_global_routers_break_additivity_at_every_gap_and_do_not_decay(self):
        for variant in ("global", "local_global"):
            with self.subTest(variant=variant):
                d = deltas(variant)
                self.assertTrue(all(abs(v) > NONZERO for v in d.values()), d)
                self.assertLess(max(d.values()) - min(d.values()), ZERO, d)


if __name__ == "__main__":
    unittest.main()
