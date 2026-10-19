"""scripts/inference_timing.py: input selection, the timer, the energy-only switch and the parameter counts."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "tests", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inference_timing as it  # noqa: E402
from test_local_backbone import build, fragments, generic  # noqa: E402
from test_router_check import batch_of  # noqa: E402


class SelectionTests(unittest.TestCase):
    rows = np.array([3, 5, 8, 9, 12, 20, 21])
    natoms = np.array([24, 26, 100, 25, 200, 340, 99])

    def test_bin_takes_the_closest_sizes_and_fits_the_batch(self):
        # closest first (25, 24, 26, 99, 100), until the next one (200) would pass 450 atoms
        self.assertEqual(it.pick_bin(self.rows, self.natoms, 25, 450), [3, 5, 8, 9, 21])
        self.assertEqual(it.pick_bin(self.rows, self.natoms, 100, 450), [5, 8, 9, 21])   # at most 450 // 100 = 4
        self.assertEqual(it.pick_bin(self.rows, self.natoms, 100, 230), [8, 21])         # at most 230 // 100 = 2
        self.assertEqual(it.pick_bin(self.rows, self.natoms, 340, 450), [20])
        self.assertEqual(it.pick_bin(self.rows, self.natoms, 400, 300), [20])       # one structure even above the limit

    def test_bin_ties_go_to_the_lower_row(self):
        self.assertEqual(it.pick_bin(np.array([7, 2]), np.array([24, 26]), 25, 25), [2])

    def test_pack_follows_the_validation_batching(self):
        groups = it.pack(self.rows, self.natoms, 450)
        self.assertEqual(groups, [[3, 5, 8, 9, 12], [20, 21]])
        self.assertTrue(all(sum(self.natoms[np.isin(self.rows, g)]) <= 450 for g in groups))
        self.assertEqual(sum(groups, []), list(self.rows))

    def test_quartiles(self):
        self.assertEqual(it.quartiles(np.arange(1.0, 6.0)), (3.0, 2.0, 4.0))


class TimerTests(unittest.TestCase):
    def test_warmup_is_excluded_and_prepare_is_outside_the_clock(self):
        calls, syncs = [], []

        def prepare():
            calls.append("prepare")
            return len(calls)

        def call(x):
            calls.append("call")

        t = it.timed(prepare, call, warmup=3, repeats=20, sync=lambda: syncs.append(1))
        self.assertEqual(len(t), 20)
        self.assertEqual(calls.count("call"), 23)
        self.assertEqual(len(syncs), 40)                 # one before and one after every timed call
        self.assertTrue((t >= 0).all())


class ModelTests(unittest.TestCase):
    def test_energy_only_matches_energy_forces_and_restores_the_switch(self):
        _, _, union = fragments()
        for variant in ("dense", "global", "local6"):
            model = build(variant)
            data = batch_of(generic(union))
            full = model(data.clone())
            with it.forces_off(model.backbone), torch.no_grad():
                energy = model(data.clone())
            self.assertNotIn("omol_forces", energy, variant)
            self.assertFalse(energy["omol_energy"]["energy"].requires_grad, variant)
            torch.testing.assert_close(energy["omol_energy"]["energy"], full["omol_energy"]["energy"].detach(),
                                       rtol=0, atol=1e-10, msg=variant)
            self.assertTrue(model.backbone.regress_config.forces, variant)
            self.assertIn("omol_forces", model(data.clone()), variant)

    def test_parameter_counts_match_the_configured_model(self):
        # the small test models: Dense has no MOLE layer; Global and Local-6 share parameters and keep 1 of 8 slices
        dense, glob, local = (it.param_counts(build(v)) for v in ("dense", "global", "local6"))
        self.assertEqual(dense[0], dense[1])
        self.assertEqual(glob, local)
        from fairchem.core.models.uma.nn.mole import MOLE
        bank = sum(m.weights.numel() for m in build("global").modules() if isinstance(m, MOLE))
        self.assertEqual(glob[0] - glob[1], bank * 7 // 8)


if __name__ == "__main__":
    unittest.main()
