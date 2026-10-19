"""scripts/router_check.py: the fixed-router ablation, the router-only pass and the statistics, on tiny float64 models."""

import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "tests", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_local_backbone import DEVICE, build, fragments, generic, run  # noqa: E402
import router_check as rc  # noqa: E402

UNIFORM = np.full(rc.K, 1.0 / rc.K + 0.005)


def batch_of(*structures):
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    items = []
    for atoms in structures:
        atoms = atoms.copy()
        atoms.info.update(charge=0, spin=1)
        items.append(AtomicData.from_ase(atoms, task_name="omol", r_edges=False, r_data_keys=["spin", "charge"],
                                         r_forces=False, r_stress=False, molecule_cell_size=120.0,
                                         target_dtype=torch.float64))
    return atomicdata_list_to_batch(items).to(DEVICE)


class RouterCheckTests(unittest.TestCase):
    def test_uniform_router_equals_one_averaged_expert(self):
        # alpha = 0.13 for every expert must equal the trained router on a copy whose experts all hold
        # 0.13 * sum_k W_k / 1.04 (the trained alpha always sums to 1.04)
        _, _, union = fragments()
        union = generic(union)
        for variant in ("global", "local6"):
            model = build(variant)
            with rc.fixed_alpha(model.backbone, UNIFORM):
                e1, f1 = run(model, union)
            averaged = copy.deepcopy(model)
            for m in averaged.modules():
                if hasattr(m, "weights") and hasattr(m, "global_mole_tensors"):
                    m.weights.data[:] = (0.13 * m.weights.data.sum(0) / 1.04)[None]
            e2, f2 = run(averaged, union)
            torch.testing.assert_close(e1, e2, rtol=0, atol=1e-10, msg=variant)
            torch.testing.assert_close(f1, f2, rtol=0, atol=1e-10, msg=variant)
            e3, _ = run(model, union)               # the router is restored after the context
            self.assertGreater(abs(e3.item() - e1.item()), 1e-8, variant)

    def test_route_equals_the_coefficients_of_a_forward_pass(self):
        a, b, union = fragments()
        for variant in ("global", "local6"):
            model = build(variant)
            record = []
            with rc.fixed_alpha(model.backbone, None, record):
                model(batch_of(a, union, b))
            with torch.no_grad():
                routed = rc.route(model.backbone, batch_of(a, union, b))
            expected = 3 if variant == "global" else len(a) + len(union) + len(b)
            self.assertEqual(routed.shape, (expected, rc.K))
            torch.testing.assert_close(routed, record[0], rtol=0, atol=1e-12, msg=variant)
            torch.testing.assert_close(routed.sum(1), torch.full_like(routed[:, 0], 1.04), rtol=0, atol=1e-12)

    def test_router_statistics(self):
        p = np.vstack([np.full(rc.K, 1.0 / rc.K), np.eye(rc.K)[0]])
        s = rc.unit_stats(p)
        np.testing.assert_allclose(s["entropy"], [math.log(rc.K), 0.0], atol=1e-12)
        np.testing.assert_allclose(s["kl"], [0.0, math.log(rc.K)], atol=1e-12)
        np.testing.assert_allclose(s["l1"], [0.0, 2 * (1 - 1.0 / rc.K)], atol=1e-12)
        # two structures of two atoms: structure 0 holds the uniform p twice, structure 1 two different one-hots
        p4 = np.vstack([p[0], p[0], np.eye(rc.K)[0], np.eye(rc.K)[1]])
        out = rc.router_summary(p4, np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(out["within_l1_mean"], (0 + 0 + 1 + 1) / 4)
        self.assertAlmostEqual(out["within_std_mean"], (0 + 2 * 0.5 / rc.K) / 2)

    def test_expert_statistics(self):
        rng = np.random.default_rng(0)
        same = np.repeat(rng.normal(size=(1, 5, 4)), rc.K, axis=0)
        out = rc.expert_stats(same)
        self.assertAlmostEqual(out["cos_mean"], 1.0)
        self.assertAlmostEqual(out["spread"], 0.0)
        w0 = rng.normal(size=(rc.K, 5, 4))
        out = rc.expert_stats(w0 + same, w0)          # identical updates on different starts
        self.assertAlmostEqual(out["upd_cos_min"], 1.0)
        self.assertAlmostEqual(out["upd_spread"], 0.0)
        self.assertAlmostEqual(out["dev_over_dev0"], 1.0)


if __name__ == "__main__":
    unittest.main()
