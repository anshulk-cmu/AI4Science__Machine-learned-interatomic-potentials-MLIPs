"""Final statistics: input integrity, exact bootstrap weights and CLI coverage."""
import contextlib
import io
import itertools
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_statistics as ps

METRIC = "force_mae_eV_per_A"


def fixture(n=5):
    rows = []
    for i, model in enumerate(ps.MODELS):
        base = 10 if model == "dense" else 8 if model == "global" else 8.1 + i / 100
        for seed in range(n):
            rows.append({"variant": model, "seed": seed,
                         METRIC: base + seed / 100,
                         "energy_mae_per_atom_eV": (base + seed / 100) / 2})
    return pd.DataFrame(rows)


class FinalStatistics(unittest.TestCase):
    def test_final_grid_and_alias(self):
        df = fixture().replace({"variant": {"frozen": "global_frozen"}})
        checked = ps.validate_data(df, require_final=True)
        self.assertEqual(set(checked.variant), set(ps.MODELS))
        self.assertEqual(len(checked), 40)

    def test_bad_inputs_fail(self):
        df = fixture()
        cases = [df.drop(columns=METRIC), df.iloc[1:], pd.concat([df, df.iloc[:1]]),
                 df[df.variant != "global"], df.replace({"variant": {"element": "typo"}}),
                 df.assign(seed=0), df.iloc[:0]]
        for value in (float("nan"), float("inf"), -1, "bad"):
            bad = df.copy()
            bad[METRIC] = bad[METRIC].astype(object)
            bad.loc[0, METRIC] = value
            cases.append(bad)
        cases.append(df.assign(seed=df.seed + .5))
        cases.append(df.assign(seed=df.seed.where(df.index != 0, 99)))
        for bad in cases:
            with self.subTest(data=bad.head().to_dict()), self.assertRaises(ValueError):
                ps.validate_data(bad)

    def test_final_mode_rejects_partial_grid_and_wrong_seed_ids(self):
        for df in (fixture(3), fixture().query("variant != 'element'"), fixture().assign(seed=lambda x: x.seed + 1)):
            with self.assertRaisesRegex(ValueError, "all 8 variants"):
                ps.validate_data(df, require_final=True)

    def test_equal_counts_with_different_seeds_fail(self):
        df = fixture()
        df.loc[(df.variant == "element") & (df.seed == 4), "seed"] = 5
        with self.assertRaisesRegex(ValueError, "Seed set"):
            ps.validate_data(df)

    def test_interval_uses_observed_seed_count(self):
        x = np.array([1., 2., 4., 5., 8.])
        half = stats.t.ppf(.975, 4) * x.std(ddof=1) / np.sqrt(5)
        np.testing.assert_allclose(ps.interval(x), [x.mean() - half, x.mean() + half])

    def test_compressed_bootstrap_multiplicities(self):
        means, weights = ps.bootstrap_means(np.arange(5, dtype=float))
        self.assertEqual(len(means), 126)
        self.assertEqual(weights.sum(), 5 ** 5)
        self.assertEqual(weights.max(), 120)
        self.assertEqual(int(weights.sum()) ** 3, 30_517_578_125)

    def test_weighted_percentile_matches_expansion_and_interpolation(self):
        x = np.array([8., 1., 3., 3.])
        weights = np.array([3, 1, 9, 2])
        for q in (0, 5, 50, 95, 100):
            self.assertAlmostEqual(ps.weighted_percentile(x, weights, q), np.percentile(np.repeat(x, weights), q))
        self.assertNotEqual(ps.weighted_percentile(x, weights), np.percentile(x, 5))

    def test_exact_matches_brute_force_including_conditioning(self):
        for n in (2, 3):
            d = np.array([1., 4., 7.][:n])
            g = np.array([2., 3., 6.][:n])
            x = np.array([1., 2., 5.][:n])
            selections = list(itertools.product(range(n), repeat=n))
            means = [np.array([v[list(c)].mean() for c in selections]) for v in (d, g, x)]
            a, b, c = np.meshgrid(*means, indexing="ij")
            ok = a > b
            expected = np.percentile(ps.retention(a[ok], b[ok], c[ok]), 5)
            result, fraction = ps.exact_bound(d, g, x)
            self.assertAlmostEqual(result, expected, places=12)
            self.assertAlmostEqual(fraction, ok.mean(), places=12)

    def test_five_seed_t_draws_match_df_four(self):
        df = fixture()
        draws, seed = 1000, 31
        rng = np.random.default_rng(seed)
        v = {m: ps.values(df, m, METRIC) for m in ("dense", "global", "local6")}
        for x in v.values():
            rng.normal(x.mean(), x.std(ddof=1) / np.sqrt(5), draws)
        samples = {m: x.mean() + x.std(ddof=1) / np.sqrt(5) * rng.standard_t(4, draws) for m, x in v.items()}
        ok = samples['dense'] > samples['global']
        expected = np.percentile(ps.retention(samples['dense'][ok], samples['global'][ok], samples['local6'][ok]), 5)
        actual = ps.bounds(df, METRIC, "local6", draws, seed, skip_exact=True)
        self.assertEqual(actual['df'], 4)
        self.assertNotIn('t2', actual)
        self.assertEqual(actual['t'], expected)

    def test_row_order_does_not_change_bounds(self):
        df = fixture(3)
        self.assertEqual(ps.bounds(df, METRIC, "element", 1000),
                         ps.bounds(df.sample(frac=1, random_state=4), METRIC, "element", 1000))

    def test_no_positive_global_gain_is_unavailable_without_divide_warning(self):
        df = fixture(3)
        df[METRIC] = 1.
        with np.errstate(all="raise"):
            result = ps.bounds(df, METRIC, "local6", 1000)
        for method in ("normal", "t", "exact"):
            self.assertTrue(np.isnan(result[method]))
            self.assertEqual(result['accepted'][method], 0)

    def test_exact_limit_and_explicit_skip(self):
        with self.assertRaisesRegex(ValueError, "skip-exact"):
            ps.bounds(fixture(6), METRIC, "element", 1000)
        result = ps.bounds(fixture(6), METRIC, "element", 1000, skip_exact=True)
        self.assertNotIn("exact", result)
        self.assertEqual(result['df'], 5)

    def test_bad_draw_count_and_baseline_model(self):
        for draws in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                ps.bounds(fixture(), METRIC, "element", draws)
        with self.assertRaises(ValueError):
            ps.bounds(fixture(), METRIC, "dense", 1000)

    def run_cli(self, df, *args):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            df.to_csv(path, index=False)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                ps.main([str(path), "--draws", "1000", *args])
            return output.getvalue()

    def test_cli_all_eight_variants_and_five_seeds(self):
        text = self.run_cli(fixture().sample(frac=1, random_state=8), "--require-final")
        self.assertIn("Variants: 8; seeds per variant: 5", text)
        self.assertEqual(text.count("Welch "), 57)  # header plus 28 pairs per metric
        self.assertEqual(text.count("retention of the Global gain"), 12)
        self.assertEqual(text.count("t(4)"), 12)
        self.assertEqual(text.count("accepted Dense > Global"), 12)
        self.assertEqual(text.count("PASS (normal lower bound >= 0.5)"), 2)
        self.assertNotIn("nan", text)
        for model in ps.MODELS:
            self.assertIn(model, text)
        self.assertIn("per seed 0:", text)

    def test_cli_reports_undefined_gain(self):
        df = fixture(3)
        for metric in ps.METRICS:
            df[metric] = 1.
        text = self.run_cli(df, "--skip-exact")
        self.assertIn("0:undefined", text)
        self.assertEqual(text.count("UNAVAILABLE (normal lower bound >= 0.5)"), 2)
        self.assertNotIn("PASS", text)

    def test_cli_validation_happens_before_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.run_cli(fixture(3), "--require-final")
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
