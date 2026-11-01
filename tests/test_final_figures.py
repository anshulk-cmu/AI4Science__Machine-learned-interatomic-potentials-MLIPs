"""Check plotted values, uncertainty, coverage and figure command entry points."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figure_common as fc
import make_accuracy_figure as accuracy
import make_ensemble_figure as ensemble


def metrics(n=5):
    return pd.DataFrame([
        {"variant": model, "seed": seed, "energy_mae_per_atom_eV": .014 - i * .0001 + seed * .00002,
         "force_mae_eV_per_A": .04 - i * .0002 + seed * .00004}
        for i, model in enumerate(fc.STYLE) for seed in range(n)])


def curves(fin, steps=(5000, 25000, 50000)):
    return pd.DataFrame([
        {"variant": row.variant, "seed": row.seed, "step": step,
         accuracy.CURVE_FORCE: row.force_mae_eV_per_A + .1 * (1 - step / max(steps)),
         accuracy.TRAIN: .1 + .3 * (1 - step / max(steps))}
        for row in fin.itertuples() for step in steps])


def ensemble_metrics(n=5):
    return pd.concat([metrics(n).assign(method=m, eval_split="report") for m in ensemble.METHODS], ignore_index=True)


class FinalFigures(unittest.TestCase):
    def tearDown(self):
        fc.plt.close("all")

    def test_seed_counts_offsets_and_intervals(self):
        for n in (3, 5):
            y = np.arange(n, dtype=float)
            mean, half = fc.mean_interval(y)
            self.assertEqual(mean, y.mean())
            self.assertAlmostEqual(half, stats.t.ppf(.975, n - 1) * y.std(ddof=1) / np.sqrt(n))
            self.assertEqual(len(fc.offsets(n)), n)
            self.assertAlmostEqual(fc.offsets(n).mean(), 0)
        self.assertIsNone(fc.mean_interval([1., 2.])[1])
        self.assertIsNone(fc.mean_interval([1.])[1])
        self.assertIsNone(fc.mean_interval(np.arange(5.), descriptive=True)[1])

    def test_all_variants_have_unique_colors_and_markers(self):
        self.assertEqual(list(fc.STYLE), fc.MODELS)
        self.assertEqual(len({v[1] for v in fc.STYLE.values()}), 8)
        self.assertEqual(len({v[2] for v in fc.STYLE.values()}), 8)

    def test_final_points_and_means_match_shuffled_csv(self):
        df = metrics().sample(frac=1, random_state=1)
        fig = accuracy.build_figure(df)
        for ax, metric in zip(fig.axes, fc.METRICS):
            scatters = [c for c in ax.collections if hasattr(c, "get_sizes")]
            self.assertEqual(len(scatters), 8)
            for i, (scatter, model) in enumerate(zip(scatters, fc.STYLE)):
                expected = df[df.variant == model].sort_values("seed")[metric].to_numpy()
                np.testing.assert_allclose(scatter.get_offsets()[:, 1], expected)
                np.testing.assert_allclose(scatter.get_offsets()[:, 0], i + fc.offsets(5))
                container = ax.containers[i]
                self.assertAlmostEqual(container.lines[0].get_ydata()[0], expected.mean())
                segment = container.lines[2][0].get_segments()[0]
                _, half = fc.mean_interval(expected)
                np.testing.assert_allclose(segment[:, 1], expected.mean() + np.array([-half, half]))

    def test_curve_means_ranges_and_baseline_gain(self):
        fin = metrics()
        cur = curves(fin)
        fig = accuracy.build_figure(fin, cur)
        self.assertEqual(len(fig.legends[0].texts), 8)
        self.assertEqual(len(fig.axes[0].lines), 8)
        for line, model in zip(fig.axes[1].lines, fc.STYLE):
            expected = cur[cur.variant == model].groupby("step")[accuracy.CURVE_FORCE].mean()
            np.testing.assert_allclose(line.get_ydata(), expected)
        vertices = fig.axes[1].collections[0].get_paths()[0].vertices
        expected = cur[cur.variant == "dense"][accuracy.CURVE_FORCE]
        self.assertAlmostEqual(vertices[:, 1].min(), expected.min())
        self.assertAlmostEqual(vertices[:, 1].max(), expected.max())
        gain = fig.axes[2].lines[3]  # Local-6
        dense = cur[cur.variant == "dense"].groupby("step")[accuracy.CURVE_FORCE].mean()
        local = cur[cur.variant == "local6"].groupby("step")[accuracy.CURVE_FORCE].mean()
        np.testing.assert_allclose(gain.get_ydata(), ((dense - local) * 1000).loc[gain.get_xdata()])
        self.assertIn("50,000", fig.axes[3].get_title())

    def test_long_run_axes_and_no_confidence_intervals(self):
        fin = metrics(2)
        fig = accuracy.build_figure(fin, curves(fin, (50000, 150000, 300000)), long_run=True)
        self.assertGreater(fig.axes[0].get_xlim()[1], 300000)
        self.assertIn("300,000", fig.axes[3].get_title())
        self.assertTrue(all(not c.has_yerr for c in fig.axes[3].containers))
        self.assertNotIn("95%", " ".join(t.get_text() for t in fig.texts))

    def test_missing_duplicate_and_nonfinite_metrics_rejected(self):
        df = metrics()
        bads = [df.iloc[1:], pd.concat([df, df.iloc[:1]]), df.assign(force_mae_eV_per_A=np.nan),
                df.assign(seed=1.5), df.replace({"variant": {"element": "unknown"}})]
        for bad in bads:
            with self.assertRaises(ValueError):
                accuracy.build_figure(bad)

    def test_incomplete_or_mismatched_curves_rejected(self):
        fin = metrics()
        cur = curves(fin)
        bads = [cur.iloc[1:], pd.concat([cur, cur.iloc[:1]]), cur[cur.variant != "element"],
                cur.assign(val_force_mae_eV_per_A=0), cur.assign(step=1.5)]
        for bad in bads:
            with self.assertRaises(ValueError):
                accuracy.build_figure(fin, bad)
        with self.assertRaisesRegex(ValueError, "last curve"):
            accuracy.build_figure(fin.assign(step=300000), cur)

    def test_strict_final_grid_and_alias(self):
        checked = fc.validate_metrics(metrics().replace({"variant": {"frozen": "global_frozen"}}), True)
        self.assertIn("frozen", set(checked.variant))
        with self.assertRaisesRegex(ValueError, "8 variants"):
            fc.validate_metrics(metrics(3), True)

    def test_metrics_only_step_labels_for_unequal_budgets(self):
        df = metrics(2)
        df["step"] = np.where(df.variant == "element", 200000, 300000)
        fig = accuracy.build_figure(df, long_run=True)
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        self.assertIn("200,000", labels[2])
        self.assertIn("300,000", labels[0])

    def test_ensemble_methods_and_points(self):
        fig = ensemble.build_figure(ensemble_metrics(2))
        self.assertEqual(len(fig.legends[0].texts), 4)
        for ax in fig.axes:
            scatters = [c for c in ax.collections if hasattr(c, "get_sizes")]
            self.assertEqual(len(scatters), 32)
            self.assertTrue(all(len(c.get_offsets()) == 2 for c in scatters))
            self.assertTrue(all(not c.has_yerr for c in ax.containers))

    def test_ensemble_requires_matching_methods_and_report_set(self):
        df = ensemble_metrics()
        bads = [df.iloc[1:], df[df.method != "best"], df[df.method.isin(["best", "final"])],
                df.assign(eval_split="selection"), df.assign(method="average"),
                df.drop(columns="eval_split"), pd.concat([df, df.iloc[:1]])]
        for bad in bads:
            with self.assertRaises(ValueError):
                ensemble.validate_ensemble(bad)

    def test_one_ensemble_method_and_variant_supported(self):
        df = ensemble_metrics(3)
        df = df[(df.variant == "element") & (df.method != "prediction_average")]
        fig = ensemble.build_figure(df)
        self.assertEqual(len(fig.legends[0].texts), 3)
        self.assertTrue(all(c.has_yerr for c in fig.axes[0].containers))

    def test_cli_legacy_final_long_and_ensemble_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            fin, cur, ens = out / "metrics.csv", out / "curves.csv", out / "ensemble.csv"
            metrics().to_csv(fin, index=False)
            curves(metrics()).to_csv(cur, index=False)
            ensemble_metrics(2).to_csv(ens, index=False)
            commands = [
                ["make_accuracy_figure.py", str(out / "legacy.pdf")],
                ["make_accuracy_figure.py", str(out / "final.png"), "--metrics", str(fin), "--curves", str(cur), "--require-final"],
                ["make_accuracy_figure.py", str(out / "long.svg"), "--metrics", str(fin), "--metrics-only", "--long-run"],
                ["make_ensemble_figure.py", str(ens), str(out / "ensemble.pdf")],
            ]
            for script, *args in commands:
                result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script), *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("Warning", result.stderr)
            for filename in ("legacy.pdf", "final.png", "long.svg", "ensemble.pdf"):
                self.assertGreater((out / filename).stat().st_size, 10000)

    def test_cli_avoids_silent_pilot_curve_fallback(self):
        with self.assertRaisesRegex(ValueError, "explicit --metrics"):
            accuracy.main("unused.png", curves="custom.csv")
        with self.assertRaisesRegex(ValueError, "Specify --curves"):
            accuracy.main("unused.png", metrics="custom.csv")
        with self.assertRaisesRegex(ValueError, "explicit --metrics"):
            accuracy.main("unused.png", long_run=True, metrics_only=True)

    def test_input_files_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            metrics().to_csv(path, index=False)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                accuracy.main(path, metrics=path, metrics_only=True)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
