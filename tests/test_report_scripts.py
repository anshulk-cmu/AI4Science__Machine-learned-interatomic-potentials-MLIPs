"""Tests of the report scripts: the pilot statistics, the audit summary table and the figures."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_statistics as ps  # noqa: E402


class PilotStatistics(unittest.TestCase):
    df = pd.read_csv(ROOT / "results" / "pilot_final_metrics.csv")

    def test_inputs_are_complete(self):
        counts = self.df.groupby("variant").seed.apply(list)
        for model in ("dense", "global", "local6", "frozen"):
            self.assertEqual(counts[model], [0, 1, 2])

    def test_means_and_retention_match_the_report(self):
        force = lambda m: ps.values(self.df, m, "force_mae_eV_per_A")  # noqa: E731
        self.assertAlmostEqual(force("dense").mean(), 0.038795, places=6)
        self.assertAlmostEqual(ps.retention(force("dense").mean(), force("global").mean(), force("local6").mean()), 1.0075, places=4)
        self.assertAlmostEqual(ps.retention(force("dense").mean(), force("global").mean(), force("frozen").mean()), 0.8422, places=4)

    def test_lower_bounds_match_the_report(self):
        b = ps.bounds(self.df, "force_mae_eV_per_A", "local6", 200_000)
        self.assertAlmostEqual(b["normal"], 0.837, delta=0.005)
        self.assertAlmostEqual(b["exact"], 0.859, delta=0.001)
        self.assertTrue(0.65 < b["t2"] < 0.72)
        e = ps.bounds(self.df, "energy_mae_per_atom_eV", "local6", 200_000)
        self.assertAlmostEqual(e["exact"], 0.763, delta=0.001)


class ReportFigures(unittest.TestCase):
    def test_figures_and_table_are_made_from_the_result_files(self):
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_midterm_figures.py"), str(ROOT / "results"), out],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in ("audit_combined.png", "size_consistency_training.png", "router_check.png"):
                self.assertGreater((Path(out) / name).stat().st_size, 10_000)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_summary_table.py"), str(ROOT / "results" / "summary.csv")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(r.stdout.strip().splitlines()), 12)


if __name__ == "__main__":
    unittest.main()
