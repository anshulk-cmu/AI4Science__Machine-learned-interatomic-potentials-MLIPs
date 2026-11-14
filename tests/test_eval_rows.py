"""CPU checks of scripts/eval_rows.py: resume after a cut line, shard merge, and the standard errors."""
import json
import math
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_rows as er  # noqa: E402


class EvalRowsTest(unittest.TestCase):
    def test_done_rows_drops_cut_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "s.csv"
            out.write_text(",".join(er.FIELDS) + "\n3,2,1.0,1.5,0.25,0.6,6\n5,4,1.0")
            self.assertEqual(er.done_rows(out), {3})
            self.assertTrue(out.read_text().endswith("0.6,6\n"))
            out.write_text(",".join(er.FIELDS) + "\n3,2,1.0\n5,4,1.0,1.5,0.25,0.6,6\n")
            with self.assertRaises(ValueError):
                er.done_rows(out)

    def test_shards_cover_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = Path(tmp) / "rows.txt"
            rows.write_text("\n".join(map(str, range(10))) + "\n")
            parts = [er.shard_rows(rows, s, 3) for s in range(3)]
            self.assertEqual(sorted(sum(parts, [])), list(range(10)))

    def test_summary_and_merge(self):
        records = [(0.1, 3.0, 6), (0.3, 9.0, 12), (0.2, 3.0, 3)]
        s = er.summarize(records)
        self.assertAlmostEqual(s["energy_mae_per_atom_eV"], 0.2)
        self.assertAlmostEqual(s["energy_se"], 0.1 / math.sqrt(3))
        self.assertAlmostEqual(s["force_mae_eV_per_A"], 15 / 21)
        r = 15 / 21
        resid = [3 - 6 * r, 9 - 12 * r, 3 - 3 * r]
        self.assertAlmostEqual(s["force_se"], math.sqrt(sum(x * x for x in resid) / 6) / 7)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = tmp / "rows.txt"
            rows.write_text("1\n2\n3\n")
            meta = dict(checkpoint_sha256="c", rows_sha256=er.ck.sha256(rows), dtype="float32", num_shards=2, data_rows=9)
            for shard, lines in ((0, [(1, records[0]), (3, records[2])]), (1, [(2, records[1])])):
                path = tmp / f"s{shard}.csv"
                path.write_text(",".join(er.FIELDS) + "\n" + "".join(
                    f"{row},1,0,0,{e},{f},{c}\n" for row, (e, f, c) in lines))
                path.with_suffix(".json").write_text(json.dumps(dict(meta, shard=shard)))
            table = tmp / "t.csv"
            args = Namespace(rows=rows, label="dense", eval_split="x", table=table, shards=[tmp / "s0.csv", tmp / "s1.csv"])
            er.merge(args)
            self.assertIn("dense,x,float32,3,", table.read_text())
            with self.assertRaises(ValueError):          # a missing shard
                er.merge(Namespace(**{**vars(args), "shards": [tmp / "s0.csv"], "table": None}))
            other = tmp / "o0.csv"                       # model A: every error 0.1 higher on one shard layout
            other.write_text(",".join(er.FIELDS) + "
" + "".join(
                f"{row},1,0,0,{e + 0.1},{f + c * 0.1},{c}
" for row, (e, f, c) in zip((1, 2, 3), records)))
            other.with_suffix(".json").write_text(json.dumps(dict(meta, num_shards=1, shard=0)))
            d = er.paired(Namespace(rows=rows, a=[other], b=args.shards))
            self.assertAlmostEqual(d["energy_diff_eV_per_atom"], 0.1)
            self.assertAlmostEqual(d["force_diff_eV_per_A"], 0.1)
            self.assertAlmostEqual(d["force_diff_se"], 0.0)
            with self.assertRaises(ValueError):          # a duplicated shard
                er.merge(Namespace(**{**vars(args), "shards": [tmp / "s0.csv", tmp / "s0.csv"], "table": None}))


if __name__ == "__main__":
    unittest.main()
