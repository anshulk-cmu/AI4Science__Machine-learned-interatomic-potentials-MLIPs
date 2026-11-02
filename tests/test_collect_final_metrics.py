"""Final-metrics ingestion contracts, using temporary logs and real protobuf records."""
import contextlib
import csv
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_final_metrics as cm  # noqa: E402

ENERGY, FORCE, LOSS = cm.METRICS


def evaluation(energy=0.012345678901, force=0.034567890123, loss=0.056789012345):
    return {ENERGY: energy, FORCE: force, LOSS: loss}


def log_eval(values=None, header=True):
    values = evaluation() if values is None else values
    text = "Finished aggregating metrics: \n" if header else ""
    return text + "".join(f"  {key}: {value}\n" for key, value in values.items())


def write_wandb(path, rows, large=False):
    """Make protocol-buffer records in W&B's v0 on-disk framing, with fragments."""
    from wandb.proto.wandb_internal_pb2 import Record

    path.parent.mkdir(parents=True, exist_ok=True)
    content = bytearray(struct.pack("<4sHB", b":W&B", 0xBEE1, 0))
    for row in rows:
        record = Record()
        for key, value in row.items():
            record.history.item.add(key=key, value_json=json.dumps(value))
        if large:
            record.history.item.add(key="unused_padding", value_json=json.dumps("x" * 100_000))
        payload = record.SerializeToString()
        first = True
        while payload:
            available = 32768 - len(content) % 32768
            if available < 7:
                content.extend(b"\0" * available)
                available = 32768
            fragment, payload = payload[:available-7], payload[available-7:]
            kind = (1 if first else 4) if not payload else (2 if first else 3)
            checksum = zlib.crc32(bytes([kind]) + fragment) & 0xFFFFFFFF
            content.extend(struct.pack("<IHB", checksum, len(fragment), kind))
            content.extend(fragment)
            first = False
    path.write_bytes(content)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_dir(self, name="run", variant="element", seed=0, steps=200, text=None, data="fixed pilot lists"):
        run = self.root / name
        (run / "logs").mkdir(parents=True)
        (run / "settings.txt").write_text(
            f"variant={variant} steps={steps} seed={seed} data={data} eval_every=200 ckpt_every=200\n",
            encoding="utf-8")
        if text is None:
            text = f"{{'train/loss': 0.1, 'train/step': {steps-1}}}\n" + log_eval()
        (run / "logs" / "train.log").write_text(text, encoding="utf-8")
        return run

    def collect(self, run, **kwargs):
        return cm.collect_run({"path": str(run)}, 200, **kwargs)

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cm.main(list(map(str, args)))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_log_scientific_notation_and_last_evaluation_not_best(self):
        text = "{'train/step': 199}\n" + log_eval(evaluation(0.001, 0.001, 0.001))
        text += log_eval({ENERGY: "1.23e-2", FORCE: "3.45E-2", LOSS: "+6.7e-2"})
        row, provenance = self.collect(self.run_dir(text=text))
        self.assertEqual(list(row), cm.COLUMNS)
        self.assertEqual(row["energy_mae_per_atom_eV"], 0.0123)
        self.assertEqual(row["force_mae_eV_per_A"], 0.0345)
        self.assertEqual(provenance["precision"], "rounded_train_log")

    def test_native_sorted_log_block_with_extra_metrics(self):
        text = ("{'train/step': 199}\nFinished aggregating metrics: \n"
                "  val/atoms_per_second: 20.0000\n  val/epoch: 0.0000\n  val/loss: 0.0600\n"
                "  val/omol.val,omol_energy,mae: 1.2345\n"
                f"  {ENERGY}: 0.0120\n  {FORCE}: 0.0340\nTraining Completed 200 steps\n")
        row, _ = self.collect(self.run_dir(text=text))
        self.assertEqual(row["val_loss"], 0.06)

    def test_partial_repeat_is_not_filled_from_previous_evaluation(self):
        for header in (True, False):
            with self.subTest(header=header):
                text = "{'train/step': 199}\n" + log_eval() + log_eval({ENERGY: 0.01}, header=header)
                run = self.run_dir(name=f"partial{header}", text=text)
                with self.assertRaisesRegex(cm.CollectionError, "incomplete final validation"):
                    self.collect(run)

    def test_empty_final_validation_block_is_incomplete(self):
        run = self.run_dir(text="{'train/step': 199}\n" + log_eval() + log_eval({}))
        with self.assertRaisesRegex(cm.CollectionError, "incomplete final validation"):
            self.collect(run)

    def test_early_stop_and_unknown_step_are_rejected(self):
        for i, prefix in enumerate(("{'train/step': 198}\n", "")):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(cm.CollectionError, "expected 200"):
                self.collect(self.run_dir(name=f"early{i}", text=prefix + log_eval()))

    def test_training_after_validation_invalidates_older_metrics(self):
        run = self.run_dir(text="{'train/step': 199}\n" + log_eval() + "{'train/step': 150}\n")
        with self.assertRaisesRegex(cm.CollectionError, "no validation after"):
            self.collect(run)

    def test_resume_in_appended_log_uses_last_attempt(self):
        text = "{'train/step': 199}\n" + log_eval()
        text += "{'train/step': 100}\n{'train/step': 199}\n" + log_eval(evaluation(0.02, 0.04, 0.06))
        row, _ = self.collect(self.run_dir(text=text))
        self.assertEqual(row["energy_mae_per_atom_eV"], 0.02)

    def test_invalid_metrics_are_rejected(self):
        for i, invalid in enumerate(("nan", "inf", "-0.2", "1e", "not_a_number")):
            with self.subTest(invalid=invalid), self.assertRaises(cm.CollectionError):
                self.collect(self.run_dir(name=f"bad{i}", text="{'train/step': 199}\n" +
                                          log_eval(evaluation(force=invalid))))

    def test_failed_training_stage_is_rejected(self):
        run = self.run_dir()
        (run / "summary.txt").write_text("END   train exit 1 (log: /tmp/train.log)\n")
        with self.assertRaisesRegex(cm.CollectionError, "training stage failed"):
            self.collect(run)

    def test_step_budget_and_variant_alias(self):
        row, _ = self.collect(self.run_dir(variant="global_frozen"))
        self.assertEqual(row["variant"], "frozen")
        with self.assertRaisesRegex(cm.CollectionError, "configured for 300000"):
            self.collect(self.run_dir(name="long", steps=300_000))

    def test_resolved_canonical_config_fallback(self):
        import yaml
        run = self.run_dir()
        (run / "settings.txt").unlink()
        config = {"job": {"seed": 3}, "runner": {"max_steps": 200, "train_eval_unit": {"model": {
            "backbone": {"model": "training.local_global_moe.eSCNMDLocalGlobalMoeBackbone"}}}}}
        (run / "canonical_config.yaml").write_text(yaml.safe_dump(config))
        row, p = self.collect(run)
        self.assertEqual((row["variant"], row["seed"]), ("local_global", 3))
        self.assertTrue(p["metadata_source"].endswith("canonical_config.yaml"))

    def test_settings_missing_or_conflicting_identity_fail(self):
        run = self.run_dir()
        (run / "settings.txt").write_text("variant=element seed=0 steps=200 seed=1\n")
        with self.assertRaisesRegex(cm.CollectionError, "conflicting seed"):
            self.collect(run)

    def test_wandb_binary_preserves_precision_and_ignores_internal_step(self):
        rounded = {k: round(v, 4) for k, v in evaluation().items()}
        run = self.run_dir(text="{'train/step': 199}\n" + log_eval(rounded))
        binary = run / "wandb" / "run-test.wandb"
        write_wandb(binary, [{"train/step": 199, "_step": 500}, dict(evaluation(), _step=1000)])
        before = binary.read_bytes()
        row, p = self.collect(run)
        self.assertEqual(row["energy_mae_per_atom_eV"], evaluation()[ENERGY])
        self.assertEqual(p["precision"], "full_precision_history")
        self.assertEqual(binary.read_bytes(), before)

    def test_wandb_fragmented_protobuf_records(self):
        run = self.run_dir()
        write_wandb(run / "run-test.wandb", [{"train/step": 199}, evaluation()], large=True)
        row, _ = self.collect(run)
        self.assertEqual(row["force_mae_eV_per_A"], evaluation()[FORCE])

    def test_corrupt_or_truncated_binary_does_not_silently_fall_back(self):
        for i, truncate in enumerate((False, True)):
            run = self.run_dir(name=f"broken{i}")
            path = run / "test.wandb"
            write_wandb(path, [{"train/step": 199}, evaluation()])
            raw = bytearray(path.read_bytes())
            if truncate:
                raw = raw[:-3]
            else:
                raw[-1] ^= 1
            path.write_bytes(raw)
            with self.subTest(truncate=truncate), self.assertRaisesRegex(cm.CollectionError, "corrupt|truncated"):
                self.collect(run)
            row, p = self.collect(run, source="log")
            self.assertEqual(p["precision"], "rounded_train_log")

    def test_history_internal_step_alone_does_not_prove_training_step(self):
        run = self.run_dir()
        (run / "wandb-history.jsonl").write_text(json.dumps(dict(evaluation(), _step=200)) + "\n")
        with self.assertRaisesRegex(cm.CollectionError, "expected 200"):
            self.collect(run, source="wandb")

    def test_history_can_use_explicit_completed_validation_step(self):
        run = self.run_dir()
        (run / "wandb-history.jsonl").write_text(json.dumps(dict(evaluation(), **{"val/step": 200})) + "\n")
        self.assertEqual(self.collect(run)[0]["val_loss"], evaluation()[LOSS])

    def test_history_and_log_disagreement_is_reported(self):
        run = self.run_dir()
        write_wandb(run / "test.wandb", [{"train/step": 199}, evaluation(force=0.5)])
        with self.assertRaisesRegex(cm.CollectionError, "disagree"):
            self.collect(run)

    def test_summary_json_is_not_used_as_validation_history(self):
        run = self.run_dir()
        (run / "logs" / "train.log").unlink()
        (run / "wandb-summary.json").write_text(json.dumps(evaluation()))
        with self.assertRaisesRegex(cm.CollectionError, "no usable metric source"):
            self.collect(run)

    def test_manifest_orders_resume_segments_and_rejects_stale_final(self):
        run = self.run_dir()
        first, last = run / "a.wandb", run / "b.wandb"
        write_wandb(first, [{"train/step": 199}, evaluation()])
        write_wandb(last, [{"train/step": 150}, evaluation()])
        with self.assertRaisesRegex(cm.CollectionError, "multiple W&B"):
            self.collect(run)
        manifest = self.root / "runs.json"
        manifest.write_text(json.dumps([{"path": "run", "wandb": ["a.wandb", "b.wandb"]}]))
        out = self.root / "out.csv"
        code, _, err = self.invoke("--run-manifest", manifest, "--output", out, "--steps", 200,
                                   "--source", "wandb", "--allow-partial")
        self.assertEqual(code, 1)
        self.assertIn("expected 200", err)
        self.assertFalse(out.exists())
        write_wandb(last, [{"train/step": 150}, {"train/step": 199}, evaluation(force=0.04)])
        code, _, err = self.invoke("--run-manifest", manifest, "--output", out, "--steps", 200,
                                   "--source", "wandb", "--allow-partial")
        self.assertEqual(code, 0, err)
        with out.open(newline="") as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]["force_mae_eV_per_A"], "0.04")

    def test_explicit_log_segments_must_end_with_completed_validation(self):
        run = self.run_dir()
        (run / "logs" / "resume.log").write_text("{'train/step': 150}\n")
        spec = {"path": str(run), "logs": ["logs/train.log", "logs/resume.log"]}
        with self.assertRaisesRegex(cm.CollectionError, "no validation after"):
            cm.collect_run(spec, 200, "log")

    def test_duplicate_aliases_seeds_and_missing_pairs(self):
        a = self.run_dir(name="a", variant="global_frozen")
        b = self.run_dir(name="b", variant="frozen")
        rows = [self.collect(a), self.collect(b)]
        with self.assertRaisesRegex(cm.CollectionError, "duplicate"):
            cm.validate(rows, list(cm.VARIANTS), 5, True)
        with self.assertRaisesRegex(cm.CollectionError, "missing variant"):
            cm.validate(rows[:1], list(cm.VARIANTS), 5)
        self.assertEqual(len(cm.validate(rows[:1], list(cm.VARIANTS), 5, True)), 39)
        with self.assertRaisesRegex(cm.CollectionError, "unexpected"):
            cm.validate([self.collect(self.run_dir(name="wrongseed", seed=5))], list(cm.VARIANTS), 5, True)

    def test_mixed_data_is_rejected(self):
        rows = [self.collect(self.run_dir(name="a", seed=0)),
                self.collect(self.run_dir(name="b", seed=1, data="different lists"))]
        with self.assertRaisesRegex(cm.CollectionError, "different training/validation"):
            cm.validate(rows, list(cm.VARIANTS), 5, True)

    def test_merge_existing_15_rows_with_25_new_runs_produces_40(self):
        old = ROOT / "results" / "pilot_final_metrics.csv"
        before = hashlib.sha256(old.read_bytes()).hexdigest()
        runs = []
        for variant in cm.VARIANTS:
            seeds = range(5) if variant in ("element", "local12", "local_global") else (3, 4)
            for seed in seeds:
                runs.append(self.run_dir(name=f"{variant}{seed}", variant=variant, seed=seed, steps=50_000))
        out = self.root / "final.csv"
        code, stdout, stderr = self.invoke(*runs, "--merge", old, "--output", out)
        self.assertEqual(code, 0, stderr)
        self.assertIn("Wrote 40 rows", stdout)
        with out.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 40)
        self.assertEqual(list(rows[0]), cm.COLUMNS)
        self.assertEqual(float(rows[0]["energy_mae_per_atom_eV"]), 0.014104768689911168)
        info = json.loads(out.with_suffix(".provenance.json").read_text())
        self.assertTrue(info["complete"])
        self.assertEqual(sum(not row["step_verified"] for row in info["rows"]), 15)
        self.assertEqual(before, hashlib.sha256(old.read_bytes()).hexdigest())
        self.assertEqual(len(cm.merge_csv(out, 50_000)), 40)
        with self.assertRaisesRegex(cm.CollectionError, "cannot mix"):
            cm.merge_csv(out, 300_000)

    def test_missing_runs_fail_without_changing_existing_output(self):
        run = self.run_dir()
        out = self.root / "out.csv"
        out.write_text("previous output\n")
        code, _, err = self.invoke(run, "--output", out, "--steps", 200)
        self.assertEqual(code, 1)
        self.assertIn("missing variant/seed", err)
        self.assertEqual(out.read_text(), "previous output\n")
        self.assertFalse(out.with_suffix(".provenance.json").exists())

    def test_partial_report_records_gaps_and_validates_provenance_hash(self):
        run = self.run_dir()
        out = self.root / "out.csv"
        code, _, err = self.invoke(run, "--output", out, "--steps", 200, "--allow-partial")
        self.assertEqual(code, 0, err)
        info = json.loads(out.with_suffix(".provenance.json").read_text())
        self.assertFalse(info["complete"])
        self.assertEqual(len(info["missing"]), 39)
        out.write_text(out.read_text() + "\n")
        with self.assertRaisesRegex(cm.CollectionError, "provenance does not match"):
            cm.merge_csv(out, 200)

    def test_output_cannot_replace_merged_input(self):
        old = self.root / "old.csv"
        old.write_bytes((ROOT / "results" / "pilot_final_metrics.csv").read_bytes())
        original = old.read_bytes()
        code, _, err = self.invoke("--merge", old, "--output", old, "--allow-partial")
        self.assertEqual(code, 1)
        self.assertIn("overwrite an input", err)
        self.assertEqual(old.read_bytes(), original)

    def test_best_metric_prefixes_are_not_final_validation(self):
        run = self.run_dir(text="{'train/step': 199}\n" + log_eval() + "best_val/loss: 0.0001\n")
        self.assertEqual(self.collect(run)[0]["val_loss"], evaluation()[LOSS])

    def test_history_does_not_hide_incomplete_latest_log_validation(self):
        run = self.run_dir(text="{'train/step': 199}\n" + log_eval() + log_eval({ENERGY: 0.01}))
        write_wandb(run / "test.wandb", [{"train/step": 199}, evaluation()])
        with self.assertRaisesRegex(cm.CollectionError, "incomplete final validation"):
            self.collect(run)

    def test_long_run_seed_count_can_be_configured(self):
        runs = [self.run_dir(name=f"long{seed}", seed=seed, steps=300_000) for seed in (0, 1)]
        out = self.root / "long.csv"
        code, _, err = self.invoke(*runs, "--output", out, "--steps", 300_000,
                                   "--expected-seeds", 2, "--variants", "element")
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out.with_suffix(".provenance.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
