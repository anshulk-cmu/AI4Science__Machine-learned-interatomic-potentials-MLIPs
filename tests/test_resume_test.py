"""Resume test tooling on CPU: the composed overrides, the batch probe, fairchem's choice of the checkpoint to resume from,
and the comparison script (docs/resume-test.md)."""

import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_training_config import compose  # noqa: E402  (also stubs fcntl on Windows)

import resume_compare as rc  # noqa: E402
from training import resume_probe as rp  # noqa: E402
from fairchem.core.units.mlip_unit.mlip_unit import MLIPTrainEvalUnit  # noqa: E402


def batch(order=(0, 1)):
    sids = ["a", "b"]
    pos = torch.arange(15, dtype=torch.float32).view(5, 3)
    parts = {0: (sids[0], pos[:2], [1, 8]), 1: (sids[1], pos[2:], [6, 1, 1])}
    picked = [parts[i] for i in order]
    return {"sid": [p[0] for p in picked], "natoms": torch.tensor([len(p[2]) for p in picked]),
            "atomic_numbers": torch.tensor([z for p in picked for z in p[2]]), "pos": torch.cat([p[1] for p in picked])}


class OverrideTests(unittest.TestCase):
    def test_fixed_run_folder_probe_unit_and_schedule(self):
        cfg = compose("dense", "+job.timestamp_id=B", "training.steps=400", "training.ckpt_every=100",
                      "training.eval_every=1000000", "runner.train_eval_unit._target_=training.resume_probe.ProbeTrainEvalUnit",
                      "+runner.train_eval_unit.batch_log=/x/probe/seg_1.csv")
        self.assertEqual(cfg.job.timestamp_id, "B")
        self.assertEqual(Path(cfg.job.metadata.checkpoint_dir).parts[-2:], ("B", "checkpoints"))
        self.assertIsNone(cfg.job.runner_state_path)          # so TrainEvalRunner.load_state(None) searches that folder
        unit = cfg.runner.train_eval_unit
        self.assertEqual((unit._target_, unit.batch_log), ("training.resume_probe.ProbeTrainEvalUnit", "/x/probe/seg_1.csv"))
        self.assertEqual((cfg.runner.max_steps, unit.cosine_lr_scheduler_fn.steps), (400, 400))
        self.assertTrue(issubclass(rp.ProbeTrainEvalUnit, rp.GuardedTrainEvalUnit))

    def test_without_a_fixed_id_every_launch_gets_a_new_folder(self):
        a, b = compose("dense").job.timestamp_id, (time.sleep(1.1), compose("dense"))[1].job.timestamp_id
        self.assertNotEqual(a, b)


class ProbeTests(unittest.TestCase):
    def test_hash_depends_on_content_and_order(self):
        self.assertEqual(rp.batch_hash(batch()), rp.batch_hash(batch()))
        self.assertNotEqual(rp.batch_hash(batch()), rp.batch_hash(batch((1, 0))))
        moved = batch()
        moved["pos"] = moved["pos"] + 1e-3
        self.assertNotEqual(rp.batch_hash(batch()), rp.batch_hash(moved))

    def test_train_step_writes_one_row(self):
        from torchtnt.framework.unit import AppStateMixin

        with tempfile.TemporaryDirectory() as tmp:
            unit = object.__new__(rp.ProbeTrainEvalUnit)
            AppStateMixin.__init__(unit)
            unit.gc_every, unit.max_oom_retries = 0, 0
            unit.oom_events, unit.oom_recovered = {"train": 0, "eval": 0}, {"train": 0, "eval": 0}
            unit.optimizer = types.SimpleNamespace(param_groups=[{"lr": 4e-5}], zero_grad=lambda **k: None)
            unit.logger, unit.last_grad_norm = None, 2.5
            unit.train_progress = types.SimpleNamespace(num_steps_completed=7, num_epochs_completed=1, num_steps_completed_in_epoch=3)
            unit.batch_log = Path(tmp) / "seg_1.csv"
            unit.batch_log.write_text(rp.COLUMNS + "\n", encoding="utf-8")

            def fake_step(self, state, data):
                self.last_loss = 0.123456789012345

            with mock.patch.object(MLIPTrainEvalUnit, "train_step", fake_step):
                unit.train_step(None, batch())
            rows = rc.read_segments(unit.batch_log)[0]
        self.assertEqual(list(rows), [7])
        r = rows[7]
        self.assertEqual((r["epoch"], r["step_in_epoch"], r["structures"], r["atoms"]), (1, 3, 2, 5))
        self.assertEqual((r["loss"], r["lr"], r["batch_sha256"]), (0.123456789012345, 4e-5, rp.batch_hash(batch())))


class ViableCheckpointTests(unittest.TestCase):
    def test_half_written_folder_is_skipped(self):
        from fairchem.core.components.train.train_runner import get_most_recent_viable_checkpoint_path

        with tempfile.TemporaryDirectory() as tmp:
            for i, name in enumerate(("step_100", "step_200", "step_300")):
                d = Path(tmp) / name
                d.mkdir()
                (d / "__0_0.distcp").write_bytes(b"x")
                if name != "step_300":
                    (d / ".metadata").write_bytes(b"m")
                os.utime(d, (1000 + i, 1000 + i))
            self.assertEqual(Path(get_most_recent_viable_checkpoint_path(tmp)).name, "step_200")


def write_run(root, name, segments, loss_noise=0.0, weight=1.0, mae=0.04, hash_bad_at=None):
    """A synthetic run folder: segments is a list of (job id, first step, last step)."""
    run = Path(root) / name
    (run / "probe").mkdir(parents=True)
    for job, first, last in segments:
        lines = [rp.COLUMNS]
        for s in range(first, last + 1):
            digest = "bad" if s == hash_bad_at else f"h{s}"
            lines.append(f"{s},0,{s},8,400,{digest},{1.0 + 0.001 * s + loss_noise},{1e-4 * (s + 1)},1.0")
        (run / "probe" / f"seg_{job}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        events = {"event": "eval", "step": last + 1, "val/loss": 0.06, "val/omol.val,omol_forces,mae": mae}
        (run / "probe" / f"seg_{job}.events.jsonl").write_text(json.dumps(events) + "\n", encoding="utf-8")
    final = run / "train" / name / "checkpoints" / "final"
    final.mkdir(parents=True)
    state = {"w": torch.full((4,), weight)}
    torch.save(types.SimpleNamespace(model_state_dict=state, ema_state_dict=state), final / "inference_ckpt.pt")
    return run


class CompareTests(unittest.TestCase):
    def test_later_segment_overrides_from_its_first_step(self):
        merged, starts = rc.merge_segments([{0: "a0", 1: "a1", 2: "a2"}, {1: "b1", 2: "b2", 3: "b3"}])
        self.assertEqual((merged, starts), ({0: "a0", 1: "b1", 2: "b2", 3: "b3"}, [0, 1]))

    def test_line_cut_by_a_kill_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "seg_1.csv"
            p.write_text(rp.COLUMNS + "\n0,0,0,8,400,h0,1.0,1e-4,1.0\n1,0,1,8,4", encoding="utf-8")
            self.assertEqual([list(s) for s in rc.read_segments(p)], [[0]])

    def test_two_segments_in_one_file_are_split_where_the_step_goes_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "seg_1.csv"
            rows = [f"{s},0,{s},8,400,h{s},1.0,1e-4,1.0" for s in (0, 1, 2, 3, 2, 3, 4)]
            p.write_text("\n".join([rp.COLUMNS, *rows]) + "\n", encoding="utf-8")
            self.assertEqual([list(s) for s in rc.read_segments(p)], [[0, 1, 2, 3], [2, 3, 4]])

    def checks(self, **test_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            refs = [rc.load_run(write_run(tmp, "A1", [(1, 0, 9)])),
                    rc.load_run(write_run(tmp, "A2", [(2, 0, 9)], loss_noise=1e-3, weight=1.01, mae=0.0401))]
            segments = test_kwargs.pop("segments", [(1, 0, 7), (2, 5, 9)])
            test = rc.load_run(write_run(tmp, "B", segments, **test_kwargs))
            return {c["check"]: c["pass"] for c in rc.compare(refs, test, 10, list(range(test["starts"][-1], 10)))}

    def test_resumed_run_within_the_noise_passes(self):
        result = self.checks(loss_noise=5e-4, weight=1.005, mae=0.04005)
        self.assertTrue(all(result.values()), result)

    def test_wrong_batch_after_resume_fails(self):
        result = self.checks(hash_bad_at=6)
        self.assertFalse(result["batch_sha256 equal to every reference at every step"])

    def test_drift_beyond_three_times_the_noise_fails(self):
        result = self.checks(weight=1.2)
        self.assertFalse(result["final model_state_dict, relative L2 difference"])

    def test_run_that_was_never_stopped_fails(self):
        result = self.checks(segments=[(1, 0, 9)])
        self.assertFalse(result["run was resumed (two or more segments)"])

    def test_missing_step_fails(self):
        result = self.checks(segments=[(1, 0, 4), (2, 6, 9)])
        self.assertFalse(result["steps 0 to N-1 all present once, after merging segments"])

    def test_epoch_check(self):
        rows = {s: {"epoch": 0 if s < 4 else 1, "batch_sha256": h} for s, h in enumerate("abcdbca")}
        self.assertTrue(all(c["pass"] for c in rc.epoch_check({"name": "E", "rows": rows})))
        rows[5]["batch_sha256"] = "z"
        self.assertFalse(all(c["pass"] for c in rc.epoch_check({"name": "E", "rows": rows})))


if __name__ == "__main__":
    unittest.main()
