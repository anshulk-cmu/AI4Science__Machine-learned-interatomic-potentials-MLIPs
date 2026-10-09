"""Out-of-memory guard of the training unit: skip and count an OOM batch, stop after too many in a row, free memory."""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import fcntl  # noqa: F401
except ImportError:
    _stub = types.ModuleType("fcntl")
    for _name in ("LOCK_EX", "LOCK_SH", "LOCK_UN", "LOCK_NB"):
        setattr(_stub, _name, 0)
    _stub.flock = lambda *args, **kwargs: None
    sys.modules["fcntl"] = _stub

from training import guarded_unit as gu  # noqa: E402
from fairchem.core.units.mlip_unit.mlip_unit import MLIPTrainEvalUnit  # noqa: E402


class FakeBatch:
    natoms = torch.tensor([30, 20])
    edge_index = torch.zeros(2, 1000, dtype=torch.long)

    def __contains__(self, key):
        return key in ("natoms", "edge_index")


def make_unit(gc_every=50, max_consecutive_oom=3, steps=1):
    from torchtnt.framework.unit import AppStateMixin

    unit = object.__new__(gu.GuardedTrainEvalUnit)          # skip the heavy fairchem constructor
    AppStateMixin.__init__(unit)                            # the base class every attribute assignment relies on
    unit.gc_every, unit.max_consecutive_oom = gc_every, max_consecutive_oom
    unit.skipped_train_batches = unit.skipped_eval_batches = unit._consecutive_oom = 0
    unit.optimizer, unit.scheduler, unit.logger = mock.Mock(), mock.Mock(), None
    unit.train_progress = types.SimpleNamespace(num_steps_completed=steps)
    return unit


OOM = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


class ClassificationTests(unittest.TestCase):
    def test_is_oom(self):
        self.assertTrue(gu.is_oom(OOM))
        self.assertTrue(gu.is_oom(RuntimeError("CUDA driver error: out of memory")))
        self.assertTrue(gu.is_oom(RuntimeError("CUDA error: Out Of Memory")))
        self.assertFalse(gu.is_oom(RuntimeError("shape mismatch")))
        self.assertFalse(gu.is_oom(ValueError("out of memory")))

    def test_is_a_train_eval_unit_and_named_in_the_config(self):
        self.assertTrue(issubclass(gu.GuardedTrainEvalUnit, MLIPTrainEvalUnit))
        sys.path.insert(0, str(ROOT / "tests"))
        from test_training_config import compose

        cfg = compose("dense").runner.train_eval_unit
        self.assertEqual(cfg._target_, "training.guarded_unit.GuardedTrainEvalUnit")
        self.assertEqual((cfg.gc_every, cfg.max_consecutive_oom), (50, 5))


class TrainStepTests(unittest.TestCase):
    def test_oom_batch_is_skipped_counted_and_the_schedule_advances(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=[OOM, None]) as parent, \
                mock.patch.object(gu, "free_memory") as free:
            unit.train_step(None, FakeBatch())
            self.assertEqual((unit.skipped_train_batches, unit._consecutive_oom), (1, 1))
            unit.optimizer.zero_grad.assert_called_once_with(set_to_none=True)
            unit.scheduler.step.assert_called_once()
            free.assert_called()
            unit.train_step(None, FakeBatch())                      # a good batch resets the run of failures
            self.assertEqual((unit.skipped_train_batches, unit._consecutive_oom), (1, 0))
            self.assertEqual(parent.call_count, 2)

    def test_too_many_in_a_row_stops_the_run(self):
        unit = make_unit(max_consecutive_oom=3)
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=OOM), mock.patch.object(gu, "free_memory"):
            unit.train_step(None, FakeBatch())
            unit.train_step(None, FakeBatch())
            with self.assertRaisesRegex(RuntimeError, "3 out-of-memory batches in a row"):
                unit.train_step(None, FakeBatch())

    def test_other_errors_pass_through_untouched(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=ValueError("bad batch")):
            with self.assertRaisesRegex(ValueError, "bad batch"):
                unit.train_step(None, FakeBatch())
        self.assertEqual(unit.skipped_train_batches, 0)
        unit.optimizer.zero_grad.assert_not_called()

    def test_memory_is_freed_every_gc_every_steps(self):
        with mock.patch.object(MLIPTrainEvalUnit, "train_step"), mock.patch.object(gu, "free_memory") as free:
            for steps, expected in ((49, 0), (50, 1), (100, 1), (7, 0)):
                free.reset_mock()
                make_unit(gc_every=50, steps=steps).train_step(None, FakeBatch())
                self.assertEqual(free.call_count, expected, steps)
            free.reset_mock()
            make_unit(gc_every=0, steps=50).train_step(None, FakeBatch())
            self.assertEqual(free.call_count, 0)

    def test_driver_oom_text_is_handled_like_a_torch_oom(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=RuntimeError("CUDA driver error: out of memory")), \
                mock.patch.object(gu, "free_memory"):
            unit.train_step(None, FakeBatch())
        self.assertEqual(unit.skipped_train_batches, 1)

    def test_skip_is_logged_with_the_batch_size(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=OOM), mock.patch.object(gu, "free_memory"), \
                self.assertLogs(level="WARNING") as logs:
            unit.train_step(None, FakeBatch())
        self.assertIn("50 atoms, 1000 edges", logs.output[0])


class EvalStepTests(unittest.TestCase):
    def test_eval_oom_is_skipped_and_counted_without_stopping(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "eval_step", side_effect=[OOM, OOM, OOM, OOM, None]), \
                mock.patch.object(gu, "free_memory"):
            for _ in range(5):
                unit.eval_step(None, FakeBatch())
        self.assertEqual((unit.skipped_eval_batches, unit.skipped_train_batches), (4, 0))


if __name__ == "__main__":
    unittest.main()
