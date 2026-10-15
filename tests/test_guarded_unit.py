"""Out-of-memory monitor of the training unit: log, clean up and retry the same batch, never skip it, stop loudly if it still fails."""

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


def make_unit(gc_every=50, max_oom_retries=2, steps=1):
    from torchtnt.framework.unit import AppStateMixin

    unit = object.__new__(gu.GuardedTrainEvalUnit)          # skip the heavy fairchem constructor
    AppStateMixin.__init__(unit)                            # the base class every attribute assignment relies on
    unit.gc_every, unit.max_oom_retries = gc_every, max_oom_retries
    unit.oom_events = {"train": 0, "eval": 0}
    unit.oom_recovered = {"train": 0, "eval": 0}
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
        self.assertEqual((cfg.gc_every, cfg.max_oom_retries), (50, 2))
        self.assertNotIn("max_consecutive_oom", cfg)


class TrainStepTests(unittest.TestCase):
    def test_oom_batch_is_cleaned_up_and_retried_never_skipped(self):
        unit = make_unit()
        batch = FakeBatch()
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=[OOM, None]) as parent, \
                mock.patch.object(gu, "free_memory") as free:
            unit.train_step(None, batch)
        self.assertEqual(parent.call_count, 2)                       # the same batch ran again
        self.assertEqual([c.args for c in parent.call_args_list], [(unit, None, batch)] * 2)
        self.assertEqual((unit.oom_events["train"], unit.oom_recovered["train"]), (1, 1))
        unit.optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        free.assert_called()
        unit.scheduler.step.assert_not_called()                      # nothing is skipped, so the schedule is not touched

    def test_a_clean_step_records_nothing(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "train_step") as parent, mock.patch.object(gu, "free_memory") as free:
            unit.train_step(None, FakeBatch())
        self.assertEqual(parent.call_count, 1)
        self.assertEqual((unit.oom_events, unit.oom_recovered), ({"train": 0, "eval": 0}, {"train": 0, "eval": 0}))
        unit.optimizer.zero_grad.assert_not_called()
        free.assert_not_called()

    def test_the_run_stops_loudly_when_the_retries_fail(self):
        unit = make_unit(max_oom_retries=2)
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=OOM) as parent, mock.patch.object(gu, "free_memory"):
            with self.assertRaisesRegex(RuntimeError, "50 atoms, 1000 edges.*does not fit this GPU"):
                unit.train_step(None, FakeBatch())
        self.assertEqual(parent.call_count, 3)                       # the first try and two retries
        self.assertEqual((unit.oom_events["train"], unit.oom_recovered["train"]), (3, 0))

    def test_other_errors_pass_through_untouched(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=ValueError("bad batch")) as parent:
            with self.assertRaisesRegex(ValueError, "bad batch"):
                unit.train_step(None, FakeBatch())
        self.assertEqual(parent.call_count, 1)
        self.assertEqual(unit.oom_events["train"], 0)
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
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=[RuntimeError("CUDA driver error: out of memory"), None]), \
                mock.patch.object(gu, "free_memory"):
            unit.train_step(None, FakeBatch())
        self.assertEqual((unit.oom_events["train"], unit.oom_recovered["train"]), (1, 1))

    def test_each_event_is_logged_with_the_batch_size(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=[OOM, None]), mock.patch.object(gu, "free_memory"), \
                self.assertLogs(level="WARNING") as logs:
            unit.train_step(None, FakeBatch())
        self.assertIn("50 atoms, 1000 edges", logs.output[0])
        self.assertIn("event 1", logs.output[0])
        self.assertIn("1 recovered so far", logs.output[1])

    def test_the_cleanup_runs_after_the_error_is_out_of_scope(self):
        # The traceback of an exception holds the tensors of the failed step, so the cleanup must not run inside the except block.
        unit = make_unit()
        seen = []

        def free():
            import sys as _sys
            seen.append(_sys.exc_info()[0])

        with mock.patch.object(MLIPTrainEvalUnit, "train_step", side_effect=[OOM, None]), mock.patch.object(gu, "free_memory", side_effect=free):
            unit.train_step(None, FakeBatch())
        self.assertEqual(seen, [None])


class EvalStepTests(unittest.TestCase):
    def test_eval_oom_is_retried_never_skipped(self):
        unit = make_unit()
        with mock.patch.object(MLIPTrainEvalUnit, "eval_step", side_effect=[OOM, OOM, None]) as parent, mock.patch.object(gu, "free_memory"):
            unit.eval_step(None, FakeBatch())
        self.assertEqual(parent.call_count, 3)
        self.assertEqual((unit.oom_events["eval"], unit.oom_recovered["eval"], unit.oom_events["train"]), (2, 1, 0))

    def test_eval_stops_loudly_too(self):
        unit = make_unit(max_oom_retries=1)
        with mock.patch.object(MLIPTrainEvalUnit, "eval_step", side_effect=OOM), mock.patch.object(gu, "free_memory"):
            with self.assertRaisesRegex(RuntimeError, "eval batch"):
                unit.eval_step(None, FakeBatch())


if __name__ == "__main__":
    unittest.main()
