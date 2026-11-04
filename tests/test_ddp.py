"""Two-rank data-parallel mode (docs/ddp.md) on CPU: the split sampler on synthetic atom counts, and the rank-safe parts of
the training unit and the probe with two real processes on gloo (Linux; native Windows torch has no gloo device)."""

import datetime
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_training_config  # noqa: E402,F401  (stubs fcntl on Windows)
import ddp_batch_stats as bs  # noqa: E402
from fairchem.core.datasets.samplers.max_atom_distributed_sampler import MaxAtomDistributedBatchSampler  # noqa: E402
from training.split_sampler import SplitBatchSampler, split_batch  # noqa: E402


def natoms(n=6000, seed=3):
    """Atom counts shaped like OMol25: mostly small molecules, a tail up to 350 atoms."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.lognormal(3.7, 0.7, n).astype(np.int64), 2, 350)


class SplitBatchTests(unittest.TestCase):
    def test_parts_are_balanced_disjoint_and_keep_the_batch_order(self):
        batch, sizes = [10, 11, 12, 13, 14], [300, 50, 40, 30, 20]
        a, b = split_batch(batch, sizes, 2)
        self.assertEqual((a, b), ([10], [11, 12, 13, 14]))
        self.assertEqual(split_batch([1, 2, 3, 4], [5, 5, 5, 5], 2), [[1, 3], [2, 4]])
        self.assertEqual(sorted(sum(split_batch(list(range(9)), list(range(9, 0, -1)), 3), [])), list(range(9)))

    def test_a_batch_with_fewer_structures_than_ranks_goes_whole_to_every_rank(self):
        self.assertEqual(split_batch([7], [320], 2), [[7], [7]])


class SplitSamplerTests(unittest.TestCase):
    def test_every_step_holds_exactly_the_one_gpu_batch(self):
        ds = bs.Counts(natoms())
        for shuffle, epoch, start in ((True, 0, 0), (True, 3, 17), (False, 0, 0)):
            with self.subTest(shuffle=shuffle, epoch=epoch, start=start):
                result = bs.split_check(ds, seed=1, shuffle=shuffle, epoch=epoch, start=start)
                self.assertEqual(result["mismatched_steps"], 0)
                self.assertTrue(result["same_length_as_one_gpu"])
                self.assertEqual(result["steps_compared"], result["steps"] - start)
                self.assertLess(result["atoms_per_rank"]["mean"], 240)

    def test_length_order_and_resume_follow_the_one_gpu_sampler(self):
        ds = bs.Counts(natoms())
        one = MaxAtomDistributedBatchSampler(ds, 450, 1, 0, 5)
        part = SplitBatchSampler(ds, 450, 2, 1, 5)
        self.assertEqual((len(part), part.all_batches), (len(one), one.all_batches))
        for s in (one, part):
            s.set_epoch_and_start_iteration(2, 40)
        first_whole, first_part = next(iter(one)), next(iter(part))
        self.assertTrue(set(first_part) <= set(first_whole))

    def test_largest_parts_cover_every_step_across_epochs(self):
        counts = natoms(1500)
        edges = counts * 30
        steps = len(bs.MaxAtomDistributedBatchSampler(bs.Counts(counts), 450, 1, 0, 2)) + 7      # into the second epoch
        top = bs.largest_parts(edges, counts, seed=2, steps=steps)
        self.assertEqual(top[0]["edges"], max(t["edges"] for t in top))
        self.assertEqual(top[0]["edges"], 30 * top[0]["atoms"])
        self.assertTrue(all(0 <= t["step"] < steps and t["rank"] in (0, 1) for t in top))
        self.assertGreaterEqual(top[0]["atoms"], int(counts.max()))      # the largest structure is a part, alone or not

    def test_stats_report_what_225_atoms_per_rank_would_drop(self):
        counts = natoms()
        out = bs.stats(counts, seed=0, shuffle=True)
        self.assertEqual(out["two_ranks_225_each"]["dropped_structures"], int((counts > 225).sum()))
        self.assertGreater(out["two_ranks_225_each"]["dropped_structures"], 0)
        self.assertEqual(out["one_gpu_450"]["dropped_structures"], 0)
        self.assertEqual(out["split_two_ranks"]["steps"], out["one_gpu_450"]["steps"])
        self.assertEqual(out["split_two_ranks"]["mismatched_steps"], 0)
        self.assertEqual(out["split_two_ranks_resumed_in_epoch1"]["mismatched_steps"], 0)


class FingerprintTests(unittest.TestCase):
    def test_a_copy_is_found_by_content_not_by_the_empty_sid(self):
        from training.guarded_unit import batch_fingerprint
        from training.resume_probe import set_hash, structure_hashes

        a, b = Batch(["a", "b"], [3, 2]), Batch(["c", "d"], [3, 2])
        self.assertEqual(a["sid"], b["sid"])                                     # what fairchem gives for OMol25 rows
        self.assertNotEqual(batch_fingerprint(a), batch_fingerprint(b))
        self.assertEqual(batch_fingerprint(a), batch_fingerprint(Batch(["a", "b"], [3, 2])))
        self.assertEqual(set_hash(structure_hashes(a)), set_hash(structure_hashes(Batch(["b"], [2])) + structure_hashes(Batch(["a"], [3]))))
        self.assertNotEqual(set_hash(structure_hashes(a)), set_hash(structure_hashes(b)))

    def test_charge_and_spin_tell_same_geometry_structures_apart(self):
        from training.guarded_unit import batch_fingerprint
        from training.resume_probe import structure_hashes

        def with_state(charge, spin):
            b = Batch(["a"], [3])
            b.update(charge=torch.tensor([charge]), spin=torch.tensor([spin]))
            return b

        base = with_state(0, 1)
        for other in (with_state(1, 1), with_state(0, 3)):
            self.assertNotEqual(batch_fingerprint(base), batch_fingerprint(other))
            self.assertNotEqual(structure_hashes(base), structure_hashes(other))
        self.assertEqual(batch_fingerprint(base), batch_fingerprint(with_state(0, 1)))


def gloo_works():
    try:
        torch.distributed.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        return True
    except Exception:
        return False


OOM = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def bare_unit(cls):
    from torchtnt.framework.unit import AppStateMixin

    unit = object.__new__(cls)
    AppStateMixin.__init__(unit)
    unit.gc_every, unit.max_oom_retries = 0, 2
    unit.oom_events, unit.oom_recovered = {"train": 0, "eval": 0}, {"train": 0, "eval": 0}
    unit.optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-4}], zero_grad=lambda **k: None)
    unit.logger, unit.last_grad_norm = None, 1.5
    unit.train_progress = types.SimpleNamespace(num_steps_completed=0, num_epochs_completed=0, num_steps_completed_in_epoch=0)
    return unit


class Batch(dict):
    """Structures named by letters, each with its own positions. The sid is empty, as for every OMol25 row."""

    def __init__(self, names, sizes):
        pos = torch.cat([torch.full((n, 3), float(ord(s))) for s, n in zip(names, sizes)])
        super().__init__(sid=[""] * len(names), natoms=torch.tensor(sizes), pos=pos,
                         atomic_numbers=torch.ones(sum(sizes), dtype=torch.long))
        self.names = names


def case_oom(rank, tmp):
    from fairchem.core.units.mlip_unit.mlip_unit import MLIPTrainEvalUnit
    from training import guarded_unit as gu

    unit = bare_unit(gu.GuardedTrainEvalUnit)
    calls = []

    def step(self, state, data):
        calls.append(1)
        if rank == 0:
            raise OOM
        torch.distributed.all_reduce(torch.ones(1))          # the collective rank 0 never reaches

    t0 = time.time()
    with mock.patch.object(MLIPTrainEvalUnit, "train_step", step), mock.patch.object(gu, "free_memory") as free:
        try:
            unit.train_step(None, Batch(["a"], [5]))
            outcome = "finished"
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
    return {"outcome": outcome, "calls": len(calls), "freed": free.call_count, "seconds": time.time() - t0, "at": time.time()}


def case_eval_probe_logs(rank, tmp):
    import logging
    from fairchem.core.common import distutils
    from fairchem.core.units.mlip_unit._metrics import Metrics
    from fairchem.core.units.mlip_unit.mlip_unit import MLIPTrainEvalUnit
    from training import guarded_unit as gu
    from training import resume_probe as rp

    unit = bare_unit(gu.GuardedTrainEvalUnit)
    value = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    unit.eval_unit = types.SimpleNamespace(running_metrics={"forces": {"omol.val": {"mae": Metrics()}}}, total_loss_metrics=Metrics())

    def eval_step(self, state, data):
        self.eval_unit.running_metrics["forces"]["omol.val"]["mae"] += Metrics(total=sum(value[s] for s in data.names), numel=len(data.names))
        self.eval_unit.total_loss_metrics += Metrics(metric=10.0 * (rank + 1), total=10.0 * (rank + 1), numel=1)

    seen = {}

    def epoch_end(self, state):
        m = self.eval_unit.running_metrics["forces"]["omol.val"]["mae"]
        seen.update(mae_total=distutils.all_reduce(m.total), mae_numel=distutils.all_reduce(m.numel),
                    val_loss=float(self.eval_unit.total_loss_metrics.metric))

    with mock.patch.object(MLIPTrainEvalUnit, "eval_step", eval_step), mock.patch.object(MLIPTrainEvalUnit, "on_eval_epoch_end", epoch_end), \
            mock.patch.object(MLIPTrainEvalUnit, "on_eval_epoch_start", lambda self, state: None):
        unit.on_eval_epoch_start(None)
        unit.eval_step(None, Batch(["a", "b"] if rank == 0 else ["c", "e"], [3, 3] if rank == 0 else [4, 2]))   # split in two parts of two
        unit.eval_step(None, Batch(["d"], [9]))                                                                  # held by both ranks
        unit.on_eval_epoch_end(None)
    seen["rows"] = unit.eval_rows

    probe = bare_unit(rp.ProbeTrainEvalUnit)
    log = Path(tmp) / "probe" / "seg_1.csv"
    probe.world, probe.rank = 2, rank
    probe.batch_log = rp.probe_file(log.parent / f"rank{rank}" / log.name)
    probe.global_log = rp.probe_file(log.parent / "global" / log.name) if rank == 0 else None
    probe.memory_log = probe.batch_log.parent / f"mem_{log.name}"

    def train_step(self, state, data):
        self.last_loss = 1.0 + rank

    batches = [Batch(["a"] if rank == 0 else ["b", "c"], [5] if rank == 0 else [2, 3]), Batch(["d"], [7])]
    with mock.patch.object(MLIPTrainEvalUnit, "train_step", train_step):
        for step, data in enumerate(batches):
            probe.train_progress.num_steps_completed = step
            probe.train_step(None, data)

    logging.getLogger().setLevel(logging.INFO)
    with mock.patch.object(MLIPTrainEvalUnit, "__init__", lambda self, *a, **k: None), mock.patch.object(gu, "lr_scaled", lambda f: f):
        gu.GuardedTrainEvalUnit.__init__(bare_unit(gu.GuardedTrainEvalUnit), optimizer_fn=None)
    seen["log_level"] = logging.getLogger().level
    return seen


def worker(rank, store, case, out):
    os.environ["CURRRENT_DEVICE_TYPE"] = "cpu"          # fairchem's distutils name, misspelled upstream
    torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                                         timeout=datetime.timedelta(seconds=30))
    try:
        result = globals()[case](rank, os.path.dirname(out))
    finally:
        Path(f"{out}.{rank}").write_text(json.dumps(result), encoding="utf-8")
    torch.distributed.destroy_process_group()


@unittest.skipUnless(gloo_works(), "needs a working gloo backend (Linux)")
class TwoProcessTests(unittest.TestCase):
    def spawn(self, case):
        ctx = torch.multiprocessing.get_context("spawn")
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "result")
        procs = [ctx.Process(target=worker, args=(r, os.path.join(tmp, "store"), case, out)) for r in (0, 1)]
        t0 = time.time()
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        self.assertFalse(any(p.is_alive() for p in procs), "a rank hung")
        return [json.loads(Path(f"{out}.{r}").read_text(encoding="utf-8")) for r in (0, 1)], time.time() - t0, tmp

    def test_an_oom_on_one_rank_stops_it_at_once_and_the_other_rank_is_released(self):
        (r0, r1), _, _ = self.spawn("case_oom")
        self.assertIn("OOM: train batch ran out of memory on rank 0 of 2", r0["outcome"])
        self.assertIn("the run stops", r0["outcome"])
        self.assertEqual((r0["calls"], r0["freed"]), (1, 0))       # no retry, no cleanup that could hide the error
        self.assertNotEqual(r1["outcome"], "finished")                 # its collective failed when rank 0 left
        # released within the 30 s process-group timeout of rank 0's error (in a real run torch elastic kills it after 30 s)
        self.assertLess(r1["at"] - r0["at"], 45, r1["outcome"])

    def test_eval_counts_a_shared_batch_once_and_reduces_the_loss_probe_and_logs(self):
        (r0, r1), _, tmp = self.spawn("case_eval_probe_logs")
        for r in (r0, r1):
            self.assertEqual((r["mae_total"], r["mae_numel"]), (15.0, 5))      # a, b, c, e, d once each
            self.assertAlmostEqual(r["val_loss"], 15.0)                         # (10 + 20) / 2 ranks, per step
        self.assertEqual(r0["rows"] + r1["rows"], 5)
        import logging
        self.assertEqual((r0["log_level"], r1["log_level"]), (logging.INFO, logging.WARNING))
        from training import resume_probe as rp
        rows = (Path(tmp) / "probe" / "global" / "seg_1.csv").read_text(encoding="utf-8").splitlines()
        first, second = (line.split(",") for line in rows[1:])
        whole = rp.set_hash(rp.structure_hashes(Batch(["b", "c"], [2, 3])) + rp.structure_hashes(Batch(["a"], [5])))
        self.assertEqual((first[3], first[4], first[5], float(first[6])), ("3", "10", whole, 1.5))
        self.assertEqual((second[3], second[4], second[5]), ("1", "7", rp.set_hash(rp.structure_hashes(Batch(["d"], [7])))))
        for rank in (0, 1):
            self.assertEqual(len((Path(tmp) / "probe" / f"rank{rank}" / "seg_1.csv").read_text(encoding="utf-8").splitlines()), 3)
            self.assertTrue((Path(tmp) / "probe" / f"rank{rank}" / "mem_seg_1.csv").exists())
        self.assertFalse((Path(tmp) / "probe" / "seg_1.csv").exists())


if __name__ == "__main__":
    unittest.main()
