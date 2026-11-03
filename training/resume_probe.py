"""Test-only training unit for the resume test (docs/resume-test.md, docs/ddp.md). Not used by any production config.

It trains exactly like GuardedTrainEvalUnit and also writes, for every step, one CSV row: the step, the epoch, the
step in the epoch, the batch size, a sha256 of the batch content, the loss, the learning rate and the gradient norm.
It also writes one JSON line when training starts (where a resumed run picks up) and after every validation (the
metrics at full precision; the training log rounds them to 4 decimals).

Rank 0 also writes probe/global/<name>: one row per step for the whole step over all ranks, with an order-free sha256
of its structures (so a run on W ranks compares with a run on one GPU), the structure and atom counts of all ranks
and the global loss (the mean of the rank losses, which DDPMTLoss scales to the global batch). With several ranks
the per-rank rows go to probe/rank<r>/<name> and batch_log itself is not written. Next to each per-rank file,
mem_<name> holds the peak GPU memory of every step, and the events file gets a sha256 of the model and EMA weights at
the end of training (equal on every rank when the gradients were synchronised).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from fairchem.core.common import distutils

from training.guarded_unit import GuardedTrainEvalUnit

COLUMNS = "step,epoch,step_in_epoch,structures,atoms,batch_sha256,loss,lr,grad_norm"


def batch_hash(data) -> str:
    """sha256 of the structure ids, atom counts, atomic numbers and positions of a batch, in batch order."""
    h = hashlib.sha256()
    h.update("\n".join(map(str, data["sid"])).encode("utf-8"))
    for key in ("natoms", "atomic_numbers", "pos"):
        h.update(data[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def structure_hashes(data) -> list[str]:
    """sha256 of the atomic numbers and positions of each structure (OMol25 rows have an empty sid, see guarded_unit.py)."""
    out, start = [], 0
    for n in data["natoms"].tolist():
        h = hashlib.sha256()
        for key in ("atomic_numbers", "pos"):
            h.update(data[key][start:start + n].detach().cpu().contiguous().numpy().tobytes())
        out.append(h.hexdigest())
        start += n
    return out


def set_hash(hashes) -> str:
    """Order-free sha256 of a set of structures."""
    return hashlib.sha256("\n".join(sorted(hashes)).encode("utf-8")).hexdigest()


def probe_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(COLUMNS + "\n", encoding="utf-8")
    return path


class ProbeTrainEvalUnit(GuardedTrainEvalUnit):
    def __init__(self, *args, batch_log: str, **kwargs):
        super().__init__(*args, **kwargs)
        log = Path(batch_log)
        self.world, self.rank = distutils.get_world_size(), distutils.get_rank()
        self.batch_log = probe_file(log if self.world == 1 else log.parent / f"rank{self.rank}" / log.name)
        self.global_log = probe_file(log.parent / "global" / log.name) if self.rank == 0 else None
        self.event_log = self.batch_log.with_suffix(".events.jsonl")
        self.memory_log = self.batch_log.parent / f"mem_{log.name}"
        evaluate = self.eval_unit.on_eval_epoch_end

        def logged_eval(state):
            metrics = evaluate(state)
            self._event("eval", **{k: float(v) for k, v in metrics.items()})
            return metrics

        self.eval_unit.on_eval_epoch_end = logged_eval

    def _event(self, name: str, **fields) -> None:
        p = self.train_progress
        row = {"event": name, "step": p.num_steps_completed, "epoch": p.num_epochs_completed,
               "step_in_epoch": p.num_steps_completed_in_epoch, **fields}
        for path in {self.event_log, self.global_log and self.global_log.with_suffix(".events.jsonl")} - {None}:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def on_train_start(self, state) -> None:
        super().on_train_start(state)   # loads a checkpoint here when the run resumes
        self._event("train_start")

    def on_train_end(self, state) -> None:
        h = hashlib.sha256()
        for module in (self.model, self.ema_model):
            for name, value in sorted(module.state_dict().items()):
                h.update(name.encode("utf-8"))
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
        self._event("train_end", weights_sha256=h.hexdigest())
        super().on_train_end(state)

    def train_step(self, state, data) -> None:
        p = self.train_progress
        step, epoch, in_epoch = p.num_steps_completed, p.num_epochs_completed, p.num_steps_completed_in_epoch
        lr = self.optimizer.param_groups[0]["lr"]   # the rate this step uses; the scheduler steps at its end
        digest = batch_hash(data)
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        super().train_step(state, data)
        structures, atoms = int(data["natoms"].numel()), int(data["natoms"].sum())
        mib = [torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20] if cuda else [-1, -1]
        edges = int(data["edge_index"].shape[1]) if "edge_index" in data else -1
        new = not self.memory_log.exists()
        with self.memory_log.open("a", encoding="utf-8") as f:
            f.write(("step,structures,atoms,edges,max_allocated_mib,max_reserved_mib\n" if new else "")
                    + f"{step},{structures},{atoms},{edges},{mib[0]:.0f},{mib[1]:.0f}\n")
        common = (repr(lr), repr(self.last_grad_norm))
        with self.batch_log.open("a", encoding="utf-8") as f:
            f.write(",".join(map(str, (step, epoch, in_epoch, structures, atoms, digest, repr(self.last_loss), *common))) + "\n")
        hashes, loss, counts = structure_hashes(data), self.last_loss, [structures, atoms]
        if self.world > 1:
            gathered = [None] * self.world
            torch.distributed.all_gather_object(gathered, (hashes, loss, counts))
            loss = sum(g[1] for g in gathered) / self.world
            parts = [g for i, g in enumerate(gathered) if g[0] not in [h[0] for h in gathered[:i]]]   # a batch every rank holds counts once
            hashes = [s for g in parts for s in g[0]]
            counts = [sum(g[2][i] for g in parts) for i in (0, 1)]
        if self.global_log:
            with self.global_log.open("a", encoding="utf-8") as f:
                f.write(",".join(map(str, (step, epoch, in_epoch, *counts, set_hash(hashes), repr(loss), *common))) + "\n")
