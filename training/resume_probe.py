"""Test-only training unit for the resume test (docs/resume-test.md). Not used by any production config.

It trains exactly like GuardedTrainEvalUnit and also writes, for every step, one CSV row: the step, the epoch, the
step in the epoch, the batch size, a sha256 of the batch content, the loss, the learning rate and the gradient norm.
It also writes one JSON line when training starts (where a resumed run picks up) and after every validation (the
metrics at full precision; the training log rounds them to 4 decimals).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.guarded_unit import GuardedTrainEvalUnit

COLUMNS = "step,epoch,step_in_epoch,structures,atoms,batch_sha256,loss,lr,grad_norm"


def batch_hash(data) -> str:
    """sha256 of the structure ids, atom counts, atomic numbers and positions of a batch, in batch order."""
    h = hashlib.sha256()
    h.update("\n".join(map(str, data["sid"])).encode("utf-8"))
    for key in ("natoms", "atomic_numbers", "pos"):
        h.update(data[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ProbeTrainEvalUnit(GuardedTrainEvalUnit):
    def __init__(self, *args, batch_log: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_log = Path(batch_log)
        self.event_log = self.batch_log.with_suffix(".events.jsonl")
        self.batch_log.parent.mkdir(parents=True, exist_ok=True)
        if not self.batch_log.exists():
            self.batch_log.write_text(COLUMNS + "\n", encoding="utf-8")
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
        with self.event_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def on_train_start(self, state) -> None:
        super().on_train_start(state)   # loads a checkpoint here when the run resumes
        self._event("train_start")

    def train_step(self, state, data) -> None:
        p = self.train_progress
        step, epoch, in_epoch = p.num_steps_completed, p.num_epochs_completed, p.num_steps_completed_in_epoch
        lr = self.optimizer.param_groups[0]["lr"]   # the rate this step uses; the scheduler steps at its end
        digest = batch_hash(data)
        super().train_step(state, data)
        row = (step, epoch, in_epoch, int(data["natoms"].numel()), int(data["natoms"].sum()), digest,
               repr(self.last_loss), repr(lr), repr(self.last_grad_norm))
        with self.batch_log.open("a", encoding="utf-8") as f:
            f.write(",".join(map(str, row)) + "\n")
