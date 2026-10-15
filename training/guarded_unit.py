"""Training unit that watches for CUDA out-of-memory errors, cleans up and retries, and never skips a batch.

When a batch runs out of memory the unit logs an event with the memory state, clears the gradients, collects Python
garbage, empties the CUDA cache, and runs the same batch again, up to `max_oom_retries` times. A batch that still fails
stops the run with a clear message, so a real memory problem is seen instead of hidden. Every `gc_every` steps Python
garbage is collected and the CUDA cache is emptied. Single process only.
"""
from __future__ import annotations

import gc
import logging

import torch

from fairchem.core.units.mlip_unit.mlip_unit import MLIPTrainEvalUnit


def is_oom(exc: BaseException) -> bool:
    """True for torch.OutOfMemoryError and for the RuntimeError text CUDA raises, for example 'CUDA driver error: out of memory'."""
    return isinstance(exc, torch.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_note() -> str:
    if not torch.cuda.is_available():
        return ""
    gib = 2 ** 30
    return (f"GPU memory: {torch.cuda.memory_allocated() / gib:.1f} GiB allocated, {torch.cuda.memory_reserved() / gib:.1f} GiB reserved, "
            f"{torch.cuda.max_memory_allocated() / gib:.1f} GiB peak, {torch.cuda.get_device_properties(0).total_memory / gib:.1f} GiB total")


class GuardedTrainEvalUnit(MLIPTrainEvalUnit):
    def __init__(self, *args, gc_every: int = 50, max_oom_retries: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.gc_every = gc_every
        self.max_oom_retries = max_oom_retries
        self.oom_events = {"train": 0, "eval": 0}          # out-of-memory errors seen
        self.oom_recovered = {"train": 0, "eval": 0}       # batches that ran after a cleanup and a retry

    def _run(self, kind: str, step, data) -> None:
        for attempt in range(self.max_oom_retries + 1):
            try:
                step()
            except Exception as exc:
                if not is_oom(exc):
                    raise
                message = str(exc)
            else:
                if attempt:
                    self.oom_recovered[kind] += 1
                    logging.warning(f"OOM: {kind} batch ran after {attempt} cleanup and retry, {self.oom_recovered[kind]} recovered so far")
                return
            # Outside the except block, so the tensors of the failed step can be freed by the cleanup below.
            self.oom_events[kind] += 1
            self.optimizer.zero_grad(set_to_none=True)
            free_memory()
            natoms = int(data.natoms.sum()) if "natoms" in data else -1
            edges = int(data.edge_index.shape[1]) if "edge_index" in data else -1
            if kind == "train" and self.logger:
                self.logger.log({"train/oom_events": self.oom_events["train"]}, commit=False)
            logging.warning(f"OOM: {kind} batch ({natoms} atoms, {edges} edges) ran out of memory, event {self.oom_events[kind]}; "
                            f"cleaned up, attempt {attempt + 1} of {self.max_oom_retries + 1}. {memory_note()} ({message})")
        raise RuntimeError(f"A {kind} batch ({natoms} atoms, {edges} edges) still ran out of memory after {self.max_oom_retries} "
                           "cleanups and retries: it does not fit this GPU. Use a GPU with more memory or lower training.max_atoms. "
                           f"Last error: {message}")

    def train_step(self, state, data) -> None:
        self._run("train", lambda: MLIPTrainEvalUnit.train_step(self, state, data), data)
        if self.gc_every and self.train_progress.num_steps_completed % self.gc_every == 0:
            free_memory()

    def eval_step(self, state, data) -> None:
        self._run("eval", lambda: MLIPTrainEvalUnit.eval_step(self, state, data), data)

    def on_train_end(self, state) -> None:
        e, r = self.oom_events, self.oom_recovered
        logging.info(f"OOM events: {e['train']} train ({r['train']} recovered), {e['eval']} eval ({r['eval']} recovered); no batch was skipped")
        super().on_train_end(state)
