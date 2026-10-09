"""Training unit that survives a CUDA out-of-memory error and returns memory to the pool.

An out-of-memory batch is skipped (gradients cleared, caches emptied, the learning-rate schedule still advanced) and
counted. After `max_consecutive_oom` skipped batches in a row the run stops with a clear message, because the batch is
then simply too big for the GPU. Every `gc_every` steps Python garbage is collected and the CUDA cache is emptied.
Single process only: a skipped batch on one rank of several would desynchronize the others.
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


class GuardedTrainEvalUnit(MLIPTrainEvalUnit):
    def __init__(self, *args, gc_every: int = 50, max_consecutive_oom: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.gc_every = gc_every
        self.max_consecutive_oom = max_consecutive_oom
        self.skipped_train_batches = 0
        self.skipped_eval_batches = 0
        self._consecutive_oom = 0

    def _skip(self, kind: str, data, exc: BaseException) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        free_memory()
        natoms = int(data.natoms.sum()) if "natoms" in data else -1
        edges = int(data.edge_index.shape[1]) if "edge_index" in data else -1
        if kind == "train":
            self.skipped_train_batches += 1
            self._consecutive_oom += 1
            if self.scheduler is not None:
                self.scheduler.step()                 # keep the schedule aligned with the step count
            if self.logger:
                self.logger.log({"train/oom_skipped_batches": self.skipped_train_batches}, commit=False)
            count = self.skipped_train_batches
        else:
            self.skipped_eval_batches += 1
            count = self.skipped_eval_batches
        logging.warning(f"OOM: skipped {kind} batch ({natoms} atoms, {edges} edges), {count} skipped so far: {exc}")
        if kind == "train" and self._consecutive_oom >= self.max_consecutive_oom:
            raise RuntimeError(f"{self._consecutive_oom} out-of-memory batches in a row: lower training.max_atoms "
                               "or use a GPU with more memory") from exc

    def train_step(self, state, data) -> None:
        try:
            super().train_step(state, data)
            self._consecutive_oom = 0
        except Exception as exc:
            if not is_oom(exc):
                raise
            self._skip("train", data, exc)
        if self.gc_every and self.train_progress.num_steps_completed % self.gc_every == 0:
            free_memory()

    def eval_step(self, state, data) -> None:
        try:
            super().eval_step(state, data)
        except Exception as exc:
            if not is_oom(exc):
                raise
            self._skip("eval", data, exc)

    def on_train_end(self, state) -> None:
        logging.info(f"OOM-skipped batches: {self.skipped_train_batches} train, {self.skipped_eval_batches} eval")
        super().on_train_end(state)
