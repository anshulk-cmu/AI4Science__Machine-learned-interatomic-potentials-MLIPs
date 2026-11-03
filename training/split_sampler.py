"""Batch sampler for data-parallel runs that keeps the single-GPU batches (docs/ddp.md).

fairchem's MaxAtomDistributedBatchSampler gives each rank whole batches of up to max_atoms atoms, so W ranks train on W
batches per step. This sampler builds the batches exactly as one GPU would (the parent class with one replica: same seed,
same max_atoms, same order, same resume position) and gives rank r part r of each batch. The parts are balanced by atom
count. A batch with fewer structures than ranks goes whole to every rank: DDPMTLoss then counts it W times in both the sum
and the sample count, so the averaged gradient is still the single-GPU one. The eval unit counts such a copy once
(training/guarded_unit.py).
"""
from __future__ import annotations

from fairchem.core.datasets.samplers.max_atom_distributed_sampler import MaxAtomDistributedBatchSampler


def split_batch(batch: list[int], natoms: list[int], parts: int) -> list[list[int]]:
    """Largest first into the part with the fewest atoms (ties: lowest part); each part keeps the batch order."""
    if len(batch) < parts:
        return [list(batch)] * parts
    sums, chosen = [0] * parts, [[] for _ in range(parts)]
    for pos in sorted(range(len(batch)), key=lambda i: -natoms[i]):
        k = sums.index(min(sums))
        sums[k] += natoms[pos]
        chosen[k].append(pos)
    return [[batch[i] for i in sorted(c)] for c in chosen]


class SplitBatchSampler(MaxAtomDistributedBatchSampler):
    def __init__(self, dataset, max_atoms: int, num_replicas: int, rank: int, seed: int,
                 shuffle: bool = True, drop_last: bool = False, min_atoms: int = 0) -> None:
        super().__init__(dataset, max_atoms, 1, 0, seed, shuffle, drop_last, min_atoms)
        self.parts, self.part = num_replicas, rank

    def __iter__(self):
        for batch in super().__iter__():
            natoms = [int(n) for n in self.dataset.get_metadata("natoms", batch)]
            yield split_batch(batch, natoms, self.parts)[self.part]
