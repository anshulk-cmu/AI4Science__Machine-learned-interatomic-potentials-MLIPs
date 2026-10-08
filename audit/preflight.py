"""Print the facts the plan marks as unknown, before any real run."""
import sys

import torch

import geometry as g
import models


def n5():
    """Does index_reduce_ mean divide by N or N + 1 when include_self is True?"""
    dst = torch.zeros(1, 3)
    src = torch.ones(4, 3)
    out = dst.index_reduce_(0, torch.zeros(4, dtype=torch.long), src, "mean", include_self=True)
    print("N5 mean of four ones with include_self=True:", out[0, 0].item(), "(0.8 means divides by N+1)")


def one(family, model, task, dtype):
    energy, cutoff = models.load(family, model, task, dtype)
    pair = g.stacked(g.ideal_slab("Al"), g.ideal_slab("Ni"), 100)
    e = energy(pair)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"{family} {model} {task} {dtype}: cutoff {cutoff:g} A, E(union, d=100) = {e:.6f} eV, peak {mem:.2f} GiB")


if __name__ == "__main__":
    n5()
    g.selftest()
    if len(sys.argv) > 1:                      # preflight.py <family> <model> <task> <dtype>
        one(*sys.argv[1:5])
