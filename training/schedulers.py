"""Cosine learning-rate schedule whose linear warmup is counted in optimizer steps."""

import torch
from fairchem.core.modules.scheduler import CosineLRLambda

WARMUP_FRACTION = 0.05


def cosine_with_warmup(optimizer, n_iters_per_epoch, steps, warmup_steps=None,
                       warmup_factor=0.2, lr_min_factor=0.01):
    """Drop-in for fairchem's _get_consine_lr_scheduler with the same call signature.

    fairchem sizes warmup as warmup_epochs * n_iters_per_epoch, so on the 4M split a
    fraction of an epoch can outlast a short step budget. Here warmup ignores the
    dataloader length. warmup_steps None means round(WARMUP_FRACTION * steps).
    """
    del n_iters_per_epoch
    if warmup_steps is None:
        warmup_steps = round(WARMUP_FRACTION * steps)
    if not 0 <= warmup_steps <= steps:
        raise ValueError(f"warmup_steps {warmup_steps} must lie in [0, {steps}]")
    lambda_fn = CosineLRLambda(
        warmup_epochs=max(int(warmup_steps), 1),  # CosineLRLambda divides by it
        warmup_factor=warmup_factor,
        epochs=steps,
        lr_min_factor=lr_min_factor,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
