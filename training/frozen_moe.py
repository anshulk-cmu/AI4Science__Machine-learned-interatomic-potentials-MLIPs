"""Global MoLE with a frozen router: one constant expert mixture for every structure, for the whole run.

The router output alpha is replaced through the backbone's coefficient norm, the same hook that
scripts/router_check.py (`fixed_alpha`) uses at inference, so no fairchem code changes. The routing MLP
and the composition embedding stay in the model, so the parameters, the state dict and the initial
weights equal Global for the same seed. They get requires_grad False: they receive no gradient, so
AdamW never steps them and never applies weight decay to them, and DDP does not wait for their gradients.
The constant is part of the model config, so the inference checkpoint rebuilds the same model.
"""
from __future__ import annotations

import functools

import torch

from fairchem.core.common.registry import registry
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

ROUTER = ("routing_mlp.", "composition_embedding.")   # the parameters that only the router uses


def constant_alpha(logits, value):
    return torch.full_like(logits, value)


@registry.register_model("escnmd_frozen_moe_backbone")
class eSCNMDFrozenMoeBackbone(eSCNMDMoeBackbone):
    def __init__(self, fixed_alpha: float = 0.13, **kwargs):
        super().__init__(**kwargs)
        if self.num_experts == 0 or not self.use_composition_embedding:
            raise ValueError("a frozen router needs experts and use_composition_embedding")
        self.fixed_alpha = fixed_alpha
        self.mole_expert_coefficient_norm = functools.partial(constant_alpha, value=fixed_alpha)
        for name, p in self.named_parameters():
            if name.startswith(ROUTER):
                p.requires_grad_(False)
