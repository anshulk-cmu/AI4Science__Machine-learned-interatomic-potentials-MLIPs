"""Dense with the SO(2) linear weights at the scale and step size of the frozen-router model (docs/scaled-dense.md).

global_frozen mixes the K = 8 experts of each MoLE layer with a = 0.13, so W_eff = a * sum_k W_k. Every expert gets the
gradient a * G (G = dL/dW_eff), so AdamW gives every expert the same update G / (sqrt(v_G) + eps / a), and W_eff moves
K * a = 1.04 times that, while decoupled weight decay shrinks W_eff by (1 - lr * wd) per step. This backbone puts the
same rule on the Dense weights that MoLE replaces (fc_m0 and so2_m_conv[i].fc of every SO2_Convolution, as
recursive_replace_so2_linear finds them; not their biases, which MoLE shares across experts):

- they start at a * sqrt(K) times U(-k, k), the scale of W_eff at step 0 (8 independent experts);
- `lr_scaled` gives them their own AdamW group with lr * K * a, weight decay / (K * a) and eps / a.

The scaling multiplies the Dense draws, so no extra random numbers are drawn and every other parameter equals Dense
for the same seed.
"""
from __future__ import annotations

import functools
import math

import torch

from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution


class eSCNMDScaledBackbone(eSCNMDBackbone):
    def __init__(self, frozen_alpha: float = 0.13, frozen_experts: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.frozen_alpha, self.frozen_experts = frozen_alpha, frozen_experts
        self.init_scale = frozen_alpha * math.sqrt(frozen_experts)
        self.lr_scale = frozen_alpha * frozen_experts
        self.eps_scale = 1 / frozen_alpha
        for conv in (m for m in self.blocks.modules() if isinstance(m, SO2_Convolution)):
            # Dense's SO2_m_Conv shrinks its fc by 1/sqrt(2) at init; a MoLE expert (init_linear) does not
            for layer, dense_factor in [(conv.fc_m0, 1.0)] + [(m.fc, 1 / math.sqrt(2)) for m in conv.so2_m_conv]:
                layer.weight.data.mul_(self.init_scale / dense_factor)
                layer.weight.lr_scale, layer.weight.eps_scale = self.lr_scale, self.eps_scale

    def scaled_parameters(self):
        return {n: p for n, p in self.named_parameters() if hasattr(p, "lr_scale")}


def lr_scaled(optimizer_fn):
    """Wrap an optimizer partial so that parameters tagged with `lr_scale` get their own group.

    That group has lr * lr_scale, weight_decay / lr_scale (so the decoupled decay per step, lr * weight_decay, stays
    the Dense value) and eps * eps_scale. A model without tags gets exactly the call it got before. The partial's
    keywords are kept, because fairchem reads weight_decay from them.
    """
    return functools.partial(_grouped, optimizer_fn, **optimizer_fn.keywords)


def _grouped(optimizer_fn, params, **keywords):
    params = list(params)
    tagged = [p for p in params if isinstance(p, torch.Tensor) and hasattr(p, "lr_scale")]
    if not tagged:
        return optimizer_fn(params=params)
    scales = {(p.lr_scale, p.eps_scale) for p in tagged}
    if len(scales) != 1:
        raise ValueError(f"one (lr_scale, eps_scale) expected, got {scales}")
    lr_scale, eps_scale = scales.pop()
    ids = {id(p) for p in tagged}
    optimizer = optimizer_fn(params=[{"params": [p for p in params if id(p) not in ids]},
                                     {"params": tagged, "lr": keywords["lr"] * lr_scale,
                                      "weight_decay": keywords.get("weight_decay", 0.0) / lr_scale}])
    optimizer.param_groups[1]["eps"] *= eps_scale
    return optimizer
