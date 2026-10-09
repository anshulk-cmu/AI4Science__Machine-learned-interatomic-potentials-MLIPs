"""Standalone linear-expert reference; fairchem SO(2) integration pending."""
import torch
from torch import nn
from .router import RoutingResult, edge_coefficients


class LinearExperts(nn.Module):
    """Each instance owns its expert matrices. Bias is shared, counted once.

    x: [E,I]; weights: [K,O,I]. No activation occurs before output mixing.
    Mixed-output mode is a reference; it materializes [E,K,O]. Merging creates
    [N,O,I] for Local or [B,O,I] for Global, then gathers an [E,O,I] tensor of
    per-edge matrices, so its memory scales with [E,O,I], not only [N,O,I].
    This is not an optimized grouped GEMM implementation.
    """
    def __init__(self, in_features, out_features, num_experts=8, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=in_features**-0.5)

    def forward(self, x, routing: RoutingResult, edge_index, batch, mode="expert_output_mix"):
        alpha = edge_coefficients(routing, edge_index, batch)
        if x.ndim != 2 or x.shape != (edge_index.shape[1], self.weight.shape[2]):
            raise ValueError("x must have shape [E,in_features]")
        if routing.coefficients.ndim != 2 or alpha.shape[1] != self.weight.shape[0]:
            raise ValueError("coefficient expert dimension must match the expert bank")
        if mode == "expert_output_mix":
            outputs = torch.einsum("koi,ei->eko", self.weight, x)
            y = torch.einsum("ek,eko->eo", alpha, outputs)
        elif mode in ("structure_merge", "atom_merge"):
            expected_scope = "structure" if mode == "structure_merge" else "atom"
            if routing.scope != expected_scope:
                raise ValueError(f"{mode} requires {expected_scope} coefficients")
            merged = torch.einsum("nk,koi->noi", routing.coefficients, self.weight)
            receiver = edge_index[1]
            index = receiver if routing.scope == "atom" else batch[receiver]
            y = torch.einsum("eoi,ei->eo", merged[index], x)
        else:
            raise ValueError("mode must be expert_output_mix, structure_merge, or atom_merge")
        return y if self.bias is None else y + self.bias
