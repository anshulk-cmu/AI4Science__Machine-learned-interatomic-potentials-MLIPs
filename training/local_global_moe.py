"""Per-atom routing from joined local and structure-global compositions."""
from __future__ import annotations

import torch
from torch import nn

from fairchem.core.common.registry import registry

from training.local_moe import eSCNMDLocalMoeBackbone


@registry.register_model("escnmd_local_global_moe_backbone")
class eSCNMDLocalGlobalMoeBackbone(eSCNMDLocalMoeBackbone):
    """Route on [local composition, global composition, charge/spin/task context].

    Both compositions share one element embedding. Global means use the atoms of
    each structure (UMA 1.1's N divisor), never all atoms in a minibatch. Experts
    still mix per receiving atom through LocalMOLE, so merging into a single
    structure-level linear layer and graph parallelism remain unsupported.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        first = self.routing_mlp[0]
        self.routing_mlp[0] = nn.Linear(
            first.in_features + self.composition_embedding.embedding_dim,
            first.out_features,
            bias=first.bias is not None,
            device=first.weight.device,
            dtype=first.weight.dtype,
        )

    def _composition_features(self, embeddings, batch, neighbors, nsystems):
        local = super()._composition_features(embeddings, batch, neighbors, nsystems)
        totals = embeddings.new_zeros(nsystems, embeddings.shape[-1]).index_add(0, batch, embeddings)
        counts = torch.bincount(batch, minlength=nsystems).to(embeddings.dtype)
        global_mean = totals / counts[:, None]
        return torch.cat((local, global_mean[batch]), dim=-1)
