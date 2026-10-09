"""Local routing for the fairchem UMA backbone.

Global routing (eSCNMDMoeBackbone) mixes the experts once per structure from the mean element
embedding. Here the mean is replaced by a distance-weighted neighborhood composition per atom
(local_routing.local_composition), and every message edge mixes the experts with the
coefficients of its receiving atom. The routing MLP, normalization, expert weights and every
other layer are fairchem's own, so Global and Local differ only in what the router averages.
"""
from __future__ import annotations

import torch

from fairchem.core.common.registry import registry
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.uma.nn.mole import MOLE
from fairchem.core.common import gp_utils

from local_routing import RoutingNeighbors, local_composition


class LocalMOLE(MOLE):
    """MOLE with the same parameters, mixing the expert outputs per edge instead of merging per structure."""

    def forward(self, x):
        g = self.global_mole_tensors
        # activation checkpointing feeds the edges in chunks, ac_start_idx is the chunk offset
        alpha = g.edge_coefficients[g.ac_start_idx : g.ac_start_idx + x.shape[0]]
        if alpha.shape[0] != x.shape[0]:
            raise ValueError("edge coefficients do not cover this chunk of edges")
        flat = x.reshape(x.shape[0], -1, x.shape[-1])
        mixed = torch.einsum("emi,koi->emko", flat, self.weights)
        y = torch.einsum("emko,ek->emo", mixed, alpha.to(mixed.dtype))
        y = y.reshape(*x.shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias


@registry.register_model("escnmd_local_moe_backbone")
class eSCNMDLocalMoeBackbone(eSCNMDMoeBackbone):
    def __init__(self, routing_radius: float = 6.0, routing_power: int = 3, **kwargs):
        super().__init__(**kwargs)
        if self.num_experts == 0 or not self.use_composition_embedding:
            raise ValueError("local routing needs experts and use_composition_embedding")
        if self.composition_dropout > 0:
            raise ValueError("composition_dropout is not defined for local routing")
        if routing_radius > self.cutoff:
            raise NotImplementedError("a routing radius above the message cutoff needs its own neighbor list")
        self.routing_radius, self.routing_power = routing_radius, routing_power
        for module in self.modules():
            if type(module) is MOLE:
                module.__class__ = LocalMOLE      # same parameters and state dict
        self._routing_inputs = None
        self._routing_graph = None

    def merge_MOLE_model(self, data):
        raise NotImplementedError("local routing cannot be merged into one linear layer")

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb):
        # no edges yet: keep the inputs and route once the graph exists, in set_MOLE_sizes
        self._routing_inputs = (atomic_numbers_full, batch_full, csd_mixed_emb)

    def _generate_graph(self, data_dict):
        graph = super()._generate_graph(data_dict)
        self._routing_graph = graph
        return graph

    def set_MOLE_sizes(self, nsystems, batch_full, edge_index):
        atomic_numbers, batch, csd_mixed_emb = self._routing_inputs
        graph = self._routing_graph
        self._routing_inputs = self._routing_graph = None
        if gp_utils.initialized():
            raise NotImplementedError("local routing does not support graph parallelism")
        with torch.autocast(device_type=atomic_numbers.device.type, enabled=False):
            neighbors = RoutingNeighbors(graph["edge_index"], graph["edge_distance"])
            composition = local_composition(
                self.composition_embedding(atomic_numbers), batch, neighbors,
                self.routing_radius, self.routing_power,
            )
            raw = self.routing_mlp(torch.cat((composition, csd_mixed_emb[batch]), dim=-1))
            coefficients = self.mole_expert_coefficient_norm(self.mole_dropout(raw))
        g = self.global_mole_tensors
        g.expert_mixing_coefficients = coefficients                    # [N,K], only used for logging
        g.edge_coefficients = coefficients[graph["edge_index"][1]]     # [E,K], receiving atom
