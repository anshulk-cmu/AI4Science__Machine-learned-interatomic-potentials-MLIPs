"""Standalone router and adapter; fairchem SO(2) integration pending."""
from dataclasses import dataclass
from typing import Literal
import math

import torch
from torch import Tensor, nn


@dataclass
class RoutingNeighbors:
    # Directed neighbor edges: row 0 neighbor/sender, row 1 center/receiver.
    # Complete within the routing radius, no duplicate rows, no zero-distance self
    # edges. A self edge at d > 0 is a periodic image and is allowed. Distances
    # must retain position gradients.
    edge_index: Tensor
    distances: Tensor


@dataclass
class RoutingInputs:
    atomic_numbers: Tensor  # [N], int64
    positions: Tensor      # [N,3], angstroms
    batch: Tensor          # [N], int64, structure IDs
    task: Tensor           # [B], shared encoder's task representation
    charge: Tensor         # [B], total charge
    spin: Tensor           # [B], spin multiplicity (2S + 1)
    neighbors: RoutingNeighbors | None = None


@dataclass
class RoutingResult:
    coefficients: Tensor
    scope: Literal["structure", "atom"]


def reference_neighbors(positions: Tensor, batch: Tensor, radius: float) -> RoutingNeighbors:
    """O(N^2) non-periodic helper for small tests, not production training.

    Build neighbors independently from the message graph, especially for Local-12.
    Only discrete pair selection is nondifferentiable; distances retain gradients.
    It never makes self edges, since without periodic images none can exist.
    """
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if positions.ndim != 2 or positions.shape[1] != 3 or batch.shape != positions.shape[:1]:
        raise ValueError("expected positions [N,3] and batch [N]")
    with torch.no_grad():
        squared = (positions[:, None] - positions[None, :]).square().sum(-1)
        mask = (batch[:, None] == batch[None, :]) & (squared < radius**2)
        mask.fill_diagonal_(False)
        receiver, sender = mask.nonzero(as_tuple=True)
    distances = torch.linalg.vector_norm(positions[sender] - positions[receiver], dim=-1)
    return RoutingNeighbors(torch.stack((sender, receiver)), distances)


def local_composition(embeddings: Tensor, batch: Tensor, neighbors: RoutingNeighbors,
                      radius: float = 6.0, power: int = 3, check_duplicates: bool = True) -> Tensor:
    """Smooth sparse composition average with the central atom counted once.

    A zero-distance self edge is rejected because the central atom is already
    included. A self edge at d > 0 is a periodic image and adds its weight.
    Rows with the same (receiver, sender, distance) are rejected as duplicates. Pass
    check_duplicates=False for a periodic graph: two images of one atom pair can lie at
    exactly the same distance (opposite cell shifts), and the row carries no cell offset.
    """
    if not math.isfinite(radius) or radius < 0 or not isinstance(power, int) or power < 3:
        raise ValueError("radius must be finite/nonnegative and power an integer >=3")
    if radius == 0:
        return embeddings
    edges, distances = neighbors.edge_index, neighbors.distances
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError("neighbor edge_index must be int64 [2,E]")
    if distances.shape != (edges.shape[1],):
        raise ValueError("neighbor distances must have shape [E]")
    if distances.dtype != embeddings.dtype or distances.device != embeddings.device:
        raise ValueError("neighbor distances and embeddings must have matching dtype/device")
    if edges.numel() and bool(((edges < 0) | (edges >= embeddings.shape[0])).any()):
        raise ValueError("neighbor index outside atom range")
    sender, receiver = edges
    if bool(((sender == receiver) & (distances == 0)).any()):
        raise ValueError("omit zero-distance self edges: central atom is already included")
    # float64 holds int64 indices exactly below 2**53 and every float32/float64 distance.
    if check_duplicates:
        rows = torch.stack((receiver.double(), sender.double(), distances.detach().double()), dim=1)
        if torch.unique(rows, dim=0).shape[0] != rows.shape[0]:
            raise ValueError("duplicate neighbor edges")
    if bool((batch[sender] != batch[receiver]).any()):
        raise ValueError("routing neighbors cannot cross structure boundaries")
    if not bool(torch.isfinite(distances).all()) or bool((distances < 0).any()):
        raise ValueError("distances must be finite and nonnegative")
    weights = (1 - distances.square() / radius**2).clamp_min(0).pow(power)
    numerator = embeddings.index_add(0, receiver, weights[:, None] * embeddings[sender])
    denominator = embeddings.new_ones(embeddings.shape[0]).index_add(0, receiver, weights)
    return numerator / denominator[:, None]


class CompositionRouter(nn.Module):
    """Shared Global/Local reference, provisional softmax and shared coefficients.

    For conditioned runs, pass the shared nn.Module conditioner, accepting
    (task, charge, spin) and returning [B,context_dim]. Charge/spin encodings belong to
    the shared backbone. context_dim=0 requires explicit allow_unconditioned=True and
    is only intended for component tests. Architecture dimensions are provisional.
    """
    def __init__(self, variant: Literal["global", "local"], embedding_dim=16,
                 hidden_dim=32, num_experts=8, radius=6., power=3,
                 context_dim=0, conditioner: nn.Module | None = None,
                 allow_unconditioned=False):
        super().__init__()
        if variant not in ("global", "local"):
            raise ValueError("variant must be global or local; Dense has no router")
        if not math.isfinite(radius) or radius < 0 or not isinstance(power, int) or power < 3:
            raise ValueError("invalid radius or envelope power")
        if context_dim < 0 or (context_dim > 0 and conditioner is None):
            raise ValueError("conditioned routing requires a shared conditioner")
        if context_dim == 0 and (not allow_unconditioned or conditioner is not None):
            raise ValueError("explicitly enable unconditioned component-test mode")
        self.variant, self.radius, self.power = variant, radius, power
        self.context_dim, self.conditioner = context_dim, conditioner
        self.element_embedding = nn.Embedding(119, embedding_dim)
        self.gate = nn.Sequential(nn.Linear(embedding_dim + context_dim, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, num_experts))

    def forward(self, inputs: RoutingInputs) -> RoutingResult:
        z, positions, batch = inputs.atomic_numbers, inputs.positions, inputs.batch
        n, b = z.numel(), inputs.charge.numel()
        if z.shape != (n,) or z.dtype != torch.long or batch.shape != (n,) or batch.dtype != torch.long:
            raise ValueError("atomic_numbers and batch must be int64 [N]")
        if positions.shape != (n, 3) or not bool(torch.isfinite(positions).all()):
            raise ValueError("positions must be finite [N,3]")
        if any(x.shape != (b,) for x in (inputs.task, inputs.charge, inputs.spin)):
            raise ValueError("task, charge, spin must each have shape [B]")
        if n == 0 or b == 0 or bool(((batch < 0) | (batch >= b)).any()):
            raise ValueError("nonempty structures and valid batch indices required")
        if bool(((z < 1) | (z > 118)).any()):
            raise ValueError("atomic numbers must be between 1 and 118")
        emb = self.element_embedding(z)
        if positions.dtype != emb.dtype or positions.device != emb.device:
            raise ValueError("positions and router must have matching dtype/device")
        counts = torch.bincount(batch, minlength=b)
        if bool((counts == 0).any()):
            raise ValueError("each declared structure must contain atoms")
        if self.variant == "global":
            composition = emb.new_zeros(b, emb.shape[1]).index_add(0, batch, emb)
            composition = composition / counts[:, None]
            scope = "structure"
        else:
            if inputs.neighbors is None:
                raise ValueError("local routing requires a geometry-helper neighbor list")
            if (torch.is_grad_enabled() and positions.requires_grad
                    and not inputs.neighbors.distances.requires_grad):
                raise ValueError("neighbor distances must be computed from positions with autograd")
            composition = local_composition(emb, batch, inputs.neighbors, self.radius, self.power)
            scope = "atom"
        if self.context_dim:
            context = self.conditioner(inputs.task, inputs.charge, inputs.spin)
            if context.shape != (b, self.context_dim):
                raise ValueError("conditioner must return [B,context_dim]")
            if self.variant == "local":
                context = context[batch]
            composition = torch.cat((composition, context), dim=-1)
        return RoutingResult(self.gate(composition).softmax(-1), scope)


def edge_coefficients(result: RoutingResult, edge_index: Tensor, batch: Tensor) -> Tensor:
    """Map a scoped result to directed message edges without expanding matrices."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise ValueError("edge_index must be int64 [2,E]")
    if edge_index.numel() and bool(((edge_index < 0) | (edge_index >= batch.numel())).any()):
        raise ValueError("edge index outside atom range")
    sender, receiver = edge_index
    if bool((batch[sender] != batch[receiver]).any()):
        raise ValueError("message edges cannot cross structure boundaries")
    rows = result.coefficients.shape[:1]
    if result.scope == "atom":
        if rows != (batch.numel(),):
            raise ValueError("atom coefficients must have one row per atom")
        return result.coefficients[receiver]
    if result.scope == "structure":
        if rows != ((int(batch.max()) + 1 if batch.numel() else 0),):
            raise ValueError("structure coefficients must have one row per structure")
        return result.coefficients[batch[receiver]]
    raise ValueError("unknown routing scope")
