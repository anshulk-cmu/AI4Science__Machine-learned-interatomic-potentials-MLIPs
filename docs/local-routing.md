# Local router

The router and a linear-expert reference are implemented in `local_routing/`. They are integrated into fairchem 2.21.0
as the Local-6 backbone (`training/local_moe.py`, `configs/variant/local6.yaml`; see `docs/TRAINING.md`). Local-6 was
trained in three 50,000-step pilots (seeds 0 to 2) and in one 400,000-step long run (seed 0); results in `README.md` and
`docs/final-evaluation.md`. (Updated 2026-11-16; the first version of this page was written before any training.)

## What the trained models use

The trained backbones (`training/local_moe.py` for Local-6, Local-12 and Element; `training/local_global_moe.py` for
Local+Global) use only `local_routing.local_composition` from this package. The router itself is fairchem's: the per-atom
composition (128 values) and the charge, spin and task embedding (128 values) go through `routing_mlp`, which is
Linear 256 -> 16, SiLU, Linear 16 -> 16, SiLU, Linear 16 -> 8, SiLU, and the 8 coefficients are softmax + 0.005
(measured from the installed fairchem 2.21.0, `fairchem/core/models/uma/nn/mole_utils.py` and `mole.py`). Global uses the
same MLP on the composition of the whole structure. Local+Global widens the first layer to 384 -> 16 and feeds it the local
composition, the structure mean composition and the context. The reference `CompositionRouter` below is not used in training.

## Interface of the reference router

`CompositionRouter.forward(RoutingInputs)` returns `RoutingResult`:

- Global: coefficients `[B, 8]`, scope `structure`.
- Local: coefficients `[N, 8]`, scope `atom`.
- Dense bypasses routing.

Inputs contain atomic numbers `[N]`, positions `[N,3]`, structure indices `[N]`,
and task, total charge, and spin multiplicity `[B]`. Local also requires a
`RoutingNeighbors` object. Indices are int64; coordinates are in angstroms.
Use compatible dtypes and devices throughout.

Neighbor indices are `[2,E]`: row 0 sends, row 1 receives. Include each directed
pair once per image, with no zero-distance self edges. A self edge at a distance
above zero is a periodic image of the central atom and is allowed, as is the same
pair at several distances. The composition calculation adds the central atom
with weight one. Supply all neighbors within the routing radius, even if that
radius exceeds the message cutoff. The receiver convention is agreed by the team.

The neighbor-list contract: the list is complete within the routing radius and
is not truncated by `max_neighbors`, the distances carry gradients to positions,
and no two rows of (receiver, sender, distance) are identical. The router raises
`ValueError` on duplicate rows, on detached distances when positions require
gradients, and on distances whose dtype or device differ from the router's.
`local_composition(..., check_duplicates=False)` skips the duplicate check. The fairchem backbone
(`training/local_moe.py`) uses it, because a periodic graph can hold two images of one atom pair at exactly
the same distance (opposite cell shifts) and the rows carry no cell offset.

**The trained backbone does not meet the "not truncated" part in every case.** `training/local_moe.py` builds a separate,
uncapped routing graph only when the routing radius is larger than the message cutoff (Local-12). Local-6 and
Local+Global (radius 6 A = cutoff 6 A) route on the message graph, and the data loader keeps at most the 300 nearest
neighbours of each atom (`max_neighbors: 300` in `configs/train.yaml`). So the routing composition of an atom is cut
when 301 or more atoms lie within 6 A of it. Whether any OMol25 structure reaches this was not measured. This does not
affect size consistency: the cut keeps the nearest neighbours, all within 6 A, so atoms farther than 6 A never enter
(reasoning).

**Additivity threshold.** A model with message cutoff r_c and routing radius R is exactly additive for two fragments
whose closest atoms are farther apart than max(R, r_c): no message edge and no routing neighbour crosses between them.
A threshold of R + r_c is also safe, but loose. The audit code uses the tight form, max(R, r_c) plus a 1 A margin
(`audit/models.py`, `interaction_radius`; reasoning for the bound, the code is measured).

Distances must retain gradients to positions. `reference_neighbors` does this
for small, non-periodic systems, using an O(N^2) search. It never makes self
edges, since a non-periodic system has no images. The composition calculation
itself uses sparse indexed sums. Training will need the shared geometry helper.

`include_self: true` in `configs/config_draft.yaml` means the central atom counts
with weight one in its own average. It is unrelated to fairchem's `include_self`
option, where version 1.0 divides by N+1.

`edge_coefficients` maps atom or structure coefficients to receiving edges.
`LinearExperts` implements one ordinary linear operation with a shared bias.
It is a test reference, not a drop-in SO(2) convolution.

## Draft config

Yiwei's supplied draft is saved unchanged in `configs/config_draft.yaml`. It is
not a runnable fairchem config. Null entries remain unresolved; this package does
not load it or fill in training defaults.

The expert execution names match the draft:

| Mode | Scope | Calculation |
|---|---|---|
| `expert_output_mix` | Either | Apply experts, then mix before activation |
| `structure_merge` | Structure | Merge once per structure |
| `atom_merge` | Atom | Merge once per receiving atom |

Local-6 starts with `expert_output_mix`, as proposed in the YAML. Both merge
modes gather matrices per edge in this reference; neither is an optimized kernel.

The reference code uses radius 6, envelope power 3, eight experts, and shared coefficients.
Softmax, embedding size 16, and router hidden size 32 are provisional settings of the reference only. The trained
models use fairchem's router (Section "What the trained models use": composition size 128, hidden size 16).

Pass the shared task/charge/spin encoder as `conditioner`, with signature
`conditioner(task, charge, spin) -> [B, context_dim]`. No new encoding is defined
here. Unconditioned tests require explicit `allow_unconditioned=True`. Local
coefficients of a fragment do not depend on distant atoms in the same structure,
except through this structure-level context: a different total charge changes
the coefficients of every atom.

The repository pins fairchem 2.21.0 and PyTorch 2.8.0+cu128, while the draft's
version/commit fields are null. Reconcile those fields when the shared backbone
is available; do not assume a source commit from a version.

## Tests

From the repository root, in an environment with PyTorch:

```sh
python -m unittest discover -s tests -v
```

Seventeen component tests (`tests/test_local_routing.py`, still 17 on 2026-11-16) cover composition arithmetic, locality, batch isolation,
receiver mapping, cutoff smoothness, first/second derivatives, conditioning,
agreement between execution modes, and a toy energy-and-force optimizer step.
The toy scalar is not a molecular potential.

All 17 tests passed again on 8 October 2026, on macOS CPU with Python 3.12.14
and PyTorch 2.8.0. PyTorch warned that NumPy was absent in the temporary test
environment; these tests use tensor operations and do not require NumPy.
MD stability remains untested. CUDA, the fairchem integration and full-model forces are tested by
`tests/test_local_backbone.py`.

`tests/test_local_routing_contracts.py` adds 27 tests (28 on 2026-11-16): the neighbor-list and
coefficient contracts above, the error paths, float32 paths, unequal structure
sizes, interleaved batch ids, permuted edges, isolated atoms, envelope powers 4
and 5, radius 0, and a pair at exactly the radius. Set `ROUTER_TEST_DEVICE=cuda`
to run both files on a GPU; the default is `cpu`. All 44 tests passed on
8 October 2026 on Windows CPU with Python 3.12.15 and PyTorch 2.8.0+cu128.

## Handoff and dates

Local-6 is integrated and tested. On 9 October 2026 the full suite had 101 tests (`python -m unittest discover -s tests`,
Windows CPU, Python 3.12, PyTorch 2.8.0+cu128); the current count is in `README.md`. The Linux CPU and CUDA runs were done
earlier on Bridges-2. The section below is the plan of October 2026, kept as written; Local-6 was trained on time.

Dates for Pragati:

- 11 October: report progress and blockers.
- 12 to 18 October: the Local-6 pilot, `slurm/pilot_local6.job` on an H100 (`h100-80`). It needs a working PSC login.
  If that is still blocked, Anshul submits it.

If Local-6 cannot be trained by the deadline, the agreed midterm fallback is to report it as implemented but not
trained. Dense and Global remain the midterm baseline.
