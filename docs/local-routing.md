# Local router

The router and a linear-expert reference are implemented in `local_routing/`. They are integrated into fairchem 2.21.0
as the Local-6 backbone (`training/local_moe.py`, `configs/variant/local6.yaml`; see `docs/TRAINING.md`). Local-6 has
been trained for 30 steps on real OMol25 data on Bridges-2. Long training runs are still to be done.

## Interface

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

The code uses radius 6, envelope power 3, eight experts, and shared coefficients.
Softmax, embedding size 16, and router hidden size 32 are provisional; the YAML
leaves normalization and hidden dimensions open. These are not measured 6M-model settings.

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

Seventeen component tests cover composition arithmetic, locality, batch isolation,
receiver mapping, cutoff smoothness, first/second derivatives, conditioning,
agreement between execution modes, and a toy energy-and-force optimizer step.
The toy scalar is not a molecular potential.

All 17 tests passed again on 8 October 2026, on macOS CPU with Python 3.12.14
and PyTorch 2.8.0. PyTorch warned that NumPy was absent in the temporary test
environment; these tests use tensor operations and do not require NumPy.
MD stability remains untested. CUDA, the fairchem integration and full-model forces are tested by
`tests/test_local_backbone.py`.

`tests/test_local_routing_contracts.py` adds 27 tests: the neighbor-list and
coefficient contracts above, the error paths, float32 paths, unequal structure
sizes, interleaved batch ids, permuted edges, isolated atoms, envelope powers 4
and 5, radius 0, and a pair at exactly the radius. Set `ROUTER_TEST_DEVICE=cuda`
to run both files on a GPU; the default is `cpu`. All 44 tests passed on
8 October 2026 on Windows CPU with Python 3.12.15 and PyTorch 2.8.0+cu128.

## Handoff and dates

Local-6 is integrated and tested. The full suite has 101 tests (`python -m unittest discover -s tests`), run on 9 October
2026 on Windows CPU (Python 3.12, PyTorch 2.8.0+cu128). The Linux CPU and CUDA runs were done earlier on Bridges-2.

Dates for Pragati:

- 11 October: report progress and blockers.
- 12 to 18 October: the Local-6 pilot, `slurm/pilot_local6.job` on an L40S (`l40s-48`). It needs a working PSC login.
  If that is still blocked, Anshul submits it.

If Local-6 cannot be trained by the deadline, the agreed midterm fallback is to report it as implemented but not
trained. Dense and Global remain the midterm baseline.
