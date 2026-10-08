# Local router

**Standalone component with fairchem integration pending.**

The router and a linear-expert reference are implemented. The SO(2) layer
integration and OMol25 training runs remain to be done.

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
pair once, with no self edges. The composition calculation adds the central atom
with weight one. Supply all neighbors within the routing radius, even if that
radius exceeds the message cutoff. The receiver convention is agreed by the team.

Distances must retain gradients to positions. `reference_neighbors` does this
for small, non-periodic systems, using an O(N^2) search. The composition calculation
itself uses sparse indexed sums. Training will need the shared geometry helper.

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
Softmax and the small embedding/MLP dimensions are provisional; the YAML leaves
normalization and hidden dimensions open. These are not measured 6M-model settings.

Pass the shared task/charge/spin encoder as `conditioner`, with signature
`conditioner(task, charge, spin) -> [B, context_dim]`. No new encoding is defined
here. Unconditioned tests require explicit `allow_unconditioned=True`.

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
CUDA, fairchem integration, full-model forces, and MD stability remain untested.

## Handoff and dates

Yiwei is targeting a usable shared pipeline on 8 October: fixed pilot data,
Dense training/validation, W&B, checkpoints, and model integration points.
Local-6 integration depends on that code. His travel starts 9 October, so use
the 8 October meeting to confirm the encoder, geometry helper, exact SO(2)
modules, coefficient normalization, and any unfinished pipeline work.

Anshul accepted these dates for Pragati:

- 8 October: agree on the interface and implementation approach.
- 11 October: report progress and blockers.
- 15 October: router merged; Local-6 runs a few batches; force checks complete.
- 16–18 October: short pilot if integration and GPU access are ready.

If Local-6 is not running by 15 October, the agreed midterm fallback is to report
it as implemented but not trained. Dense and Global remain the midterm baseline.
Describe exactly which components are implemented; this package alone does not
establish a working Local-6 backbone.
