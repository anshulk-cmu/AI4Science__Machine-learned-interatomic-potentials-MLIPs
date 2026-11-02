# Local+Global router

Date: 2026-11-01. Config: `configs/variant/local_global.yaml`.
Backbone: `training.local_global_moe.eSCNMDLocalGlobalMoeBackbone`.

## Definition

The new class extends `eSCNMDLocalMoeBackbone`. Each atom's routing MLP reads:

```text
[local composition, global composition of this structure, charge/spin/task context]
```

Both compositions share the same trainable element embedding. The local part is
the existing smooth average within 6 A, including the central atom. The global
part is the mean embedding of all atoms in this structure, divided by its atom
count N (the UMA 1.1 convention). It does not average across the minibatch and
does not count periodic images as extra atoms in the global mean.

The message cutoff stays 6 A. The local composition retains position gradients.
The global mean retains gradients to the shared element embedding. Each atom
gets eight coefficients; message edges use the receiving atom's coefficients.
The existing fairchem normalization, `softmax + 0.005`, gives a sum of 1.04.

A small `_composition_features` hook in the Local backbone lets the new class
add the global input. Graph construction, temporary-state cleanup, per-edge
LocalMOLE execution and activation-checkpoint chunks remain in the parent.
The existing Local-6 and Element paths continue to use the local feature alone.

The config inherits Local-6 and changes only the backbone class. At the default
width, the first routing layer changes from `Linear(256, 16)` to
`Linear(384, 16)`. All later layers keep their original widths and activations.
The class derives dimensions, device and dtype from the existing layer.

## Expected physical behavior

At fixed charge, spin and task, adding a distant fragment with different
composition leaves the local features unchanged but changes the global features.
Thus Local+Global is generally not size consistent even at 100 A separation.
The tests use a fixed responsive router to demonstrate this behavior. Nonzero
additivity error is not guaranteed for every possible set of learned weights;
for example, a model that ignores the global input recovers local behavior.

Separate structures in one minibatch must not affect one another. This is
different from putting two disconnected fragments inside the same structure.

Like Local-6, this class does not support graph parallelism, composition dropout,
merging into a single structure-level linear layer, or a routing radius above
the message cutoff without a separate neighbor graph.

## Parameters and checkpoints

The extra global input adds 128 x 16 = 2,048 weights to the first routing layer.

| Quantity | Count |
|---|---:|
| Total | 37,654,313 |
| Expert bank | 36,179,968 |
| Router, including shared composition embedding | 19,368 |
| MOLE layers | 24 |
| Effective active count under the project's existing convention | 6,000,425 |

The active count is not a FLOP or memory measurement. All expert outputs are
evaluated per edge by LocalMOLE. `tests/test_training_config.py` checks these
counts on the full-size model.

New inference checkpoints save the new class path and rebuild the larger MLP.
An existing Local-6 checkpoint cannot be loaded directly with `strict=True`
because the first layer has a different shape. A test-only conversion inserts
zero global-input weights and reproduces Local-6 energies and forces; production
training starts from the normal initialization of the new class.

## Validation

From the repository root, with the project environment:

```bash
python -B -m unittest discover -s tests -p 'test_local_global_backbone.py' -v
python -B -m unittest discover -s tests -v
```

The new tests cover manual composition arithmetic and input ordering, receiver
mapping, normalization, temporary-state cleanup, batch isolation, non-additivity
at 40 and 100 A, recovery of Local-6 when global weights are zero, finite-difference
forces, gradient updates for local/global/context inputs, charge/spin/task
conditioning, rotation/translation/permutation of routing coefficients, and a
real fairchem inference-checkpoint round trip including EMA weights.

On 2026-11-01, all nine new tests passed on CPU. The full regression suite then
passed all 188 tests in 71.295 seconds, including the existing Local-6 and Element
tests and the full-size parameter counts above. `git diff --check` also passed.
The environment was native Windows with the project's `omol25-env` and the
existing test-only `fcntl` shim. CUDA tests and production training are not part
of this result. Windows cache/TEMP setup is documented in `docs/element-router.md`.

## 200-step training acceptance (pending)

No production training or GPU memory measurement has been performed for this
variant. Native Windows lacks the POSIX/NCCL support required by the project's
CUDA training CLI; see `docs/TRAINING.md`. Unit tests do not replace this run.

On an allocated Linux GPU node, Anshul can run from the repository root:

```bash
PILOT_REPO="$PWD" PILOT_RUN_PREFIX=smoke_local_global \
PILOT_STEPS=200 PILOT_SEED=0 \
PILOT_TRAIN_SAMPLES=256 PILOT_VAL_SAMPLES=64 \
PILOT_EVAL_EVERY=200 PILOT_CKPT_EVERY=200 \
bash slurm/pilot.sh local_global
```

This uses a small random OMol25 subset. Unset the two sample-count variables
for the fixed pilot manifests. The runner accepts `local_global` and temporarily
uses the Local-6 44,000 MiB guard (H100-80 or L40S-48). This guard is provisional,
not a measured minimum for the new model.

Record the run directory, GPU, peak memory, runtime, finite final metrics and
final inference checkpoint after 200 completed steps. The size-consistency stage
must execute and report its Delta values; Local+Global is not required to pass
an additivity tolerance. Review nonzero Delta as the expected global-routing
effect, not as a training failure.
