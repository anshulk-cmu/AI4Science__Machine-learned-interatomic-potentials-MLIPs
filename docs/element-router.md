# Element router (radius 0)

Date: 2026-11-01. Implementation: `configs/variant/element.yaml`.

## Definition

Element inherits the complete Local-6 config and changes only `routing_radius` to
0.0. It uses `training.local_moe.eSCNMDLocalMoeBackbone`. The existing
`local_composition` zero-radius branch returns the central atom's element
embedding. No new router class or fairchem patch is needed.

The router reads `[element_embedding(Z_i), charge_spin_task_context(structure_i)]`.
At fixed charge, spin and task, atoms of the same element have the same expert
coefficients, regardless of positions or neighboring elements. The production
task mapping is still `omol: omol`.

The message cutoff stays 6 A. Energy and forces still depend on geometry through
the backbone. The router outputs eight coefficients per atom. Message edges use
the coefficients of the receiving atom. fairchem 2.21 normalizes with
`softmax + 0.005`, so these eight coefficients sum to 1.04, not 1.

For non-interacting fragments with the same conditioning as their union, the
energy is additive. This statement does not cover changing the total charge,
spin or task between the fragments and the union.

## Parameter counts

The architecture and state-dict layout are the same as Local-6 and Global.
`tests/test_training_config.py` checks the full-size model:

| Quantity | Count |
|---|---:|
| Total | 37,652,265 |
| Expert bank | 36,179,968 |
| Router, including composition embedding | 17,320 |
| MOLE layers | 24 |
| Effective active count, using the existing project counting rule | 5,998,377 |

The active count is a parameter accounting convention. LocalMOLE evaluates all
expert outputs per edge; this number does not measure FLOPs or resident memory.

## Tests

From the repository root in the project's environment:

```bash
python -B -m unittest discover -s tests -p 'test_element_router.py' -v
python -B -m unittest discover -s tests -p 'test_training_config.py' -v
python -B -m unittest discover -s tests -v
```

The Element tests use small float64 models and no external dataset. They check
config inheritance, native normalization, same-element routing, independence
from geometry and neighboring elements, charge/spin/task conditioning, batch
isolation, additivity at 100 A, finite-difference forces, an energy-and-force
optimizer step, and a state-dict round trip. Set `ROUTER_TEST_DEVICE=cuda` to run
these tests on a GPU.

On 2026-11-01, the complete CPU suite passed: 179 tests in 65.736 seconds,
including all eight Element tests and the full-size parameter-count check.
The environment was native Windows with the project's `omol25-env`; the model
tests use the existing test-only `fcntl` shim. CUDA tests and production training
are not included in this result.

On Windows, set `FAIRCHEM_CACHE_DIR`, `WARP_CACHE_PATH` and `MPLCONFIGDIR` to
writable cache directories before running the tests. If a sandbox gives TEMP a
path containing braces, set `TEMP` and `TMP` to a writable plain directory too:
the existing shared test helper passes temporary paths through Hydra overrides.

## 200-step training acceptance (pending)

The full training smoke run has not been performed. Native Windows cannot run
the project's CUDA training CLI because it requires POSIX `fcntl` and NCCL;
see `docs/TRAINING.md`. CPU unit tests do not replace this acceptance run.

On an allocated Linux GPU node, Anshul can run the following from the repository
root. This invokes the existing runner; it does not submit a Slurm job:

```bash
PILOT_REPO="$PWD" PILOT_RUN_PREFIX=smoke_element \
PILOT_STEPS=200 PILOT_SEED=0 \
PILOT_TRAIN_SAMPLES=256 PILOT_VAL_SAMPLES=64 \
PILOT_EVAL_EVERY=200 PILOT_CKPT_EVERY=200 \
bash slurm/pilot.sh element
```

This smoke run uses a small random OMol25 subset. For the matched pilot recipe,
unset `PILOT_TRAIN_SAMPLES` and `PILOT_VAL_SAMPLES`; the script then verifies the
fixed pilot manifests before training. Use a new run prefix for each experiment.

`slurm/pilot.sh` accepts `element` and temporarily uses Local-6's conservative
44,000 MiB GPU guard (H100-80 or L40S-48). Element uses the same per-edge expert
execution, but its actual peak memory has not been measured. The local RTX 3070
has 8 GiB and is not a validated GPU for this recipe.

Before marking training acceptance complete, record the run directory, GPU,
peak memory, elapsed time, final validation metrics, final checkpoint and the
size-consistency result. Check that all 200 steps complete with finite metrics
and that the final checkpoint passes the existing size-consistency stage.
