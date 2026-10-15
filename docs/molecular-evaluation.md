# Molecular checkpoint evaluation

This evaluates trained Dense, Global and Local-6 inference checkpoints. It adds a
molecular check to the existing slab audit. It does not train a model or measure MAE.
The evaluator has passed controlled-potential tests. Pilot checkpoints have not
been evaluated with it yet.

## What runs

Three fixed neutral singlet pairs: water + methane, ammonia + carbon dioxide, and
water + water. The last pair is a same-composition control. Each pair is evaluated
at bounding-plane gaps of 8, 12, 20 and 40 Å. The script checks the actual minimum
cross-fragment distance against the loaded model cutoff. Systems are nonperiodic;
isolated and combined evaluations use the same cell and fragment coordinates.
Molecules are rotated off the coordinate axes, following the backbone tests.

For each gap, save:

- `E(pair) - E(A) - E(B)`, in eV.
- Maximum force change on each fragment, in eV/Å.
- Repeated energy and force differences on identical inputs, as a numerical-noise diagnostic.

Central finite differences check every force component on isolated water and
ammonia, at steps 0.001 and 0.0003 Å. These exercise geometry-dependent routing
within bonded molecules. Inspect step convergence; repeated-call noise alone does
not establish a finite-difference tolerance. No automatic pass threshold is imposed.

This is a small diagnostic suite, not a representative OMol25 benchmark or a test
of charged/open-shell fragment additivity. A local result on these pairs does not
prove locality for all inputs. Runtime, memory, and MD drift are separate work.

## Run

Use the team's Linux/CUDA environment with fairchem-core 2.21.0. From the repo root:

```sh
python -m audit.molecular /path/to/final/inference_checkpoint.pt \
  --run-config /path/to/resolved_config.yaml \
  --label local6_seed0_step50000 \
  --output /path/to/evaluation/local6_seed0
```

Repeat with Dense and Global checkpoints, then with the other seeds. Keep the
same pairs, gaps, precision and finite-difference steps. Start with one checkpoint
per variant to check loading. The output directory must not already exist.
`--dtype float32` is available if float64 does not fit; do not silently compare
mixed precisions. The existing loader uses CUDA and resets the ASE calculator
cache on each call. The real checkpoint-loading path remains untested locally.

Outputs:

| File | Contents |
|---|---|
| `locality.csv` | One row per pair and gap, with explicit units |
| `finite_difference.csv` | Predicted and numerical forces, and absolute differences |
| `geometries.json` | Exact combined geometries and cells |
| `metadata.json` | Checkpoint/config hashes, copied config text, evaluator commit and dirty status, package versions, loaded routing class, completion status |

A failure sets the metadata status to `failed` and exits with an error. Check the
loaded routing class against the run config and confirm the checkpoint training
step from the run records. A filename or user-supplied label is not verification
of the model variant or training step. Keep outputs outside the tracked tree until
reviewed; do not commit checkpoints or private paths/configs without checking them.

```sh
python -m audit.plot_molecular /path/to/evaluation/*/locality.csv \
  --output /path/to/evaluation/locality.png
```

The plot shows absolute energy residuals and the larger force change across the
two fragments. Curves are per run; no seed confidence interval is inferred.

## Integration inspected

At upstream commit `9513431`, `training/local_moe.py` uses `local_composition` with
fairchem's element embeddings, routing MLP and coefficient normalization. It maps
weights to the receiving atom. `LocalMOLE` mixes expert outputs before activation.
Distances come from the backbone graph; the calculation does not explicitly detach
them. The existing `test_local_backbone.py` checks the force-gradient path and
locality with small random-weight models. Those tests are distinct from evaluating
trained pilot checkpoints. Local-12 still needs a separate routing neighbor list
when its radius exceeds the message cutoff.

## Evaluator tests

Only NumPy and ASE are needed for the measurement tests:

```sh
python -m unittest discover -s tests -p 'test_molecular_audit.py' -v
```

Tests use analytic toy potentials with known behavior, including a deliberate
global-composition dependence and a force-sign error. They do not require model
weights and are not evidence of trained-model performance.
