# Molecular checkpoint evaluation

This evaluates trained Dense, Global and Local-6 inference checkpoints. It adds a
molecular check to the existing slab audit. It does not train a model or measure MAE.
The evaluator has passed controlled-potential tests. It ran on the final checkpoints
of the nine pilots on 2026-10-18 (job 49437122, results at the end of this file).

## What runs

Three fixed neutral singlet pairs: water + methane, ammonia + carbon dioxide, and
water + water. The molecules come from ASE's built-in G2 set (`ase.build.molecule`),
so no data file is needed. All four are closed-shell, and the union and each part get
charge 0 and spin multiplicity 1. The last pair is a same-composition control: the
Global router averages the element embeddings over the structure, and that average
is the same for two waters as for one, so Global must also be additive there. Each pair is evaluated
at bounding-plane gaps of 8, 12, 20 and 40 Å. The script checks the actual minimum
cross-fragment distance against the loaded model cutoff. Systems are nonperiodic;
isolated and combined evaluations use the same cell and fragment coordinates.
Molecules are rotated off the coordinate axes, following the backbone tests.

For each gap, save:

- `E(pair) - E(A) - E(B)`, in eV.
- Maximum force change on each fragment, in eV/Å.
- Repeated energy and force differences on identical inputs, as a numerical-noise diagnostic.

Central finite differences check every force component on each isolated molecule
(H2O, CH4, NH3 and CO2, once each), at steps 0.001 and 0.0003 Å. These exercise geometry-dependent routing
within bonded molecules. Inspect step convergence; repeated-call noise alone does
not establish a finite-difference tolerance. No automatic pass threshold is imposed.

This is a small diagnostic suite, not a representative OMol25 benchmark or a test
of charged/open-shell fragment additivity. A local result on these pairs does not
prove locality for all inputs. Runtime, memory, and MD drift are separate work.

## Run

Use the team's environment with fairchem-core 2.21.0 and a CUDA GPU. From the repo root,
for one pilot run on Bridges-2:

```sh
RUN=/ocean/projects/che260063p/shared/runs/pilot_local6_s0_49041936
/ocean/projects/che260063p/shared/envs/mlip/bin/python -m audit.molecular \
  $RUN/train/*/checkpoints/final/inference_ckpt.pt \
  --run-config $RUN/train/*/canonical_config.yaml \
  --label local6_s0_step50000 \
  --output /path/to/evaluation/pilot_local6_s0
```

`canonical_config.yaml` is the resolved config that fairchem saves for each run.
`slurm/molecular_eval.job` runs all nine seeded pilots in one job and also writes the
combined CSVs and the figure (results below).

Repeat with Dense and Global checkpoints, then with the other seeds. Keep the
same pairs, gaps, precision and finite-difference steps. Start with one checkpoint
per variant to check loading. The output directory must not already exist.
`--dtype float32` is available if float64 does not fit; do not silently compare
mixed precisions. Use float64 for the finite differences. The rounding error of a
central difference is about eps |E| / h, with eps the relative precision. The total
energy of water is about -2080 eV. In float32 (eps = 6e-8) and h = 0.001 Å this
can reach 0.1 eV/Å, which hides any real force error; in float64 (eps = 1.1e-16) it
is about 2e-10 eV/Å. This is an estimate, not a measurement: where fairchem adds the
float64 element references decides the true float32 error. The existing loader uses
CUDA and resets the ASE calculator cache on each call, so the repeat columns are
fresh calculations (`test_loader_recomputes_identical_atoms`).

Outputs:

| File | Contents |
|---|---|
| `locality.csv` | One row per pair and gap, with explicit units |
| `finite_difference.csv` | Predicted and numerical forces, and absolute differences |
| `geometries.json` | Exact combined geometries and cells |
| `metadata.json` | Checkpoint/config hashes, copied config text, evaluator commit and dirty status, package versions, backbone class (`eSCNMDBackbone` Dense, `eSCNMDMoeBackbone` Global, `eSCNMDLocalMoeBackbone` Local-6), completion status |

A failure sets the metadata status to `failed` and exits with an error. Outside a git
checkout (the Bridges-2 copy has no `.git`) the evaluator commit is `null`. Check the
backbone class against the run config and confirm the checkpoint training
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

The tests need NumPy and ASE, torch for the loader cache test and matplotlib for the plot test:

```sh
python -m unittest discover -s tests -p 'test_molecular_audit.py' -v
```

Tests use analytic toy potentials with known behavior, including a deliberate
global-composition dependence and a force-sign error. They do not require model
weights and are not evidence of trained-model performance.

## Results: final checkpoints of the nine pilots (2026-10-18)

Job 49437122 on Bridges-2: one V100-32, account `che260063p`, COMPLETED, exit 0,
12 min 15 s (`sacct`). At 1 SU per V100 GPU-hour this is about 0.2 SU. We computed
that number from the elapsed time; `job_info` did not list the job yet. The job ran
`slurm/molecular_eval.job` on a copy of commit `3b6c7c4` (`shared/repo_pr4`). It used
the default settings: float64, gaps 8, 12, 20 and 40 Å, steps 0.001 and 0.0003 Å.

Files:

- `results/molecular_locality.csv`: 108 rows (9 runs, 3 pairs, 4 gaps).
- `results/molecular_finite_difference.csv`: 810 rows (9 runs, 45 coordinates, 2 steps).
- `results/molecular_metadata.json`: hashes, versions and backbone class of each run.
- `results/figures/molecular_locality.png`: the figure from `audit/plot_molecular.py`.

Delta = E(A+B) - E(A) - E(B), in meV, at a gap of 8 Å:

| Variant | Pair | s0 | s1 | s2 |
|---|---|---|---|---|
| Dense | H2O + CH4 | 0 | 4.5e-10 | 0 |
| Dense | NH3 + CO2 | 0 | 0 | 0 |
| Dense | H2O + H2O | 0 | 0 | 0 |
| Global | H2O + CH4 | 3.006 | 13.27 | 0.2945 |
| Global | NH3 + CO2 | 34.07 | 10.79 | 38.10 |
| Global | H2O + H2O | 0 | 0 | 0 |
| Local-6 | H2O + CH4 | -2.3e-10 | 2.3e-10 | 0 |
| Local-6 | NH3 + CO2 | -9.1e-10 | 0 | 0 |
| Local-6 | H2O + H2O | 0 | 0 | 0 |

Largest force change on either fragment, in meV/Å, at a gap of 8 Å:

| Variant | H2O + CH4 | NH3 + CO2 | H2O + H2O |
|---|---|---|---|
| Dense, 3 seeds | 9e-13 to 1.1e-12 | 1.9e-12 to 3.8e-12 | 5e-13 to 1.2e-12 |
| Global, 3 seeds | 5.07 to 5.93 | 20.9 to 46.0 | 6e-13 to 1.2e-12 |
| Local-6, 3 seeds | 9e-13 to 1.1e-12 | 1.7e-12 to 3.0e-12 | 6e-13 to 8e-13 |

Measured:

- All 9 runs completed. The backbone classes are `eSCNMDBackbone` (Dense), `eSCNMDMoeBackbone` (Global) and `eSCNMDLocalMoeBackbone` (Local-6). The cutoff is 6.0 Å.
- The smallest distance between the two fragments is 8.02 Å.
- For Dense and Local-6, the largest |Delta| is 9.1e-13 eV. This is at most 1 float64 ulp of the pair energy.
- For Dense and Local-6, the largest force change is 7.3e-15 eV/Å.
- For Global, Delta is 0.29 to 13.3 meV for H2O + CH4 and 10.8 to 38.1 meV for NH3 + CO2.
- For Global, the force change is 5.1 to 5.9 meV/Å for H2O + CH4 and 20.9 to 46.0 meV/Å for NH3 + CO2.
- For Global, Delta and the force change have the same value at all four gaps.
- For Global, H2O + H2O gives Delta = 0 exactly and a force change of at most 1.2e-15 eV/Å.
- Repeated calls on the same input gave the same energy (difference 0) and force differences of at most 8.4e-15 eV/Å.
- The largest finite-difference force error per variant is 5.0e-5 to 5.1e-5 eV/Å at h = 0.001 Å and 4.5e-6 to 4.6e-6 eV/Å at h = 0.0003 Å. The ratio of the two is 11.11 for each variant. The largest force component is 1.80 eV/Å (CO2).
- A run of the three s0 checkpoints on a local RTX 5070 Ti GPU gave the same pair energies and Delta values, digit for digit, as the V100.

Interpreted:

- Dense and Local-6 are additive on these pairs to the float64 floor. Far-apart molecules do not change their energy or forces.
- Global is not additive. A far molecule of a different composition changes the energy and the forces of the other molecule.
- The Global error does not decay with the gap. This agrees with a router that reads the composition of the whole structure and not the distance.
- The H2O + H2O control agrees with the same cause. The mean composition of two waters equals that of one water, so the Global router gives the same mixing and Delta is zero.
- The finite-difference error falls by (0.001 / 0.0003)^2 = 11.1. So the error is the O(h^2) truncation error of the central difference, and the forces are the negative gradient of the energy for all three variants, including the Local-6 router path.

Caveats:

- The Global values are meV, while the slab test gave eV. These are small molecules of similar composition, so the router input changes less. This was not tested further.
- The Global Delta changes by a factor of 45 between seeds for H2O + CH4. Three seeds do not give a stable mean.
- Three pairs of four small molecules are a diagnostic, not an OMol25 benchmark. Charged and open-shell fragments are not tested.
- The evaluator commit in the metadata is `null`, because the Bridges-2 copy has no `.git`. The copy came from commit `3b6c7c4`.
