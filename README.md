# Local Composition Routing for Mixture-of-Experts Interatomic Potentials

Course project for CMU 11-785 Introduction to Deep Learning, Fall 2026 (AI4Science track, machine-learned interatomic potentials).
Team 6: Anshul Kumar, Pragati Bhattad, Yiwei Xia. Status (15 November 2026): midterm report submitted (23 October); all 8 long training runs (seed 0, 400,000 steps) are finished on CMU Babel and tested on the full held-out test list (399,276 structures); the final report and video are next.

## Summary

A good interatomic potential gives two far-apart, non-interacting molecules a total energy equal to the sum of their separate energies.
This property is called size consistency. UMA, a widely used foundation model for molecules and materials, does not have it. Its
mixture-of-experts layers choose their weights from the average element composition of the whole structure, with no distance cutoff,
so a molecule placed anywhere changes the predicted energy of every other molecule.

We propose local composition routing: the router sees a smooth, bounded average over the neighbors of each atom instead of the whole
structure. Our main model, Local-6, uses a 6 angstrom radius. The question of the project is whether Local-6 keeps the accuracy that
routing adds over a model without experts, while being size consistent. We audit released models without training, and we train
Dense, Global (UMA-style routing), Local-6, and two controls on a fixed subset of OMol25 with the `fairchem` code.

## Results of the long runs (final training)

Eight variants, one seed (0), 400,000 steps each, trained on the 70 percent training list of `train_4M` (split by composition, `docs/data-split.md`). Validation errors are the final checkpoints evaluated again on the fixed 5,000-row monitor list, at full precision (`results/final_eval/val5k_metrics.csv`; the training log gives the same values rounded to 4 decimals in eV, `results/long_runs_validation.csv`). The gap is the additivity error of two far-apart Al and Ni slabs at a gap of 8 angstrom, from `audit/check_checkpoint.py`.

| Model | Energy MAE (meV/atom) | Force MAE (meV/A) | Gap of two far-apart slabs (meV) |
|---|---:|---:|---|
| Dense | 5.382 | 28.967 | 3e-6 (zero) |
| Scaled Dense | 5.004 | 28.195 | 4e-6 (zero) |
| Frozen | 5.120 | 27.917 | 3e-6 (zero) |
| Global | 4.918 | 27.382 | -62,350, the same at 8 to 100 A |
| Element | **4.657** | **26.192** | 3e-6 (zero) |
| **Local-6** | 4.886 | 27.310 | 3e-6 (zero) at 8 to 100 A |
| Local+Global | 4.986 | 27.591 | -9,226, the same at 8 to 100 A |
| Local-12 | 5.014 | 27.525 | 3,916 at 8 A, 320 at 10 A, 3e-6 (zero) from 15 A |

The SE of one validation MAE is 0.24 to 0.29 meV/atom and 0.61 to 0.66 meV/A (5,000 rows, row SEs), so the validation table alone does not rank the routed models.

**Test set (the full held-out list, 399,276 structures, final checkpoints, float32, `docs/final-evaluation.md`).** The SE treats each composition as one sampling unit, because the test list was split by composition (110,864 compositions; `scripts/cluster_se.py`).

| Model | Energy MAE (meV/atom) | SE | Force MAE (meV/A) | SE |
|---|---:|---:|---:|---:|
| Dense | 5.611 | 0.421 | 28.653 | 0.216 |
| Scaled Dense | 5.322 | 0.362 | 27.915 | 0.210 |
| Frozen | 5.464 | 0.451 | 27.719 | 0.211 |
| Global | 5.447 | 0.575 | 27.183 | 0.207 |
| Element | **5.144** | 0.443 | **26.074** | 0.201 |
| Local-6 | 5.337 | 0.477 | 27.190 | 0.208 |
| Local+Global | 5.517 | 0.600 | 27.513 | 0.209 |
| Local-12 | 5.348 | 0.474 | 27.259 | 0.209 |

Paired differences on the same rows (A minus B, composition SE, z = difference / SE):

| A minus B | Force (meV/A) | z | Energy (meV/atom) | z | Energy without the 804 two-atom rows | z |
|---|---:|---:|---:|---:|---:|---:|
| Global minus Dense | -1.471 (SE 0.018) | -81 | -0.164 (SE 0.180) | -0.9 | -0.378 (SE 0.036) | -10.5 |
| Local-6 minus Global | +0.007 (SE 0.011) | +0.6 | -0.110 (SE 0.110) | -1.0 | +0.013 (SE 0.027) | +0.5 |
| Scaled Dense minus Dense | -0.738 (SE 0.016) | -46 | -0.289 (SE 0.073) | -4.0 | -0.255 (SE 0.033) | -7.7 |
| Element minus Dense | -2.580 (SE 0.022) | -115 | -0.467 (SE 0.048) | -9.8 | -0.512 (SE 0.030) | -17.2 |
| Element minus Global | -1.109 (SE 0.016) | -68 | -0.303 (SE 0.150) | -2.0 | -0.135 (SE 0.034) | -3.9 |

All 14 pairs, the row SEs and the size bins: `docs/final-evaluation.md`, `results/final_test_paired_cluster.csv`, `results/final_test_by_size.csv`. Every SE covers the sampling of test structures only; every number is one training seed ([limits](docs/limits-and-provenance.md)).

- **Measured.** On test, Local-6 and Global have the same force error within the error (+0.007 meV/A, SE 0.011). Both are lower than Dense, by 1.471 (Global) and 1.464 meV/A (Local-6). Local-6 has a zero gap at every tested distance; Global has a gap of -62 eV that does not shrink with distance.
- **Measured.** No energy difference between Dense, Global and Local-6 is established on the test list (z -0.9 and -1.0). The energy differences are dominated by 804 two-atom structures (0.2 percent of the rows); one composition, `S2` (358 rows), carries 95 percent of the summed Local-6 minus Global difference. The force results are the evidence.
- **Measured.** Element routing (radius 0) has the lowest force error on test and the lowest energy error in the table, and a zero gap. Its energy gap to Dense holds with composition SEs and without the two-atom rows; its energy gap to Global is at z -2.0 on the full list and smaller without the two-atom rows (-0.135, z -3.9), so it is weaker evidence. Local-12 is not additive inside its own 12 A radius, as expected, and is zero from 15 A. Local+Global keeps a gap, 7 times smaller than Global.
- **Measured.** Scaled Dense, a Dense model with the start scale and step size of Frozen, has a test force error 0.738 meV/A below Dense without any experts, half of the 1.471 meV/A gain of Global.
- **Interpreted.** Much of the force gain of Global over Dense is reached without a router, so the gain should not be credited to routing alone. The size-consistency result is the clearer one: Local-6 and Element remove the gap of Global with no measured cost in force error.
- **Measured.** Along training, Global's gap is -9.8 meV at step 0 and +0.42 eV at step 10,000; from step 40,000 on it lies between 1.3 eV and 188 eV in size, and it changes sign 6 times over the run. Local-6 stays at 4e-6 meV or less at all 52 saved checkpoints ([`docs/size-consistency-long-runs.md`](docs/size-consistency-long-runs.md)).
- Details of the runs, the failures on Babel and the exact launch commands: [`docs/babel.md`](docs/babel.md).

## Results at the midterm

All trained results use 3 seeds, 50,000 steps, the same fixed training and validation lists, and the same backbone (about 6 million
active parameters, cutoff 6 angstrom). Errors are on 5,000 validation structures from OMol25. Numbers are means over the 3 seeds; full
per-seed values, intervals and tests are in `docs/` and in the report appendix.

| Model | Force MAE (eV/A) | Energy MAE per atom (eV) | Additivity error of the final checkpoints (Al + Ni slabs) |
|---|---:|---:|---|
| Dense (no experts) | 0.0388 | 0.0142 | 3e-6 meV (zero) |
| Global (UMA-style routing) | 0.0372 | 0.0131 | -5.1, -6.6 and +2.4 eV for seeds 0, 1, 2 |
| Local-6 | 0.0372 | 0.0130 | 2e-6 to 3e-6 meV (zero) |
| Frozen (Global with a constant router) | 0.0375 | 0.0131 | 3e-6 meV (zero) |
| Scaled Dense (Dense with the start scale and step size of Frozen) | 0.0376 | 0.0135 | 2e-6 to 3e-6 meV (zero) |

- **Audit of released models (RQ1).** MACE gives an additivity error of zero within the noise floor. ORB-v3 and all UMA models do not, by 0.5 eV to hundreds of eV for two metal slabs, and the error does not shrink with distance. Our protocol is fully specified and differs from the published one; we compare with the published values in kind, not digit by digit. See `status-wip/README.md` and `audit/README.md`.
- **Accuracy (RQ2).** Global and Local-6 reduce the force error by about 4 percent against Dense (Welch p = 0.0014 and 0.0013). We detect no difference between Global and Local-6 (p = 0.95). The retention share of the Global gain kept by Local-6 is 1.01 for force and 1.06 for energy. Its one-sided 95 percent lower bound, with the seed spread treated by a Student-t distribution (2 degrees of freedom), is 0.69 for force and 0.50 for energy; with a normal distribution it is 0.84 and 0.72. These are pilot numbers (3 seeds, 50,000 steps; `scripts/pilot_statistics.py results/pilot_final_metrics.csv`, rerun 16 November 2026).
- **Size consistency through training (RQ3).** Dense and Local-6 stay at the float64 floor at all 66 of their saved checkpoints (33 each); Global is off at all 33 of its own, 99 in total. Global is 10 to 69 meV off before training and 2.4 to 6.6 eV off after 50,000 steps, the same at gaps of 8 and 100 angstrom. See `docs/size-consistency-training.md`. The same pattern holds for far-apart molecule pairs (`docs/molecular-evaluation.md`).
- **Cost (RQ4).** At inference, with energy and forces on the same GPU, Local-6 is 3.3 to 3.5 times slower than Global (627 against 2,201 atoms per second on a V100; 1,736 against 5,708 on an H100). Global is 1.2 to 1.3 times slower than Dense. See `docs/inference-timing.md`. The nine pilots cost 135 GPU service units; the whole project about 195 of 2,950.
- **Mechanism (RQ5).** The trained router is far from uniform, but the experts differ mostly by their random start, and training moved them together (`docs/router-check.md`). A model with a constant router keeps 84 percent of the force gain of Global and 97 percent of the energy gain (`docs/frozen-router.md`); its force error is 0.7 percent higher than Global in all 3 seeds (p = 0.18). A plain Dense model with the start scale and step size that a constant router implies (no experts, no router, 5,981,057 parameters) keeps 74 percent of the force gain and 60 percent of the energy gain (`docs/scaled-dense.md`). We detect no difference between the two controls. We conclude that a large part of the gain over Dense can be reproduced without experts or routing, by start scale and step size alone. Global and Local-6 stay about 1.1 percent lower in force error than the scaled Dense, which is not significant with 3 seeds, so we cannot say that learned routing adds nothing. The two controls differ in two details that we did not test on a small model: the shape of the initial weights and gradient clipping.

## Limitations

- Three seeds per model; the proposal planned five. Intervals and bounds with 3 seeds are approximate.
- One validation set of 5,000 structures, which are larger on average than the training structures (102 against 55 atoms).
- Short training: about 1/800 to 1/870 of an official run, on a 200,000-structure subset of the 4M training split. The target normalizer (1.433569) is a placeholder taken from an official run.
- Size consistency along training uses one ideal Al + Ni geometry; the audit of released models uses 21 geometries.
- Global and Local-6 were trained on different GPU types (V100 and H100), so their training times are not comparable. The inference timings use the same GPU type for all models, on shared nodes.
- (Midterm) Element, Local-12 and Local+Global routing were not trained at the midterm; they are trained in the long runs above. All three have configs and backbone tests; see [the Element router notes](docs/element-router.md), [the Local-12 notes](docs/local12.md) and [the Local+Global notes](docs/local-global-router.md) for their pending training acceptance. The additivity test covers neutral singlet pairs only; we train on all charges and spins.

### Limits of the long runs

Full list, with the measurements behind each point: [`docs/limits-and-provenance.md`](docs/limits-and-provenance.md).

- One training seed per variant, and the GPU path is not deterministic (`deterministic: False`; steps trained twice after a resume never match exactly, and drift apart by 1 to 3 percent of the loss over thousands of steps). Run-to-run noise at 400,000 steps is not measured, so force differences near 0.1 meV/A cannot be separated from it.
- The standard errors cover the sampling of test structures by composition only. Energy differences between models are dominated by 804 two-atom structures.
- The runs cover 1.049 epochs: epoch 1 starts at step 381,159, so the last 18,841 steps are a second pass, in the same order for all runs. The validation at step 381,159 is not in `results/long_validation_curves.csv`.
- Each run used two or three code versions (five in all), recorded only as a hash; three of the five hashes map to commits, two do not. The changes between the reproduced versions touch no model, config or loss file.
- The runs resumed many times after preemptions on mixed GPU types (V100, L40S, RTX A6000, RTX 6000 Ada, RTX PRO 6000 Blackwell). The test list was evaluated in float32. The final checkpoint is reported, without selection. Size consistency along training uses one slab pair.

## Repository layout

| Folder or file | What it holds |
|---|---|
| `status-wip/` | Reports on the audit: write-up, final tables and plots |
| `audit/` | Audit code (slab test, null tests, neon probe, forcing test, molecular additivity and force checks `molecular.py`), with its own README and the exact commands |
| `local_routing/` | Local composition, router interface, receiving-atom adapter and reference linear experts |
| `training/` | Entry point for the fairchem training CLI, the Local-6, Local-12 and Element backbone (`local_moe.py`), the Local+Global backbone (`local_global_moe.py`), checkpoint selection and averaging (`checkpoints.py`), the frozen-router and scaled-Dense variants, the out-of-memory guard and the step-based scheduler |
| `configs/` | Training configuration: `train.yaml`, data configuration, the variants `dense`, `global`, `local6`, `local12`, `element`, `local_global`, `global_frozen`, `dense_scaled`, element references |
| `slurm/` | Bridges-2 and Babel job files and scripts (the long runs on Babel use `long_run.sh`, `long_run.job` and `submit_chain.sh`, see `docs/babel.md`): environment, data download, smoke tests, pilots, Wave 1 (`wave1_*.job`), long runs as chained segments (`long_run.sh`, `submit_chain.sh`), size consistency, router check, timing, molecular evaluation |
| `scripts/` | Data preparation and audit, pilot sample lists, size consistency along training, router check, inference timing, curve comparison, the pilot statistics (`pilot_statistics.py`, input `results/pilot_final_metrics.csv`) and the report figures and tables (`make_midterm_figures.py`, `audit_summary_table.py`) |
| `manifests/` | The OMol25 data manifest and the fixed training and validation lists with their hashes, including the 70/20/10 split of `train_4M` by composition for the final runs (`split_*.txt`, `docs/data-split.md`) |
| `results/` | Raw and summary result files (CSV) and figures behind every number in the reports |
| `tests/` | Unit tests (336, 3 skipped): routing, the 8 variants, the long-run chain, checkpoint tools, final metrics and statistics, audit code, training configuration, frozen router, scaled Dense, router check, molecular checks, timing |
| `docs/` | Detailed reports: `TRAINING.md` (setup and run), `local-routing.md`, `size-consistency-training.md`, `router-check.md`, `frozen-router.md`, `scaled-dense.md`, `molecular-evaluation.md`, `inference-timing.md`, `final-training-plan.md`, `data-split.md` (the 70/20/10 split of the final runs), `launch-checklist.md` (how to start the final training), `final-evaluation.md` (test and validation results of the long runs), `limits-and-provenance.md` (what the long-run numbers do not cover, and the code version of each run) |
| `requirements.txt`, `requirements-training.txt`, `requirements-mace.txt` | Environments: UMA, ORB-v3 and fairchem (Python 3.12, PyTorch 2.8 with CUDA 12.8); extra training packages; a separate environment for MACE |

Model weights, checkpoints, the OMol25 data and the run folders are not in the repository. They live on CMU Babel (`/data/user_data/anshulk/mlip`; the Bridges-2 copy went offline on 9 November 2026); see `docs/babel.md` and `docs/TRAINING.md` for the paths and for how the data is prepared.

## Reproducing the results

Collect final validation metrics from new local run folders with
`scripts/collect_final_metrics.py`; see [the collection guide](docs/final-metrics.md)
for the 5-seed/8-variant checks, W&B/log sources, resume manifests and merging the
existing pilot CSV. Long-run metrics must go into a separate table.

Analyze the collected table with
`python scripts/pilot_statistics.py results/final_metrics.csv --require-final`.
The script supports 8 variants and 5 seeds, with dynamic Student-t degrees of
freedom and weighted exact resampling; see [the statistics guide](docs/final-statistics.md).
Omit `--require-final` to reproduce the existing three-seed pilot statistics.

Plot the final table with
`python scripts/make_accuracy_figure.py results/figures/final_accuracy.pdf --metrics results/final_metrics.csv --metrics-only --require-final`.
See [the figure guide](docs/final-figures.md) for training curves, descriptive
long-run plots and `make_ensemble_figure.py` report-set comparisons.

1. Create the environments from the requirements files; the steps are in the header of each file. The UMA weights are gated on Hugging Face: log in and accept the license first. Never commit a token.
2. Run the tests: `python -m unittest discover -s tests`. All 336 tests pass on Windows, 3 skipped (2026-11-16). One configuration-hash test stores hashes measured on Windows and may need its hashes refreshed on Linux.
3. The audit, the pilots and every follow-up experiment have a job file in `slurm/` and a report in `docs/` or `audit/` that gives the exact commands, the job identifiers, the settings and the files that hold the results. Start with `docs/TRAINING.md`.
4. A run writes `settings.txt` and `summary.txt`; check that `settings.txt` shows the steps, the seed, the variant and the sample list hashes you expect before you trust the run.

## Contributions

The list follows the commit and pull request history of this repository.

- **Anshul Kumar:** the audit of released models and its benchmark code; the Bridges-2 setup; the integration of the local router into the training code; the nine pilots and their analysis; size consistency along training, the router check, the frozen-router and scaled-Dense runs, and the inference timing; review and merge of the molecular checks.
- **Pragati Bhattad:** the standalone local composition router with component tests (`local_routing/`); the molecular additivity and force checks of the final checkpoints (`audit/molecular.py`).
- **Yiwei Xia:** the first training pipeline for Dense and Global on OMol25 (configuration, data manifest and audit, training notes); an independent statistical analysis of the pilots (means, Student-t intervals, retention, Welch tests, exact seed resampling).

## Team

| Name | Email | GitHub |
|---|---|---|
| Anshul Kumar | anshulk@andrew.cmu.edu | [anshulk-cmu](https://github.com/anshulk-cmu) |
| Pragati Bhattad | pbhattad@andrew.cmu.edu | [aptenosys](https://github.com/aptenosys) |
| Yiwei Xia | yiweixia@andrew.cmu.edu | [sdfIvy](https://github.com/sdfIvy) |

## References

- Wood et al. (2025). UMA: A Family of Universal Models for Atoms. NeurIPS 38. arXiv:2506.23971.
- Levine et al. (2025). The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models. arXiv:2505.08762.
- Batatia et al. (2025). Cross Learning between Electronic Structure Theories for Unifying Molecular, Surface, and Inorganic Crystal Foundation Force Fields. arXiv:2510.25380.
- Liu et al. (2026). Mixture of Experts Architectures for Machine Learning Interatomic Potentials. npj Artificial Intelligence. arXiv:2603.07977.
- Pople, Binkley and Seeger (1976). Theoretical models incorporating electron correlation. Int. J. Quantum Chem. 10(S10), 1-19.
- Batatia et al. (2022). MACE: Higher order equivariant message passing neural networks. NeurIPS 35. arXiv:2206.07697.
- Rhodes et al. (2025). Orb-v3: atomistic simulation at scale. arXiv:2504.06231.

The report has the full reference list with the claim that each source supports.
