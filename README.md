# Local Composition Routing for Mixture-of-Experts Interatomic Potentials

Course project for CMU 11-785 Introduction to Deep Learning, Fall 2026 (AI4Science track, machine-learned interatomic potentials).
Team 6: Anshul Kumar, Pragati Bhattad, Yiwei Xia. Status: midterm submission (23 October 2026).

## Summary

A good interatomic potential gives two far-apart, non-interacting molecules a total energy equal to the sum of their separate energies.
This property is called size consistency. UMA, a widely used foundation model for molecules and materials, does not have it. Its
mixture-of-experts layers choose their weights from the average element composition of the whole structure, with no distance cutoff,
so a molecule placed anywhere changes the predicted energy of every other molecule.

We propose local composition routing: the router sees a smooth, bounded average over the neighbors of each atom instead of the whole
structure. Our main model, Local-6, uses a 6 angstrom radius. The question of the project is whether Local-6 keeps the accuracy that
routing adds over a model without experts, while being size consistent. We audit released models without training, and we train
Dense, Global (UMA-style routing), Local-6, and two controls on a fixed subset of OMol25 with the `fairchem` code.

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
- **Accuracy (RQ2).** Global and Local-6 reduce the force error by about 4 percent against Dense (Welch p = 0.0014 and 0.0013). We detect no difference between Global and Local-6 (p = 0.95). The retention share of the Global gain kept by Local-6 is 1.01 for force (one-sided 95 percent lower bound 0.84) and 1.06 for energy (bound 0.72, and about 0.50 if the seed spread is treated with a Student-t distribution). All bounds come from 3 seeds.
- **Size consistency through training (RQ3).** Dense and Local-6 stay at the float64 floor at all 66 of their saved checkpoints (33 each); Global is off at all 33 of its own, 99 in total. Global is 10 to 69 meV off before training and 2.4 to 6.6 eV off after 50,000 steps, the same at gaps of 8 and 100 angstrom. See `docs/size-consistency-training.md`. The same pattern holds for far-apart molecule pairs (`docs/molecular-evaluation.md`).
- **Cost (RQ4).** At inference, with energy and forces on the same GPU, Local-6 is 3.3 to 3.5 times slower than Global (627 against 2,201 atoms per second on a V100; 1,736 against 5,708 on an H100). Global is 1.2 to 1.3 times slower than Dense. See `docs/inference-timing.md`. The nine pilots cost 135 GPU service units; the whole project about 195 of 2,950.
- **Mechanism (RQ5).** The trained router is far from uniform, but the experts differ mostly by their random start, and training moved them together (`docs/router-check.md`). A model with a constant router keeps 84 percent of the force gain of Global and 97 percent of the energy gain (`docs/frozen-router.md`); its force error is 0.7 percent higher than Global in all 3 seeds (p = 0.18). A plain Dense model with the start scale and step size that a constant router implies (no experts, no router, 5,981,057 parameters) keeps 74 percent of the force gain and 60 percent of the energy gain (`docs/scaled-dense.md`). We detect no difference between the two controls. We conclude that a large part of the gain over Dense can be reproduced without experts or routing, by start scale and step size alone. Global and Local-6 stay about 1.1 percent lower in force error than the scaled Dense, which is not significant with 3 seeds, so we cannot say that learned routing adds nothing. The two controls differ in two details that we did not test on a small model: the shape of the initial weights and gradient clipping.

## Limitations

- Three seeds per model; the proposal planned five. Intervals and bounds with 3 seeds are approximate.
- One validation set of 5,000 structures, which are larger on average than the training structures (102 against 55 atoms).
- Short training: about 1/800 to 1/870 of an official run, on a 200,000-structure subset of the 4M training split. The target normalizer (1.433569) is a placeholder taken from an official run.
- Size consistency along training uses one ideal Al + Ni geometry; the audit of released models uses 21 geometries.
- Global and Local-6 were trained on different GPU types (V100 and H100), so their training times are not comparable. The inference timings use the same GPU type for all models, on shared nodes.
- Element, Local-12 and Local+Global routing are not trained yet. The additivity test covers neutral singlet pairs only; we train on all charges and spins.

## Repository layout

| Folder or file | What it holds |
|---|---|
| `status-wip/` | Reports on the audit: write-up, final tables and plots |
| `audit/` | Audit code (slab test, null tests, neon probe, forcing test, molecular additivity and force checks `molecular.py`), with its own README and the exact commands |
| `local_routing/` | Local composition, router interface, receiving-atom adapter and reference linear experts |
| `training/` | Entry point for the fairchem training CLI, the Local-6 backbone (`local_moe.py`), the frozen-router and scaled-Dense variants, the out-of-memory guard and the step-based scheduler |
| `configs/` | Training configuration: `train.yaml`, data configuration, the variants `dense`, `global`, `local6`, `global_frozen`, `dense_scaled`, element references |
| `slurm/` | Bridges-2 job files and scripts: environment, data download, smoke tests, pilots, size consistency, router check, timing, molecular evaluation |
| `scripts/` | Data preparation and audit, pilot sample lists, size consistency along training, router check, inference timing, curve comparison, the pilot statistics (`pilot_statistics.py`, input `results/pilot_final_metrics.csv`) and the report figures and tables (`make_midterm_figures.py`, `audit_summary_table.py`) |
| `manifests/` | The OMol25 data manifest and the fixed training and validation lists with their hashes |
| `results/` | Raw and summary result files (CSV) and figures behind every number in the reports |
| `tests/` | Unit tests (152): routing, audit code, training configuration, frozen router, scaled Dense, router check, molecular checks, timing |
| `docs/` | Detailed reports: `TRAINING.md` (setup and run), `local-routing.md`, `size-consistency-training.md`, `router-check.md`, `frozen-router.md`, `scaled-dense.md`, `molecular-evaluation.md`, `inference-timing.md` |
| `requirements.txt`, `requirements-training.txt`, `requirements-mace.txt` | Environments: UMA, ORB-v3 and fairchem (Python 3.12, PyTorch 2.8 with CUDA 12.8); extra training packages; a separate environment for MACE |

Model weights, checkpoints, the OMol25 data and the run folders are not in the repository. They live on the Bridges-2 file system of the
project; see `docs/TRAINING.md` for the paths and for how the data is prepared.

## Reproducing the results

1. Create the environments from the requirements files; the steps are in the header of each file. The UMA weights are gated on Hugging Face: log in and accept the license first. Never commit a token.
2. Run the tests: `python -m unittest discover -s tests`. All 152 tests pass on Windows (CPU). One configuration-hash test stores hashes measured on Windows and may need its hashes refreshed on Linux.
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
