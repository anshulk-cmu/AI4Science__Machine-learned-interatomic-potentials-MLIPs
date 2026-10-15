# Local Composition Routing for Mixture-of-Experts Interatomic Potentials

Local-composition routing for mixture-of-experts interatomic potentials, benchmarked against UMA on OMol25.

Course project for CMU 11-785 Introduction to Deep Learning, Fall 2026, AI4Science track (machine-learned interatomic potentials).

## Idea

UMA routes its Mixture of Linear Experts (MoLE) with an average element embedding taken over the whole structure, with no distance cutoff. For two fragments of different composition, this average differs from that of either fragment, so the energy is not additive when the fragments are far apart (size consistency). We propose restricting the routing average to a smooth, bounded neighborhood around each atom, and test whether this keeps the accuracy that routing adds.

## Plan

- Backbone: UMA's eSEN-based architecture, via the `fairchem` implementation. Only the router changes between configurations.
- Data: OMol25, official splits.
- Configurations: Dense (no experts), Global (UMA-style routing over the whole structure), Element (routing by the atom's own element), Local-6 and Local-12 (routing from the neighborhood within 6 or 12 angstrom), and Local+Global.
- Measures: energy and force error, the additivity residual on neutral singlet fragment pairs, spurious host-atom forces, and dissociation curves.

## Status (2026-10-09)

The proposal is submitted. The audit, the Local-6 model and the Bridges-2 setup are done. The pilot training runs
are next (12 to 18 October). The midterm report is due 23 October.

- **Done: the audit of released models.** We tested whether MACE, ORB-v3, UMA-S-1.1, UMA-S-1.2 and UMA-M-1.1 give an
  energy for two far-apart metal slabs that equals the sum of the parts. MACE does, exactly. ORB-v3 and all UMA
  models do not, by 0.5 eV to hundreds of eV, and the error does not shrink with distance. A forcing test shows that
  the whole-structure router average is the whole cause in UMA-S-1.1; UMA-S-1.2 has a second whole-structure term,
  charge balancing. Read `status-wip/README.md`.
- **Done: Local-6 inside fairchem.** `training/local_moe.py` and `configs/variant/local6.yaml` run Local-6 in the real
  model, with the same parameters as Global (37,652,265; Dense has 5,981,057). The router component is in
  `local_routing/`. All 107 tests pass (Windows CPU, and Linux on CPU and GPU). Read `docs/local-routing.md`.
- **Done: Bridges-2 setup.** Shared environment, OMol25 `train_4M` and validation data, job files, and 30-step test
  jobs of all three variants. On the saved checkpoints of two far-apart Al and Ni slabs, Dense and Local-6 give an
  energy error of about 3e-6 meV (zero) and Global gives about -31 meV. This is a pipeline check after 30 steps, not
  a quality result. Read `docs/TRAINING.md`.
- **Next: the pilot runs.** Dense, Global and Local-6, one job file each in `slurm/`. The settings (steps, seeds,
  fixed sample lists, normalizer, size matching, neighbor cap, `num_distance_basis`) are waiting on the team's
  answers by mail. Until then the job files hold placeholders.
- **Not done:** trained quality comparison and the size-consistency benchmark on the pilot checkpoints. Element, Local-12 and
  Local+Global are later work (decided 2026-10-11); the midterm trains Dense, Global and Local-6.

## Repository layout

| Folder or file | What it holds |
|---|---|
| `status-wip/` | Reports only: the write-up, the final tables and the plots |
| `audit/` | The audit code (slab test, null tests, neon probe `neon_probe.py`, forcing test `f1_forcing.py`, W&B upload `log_wandb.py`), with its own README and the exact commands |
| `results/` | Raw per-run energies, null tests, neon and forcing-test results, and cloud-run logs behind every number |
| `local_routing/` | Local composition, router interface, receiving-atom adapter and reference linear experts |
| `tests/` | Component tests for routing, derivatives and expert execution, and tests of the audit code and the training configuration |
| `docs/local-routing.md` | Router interface, test results and integration steps |
| `configs/config_draft.yaml` | Yiwei's draft shared configuration; unresolved settings remain open |
| `configs/` (other files) | Training configuration: `train.yaml`, the data config in `data/`, the Dense, Global and Local-6 variants in `variant/`, element references in `element_refs/` |
| `training/` | Version-checked entry point for the fairchem training CLI, the Local-6 backbone (`local_moe.py`), the out-of-memory guard and the step-based scheduler |
| `slurm/` | Bridges-2 job files and scripts: environment, data download, smoke tests, and the pilot jobs |
| `docs/TRAINING.md` | How to set up and run training |
| `scripts/` | OMol25 data preparation notes and the data audit script |
| `manifests/` | The OMol25 data manifest written by the data audit |
| `requirements.txt` | Environment for UMA, ORB-v3 and fairchem (Python 3.12, PyTorch 2.8 with CUDA 12.8) |
| `requirements-training.txt` | Extra packages for training, installed after `requirements.txt` |
| `requirements-mace.txt` | Separate environment for MACE, which cannot share one with fairchem |

## Setup

Create the environments from the two requirements files. The steps are in the header of each file. The UMA weights are
gated on Hugging Face, so log in and accept the license first. Never commit a token.

## Who does what

- Anshul Kumar: the audit, the Bridges-2 setup, the Local-6 integration into fairchem, the Global pilot run, and the size-consistency benchmark.
- Pragati Bhattad: the router interface and local composition. Anshul submits the Local-6 pilot runs; Pragati can also submit them.
- Yiwei Xia: the OMol25 training pipeline, the Dense pilot run, and the statistics.

## Team

| Name | Email | GitHub |
|---|---|---|
| Anshul Kumar | anshulk@andrew.cmu.edu | [anshulk-cmu](https://github.com/anshulk-cmu) |
| Pragati Bhattad | pbhattad@andrew.cmu.edu | [aptenosys](https://github.com/aptenosys) |
| Yiwei Xia | yiweixia@andrew.cmu.edu | [sdfIvy](https://github.com/sdfIvy) |
