# Local Composition Routing for Mixture-of-Experts Interatomic Potentials

CMU 11-785 Introduction to Deep Learning, Fall 2026. AI4Science track.
Mentor: Ron Sarma.

## Project

We study whether local composition routing can preserve the accuracy gain of
mixture-of-experts interatomic potentials while restoring size consistency.
For two non-interacting fragments, size consistency requires
`E(A + B) = E(A) + E(B)`.

UMA selects its linear-expert weights from whole-structure information, including
average composition. Adding a distant fragment can change that average and the
energy assigned to the original fragment. Local routing replaces the composition
average with a smooth average around each atom. The additivity test uses neutral,
closed-shell fragment pairs because total charge and spin remain global inputs.

The study has two parts: audit released UMA, ORB-v3, and MACE models, then train
matched models on the official OMol25 4M split. The six configurations are Dense,
Global, Element, Local-6, Local-12, and Local+Global. The shared backbone will use
fairchem's UMA/eSEN implementation, with forces from the negative energy gradient.

The main test asks whether Local-6 retains more than half of Global's force-error
improvement over Dense. Element tests whether local neighbors justify their cost.
We also measure energy error, additivity, energy drift, interaction energies,
inference speed, and memory. The plan includes 24 training runs across seeds;
these are planned experiments, not results from this branch.

## Current status

Anshul completed the first released-model slab audit on 4 October. In that setup,
MACE's additivity residual was at numerical noise, while ORB-v3 and the tested
UMA checkpoints had errors that persisted with distance. The router-version test
supports a global-routing explanation but does not establish it alone. These
are slab results, including slabs evaluated with UMA's molecular task, not a
molecular benchmark. See the [audit write-up](status-wip/README.md) for results
and limits.

Dense, Global, and Local-6 training runs remain pending in this repository.
The midterm report is due 23 October.

## This branch

**Standalone component with fairchem integration pending.**

`research/local-composition-routing` provides the router interface, local
composition calculation, receiver-based edge adapter, and ordinary linear-expert
references. It does not yet provide a complete Local-6 model or OMol25 training
results. The linear references do not replace fairchem's SO(2) layers.

| Path | Contents |
|---|---|
| `audit/` | Released-model audit, checkpoint loaders, analysis, and W&B logging; see [commands](audit/README.md) |
| `results/` | Raw energies, numerical controls, tables, plots, and cloud-run records |
| `status-wip/` | Team-facing audit report and result snapshots |
| `local_routing/router.py` | Input/result types, Global and Local routers, sparse composition sums, receiver adapter, and a small-system neighbor helper |
| `local_routing/experts.py` | Expert-output mixing, structure merging, and atom merging for an ordinary linear layer |
| `tests/test_local_routing.py` | 17 component tests, including derivatives and a toy force-loss training step |
| `configs/config_draft.yaml` | Yiwei's draft, preserved unchanged; unresolved values remain `null` |
| `docs/local-routing.md` | Tensor contract, assumptions, integration work, and agreed dates |
| `requirements.txt`, `requirements-mace.txt` | Separate project environments for fairchem/ORB and MACE |

Local routing includes the central atom with weight one and uses a smooth
polynomial envelope. Each directed message edge uses the receiving atom's
coefficients. Position gradients pass through distances, composition, and mixing.
Local-12 requires a 12 angstrom routing neighborhood even when message edges use
the shared 6 angstrom cutoff.

The reference defaults are eight experts, radius 6 angstroms, and envelope power
3. **Softmax, embedding size 16, and router hidden size 32 are provisional.**
Shared coefficients across expert operations follow the draft and still need
confirmation. Task, charge, and spin encodings must come from the shared model;
unconditioned operation is an explicit component-test option.

The YAML is a discussion draft, not an executable training configuration. This
package does not load it. The neighbor helper uses an O(N^2), non-periodic search;
training needs a scalable helper. The expert execution modes are correctness
references, with no speed or memory claim.

## Setup

Use the setup commands in the headers of `requirements.txt` and
`requirements-mace.txt`. MACE uses a separate environment because of dependency
conflicts with fairchem. Released UMA weights require Hugging Face access and
license acceptance. Keep login tokens out of the repository.

## Checks

The component tests require Python 3.10 or later and PyTorch. From the repository
root, in a PyTorch environment, run:

```sh
python -m unittest discover -s tests -v
```

The tests cover composition arithmetic, locality, batch isolation, receiver
mapping, cutoff smoothness, first and second derivatives, conditioning, and
agreement between expert execution modes. The training check uses a toy scalar,
not molecular reference energies. Passing these checks does not establish
full-model force accuracy, GPU performance, or molecular-dynamics stability.
See the [implementation notes](docs/local-routing.md) for the tested runtime.

## Next steps

1. Confirm the shared backbone, coefficient normalization, router dimensions,
   conditioning encoder, and geometry-helper contract with Yiwei.
2. Connect the adapter and expert operations to fairchem's actual SO(2) layers.
   Preserve position derivatives through conservative force evaluation.
3. Run full-model force checks and a few OMol25 training batches, then review and
   merge the integration. The agreed target is 15 October.
4. Run a short Local-6 pilot on 16-18 October if integration and GPU access are
   ready. Measure energy-plus-force inference, training time, and peak memory.

The progress checkpoint is 11 October. If Local-6 is not running by 15 October,
report the implemented components and remaining work. Dense and Global remain
the midterm baseline. The shared training pipeline is an integration dependency.

## Team

| Name | Responsibility | GitHub |
|---|---|---|
| Anshul Kumar | Released-model audit; Global and Element | [anshulk-cmu](https://github.com/anshulk-cmu) |
| Pragati Bhattad | Local routing; Local-6 and Local-12; speed and memory | [aptenosys](https://github.com/aptenosys) |
| Yiwei Xia | Shared pipeline; Dense and Local+Global; statistical and physical evaluation | [sdfIvy](https://github.com/sdfIvy) |

References: [UMA](https://arxiv.org/abs/2506.23971),
[OMol25](https://huggingface.co/facebook/OMol25),
[fairchem](https://github.com/facebookresearch/fairchem).
