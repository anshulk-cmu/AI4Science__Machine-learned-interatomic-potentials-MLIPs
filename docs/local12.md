# Local-12

Local-12 uses the same backbone and 37,652,265 parameters as Local-6. Only the routing radius changes. Message edges remain at 6 Å. Each message uses the expert coefficients of its receiving atom.

The backbone builds a second neighbour list at 12 Å. This also works when the data loader supplies the 6 Å message graph. We use fairchem 2.21's periodic graph generator (v2), without a neighbour cap for routing. Distances are computed from live positions, so force training includes derivatives through the composition weights. The existing Local-6 graph path is unchanged.

This choice keeps graph construction next to the force calculation and avoids changing the stored data. It can increase memory use, especially for periodic slabs. Graph parallelism is not supported.

## Checks

CPU tests cover the shell from 6 to 12 Å, periodic images, precomputed message edges, force finite differences, force-loss backpropagation and distant-fragment additivity. Existing Local-6 tests also pass. The new configuration uses `variant=local12`.

The audit gaps are 8, 10, 15, 25 and 100 Å. The guard stops a pair whose fragments are closer than the message cutoff (6 Å) plus a 1 Å margin. The larger of the message cutoff and the routing radius, plus 1 Å, only labels a pair: `routing-connected` or `disconnected`. The label is about graph distance only; it does not say that the energy is additive. At 8 and 10 Å, Local-12 measurements are marked `routing-connected`; they are not failures of size consistency. At 15 Å and above both graphs disconnect. Global and Local+Global have no finite routing radius that guarantees additivity.

`size_consistency_along_training.py --every-steps 50000` limits intermediate checks and always includes the last checkpoint. The CSV records the radius, disconnection status and whether additivity is expected.

## Cluster check

Measured on Bridges-2 (job 50116348, 2026-11-01, `slurm/smoke_new_variants.job`): 200 steps through `slurm/pilot.sh` on the fixed pilot lists, one H100.

| Quantity | Value |
|---|---|
| Data check, OOM events, final checkpoint | 8 of 8 PASS, 0, written; all stages passed |
| Peak GPU memory (nvidia-smi every 5 s, 200 steps and one validation) | 37,801 MiB |
| Seconds per step, steps 20 to 199 (fairchem step timer) | median 0.921, mean 1.059 |
| Delta at 8 / 10 Å (`routing-connected`) | 0.379 / 0.0343 meV |
| Delta at 15 / 25 / 100 Å (`disconnected`) | 2.8e-06 meV each |

The peak covers 200 steps only. Local-6 peaked at 26 to 29 GiB in short tests and at 43 to 45 GiB over 50,000 steps. So `pilot.sh` keeps the 79,000 MiB guard, and Local-12 runs on an H100 only.
