# Local-12

Local-12 uses the same backbone and 37,652,265 parameters as Local-6. Only the routing radius changes. Message edges remain at 6 Å. Each message uses the expert coefficients of its receiving atom.

The backbone builds a second neighbour list at 12 Å. This also works when the data loader supplies the 6 Å message graph. We use fairchem 2.21's periodic graph generator (v2), without a neighbour cap for routing. Distances are computed from live positions, so force training includes derivatives through the composition weights. The existing Local-6 graph path is unchanged.

This choice keeps graph construction next to the force calculation and avoids changing the stored data. It can increase memory use, especially for periodic slabs. Graph parallelism is not supported.

## Checks

CPU tests cover the 6–12 Å shell, periodic images, precomputed message edges, force finite differences, force-loss backpropagation and distant-fragment additivity. Existing Local-6 tests also pass. The new configuration uses `variant=local12`.

The audit gaps are 8, 10, 15, 25 and 100 Å. The guard uses `max(message cutoff, routing radius)` plus a 1 Å margin. At 8 and 10 Å, Local-12 measurements are marked `routing-connected`; they are not failures of size consistency. At 15 Å and above both graphs disconnect. Global and Local+Global have no finite routing radius that guarantees additivity.

`size_consistency_along_training.py --every-steps 50000` limits intermediate checks and always includes the last checkpoint. The CSV records the radius, disconnection status and whether additivity is expected.

## Cluster check still needed

**Measured on Bridges-2 (job 50116348, 2026-11-01, `slurm/smoke_new_variants.job`).** 200 steps through `slurm/pilot.sh` on the fixed pilot lists, one H100. Data check PASS, no OOM event, one validation, final checkpoint, all stages passed. Peak GPU memory 37,801 MiB (nvidia-smi every 5 s, 200 steps and one validation). Median 0.921 s per step, mean 1.059 (steps 20 to 199). Delta 0.379 meV at 8 A and 0.0343 meV at 10 A (routing-connected), 2.8e-06 meV at 15, 25 and 100 A. The peak is over 200 steps only, so the 80 GB guard stays.


The 200-step GPU smoke run, peak memory and throughput are not measured yet. `pilot.sh` reserves an 80 GB GPU for Local-12 (79,000 MiB guard). This is a provisional allocation, not a measured minimum. Anshul submits the smoke run after review. No jobs were submitted as part of this change.
