# Size consistency along the long runs (Global and Local-6)

Status: run on CMU Babel on 2026-11-15 for the two long runs `long_global_s0_400000` and `long_local6_s0_400000` (seed 0, 400,000 steps). Same method as the pilots (`docs/size-consistency-training.md`).

## What was measured

- **Quantity.** Delta = E(Al + Ni) - E(Al) - E(Ni) in meV for the two 4x4 slabs of `audit/check_checkpoint.py`, at gaps 8, 10, 15, 25 and 100 angstrom, float64, at every saved checkpoint (`scripts/size_consistency_along_training.py`).
- **Checkpoints.** Global 73 and Local-6 52, from step 0 to step 400,000: every 10,000 steps, plus the extra 2,000-step checkpoints that survived in the last part of each run. 365 and 260 rows.
- **Command and job files.** Two one-GPU jobs (11161338 and 11161339) ran `~/mlip_sc_training.sh <variant>`, which links one run folder into a private runs root and calls the script above. The self-check of the script passed for both: Local-6 stays below 1e-3 meV at every checkpoint, and the Delta of the final checkpoint equals the value printed by the training job.
- **Files.** `results/long_size_consistency_training.csv`, `results/long_size_consistency_training_router.csv` (router mixing weights of the union and of each slab, Global only), `results/figures/long_size_consistency_training.png`.

## Results

**Measured.**
- **Local-6:** |Delta| is at most 3.7e-6 meV at every checkpoint and every gap, including step 0. The slabs are always disconnected for its 6 angstrom radius (the smallest gap is 8 angstrom).
- **Global:** Delta is -9.8 meV at step 0 and +422 meV at step 10,000. It then grows by orders of magnitude: -79.6 eV at step 40,000, -87.7 eV at step 80,000, a maximum of +188.3 eV at step 210,000, -1.49 eV at step 280,000 and -62.35 eV at step 400,000.
- **Global changes sign 6 times** along the run at gap 8, so the final value is one point of a curve that wanders, not a settled value.
- **Global is the same at every gap.** At one checkpoint, the values at 8, 10, 15, 25 and 100 angstrom differ by at most 1e-6 meV. The gap does not shrink with distance.

**Interpreted.**
- The error comes from the router, which sees the composition of the whole structure; it is not a cutoff effect. As training moves the router and the experts, the size of the error and its sign change from checkpoint to checkpoint.
- The 400,000 step value of Global (-62 eV) should therefore not be quoted as the size of the effect. Quote the range over training (from 1 eV to 188 eV in size after step 40,000) and the fact that it never returns to zero.
- Local-6 stays on the float64 floor from the first step to the last, as in the three-seed pilots at 50,000 steps. Here this is one seed.

## Limits

- One seed per model. One slab pair (Al and Ni, ideal geometry) at every checkpoint; the final checkpoints were also checked on 21 geometries (`docs/final-evaluation.md`).
- The extra 2,000-step checkpoints exist only where they were not pruned, so the two curves have a different number of points.
