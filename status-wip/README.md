# Status (work in progress): is the energy of two far-apart pieces the sum of the parts?

Written 2026-10-04 by Anshul Kumar for the 11-785 Team 6 MLIP project. This folder is for reports only. It holds
my part of the midterm baseline: the **audit of released models**, which needs nothing from the training pipeline.
The audit was run and checked on 2026-10-04. The forcing test and the neon probe (section 7) were run on 2026-10-05.
Errors found in a review were corrected on 2026-10-08.

- `README.md` this explanation
- `results.md`, `summary.csv`, `flatness.csv`, `comparison.csv` the final tables (a snapshot)
- `figures/` the plots (a snapshot)
- Not in this folder: the code is in `../audit/` and the raw per-run energies are in `../results/`. The tables and
  plots here are copies made from them on 2026-10-04.

## 1. In one paragraph

A good interatomic model must obey one simple rule: if two pieces of matter are far apart and do not touch, the
energy of both together is the sum of their separate energies. I tested this on five released models using two metal
slabs. The purely local model (MACE) obeys the rule exactly. ORB-v3 and all three UMA models break it, by amounts
from 0.5 eV to hundreds of eV, and the error does **not** shrink when the slabs are moved farther apart. For UMA the
main cause is one global input: the router that picks the experts looks at the average ingredients of the whole
structure. A forcing test shows that the router accounts for the whole error in UMA-S-1.1. UMA-S-1.2 has a second
whole-structure term, charge balancing, so there the router is only part of it.

## 2. The idea, with an example

Picture two coffee cups, each in a different room. The heat in both is just the sum of the heat in each. A model that
says otherwise is wrong, and it will be wrong in practice too, for example for a drug molecule sitting far from the
rest of a big protein.

UMA uses a "mixture of experts": many small sub-networks, and a router that decides how much of each to use. The
router sees the average element composition of the **entire** structure. Think of a chef who picks a recipe by
looking at the whole pantry. Put a lead pan and a nickel pan in one pantry, and the chef changes the recipe for
both, even though the pans never touch. The total is then not the sum.

Our project idea is to let each atom pick its recipe from its own neighborhood only. That is called local routing.
This audit measures how large the problem is in released models, and it will later be the test that a local router
must pass.

## 3. What I did (how)

1. **Built two slabs.** Al(111) and Ni(111), 8 layers, 4x4 cells, 128 atoms each, unrelaxed. They share one
   in-plane cell, so the Al is compressed by 6.54 percent and the Ni stretched by 7.53 percent. The test compares
   calculations on the same atoms, so the strain does not matter for the question.
2. **Placed them at six gaps.** The empty space between the facing surfaces was 8, 10, 15, 25, 50 and 100 angstrom.
   Every model builds its graph with a cutoff of 6 angstrom, so at 8 angstrom no edge joins the two slabs. Stacked
   layers make each atom's receptive field layers times cutoff (MACE 12, ORB 30, UMA-S 24, UMA-M 60 angstrom), but
   information moves only along edges, and no edge crosses the gap. A check in the code stops the run if any atom of
   one slab is within the cutoff of the other.
3. **Computed three energies** for each gap: slab A alone, slab B alone, and both together. Charge and spin were the
   same in all three, so only the presence of the other slab differs.
4. **Repeated it on 21 geometries.** One ideal copy plus 20 copies where every atom was shaken by Gaussian noise with
   a standard deviation of 0.05 angstrom per Cartesian component (a mean displacement of 0.080 angstrom). The same
   shake is reused at every gap, so only the gap changes. This gives an average and a spread.
5. **Added controls.** MACE is purely local, so it must give zero. If it does not, my code is wrong. Same-metal pairs
   (Al with Al, Ni with Ni) keep the average composition unchanged, so a router that sees the whole structure should
   also give zero.
6. **Ran five models.** MACE-MP-0 medium, ORB-v3, UMA-S-1.1, UMA-S-1.2 and UMA-M-1.1. Each UMA model ran on two
   tasks: `omat` (materials) and `omol` (molecules). Float64 was used everywhere except UMA-M, which ran in float32
   on a rented 24 GB cloud GPU (terminated afterwards), because 24 GB cards run float64 far too slowly.
7. **Measured the noise.** Before reading any result, I ran null tests: the same input twice, shuffled atom order,
   shifted origin, and different empty space. The spread of those energies sets how small a number counts as zero.

## 4. The math and the logic

**The quantity.** For slab A and slab B at gap d:

    Delta(d) = E(A and B together) - E(A) - E(B)

A size-consistent model gives Delta = 0.

**Why a local model gives exactly zero.** Build a graph with an edge between two atoms when they are closer than the
cutoff r_c = 6 angstrom. An atom's energy depends only on atoms it can reach through edges. If every atom of A is
farther than r_c from every atom of B, no edge joins them, so no path does either, however many layers the network
has. Each atom then sees the same neighbors with or without the other slab, so E(A and B) = E(A) + E(B) exactly.

**Why UMA breaks it.** The router output is a set of weights alpha, computed from the mean element embedding of all
atoms in the structure, together with charge, spin and task:

    c(S) = (1/N) * sum over all atoms i of e(Z_i),    alpha = router(c(S), charge, spin, task)

For A and B far apart, each atom keeps its own neighbors but the router weights come from the union. Write
alpha_AB for the weights of the union. Then

    Delta = [E_A(alpha_AB) - E_A(alpha_A)] + [E_B(alpha_AB) - E_B(alpha_B)]

where E_F(alpha) is the energy of fragment F computed with weights alpha. Three consequences:

- **Delta does not depend on d** once d is larger than the cutoff, because d appears nowhere in this formula.
- **Same-metal pairs give zero.** If A and B have the same composition, the union has that same composition, so
  alpha_AB = alpha_A = alpha_B and both brackets vanish. This holds only if c(S) is a true mean.
- **Different metals give a nonzero Delta** of no predictable sign, because it is the change of a nonlinear function
  when its input changes.

This formula assumes the router is the only whole-structure input. That holds for UMA-S-1.1 but not for UMA-S-1.2
(section 7).

**A version quirk.** In the stored router version 1.0, the mean is computed with `include_self=True`, which makes
PyTorch divide by N + 1 instead of N. I checked this with a four-line test: the mean of four ones comes out as 0.8.
Then c(S) depends on the number of atoms even for identical composition, and the same-metal pairs are not zero.
The UMA-S-1.2 checkpoint stores no `model_version` key (a byte scan of its `data.pkl` on 2026-10-08 found 0 occurrences, against 1 each in the 1.1 checkpoints), so the
fairchem 2.21.0 class default of 1.0 applies (`escn_moe.py` line 67; the value is used only at line 149, in this
mean). UMA-S-1.1 and UMA-M-1.1 store 1.1.

**ORB-v3 is different.** Its energy is N times a nonlinear function of the mean hidden feature over all atoms. That is
also a whole-structure average passed through a nonlinear map, so it also breaks the rule, by a different route. Its
energy also has a ZBL pair-repulsion term that is aggregated by a mean over atoms, another whole-structure average.

**Noise and tolerance.** Delta combines three energies. If each has independent error sigma_E, then Delta has error
sqrt(3) * sigma_E. I count a value as zero only if it is inside plus or minus tau = 5 sigma_E, about three times
that error, where sigma_E comes from the null tests. tau is never set below 0.001 meV.

**Statistics.** For each case I have 20 shaken runs. I report the mean, the standard deviation, and a 95 percent
interval, mean plus or minus 2.093 times sd divided by sqrt(20). The 20 runs measure sensitivity to geometry, not
training variance, since the models are fixed. The decision rule, fixed before the runs:

- **VIOLATION:** the whole interval is outside plus or minus tau, and at least 19 of 20 runs have the same sign.
- **CONSISTENT:** the whole interval is inside plus or minus tau.
- **INCONCLUSIVE:** anything else.

The version-swap cells (section 7) have 5 runs, so they use t = 2.776 and need all 5 runs to share the sign of the
mean. In code the sign rule is at least ceil(0.95 n) runs. This is 19 of 20 as fixed before the runs; for n = 5 it
was 4 of 5 until 2026-10-08. Re-running the analysis with the stricter n = 5 rule changed no verdict.

## 5. Results

All values are Delta in meV at a gap of 100 angstrom (1,000 meV is 1 eV). Full tables: `results.md`.
Overview plot: `figures/overview.png`.

| model | task | Al + Ni: mean (sd) | verdict | Al + Al | Ni + Ni |
|---|---|---|---|---|---|
| MACE-MP-0 medium | none | 0.000 | consistent | 0 | 0 |
| ORB-v3 | none | -501.1 (6.4) | violation | 0.03 | 0.00 |
| UMA-S-1.1 | omat | -2,154.2 (39.7) | violation | 0 | 0 |
| UMA-S-1.1 | omol | +109,367 (922) | violation | 0 | 0 |
| UMA-S-1.2 | omat | -2,163.3 (31.9) | violation | -298.2 | -0.01 |
| UMA-S-1.2 | omol | +125,428 (708) | violation | -308.3 | +3,238 |
| UMA-M-1.1 | omat | +3,524.2 (74.4) | violation | 0.01 | 0.06 |
| UMA-M-1.1 | omol | -338,720 (10,602) | violation | 1.4 | 0.4 |

- MACE: the largest |Delta| over all 378 rows is 6.5e-9 meV, which is rounding. The pipeline works.
- Delta at a gap of 8 angstrom equals Delta at 100 angstrom, within tolerance, in 35 of 36 (model, task, case)
  cells. The exception is UMA-M-1.1 `omat` Ni + Ni: a difference of 0.049 meV, 95 percent CI [0.036, 0.062], against
  tau = 0.030 meV. That cell is flat from 8 to 25 angstrom and not flat at 50 and 100 angstrom. The difference is
  negligible next to the 3,524 meV Al + Ni effect. One untested hypothesis: the Ni + Ni cell is 128.45 angstrom tall
  at a gap of 50 angstrom, just above 2^7 = 128, where float32 resolution halves. Nothing here tests this.
- The same cell, UMA-M-1.1 `omat` Ni + Ni, is a small VIOLATION at 100 angstrom: mean 0.064 meV, CI [0.052, 0.076],
  tau 0.030 meV.
- The molecule task shows an error 50 to 100 times larger than the materials task.

Per-model plots of Delta against gap: `figures/delta_<model>_<task>.png`.

## 6. What the results mean

- **The error is not a real interaction.** It does not fade with distance, and for a gap of 8 angstrom or more the
  model cannot connect the slabs. Something global must feed in.
- **It matches the whole-structure router.** It appears when different metals are mixed. Same-metal pairs give zero
  within tolerance in UMA-S-1.1 (all four cells) and in UMA-M-1.1 `omol`; UMA-M-1.1 `omat` Ni + Ni is the small
  VIOLATION above. "Only when different metals mix" holds only for UMA-S-1.1: UMA-S-1.2 gives Al + Al of -298 and
  -308 meV and Ni + Ni `omol` of +3,238 meV, and fails in 3 of its 4 same-metal cells (`omat` Ni + Ni is
  INCONCLUSIVE). ORB Al + Al is INCONCLUSIVE (0.03 meV, CI [0.0037, 0.0551], 16 of 20 runs positive). ORB shows the
  Al + Ni pattern for its own reason, the global mean.
- **The size is large.** About 2 to 3.5 eV on the materials task is far above typical chemical accuracy of about
  0.04 eV, and about 100 to 340 eV on the molecule task.
- **Compared with the paper** (Batatia et al., arXiv 2510.25380): the sign agrees in all five printed values. The sizes
  differ, with ratios of ours to theirs of 0.71 (ORB), 4.7 (UMA-S-1.1 materials), 1.23 (UMA-S-1.1 molecules), 2.45
  (UMA-M-1.1 materials) and 0.88 (UMA-M-1.1 molecules). The paper does not give its slab geometry in enough detail,
  and its UMA-M materials row does not add up even from its own printed energies, so I did not try to match it.

## 7. Tests of the cause: the router version number, the forcing test, the neon probe

I overwrote the stored version number and re-ran the same-metal pairs (5 shaken runs plus the ideal one).

| change | task | Al + Al before to after | Ni + Ni before to after |
|---|---|---|---|
| UMA-S-1.2, 1.0 to 1.1 | omat | -298.2 to 0.05 | -0.01 to -0.02 |
| UMA-S-1.2, 1.0 to 1.1 | omol | -308.3 to 0.25 | +3,238 to +5.8 |
| UMA-S-1.1, 1.1 to 1.0 | omat | 0 to -48.3 | 0 to +0.9 |
| UMA-S-1.1, 1.1 to 1.0 | omol | 0 to +681.9 | 0 to -517.6 |

UMA-S-1.1 forced to 1.0 gives VIOLATION in 4 of 4 same-metal cells. UMA-S-1.2 forced to 1.1 gives INCONCLUSIVE in
4 of 4 (mean and sd 0.052 and 0.053, -0.019 and 0.035, 0.252 and 0.488, 5.8 and 13.1 meV). In the three cells that
were VIOLATION at 1.0, |Delta| drops by at least 99.8 percent, but not to zero. So the number switches the same-metal
error fully on in one direction and almost fully off in the other. On the same runs 1 to 5, the UMA-S-1.2 `omat`
Al + Ni mean moves only from -2,174.7 to -2,117.0 meV (-2.7 percent), so the main error is not the version quirk. It
is the whole-structure average itself.

**The forcing test (F1).** For each run, each slab alone is computed twice: once as usual, and once with the router
fed the atoms of the union as one structure. The sum of the two changes is compared with Delta at 100 angstrom. If
the router is the only whole-structure input, the residual is zero.

| model, `omat` | runs | residual (meV) |
|---|---|---|
| UMA-S-1.1 | 21 | largest \|residual\| 9.7e-10 |
| UMA-S-1.2 | 21 | -1,892 (sd 23), largest \|residual\| 1,985 |
| UMA-S-1.2, charge balancing off | 3 | largest \|residual\| 8.5e-10 |

The router is the whole cause for UMA-S-1.1. UMA-S-1.2 has a second system-wide term, charge balancing
(`balance_channels_batched` in `escn_md.py`, applied to the channels in `charge_balanced_channels`, [0, 1, 2]). It
subtracts (system sum minus charge) / N from those channels of every atom, so every atom depends on the whole
structure. Switching it off changes the model, so the Delta it gives is not a result; only the exactness of the
decomposition is. The other 3-run files are `f1_uma-s-1p1-v1` (UMA-S-1.1 forced to version 1.0, largest
|residual| 8.5e-10 meV) and `f1_uma-s-1p2-v1.1` (UMA-S-1.2 forced to 1.1, residual about -2,000 meV). F1 was not run
for UMA-M or for `omol`.

**The neon probe.** One Ne atom sits 10 or 50 angstrom above an Al slab, beyond every cutoff. A local model leaves
the forces on the slab unchanged. Averaged over the 20 shaken runs, the largest force change on a slab atom and the
mean change are: MACE about 1e-12 meV/A; ORB-v3 1.11 and 0.42; UMA-S-1.1 `omat` 1.49 and 0.71, `omol` 136.6 and
58.6; UMA-S-1.2 `omat` 6.83 and 2.55, `omol` 8.10 and 3.74 meV/A. The values at 10 and 50 angstrom agree to
within 3e-10 meV/A.
These numbers were measured again on 2026-10-09 on a V100-32 (ORB on an L40S) after the ASE cache fix and did not change.
The `noise_dforce` column is now a real floor: the largest force difference on a slab atom between the slab and a copy with
its atoms in a different order. Over the 21 runs it is at most 1.6e-12 (MACE), 6.4e-12 (ORB-v3), 1.6e-12 (UMA-S-1.1
`omat`), 1.1e-10 (UMA-S-1.1 `omol`), 1.6e-12 (UMA-S-1.2 `omat`) and 5.9e-12 meV/A (UMA-S-1.2 `omol`), about ten orders of
magnitude below every non-zero effect above. MACE's 1e-12 is at that floor, so it is zero. In the files before the rerun
this column was always 0, because the repeated call was an ASE cache hit.

## 8. Limits, stated plainly

- UMA-M ran in float32, so its tolerance is larger (about 5 meV on the molecule task). It also ran on a cloud GPU
  while UMA-S ran in float64, so any UMA-M versus UMA-S difference is confounded with precision.
- The version number is read only at `escn_moe.py` line 149 in fairchem 2.21.0, the composition mean, so the swap
  changes nothing else.
- Geometries are unrelaxed and use one specific shared cell, so the values are for this setup only. The interval is
  over shakes of one strained, unrelaxed crystal, not over different structures.
- tau uses one sigma_E pooled over all null tests. The null tests of the ten float64 labels were rerun on 2026-10-09 after
  the ASE cache fix, with 27 repeats each (the padding test N4 now also uses pads of 100 and 230 angstrom). The repeat test
  N1 is now a real recomputation: bit-identical for ORB-v3 and every UMA label, and different by at most 5.7e-14 eV for
  MACE. sigma_E is at most 0.000005 meV, so tau stays at the 0.001 meV floor, and no verdict, interval or flatness result
  changed (the summary tables are byte-identical to before). UMA-M (float32, on a cloud GPU) was not rerun: its null file
  still has the cache-hit N1, which lowers its sigma_E by at most about 4 percent (an estimate); raising every tau above
  the floor by 10 percent changes no verdict. The null tests of UMA-M do not cover float32 bias or the coordinate range of
  gaps of 50 angstrom and more.
- The comparison with the paper uses the shaken mean. With the ideal geometry instead, the ratios are 0.60 (ORB),
  5.06 (UMA-S-1.1 materials), 1.22 (UMA-S-1.1 molecules), 2.98 (UMA-M-1.1 materials) and 0.82 (UMA-M-1.1 molecules).
- ORB gives -4.8e-3 meV for the ideal Ni + Ni pair at every gap, 4.8 times the 0.001 meV floor. This is not
  explained.
- The interval is for the mean over shakes. A CONSISTENT verdict says the mean is near zero; it does not say each
  geometry is, which needs a per-geometry bound.
- The forcing test shows the router is the whole cause for UMA-S-1.1 only, and it was run on `omat` only.
- The results were uploaded to W&B on 2026-10-04 (run `60v31vlc`). Training baselines (Dense, Global, Local-6) are
  not part of this audit.

## 9. Code map

All code is in `../audit/`, short and commented only where a reason is not obvious. Raw energies for every run are
in `../results/`.

- `geometry.py` builds the slabs, the strain, the shaking, the stacked pair, and the connection check
- `models.py` one loader per model family, plus the router version override
- `nulls.py` the null tests that set the tolerance
- `run_audit.py` the main loop, one CSV per model and task
- `analyze.py` statistics, decision rule, flatness test, comparison, per-model plots
- `report.py` tables and the overview plot
- `preflight.py` the quick facts checks run before the real runs
- `neon_probe.py` the neon probe, one CSV per model and task
- `f1_forcing.py` the forcing test (F1), which feeds the router the union's atoms
- `log_wandb.py` the upload of the tables, plots and result files to W&B
- `../tests/test_audit.py` tests of the audit code on CPU, with every model replaced by a stub

The rest of the repository is not part of this audit: `../local_routing/` (the router), and the training pipeline
in `../training/`, `../configs/`, `../docs/TRAINING.md`, `../scripts/`, `../manifests/` and
`../requirements-training.txt`.

How to rerun, with the exact commands and the two environments, is in `../audit/README.md`.

## 10. What comes next, and what depends on whom

- **Needs Yiwei's pipeline:** the Dense and Global baselines. The audit already loads a checkpoint file given as
  `--model <path> --label <name>`, so it can test Dense, Global and Local-6 once the pipeline writes inference
  checkpoints.
- **Needs Pragati's router:** Local-6 must give Delta = 0 for gaps of 8 angstrom or more. This audit is the test.
- For the midterm, this audit alone gives the reproduced-baseline evidence. The Dense and Global pilots are the part
  that depends on the pipeline.
