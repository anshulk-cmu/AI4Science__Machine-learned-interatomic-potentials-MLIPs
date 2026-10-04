# Status (work in progress): is the energy of two far-apart pieces the sum of the parts?

Written 2026-10-04 by Anshul Kumar for the 11-785 Team 6 MLIP project. This folder is for reports only. It holds
my part of the midterm baseline: the **audit of released models**, which needs nothing from the training pipeline.
Everything was run and checked on 2026-10-04.

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
cause looks like one global input: the router that picks the experts looks at the average ingredients of the whole
structure. A small test on the stored router version number supports this.

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
   Every model only looks 6 angstrom around an atom, so at 8 angstrom the two slabs are not connected at all. A
   check in the code stops the run if any atom of one slab is within the cutoff of the other.
3. **Computed three energies** for each gap: slab A alone, slab B alone, and both together. Charge and spin were the
   same in all three, so only the presence of the other slab differs.
4. **Repeated it on 21 geometries.** One ideal copy plus 20 copies where every atom was shaken randomly by about
   0.05 angstrom. The same shake is reused at every gap, so only the gap changes. This gives an average and a spread.
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

**A version quirk.** In the stored router version 1.0, the mean is computed with `include_self=True`, which makes
PyTorch divide by N + 1 instead of N. I checked this with a four-line test: the mean of four ones comes out as 0.8.
Then c(S) depends on the number of atoms even for identical composition, and the same-metal pairs are not zero.
UMA-S-1.2 stores version 1.0. UMA-S-1.1 and UMA-M-1.1 store 1.1.

**ORB-v3 is different.** Its energy is N times a nonlinear function of the mean hidden feature over all atoms. That is
also a whole-structure average passed through a nonlinear map, so it also breaks the rule, by a different route.

**Noise and tolerance.** Delta combines three energies. If each has independent error sigma_E, then Delta has error
sqrt(3) * sigma_E. I count a value as zero only if it is inside plus or minus tau = 5 sigma_E, about three times
that error, where sigma_E comes from the null tests (at least 0.001 meV).

**Statistics.** For each case I have 20 shaken runs. I report the mean, the standard deviation, and a 95 percent
interval, mean plus or minus 2.093 times sd divided by sqrt(20). The 20 runs measure sensitivity to geometry, not
training variance, since the models are fixed. The decision rule, fixed before the runs:

- **VIOLATION:** the whole interval is outside plus or minus tau, and at least 19 of 20 runs have the same sign.
- **CONSISTENT:** the whole interval is inside plus or minus tau.
- **INCONCLUSIVE:** anything else.

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
- Delta at a gap of 8 angstrom equals Delta at 100 angstrom for every model, task and case, within tolerance. One
  cell differs by 0.05 meV, which is negligible next to 3,524 meV.
- The molecule task shows an error 50 to 100 times larger than the materials task.

Per-model plots of Delta against gap: `figures/delta_<model>_<task>.png`.

## 6. What the results mean

- **The error is not a real interaction.** It does not fade with distance, and for a gap of 8 angstrom or more the
  model cannot connect the slabs. Something global must feed in.
- **It matches the whole-structure router.** It appears when different metals are mixed and disappears for same-metal
  pairs in the two version-1.1 UMA models. ORB shows the same pattern for its own reason, the global mean.
- **The size is large.** About 2 to 3.5 eV on the materials task is far above typical chemical accuracy of about
  0.04 eV, and tens to hundreds of eV on the molecule task.
- **Compared with the paper** (Batatia et al., arXiv 2510.25380): the sign agrees in all five printed values. The sizes
  differ, with ratios of ours to theirs of 0.71 (ORB), 4.7 (UMA-S-1.1 materials), 1.23 (UMA-S-1.1 molecules), 2.45
  (UMA-M-1.1 materials) and 0.88 (UMA-M-1.1 molecules). The paper does not give its slab geometry in enough detail,
  and its UMA-M materials row does not add up even from its own printed energies, so I did not try to match it.

## 7. A test of the cause: change only the router version number

I overwrote the stored version number and re-ran the same-metal pairs (5 shaken runs plus the ideal one).

| change | task | Al + Al before to after | Ni + Ni before to after |
|---|---|---|---|
| UMA-S-1.2, 1.0 to 1.1 | omat | -298.2 to 0.05 | -0.01 to -0.02 |
| UMA-S-1.2, 1.0 to 1.1 | omol | -308.3 to 0.25 | +3,238 to +5.8 |
| UMA-S-1.1, 1.1 to 1.0 | omat | 0 to -48.3 | 0 to +0.9 |
| UMA-S-1.1, 1.1 to 1.0 | omol | 0 to +681.9 | 0 to -517.6 |

The same-metal error switches off and on with that one number, in both directions. The Al + Ni error barely moves,
so the main error is not the version quirk. It is the whole-structure average itself.

## 8. Limits, stated plainly

- UMA-M ran in float32, so its tolerance is larger (about 5 meV on the molecule task).
- Swapping the version number may change other version-dependent behavior that I did not inspect.
- Geometries are unrelaxed and use one specific shared cell, so the values are for this setup only.
- The same-metal check shows the router is a plausible cause. It does not prove it. A stronger test (forcing the
  router to see another fragment's composition) is not done yet.
- The results were not uploaded to W&B. Training baselines (Dense, Global, Local-6) are not part of this audit.

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

How to rerun, with the exact commands and the two environments, is in `../audit/README.md`.

## 10. What comes next, and what depends on whom

- **Mine, needs nobody:** push to W&B, the composition-forcing test, and the neon-probe version of the test.
- **Needs Yiwei's pipeline:** the Dense and Global baselines. I also need a way to load a trained checkpoint as a
  function that returns the energy of an atoms object, so this audit can test Dense, Global and Local-6.
- **Needs Pragati's router:** Local-6 must give Delta = 0 for gaps of 8 angstrom or more. This audit is the test.
- For the midterm, this audit alone gives the reproduced-baseline evidence. The Dense and Global pilots are the part
  that depends on the pipeline.
