# Final evaluation of the 8 long runs (seed 0, 400,000 steps)

Status: 2026-11-14. Branch `final-eval`. "Measured" means a job or script computed the value; "reasoning" means it was not run.
Updated 2026-11-16: standard errors by composition (the test list was split by composition), the two-atom rows, and stale lines. What
the SEs do not cover and which code trained each run: [limits-and-provenance.md](limits-and-provenance.md).

## 1. What was evaluated

- The `final` checkpoint of each long run on Babel:
  `/data/user_data/anshulk/mlip/runs/long_<variant>_s0_400000/train/*/checkpoints/final/inference_ckpt.pt`, variants `dense`,
  `dense_scaled`, `global_frozen`, `global`, `element`, `local6`, `local_global`, `local12`. One seed per variant.
- Lists ([data-split.md](data-split.md)): `split_test_eval_10k.txt` (10,000 rows of the test set), `split_test_10.txt` (all 399,276
  test rows), and `split_val_monitor_5k.txt` (the 5,000 validation rows that the runs logged every 10,000 steps). All rows are read
  from the `train_4M` folder, as the runs read them. The test lists were not used for any choice.
- Metrics (the same as `training.checkpoints.evaluate`): energy MAE is the mean over structures of |E_pred - E_DFT| / N_atoms
  (eV/atom); force MAE is the mean absolute error over all Cartesian force components (eV/A). float32 (`data-split.md`, Section 7:
  float32 and float64 agree within 4e-8 eV/atom).
- Standard error (SE), "SE rows": energy, s / sqrt(n) over structures. Force, the ratio estimator with the structure as the sampling
  unit: SE = sqrt( sum_i (a_i - F c_i)^2 / (n (n - 1)) ) / mean(c), where a_i is the summed absolute force error of structure i, c_i
  its number of components and F the force MAE. This treats every structure as an independent draw.
- "SE compositions" (added 2026-11-16, `scripts/cluster_se.py`): the test list holds whole compositions ([data-split.md](data-split.md)),
  and the structures of one composition are correlated. Both MAEs are ratios R = sum_i a_i / sum_i c_i (energy: a_i the error per atom,
  c_i = 1). With u_g = sum over the structures i of composition g of (a_i - R c_i) and G compositions,
  Var(R) = G / (G - 1) * sum_g u_g^2 / (sum_i c_i)^2. With one structure per group this equals the row SE exactly. Measured on the full
  list: G = 110,864, largest composition 9,907 rows (`C4H7N1O1`); a cluster bootstrap with 1,000 resamples agrees within 4 percent.
  The composition of every test row is in `results/test_row_composition.csv.gz` (3.1 MB; unzip it with `gunzip -k` before use, its sha256 after
  unzipping starts with `9f190bdae941c654`), so the SEs can be recomputed from git and the per-row evaluation files alone.
  The row SEs are too small by 5.1 to 7.4 times (energy MAE), 2.8 to 2.9 (force MAE), 1.9 to 7.0 (paired energy) and 1.19 to 1.69
  (paired force). Neither SE covers training noise (one seed, non-deterministic GPU path).
- Paired differences: model A minus model B on the same rows, per structure, with the same estimators applied to the differences.
  Rows that are hard for every model cancel. With row SEs, the paired SEs on the full list are 2.1 to 8.4 times smaller than the SE of
  one MAE (energy 2.1 to 4.1, force 5.3 to 8.4); with composition SEs, 1.5 to 11.2 (energy) and 9.0 to 19.3 (force). Measured.

## 2. Tools and commands

`scripts/eval_rows.py` (new). One CSV line per structure, shards (`rows[shard::num_shards]`), resume after preemption (a cut last line
is dropped), several structures per forward pass (`--batch-atoms`), a merge that refuses a missing, extra or duplicated row or shards
from different checkpoints, lists or precisions, and `paired`. `slurm/eval_rows.sh` runs one shard per array task on Babel.

```bash
# on Babel, from /home/anshulk (the job script reads the code in code-babel/AI4Science-MLIPs)
sbatch --array=0-3 --job-name=mlip_eval_local6_testfull \
  --output=/data/user_data/anshulk/mlip/logs/mlip_eval_local6_testfull_%A_%a.out \
  --export=ALL,VARIANT=local6,LIST=manifests/split_test_10.txt,TAG=testfull,NSHARDS=4 mlip_eval_rows.sh
# merge (checks every row once) and paired difference
python scripts/eval_rows.py merge --rows manifests/split_test_10.txt --label local6 --eval-split test_full \
  --table $E/testfull_metrics.csv $E/testfull/local6_s{0,1,2,3}of4.csv
python scripts/eval_rows.py paired --rows manifests/split_test_10.txt --a $E/testfull/local6_s*of4.csv --b $E/testfull/global_s*of4.csv
```

**Checks of the evaluator (measured).**

- Batched against one structure at a time, 200 rows drawn with `default_rng(0)` from `split_test_10.txt`, L40S, float32: the MAEs
  differ by at most 4.8e-9 eV/atom and 4.8e-9 eV/A (Dense) and 2.7e-9 and 2.3e-10 (Local-6); the largest difference of one structure is
  3.0e-7 eV/atom.
- On the monitor list, all 8 models give the training-time validation values of `results/long_final_metrics.csv` when rounded to 4
  decimals. So the evaluator agrees with fairchem's own validation loop on the same checkpoints and rows.

**Speed (measured, rows / wall seconds including start-up, float32).** Dense and Scaled Dense 0.025 s per structure (RTX A6000), Global
0.045 (A6000), Frozen 0.042 (A6000), Element 0.055 (RTX 6000 Ada), Local-6 0.053 (L40S), Local+Global 0.034 (RTX PRO 6000
Blackwell), Local-12 0.058 (L40S). About 0.34 s per row for all 8 models, so about 38 GPU-hours for the full test list (reasoning from
these times). This is 3 to 4 times faster than the V100 one-at-a-time estimate of `data-split.md`.

## 3. Results

**Test, full list `split_test_10.txt` (399,276 structures, 64,695,648 force components; measured, float32, final checkpoints, every row once).**

| Model | Energy MAE (meV/atom) | SE rows | SE compositions | Force MAE (meV/A) | SE rows | SE compositions |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 5.611 | 0.081 | 0.421 | 28.653 | 0.074 | 0.216 |
| Scaled Dense | 5.322 | 0.071 | 0.362 | 27.915 | 0.073 | 0.210 |
| Frozen | 5.464 | 0.080 | 0.451 | 27.719 | 0.072 | 0.211 |
| Global | 5.447 | 0.080 | 0.575 | 27.183 | 0.071 | 0.207 |
| Element | 5.144 | 0.078 | 0.443 | 26.074 | 0.071 | 0.201 |
| Local-6 | 5.337 | 0.078 | 0.477 | 27.190 | 0.071 | 0.208 |
| Local+Global | 5.517 | 0.081 | 0.600 | 27.513 | 0.072 | 0.209 |
| Local-12 | 5.348 | 0.074 | 0.474 | 27.259 | 0.072 | 0.209 |

**Paired differences on the same 399,276 structures (model A minus model B, meV; a negative value means A has the lower error;
measured).** "z comp." is the difference divided by its composition SE. Both SEs cover the sampling of test structures only, with one
training seed. Files: `results/final_test_paired_full.jsonl` (row SEs), `results/final_test_paired_cluster.csv` (both).

| A | B | Energy diff | SE rows | SE comp. | z comp. | Force diff | SE rows | SE comp. | z comp. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scaled Dense | Dense | -0.289 | 0.024 | 0.073 | -4.0 | -0.738 | 0.011 | 0.016 | -46.3 |
| Frozen | Dense | -0.147 | 0.022 | 0.057 | -2.6 | -0.935 | 0.011 | 0.014 | -66.9 |
| Global | Dense | -0.164 | 0.030 | 0.180 | -0.9 | -1.471 | 0.012 | 0.018 | -81.3 |
| Element | Dense | -0.467 | 0.025 | 0.048 | -9.8 | -2.580 | 0.013 | 0.022 | -115.2 |
| Local-6 | Dense | -0.274 | 0.025 | 0.078 | -3.5 | -1.464 | 0.012 | 0.017 | -84.5 |
| Local+Global | Dense | -0.094 | 0.031 | 0.203 | -0.5 | -1.140 | 0.012 | 0.017 | -66.5 |
| Local-12 | Dense | -0.263 | 0.024 | 0.082 | -3.2 | -1.394 | 0.012 | 0.020 | -68.1 |
| Dense | Global | +0.164 | 0.030 | 0.180 | +0.9 | +1.471 | 0.012 | 0.018 | +81.3 |
| Scaled Dense | Global | -0.125 | 0.033 | 0.235 | -0.5 | +0.732 | 0.012 | 0.019 | +37.8 |
| Frozen | Global | +0.017 | 0.023 | 0.140 | +0.1 | +0.536 | 0.010 | 0.014 | +39.0 |
| Element | Global | -0.303 | 0.029 | 0.150 | -2.0 | -1.109 | 0.012 | 0.016 | -67.9 |
| Local-6 | Global | -0.110 | 0.022 | 0.110 | -1.0 | +0.007 | 0.009 | 0.011 | +0.6 |
| Local+Global | Global | +0.070 | 0.020 | 0.053 | +1.3 | +0.330 | 0.011 | 0.013 | +25.4 |
| Local-12 | Global | -0.099 | 0.020 | 0.112 | -0.9 | +0.077 | 0.008 | 0.012 | +6.4 |

**The two-atom rows (measured).** The test list holds 804 two-atom structures (0.2 percent of the rows) in 18 compositions. Their
energy errors are about 40 to 65 times the mean (two-atom MAE 223 to 347 meV/atom against 5.1 to 5.6 overall). `S2` alone (358 rows, DFT
energy -21,671 to -21,650 eV) carries 95 percent of the summed Local-6 minus Global energy difference, and 102 percent with the
opposite sign of Global minus Dense (Global is worse on `S2`, better on the rest). The same pairs without the two-atom rows (398,472
structures, 110,846 compositions; `results/final_test_by_size.csv`, which also has the bins 2, 3 to 20 and above 20 atoms):

| A | B | Energy diff | SE comp. | z comp. | Force diff | SE comp. | z comp. |
|---|---|---:|---:|---:|---:|---:|---:|
| Scaled Dense | Dense | -0.255 | 0.033 | -7.7 | -0.734 | 0.015 | -47.6 |
| Frozen | Dense | -0.207 | 0.030 | -7.0 | -0.933 | 0.014 | -67.4 |
| Global | Dense | -0.378 | 0.036 | -10.5 | -1.471 | 0.018 | -81.9 |
| Element | Dense | -0.512 | 0.030 | -17.2 | -2.581 | 0.022 | -115.5 |
| Local-6 | Dense | -0.364 | 0.027 | -13.6 | -1.460 | 0.017 | -86.1 |
| Local+Global | Dense | -0.310 | 0.040 | -7.7 | -1.140 | 0.017 | -66.9 |
| Local-12 | Dense | -0.363 | 0.036 | -10.1 | -1.390 | 0.020 | -69.6 |
| Dense | Global | +0.378 | 0.036 | +10.5 | +1.471 | 0.018 | +81.9 |
| Scaled Dense | Global | +0.123 | 0.028 | +4.4 | +0.737 | 0.019 | +38.5 |
| Frozen | Global | +0.171 | 0.030 | +5.8 | +0.537 | 0.014 | +39.4 |
| Element | Global | -0.135 | 0.034 | -3.9 | -1.110 | 0.016 | -68.1 |
| Local-6 | Global | +0.013 | 0.027 | +0.5 | +0.010 | 0.010 | +1.0 |
| Local+Global | Global | +0.068 | 0.026 | +2.6 | +0.331 | 0.013 | +25.6 |
| Local-12 | Global | +0.015 | 0.023 | +0.7 | +0.081 | 0.011 | +7.1 |

- **Measured.** Force: every pair except Local-6 minus Global has |z comp.| of 6 or more. Local-6 minus Global is +0.007 meV/A
  (SE comp. 0.011, z +0.6): zero within the error, with and without the two-atom rows.
- **Measured.** Energy on the full list: Global minus Dense -0.164 (z -0.9), Local-6 minus Global -0.110 (z -1.0), Local+Global minus
  Dense -0.094 (z -0.5). No energy difference between Dense, Global and Local-6 is established on the test list.
- **Measured.** Element has the lowest force error (26.074 meV/A; -2.580 against Dense, -1.109 against Global) and the lowest energy
  error in the table (5.144 meV/atom). Its energy gap to Dense holds with composition SEs and without the two-atom rows (-0.467, z -9.8;
  -0.512, z -17.2). Its energy gap to Global is at z -2.0 on the full list and -0.135 (z -3.9) without the two-atom rows. Paired
  differences were computed against Dense and Global only.
- **Interpreted.** The force results are the evidence. About half of the force gain of Global over Dense (0.738 of 1.471 meV/A) is
  reached by Scaled Dense without any experts, and Local-6 keeps all of the force gain of Global. Without the two-atom rows, every
  routed model and Scaled Dense has a lower energy error than Dense, and no difference between Local-6 and Global is detected; this subset was chosen after
  the two-atom effect was seen, so it is a description, not a test. 14 pairs are shown with no correction for multiple comparisons.

**Test, `split_test_eval_10k.txt` (10,000 structures, 1,607,829 force components; measured).**

| Model | Energy MAE (meV/atom) | SE | Force MAE (meV/A) | SE |
|---|---:|---:|---:|---:|
| Dense | 5.186 | 0.29 | 29.18 | 0.48 |
| Scaled Dense | 4.907 | 0.22 | 28.53 | 0.48 |
| Frozen | 5.241 | 0.32 | 28.30 | 0.48 |
| Global | 5.265 | 0.42 | 27.85 | 0.48 |
| Element | 4.727 | 0.26 | 26.73 | 0.49 |
| Local-6 | 5.177 | 0.37 | 27.82 | 0.48 |
| Local+Global | 5.249 | 0.37 | 28.23 | 0.48 |
| Local-12 | 5.096 | 0.33 | 27.92 | 0.48 |

**Validation, `split_val_monitor_5k.txt` (5,000 structures, 825,036 force components; measured).**

| Model | Energy MAE (meV/atom) | SE | Force MAE (meV/A) | SE |
|---|---:|---:|---:|---:|
| Dense | 5.382 | 0.29 | 28.97 | 0.66 |
| Scaled Dense | 5.004 | 0.25 | 28.19 | 0.64 |
| Frozen | 5.120 | 0.26 | 27.92 | 0.63 |
| Global | 4.918 | 0.24 | 27.38 | 0.63 |
| Element | 4.657 | 0.24 | 26.19 | 0.61 |
| Local-6 | 4.886 | 0.25 | 27.31 | 0.62 |
| Local+Global | 4.986 | 0.27 | 27.59 | 0.62 |
| Local-12 | 5.014 | 0.28 | 27.52 | 0.63 |

**Paired differences on the 10,000 test rows (measured), model minus Dense and the central pair.**

| Pair | Force difference (meV/A) | SE | Energy difference (meV/atom) | SE |
|---|---:|---:|---:|---:|
| Scaled Dense - Dense | -0.64 | 0.07 | -0.28 | 0.13 |
| Frozen - Dense | -0.87 | 0.07 | +0.05 | 0.10 |
| Global - Dense | -1.33 | 0.08 | +0.08 | 0.17 |
| Element - Dense | -2.45 | 0.12 | -0.46 | 0.12 |
| Local-6 - Dense | -1.36 | 0.08 | -0.01 | 0.12 |
| Local+Global - Dense | -0.94 | 0.08 | +0.06 | 0.13 |
| Local-12 - Dense | -1.25 | 0.09 | -0.09 | 0.10 |
| Local-6 - Global | -0.03 | 0.06 | -0.09 | 0.10 |
| Element - Global | -1.12 | 0.11 | -0.54 | 0.21 |
| Frozen - Global | +0.46 | 0.07 | -0.02 | 0.14 |

All pairs on both lists: `results/final_eval/paired_test10k.jsonl` and `paired_val5k.jsonl`.

The SEs of the 10,000-row and 5,000-row tables are row SEs. They are too small for the same reason as on the full list (the rows are
grouped by composition); composition SEs were computed for the full test list only. The full list supersedes the 10,000-row list.

Reading (reasoning, from the tables above):

- Forces: the order of validation holds on test. Element is best. Local-6, Global and Local-12 are tied (Local-6 minus Global is
  -0.03 +- 0.06 meV/A on 10,000 rows, +0.007 +- 0.011 on the full list with composition SEs). Every MoE variant beats Dense.
  The frozen router keeps about two thirds of the Global force gain.
- Energies: on the monitor list, Global and Local-6 have about 0.5 meV/atom lower energy error than Dense (row SEs). On the 10,000 test
  rows the paired differences to Dense are within 1 row SE for Global and Local-6. On the full test list with composition SEs, no
  energy difference between Dense, Global and Local-6 is established (z -0.9 and -1.0, Section 3 above); the energy differences
  there are dominated by 804 two-atom structures. Element's energy gap to Dense holds on every list.

## 4. Size consistency on 21 geometries

Gap Delta = E(Al + Ni) - E(Al) - E(Ni) for the final checkpoints, float64, charge 0, spin 1 (task `omol`), on the ideal slabs (run 0)
and the 20 rattled runs of the audit protocol (`audit/check_checkpoint.py --runs 20`, the fragments Al_a and Ni_a of `run_audit.py`,
rattle 0.05 A), gaps 8, 10, 15, 25, 50 and 100 A. Script `slurm/size_consistency_final.sh`; table
`results/final_size_consistency.csv` (one row per model, run and gap) from `scripts/size_consistency_table.py`. Measured.

| Model | Mean Delta over 21 runs (meV) | SD (meV) | Range (meV) | Change with the gap inside one run |
|---|---:|---:|---|---|
| Dense, Scaled Dense, Frozen, Element, Local-6 | 2.7e-6 to 3.0e-6 | about 5e-7 | 1.9e-6 to 3.7e-6 | 0 (float64 rounding of a 256-atom energy) |
| Global | -63,711 | 428 | -64,291 to -62,350 | 0: the same value at 8 and at 100 A |
| Local+Global | -9,577 | 110 | -9,724 to -9,226 | 0: the same value at 8 and at 100 A |
| Local-12 at 8 A | 3,950 | 32 | 3,885 to 4,004 | falls to 325 (SD 7) at 10 A and to 2.7e-6 from 15 A |

- Run 0 equals the single check of the training jobs (`results/long_runs_validation.csv`): Global -62,350.3, Local+Global -9,226.24,
  Local-12 3,916.17 at 8 A and 320.07 at 10 A.
- Global and Local+Global: the gap is flat in distance and its size moves with the geometry by under 1 percent. The status column of
  the script says "disconnected" for them because it checks graph and routing radii only; a global router has no finite radius.
- Local-12 is connected through its 12 A routing radius at 8 and 10 A (status "routing-connected") and exactly additive beyond it.
  Local-6 is additive at every gap from 8 A, like Dense.

## 5. Training curves

`scripts/long_curves.py` reads the segment logs of each run in `segments.txt` order and keeps the validations at multiples of 10,000
steps (40 per run; the end-of-epoch validation at step 381,159 is dropped). `collect_final_metrics.py --source log` gives the final
validation row (the W&B file of one Local-6 segment is truncated, so `--source auto` refuses). Files:
`results/long_validation_curves.csv`, `results/long_final_metrics.csv` (training-log values, rounded to 4 decimals; not for rankings),
`results/figures/long_accuracy.png` (training loss, validation
force error, gain over Dense, final values) and `results/figures/long_val_curves.png` (validation energy and force against step).

```bash
python scripts/long_curves.py finalRuns/long_*_s0_400000 --out results/long_validation_curves.csv --plot results/figures/long_val_curves.png
python scripts/collect_final_metrics.py --run-manifest long_runs.json --steps 400000 --expected-seeds 1 --source log \
  --output results/long_final_metrics.csv
python scripts/make_accuracy_figure.py results/figures/long_accuracy.png --metrics results/long_final_metrics.csv \
  --curves results/long_validation_curves.csv --long-run
```

The size consistency along training was run on 2026-11-15 for Global and Local-6 (`scripts/size_consistency_along_training.py`, commit
`31da755`): [size-consistency-long-runs.md](size-consistency-long-runs.md). The validation at the epoch boundary (step 381,159) is not in
the curves file ([limits-and-provenance.md](limits-and-provenance.md), Section 2).

## 6. Caveats

- One seed per variant, and the GPU path is not deterministic. Differences of about 1 meV/A in force could change with another seed;
  the SEs above do not show this. Details, the code versions of each run and the other limits: [limits-and-provenance.md](limits-and-provenance.md).
- The row SEs ignore the grouping by composition; use the composition SEs (Section 1).
- The energy differences are dominated by 804 two-atom structures (Section 3).
- The final checkpoint is reported, not the K=5 rule of `checkpoint-tools.md` (no selection was run).
- Test and validation are different compositions. The test set has slightly smaller structures (median 40 against 42 atoms).
- float32. Batched evaluation (600 atoms per forward pass; 2,000 for the first rows of some jobs) changes MAEs by under 5e-9.
- The data were not checked again against published checksums. Every job checks the list hash and the `train_4M` row count
  (`check_split`), and the merge checks that every row of the list appears exactly once.
