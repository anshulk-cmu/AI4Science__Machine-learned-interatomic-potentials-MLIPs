# Final evaluation of the 8 long runs (seed 0, 400,000 steps)

Status: 2026-11-14. Branch `final-eval`. "Measured" means a job or script computed the value; "reasoning" means it was not run.

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
- Standard error (SE): energy, s / sqrt(n) over structures. Force, the ratio estimator with the structure as the sampling unit:
  SE = sqrt( sum_i (a_i - F c_i)^2 / (n (n - 1)) ) / mean(c), where a_i is the summed absolute force error of structure i, c_i its
  number of components and F the force MAE. The SE covers the sampling of rows only. It does not cover training noise (one seed).
- Paired differences: model A minus model B on the same rows, per structure, with the same two estimators applied to the
  differences. Rows that are hard for every model cancel, so these SEs are 3 to 8 times smaller than the SE of one MAE.

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

| Model | Energy MAE (meV/atom) | SE | Force MAE (meV/A) | SE |
|---|---:|---:|---:|---:|
| Dense | 5.611 | 0.081 | 28.65 | 0.074 |
| Scaled Dense | 5.322 | 0.071 | 27.92 | 0.073 |
| Frozen | 5.464 | 0.080 | 27.72 | 0.072 |
| Global | 5.447 | 0.080 | 27.18 | 0.071 |
| Element | 5.144 | 0.078 | 26.07 | 0.071 |
| Local-6 | 5.337 | 0.078 | 27.19 | 0.071 |
| Local+Global | 5.517 | 0.081 | 27.51 | 0.072 |
| Local-12 | 5.348 | 0.074 | 27.26 | 0.072 |

**Paired differences on the same 399,276 structures (model A minus model B, meV; a negative value means A has the lower error; SE of the per-structure difference; measured). The SE covers the sampling of rows only, with one training seed.**

| A | B | Energy diff (meV/atom) | SE | Force diff (meV/A) | SE |
|---|---|---:|---:|---:|---:|
| Scaled Dense | Dense | -0.289 | 0.024 | -0.738 | 0.011 |
| Frozen | Dense | -0.147 | 0.022 | -0.935 | 0.011 |
| Global | Dense | -0.164 | 0.030 | -1.471 | 0.012 |
| Element | Dense | -0.467 | 0.025 | -2.580 | 0.013 |
| Local-6 | Dense | -0.274 | 0.025 | -1.464 | 0.012 |
| Local+Global | Dense | -0.094 | 0.031 | -1.140 | 0.012 |
| Local-12 | Dense | -0.263 | 0.024 | -1.394 | 0.012 |
| Dense | Global | +0.164 | 0.030 | +1.471 | 0.012 |
| Scaled Dense | Global | -0.125 | 0.033 | +0.732 | 0.012 |
| Frozen | Global | +0.017 | 0.023 | +0.536 | 0.010 |
| Element | Global | -0.303 | 0.029 | -1.109 | 0.012 |
| Local-6 | Global | -0.110 | 0.022 | +0.007 | 0.009 |
| Local+Global | Global | +0.070 | 0.020 | +0.330 | 0.011 |
| Local-12 | Global | -0.099 | 0.020 | +0.077 | 0.008 |

- **Measured.** Local-6 against Global: force difference +0.007 meV/A (SE 0.009), the same; energy difference -0.110 meV/atom (SE 0.022), Local-6 slightly lower.
- **Measured.** Global against Dense: force -1.471 meV/A (SE 0.012), energy -0.164 meV/atom (SE 0.030). Local-6 against Dense: force -1.464 (SE 0.012), energy -0.274 (SE 0.025).
- **Measured.** Scaled Dense against Dense: force -0.738 meV/A (SE 0.011), energy -0.289 meV/atom (SE 0.024); Frozen against Dense: force -0.935 (SE 0.011). Element against Global: force -1.109 (SE 0.012), energy -0.303 (SE 0.029).
- **Interpreted.** About half of the force gain of Global over Dense (0.74 of 1.47 meV/A) is reached by Scaled Dense without any experts; Local-6 keeps all of the gain of Global. The energy gain over Dense is larger for Scaled Dense and Local-6 than for Global.

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

Reading (reasoning, from the tables above):

- Forces: the order of validation holds on test. Element is best. Local-6, Global and Local-12 are tied (Local-6 minus Global is
  -0.03 +- 0.06 meV/A). Every MoE variant beats Dense by 11 to 20 paired SE. The frozen router keeps about two thirds of the Global gain.
- Energies: on validation, Global and Local-6 beat Dense by about 0.5 meV/atom (3 to 4 SE). On test this gain is gone (within 1 SE).
  Only Element and Scaled Dense keep a small energy gain on test. The energy error has a heavy tail (Global has the largest SE), so
  the energy MAE of 10,000 rows is less stable than the force MAE.

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
`results/long_validation_curves.csv`, `results/long_final_metrics.csv`, `results/figures/long_accuracy.png` (training loss, validation
force error, gain over Dense, final values) and `results/figures/long_val_curves.png` (validation energy and force against step).

```bash
python scripts/long_curves.py finalRuns/long_*_s0_400000 --out results/long_validation_curves.csv --plot results/figures/long_val_curves.png
python scripts/collect_final_metrics.py --run-manifest long_runs.json --steps 400000 --expected-seeds 1 --source log \
  --output results/long_final_metrics.csv
python scripts/make_accuracy_figure.py results/figures/long_accuracy.png --metrics results/long_final_metrics.csv \
  --curves results/long_validation_curves.csv --long-run
```

The size consistency along training (`scripts/size_consistency_along_training.py`) was not run. The checkpoints at multiples of 10,000
steps exist on Babel.

## 6. Caveats

- One seed per variant. Differences of about 1 meV/A in force could change with another seed; the SEs above do not show this.
- The final checkpoint is reported, not the K=5 rule of `checkpoint-tools.md` (no selection was run).
- Test and validation are different compositions. The test set has slightly smaller structures (median 40 against 42 atoms).
- float32. Batched evaluation (600 atoms per forward pass; 2,000 for the first rows of some jobs) changes MAEs by under 5e-9.
- The data were not checked again against published checksums. Every job checks the list hash and the `train_4M` row count
  (`check_split`), and the merge checks that every row of the list appears exactly once.
