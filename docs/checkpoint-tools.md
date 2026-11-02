# Checkpoint selection and averaging

These tools implement the proposed K=5 rule in the final training plan. They do not train models or submit jobs. CPU tests use three small, generated fairchem checkpoints and a labelled ASE database. The tools have also been checked on three saved Local-6 pilot checkpoints from PSC (seed 0, steps 30,000, 40,000 and 50,000).

## Fixed rule and measurements

For one run, rank checkpoints from the last half of training by selection-set force MAE. Take the lowest five; ties go to the earlier step. The best checkpoint is the first in that ranking. The final checkpoint is the last saved step. Do not tune K or the rule using the report results.

The default lists are `select_val_5k.txt` and `pilot_val_5k.txt`. They must not overlap. Indices refer to the full, unchanged OMol25 validation split, in fairchem's database order. Do not pass a compacted subset as the data source. The tool records list hashes and the database file inventory. These checks detect changed lists and file sizes; they do not replace verification of the downloaded dataset against its published checksums.

Energy MAE is the mean, over structures, of absolute total-energy error divided by atom count (eV/atom). Force MAE is the mean absolute error over all Cartesian force components (eV/Å). Predictions include the checkpoint's normalizer and element references. Charge and spin come from each database row. One structure is evaluated at a time, so this is an accuracy evaluator, not a throughput benchmark.

## Commands

Run from the repository root, in the fairchem 2.21 environment. Replace the example paths with one run and the full validation split. `--variant`, `--seed` and `--total-steps` must match `settings.txt`. The current reader expects one `train/*/checkpoints` directory per run; chained runs must retain that layout. A `final` folder requires a successful run summary. Incomplete or differently arranged runs need an explicit, reviewed mapping of checkpoints to actual steps first.

```bash
python scripts/checkpoint_tools.py select \
  --run-dir /path/to/pilot_local6_s0_run \
  --variant local6 --seed 0 --total-steps 50000 \
  --data /path/to/omol25/val --device cuda \
  --output /path/to/output/selection

python scripts/checkpoint_tools.py average \
  --selection /path/to/output/selection/selection.json \
  --output /path/to/output/weight_average.pt

python scripts/checkpoint_tools.py report \
  --selection /path/to/output/selection/selection.json \
  --averaged /path/to/output/weight_average.pt \
  --data /path/to/omol25/val --device cuda \
  --output /path/to/output/report
```

Outputs are new files/directories; existing results are not overwritten. Selection writes full-precision metrics and a manifest of the selected files and hashes. Averaging writes an inference checkpoint and a provenance sidecar. Report evaluation writes four rows: `final`, `best`, `weight_average`, `prediction_average`. The CSV uses Yiwei's ensemble column names and `eval_split=report`.

## What gets averaged

Fairchem inference reads `ema_state_dict`, not the raw training weights. Weight averaging takes the arithmetic mean of EMA parameters from the selected checkpoints. It accumulates in float64, retains parameter dtypes, requires identical fixed buffers, and checks model and task configurations. The output's raw and EMA fields both hold the combined model. It is for inference only; it has no optimizer or resume state.

Prediction averaging gives every member the same constant weight. It averages energy and forces, which keeps forces consistent with the averaged energy. K models remain resident on the chosen device; memory and inference cost increase with K. Weight averaging produces one model. Neither method guarantees an accuracy gain.

The audit adapter is `audit.models.load_prediction_average`. To measure a selected ensemble on the slab benchmark:

```bash
python audit/check_checkpoint.py --prediction-average checkpoint1.pt checkpoint2.pt checkpoint3.pt
```

Use all selected paths for the final K=5 evaluation. Check the weight-averaged checkpoint with the same script, without `--prediction-average`. Disconnected local models should remain additive; global models have no such guarantee. Always measure the final combined models.

## Pilot check

The end-to-end CPU check used K=3, four selection structures and four separate report structures from the first validation shard. It exercised selection, EMA weight averaging, reload and all four report methods. This is a software check, not a comparison of ensemble accuracy or the planned K=5 experiment. The shard checksum matched the repository data manifest.

On a water–methane pair at gaps of 8, 15 and 100 Å, the largest absolute additivity residual was 2.3e-13 eV. Both averaging methods agreed with energy finite differences within 5e-7 eV/Å (displacement 1e-4 Å). Inputs, checkpoint hashes and metrics are in `results/checkpoint_smoke.json`.

All 190 branch tests passed. A separate integration check with Yiwei's unmerged Element and Local+Global changes passed 268 tests after resolving the shared backbone and Slurm variant-list edits.

## Validation still to do

1. Check the evaluator's final-checkpoint MAEs against the pilot logs, allowing for float precision and the logs' rounding.
2. Run the fixed K=5 selection on the full selection set, then evaluate all four methods on the report set.
3. Run the final slab and molecular checks. Record peak memory and wall time on an allocated compute node.

Do not run full evaluation on a PSC login node. Anshul submits cluster jobs after review.
