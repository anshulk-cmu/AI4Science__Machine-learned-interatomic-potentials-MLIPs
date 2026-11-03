# Data split for the final runs: 70 / 20 / 10 of train_4M by composition

Writing style: simple English. Short sentences. One idea in each sentence. "Measured" means that a script computed the value on the
real data. "Reasoning" means that it was not run.

## 1. The lists

All lists hold row numbers of the OMol25 `train_4M` split (3,986,754 rows), in the order that fairchem and the index file
`$OMOL_DATA_ROOT/pilot/train_4M_index.npz` use. One row number per line, sorted, LF line ends. `manifests/split_samples.json` holds
the counts, seeds and sha256 values. The written lists are the source of truth.

| File | Rows | Share of train_4M | sha256 (first 16) | Seed | Role |
|---|---:|---:|---|---:|---|
| `split_train_70.txt` | 2,791,057 | 70.008 percent | `f57e446a78c48625` | 11790 | training rows of the final runs |
| `split_val_20.txt` | 796,421 | 19.977 percent | `d963a939aef7ea9b` | 11790 | validation set (never trained on) |
| `split_test_10.txt` | 399,276 | 10.015 percent | `3f13039de47b76f7` | 11790 | test set (held out; used once at the end) |
| `split_val_monitor_5k.txt` | 5,000 | rows of `split_val_20` | `4aa3aae414296dfb` | 11791 | validation during training, every `--eval-every` steps |
| `split_val_select_10k.txt` | 10,000 | rows of `split_val_20`, none of the monitor list | `dfa6ffbd71290f91` | 11792 | checkpoint choice |
| `split_test_eval_10k.txt` | 10,000 | rows of `split_test_10` | `d7b093aa22949416` | 11793 | the final evaluation, once |

- The three main lists share no row, no composition and no calculation folder. Together they hold every row once.
- The selection list shares no row with the monitor list. Both are uniform draws from `split_val_20.txt`.
- The pilot lists (`pilot_train_200k.txt`, `pilot_val_5k.txt`) are not changed. `long_train_1200k.txt` and `select_val_5k.txt` stay in
  the repository, but the final runs do not use them (Section 7).

## 2. Why the split is by group, not by row

- One molecule has many rows in OMol25: conformers, optimization steps, and isomers with the same formula. A split by row puts rows of
  one molecule in the training set and in the test set. The test error is then too low.
- OMol25 holds compositions out of its official `val` split. We measured this on every row: 0 of the 603,008 compositions of `val`
  occur in `train_4M`. Our split follows the same convention inside `train_4M`.

## 3. Choice of the group key

We read two candidate keys from every row (job 50162814): the composition (`atoms.info["composition"]`, a formula such as
`C4H7N1O1`) and the calculation folder (the source path without the file name and without a trailing `stepN`).

| Key | Groups | Rows per group: median / p90 / p99 / largest | Largest group, share of rows | Groups of one row |
|---|---:|---|---:|---:|
| composition | 1,127,158 | 1 / 4 / 28 / 9,907 | 0.248 percent (`C4H7N1O1`) | 684,684 |
| calculation folder | 3,892,230 | 1 / 1 / 2 / 11 | 0.0003 percent | 3,807,948 |

- 409,595 compositions have rows in more than one folder (3,233,793 rows). No folder holds two compositions. So composition is the
  coarser key, and a composition split is also a folder split (measured: no folder is in two sets).
- The folder key is almost one row per group. A folder split would be close to a row split, and it would let other conformers and
  isomers of a test molecule into training (reasoning).
- 218 compositions have 1,000 rows or more. They hold 516,056 rows (12.9 percent). By main data source: `trans1x` 70 compositions
  (261,719 rows), `biomolecules` 84 (163,611), `ani2x` 21 (29,257), `reactivity` 22 (28,768), `elytes` 11 (17,362), `rgd` 10 (15,339).
  The five largest are small `trans1x` molecules (`C4H7N1O1`, `C4H8O2`, `C5H8O1`, `C5H7N1O1`, `C5H10O1`).
- The composition string has one form for every row of `train_4M` and `val`: element symbol and count, elements in alphabetical order,
  no element twice. Its atom count equals the atom count of the structure for every row (0 mismatches in 6,748,775 rows). So two
  different strings never describe the same set of atoms.
- **Decision: composition.** The largest composition holds 0.248 percent of the rows. With the border rule of Section 4, the share of a
  set can miss its target by at most the largest group (reasoning), so 0.25 points, inside the 0.5 point tolerance. No special rule for
  large groups is needed.

## 4. Method

Script: `scripts/make_split_manifests.py`. Seed 11790. Same generator as the earlier lists (`numpy.random.default_rng`).

1. **Stratum.** Each composition gets the data source (`data_ids`) of most of its rows. A tie goes to the source that sorts first.
   32,387 compositions (1,418,831 rows) have rows from more than one source; 382,428 rows follow a composition whose main source is
   not their own.
2. **Order.** One generator `default_rng(11790)` shuffles the compositions of each source in turn, in the sorted order of the source
   names.
3. **Border rule.** Inside a source, the compositions are laid out in that order on the cumulative row count. A composition goes to
   train if the middle of its rows lies below 70 percent of the source's rows, to val if below 90 percent, else to test.
4. **Subsets.** `default_rng(seed).choice(rows, n, replace=False)`, sorted: the monitor list from the validation rows (seed 11791), the
   selection list from the validation rows that are not in the monitor list (seed 11792), the test evaluation list from the test rows
   (seed 11793).

**Why the stratum (decision, with its history).** The first version had no stratum (one shuffle of all compositions). Its split with
seed 11790 had a validation set at the top of its own sampling spread: against 20 other seeds of the same rule, train against val ranked
19 or 20 of 20 on all five statistics of Section 5 (for example a `data_ids` share gap of 4.9 points at `trans1x`, null maximum 3.3
points). Stratifying by source lowers the typical gaps (null medians: `data_ids` 1.2 -> 0.6 points, KS of atoms per structure train
against val 0.023 -> 0.019, charge 0.7 -> 0.55 points). We then applied the stratified rule once, with the same seed 11790. We tried no
other seed for the final split. This change was made after we saw the statistics of the first split. It used no model result.

**Reproducibility.** The laptop and a Bridges-2 compute node wrote the same lists byte for byte from the same table (Section 5, job
50164159). The table is on Ocean (Section 9). numpy 2.4.6 on both.

## 5. Checks

All values in this section are measured (`make_split_manifests.py check`, file `results/data_split/checks.json`).

**List checks: 18 of 18 pass.** The sha256 of each list equals `split_samples.json`. The lists equal the assignment recomputed from the
table. Every row is in exactly one main list. No composition and no calculation folder is in two sets. Each share is within 0.5 points
of its target (gaps +0.008, -0.023 and +0.015 points). Each subset has the right size, lies inside its set, and the selection list shares
no row with the monitor list. The sampler check of Section 8 also passes.

**Distributions.** For each pair of sets: the KS statistic of atoms per structure, and the largest gap in the share of structures by data
source, by element present, by charge and by spin multiplicity. "Null" is the same statistic for 20 other seeds (0 to 19) of the same
rule. It shows the spread that comes from the group sampling alone. Rank: how many of the 20 null values are below this split's value.

| Pair | KS atoms (null median / max) | Data source gap, points | Element gap, points | Charge gap, points | Spin gap, points |
|---|---|---|---|---|---|
| train against val | 0.011 (0.019 / 0.032), rank 2 | 0.32 `ani2x` (0.59 / 0.93), rank 2 | 1.5 N (1.3 / 2.9), rank 12 | 0.59 at 0 (0.55 / 1.1), rank 11 | 0.24 at 1 (0.12 / 0.34), rank 17 |
| train against test | 0.039 (0.026 / 0.039), rank 20 | 0.16 `reactivity` (0.54 / 1.1), rank 0 | 2.2 N (2.5 / 4.2), rank 7 | 0.38 at 0 (0.80 / 1.7), rank 0 | 0.08 at 2 (0.13 / 0.44), rank 4 |
| val against test | 0.030 (0.032 / 0.047), rank 9 | 0.37 `ani2x` (0.60 / 1.7), rank 2 | 2.1 O (2.6 / 5.2), rank 7 | 0.46 at -2 (0.84 / 1.8), rank 2 | 0.32 at 1 (0.21 / 0.55), rank 16 |

| | train | val | test |
|---|---:|---:|---:|
| Atoms per structure, mean | 54.98 | 54.83 | 54.01 |
| Atoms per structure, 5 / 50 / 95 percent | 9 / 42 / 140 | 9 / 42 / 140 | 9 / 40 / 140 |
| Neutral, percent | 56.27 | 55.68 | 55.89 |
| Singlet, percent | 82.95 | 82.72 | 83.04 |
| Elements present | 83 | 83 | 83 |

- 14 of the 15 statistics lie inside the range of the 20 null splits. One is at the edge: the KS statistic of atoms per structure
  between train and test is 0.0395, just above the largest null value 0.0391. The test set has slightly smaller structures (median 40
  against 42 atoms). We read this as group sampling noise from the large compositions of small molecules, not as a fault of the rule
  (reasoning).
- The gaps are larger than for a split by row, because one composition can hold up to 9,907 rows (reasoning; a row split was not
  measured). This is the price of a held-out composition test. The null spread shows its size.

**Relation to the official `val` split and to the earlier lists.**

| List | Rows in train / val / test of the new split |
|---|---|
| `pilot_train_200k.txt` (train_4M) | 140,164 / 39,596 / 20,240 |
| `long_train_1200k.txt` (train_4M) | 840,211 / 239,634 / 120,155 |
| `pilot_val_5k.txt` (official val) | 0 / 0 / 0 rows whose composition is in the new sets |
| `select_val_5k.txt` (official val) | 0 / 0 / 0 rows whose composition is in the new sets |

- The official `val` split shares no composition with `train_4M` (0 of 603,008), so it shares none with the new sets either. 14
  calculation folder names occur in both `train_4M` and `val`. In each of them the two sides hold different compositions (for example
  `omol/electrolytes/solvated_090624/pyrrolidinium_mol67_solv7_1_1`: `Br8C35F4H65N2O6` in `train_4M`, `C40H84N1` in `val`). So the name is
  reused for a different calculation; it is not a shared molecule.
- **Important.** The pilots and Wave 1 trained on `pilot_train_200k.txt`. It holds 20,240 rows of the new test set and 39,596 rows of
  the new validation set. Do not evaluate pilot or Wave 1 checkpoints on the new lists. The new lists are valid only for models
  trained on `split_train_70.txt`.

**Against the real datasets (job 50164159, `check_pilot_data.py verify` with the overrides of Section 6).** 10 of 10 checks pass. For
`split_train_70.txt` and `split_val_monitor_5k.txt`: the list hash and size equal `split_samples.json`, fairchem builds exactly the
listed rows, the rows come from the `train_4M` folder, and the split has 3,986,754 rows. The same lists with the validation rows read
from `val` fail three checks (rows, folder, split size), as they must. Logs: `results/data_split/verify_samples.log` and
`verify_wrong_folder.log`.

**Reproduction.** On a Bridges-2 compute node, `write` gave all six lists and `split_samples.json` byte for byte (sha256 equal), from
the same table. `check` on the node gave the same values as on the laptop (`results/data_split/checks.json` is the node's copy).

## 6. How a long run uses the lists

`slurm/long_run.sh` reads three variables. `slurm/submit_chain.sh` passes them to every segment when they are set.

| Variable | Default (the final runs) | Meaning |
|---|---|---|
| `LONG_TRAIN_LIST` | `manifests/split_train_70.txt` | training rows |
| `LONG_VAL_LIST` | `manifests/split_val_monitor_5k.txt` | validation rows, evaluated every `--eval-every` steps |
| `LONG_VAL_SPLIT` | `train_4M` | the folder the validation rows are read from |

`slurm/pilot_data_overrides.sh` turns them into Hydra overrides. The validation dataset gets
`val_dataset.dataset_configs.omol.splits.val.src=$OMOL_DATA_ROOT/train_4M`, the index file `train_4M_index.npz` and
`subset_to` with the monitor list. The training dataset is unchanged except for its list. `pilot.sh` does not set these variables, so the
pilots and Wave 1 read `pilot_train_200k.txt` and `pilot_val_5k.txt` from `val` as before (`tests/test_long_run.py` pins the exact
overrides).

Before training, each segment runs `check_pilot_data.py verify` with `split_samples.json`, `long_samples.json` and `pilot_samples.json`.
It checks the hash and size of both lists, that fairchem builds exactly the listed rows, that the rows come from the folder the manifest
names (`train_4M` for all `split_*` lists), and the size of that split. `settings.txt` records both list names, their hashes and
`from train_4M`. A run folder that started with other lists refuses a segment with the new ones ("settings differ").

The test lists are never read by a training job.

The held chains of the 40 final runs read `shared/repo` when each segment starts. After a sync of this code they train on these
lists ([launch-checklist.md](launch-checklist.md), Section 3).

## 7. Checkpoint choice and the final evaluation

Rules (they replace the selection and report sets of `docs/final-training-plan.md` for the final runs):

1. Select on validation, report on test. Never the reverse.
2. `scripts/checkpoint_tools.py select` uses the selection list (default `split_val_select_10k.txt`) and names the test evaluation list (default
   `split_test_eval_10k.txt`) as the report list. Both index `train_4M`, so `--data` is the `train_4M` folder.
3. The test lists are used once, after the rule, K, the ensemble methods and the set of runs are fixed. No choice is made after a test
   number is seen.
4. The monitor list gives the training curves and the training-time validation values that `collect_final_metrics.py` reads. It is not
   the selection list.
5. `pilot_val_5k.txt` (official `val`, compositions held out from all of `train_4M`) is an extra out-of-composition check:
   `checkpoint_tools.py report --rows manifests/pilot_val_5k.txt --data <val folder> --eval-split official_val`. It is reported as such
   and is never used for a choice. It also links the final runs to the pilot and Wave 1 numbers, which used the same 5,000 rows.

### Time and cost of the evaluation (job 50163685)

Measured: `scripts/time_checkpoint_eval.py` runs the evaluator of `checkpoint_tools.py` (`training.checkpoints.evaluate`, one
structure at a time, energy and forces) on 300 rows drawn uniformly from `split_test_10.txt` (mean 52.6 atoms), after 10 warm-up rows,
with the final checkpoints of the seed-0 Dense and Local-6 pilots. One V100-32 (GPU-small). Files: `results/data_split/timing_*.json`.

| Model | float32, s per structure | float64 (the tool's default), s per structure |
|---|---:|---:|
| Dense | 0.0650 | 0.0743 |
| Local-6 | 0.1394 | 0.1978 |

- Loading a checkpoint took 7.1 s for the first model of the job (CUDA start) and under 1 s after. It is small against the evaluation.
- float32 and float64 gave the same MAEs within 4e-8 eV/atom (energy) and 8e-7 eV/A (force). In all 15 pilot runs the force MAE fell
  at every last-half checkpoint, by at least 3e-4 eV/A (median 1.8e-3; `results/pilot_validation_curves.csv`). So float32 is very unlikely
  to change the ranking that the rule makes (reasoning from these two measurements; the pilot values are rounded to 1e-4). **Proposal: `--dtype float32` for select and report** (the
  tool requires the same precision for both).
- These accuracy values are not results: the pilots trained on rows of this test set (Section 5).

Cost for the 40 final runs (8 variants x 5 seeds). Reasoning from the measured times, with two assumptions: Global and Frozen take 1.3 x
the Dense time (`docs/inference-timing.md`: Global is 1.2 to 1.3 times slower than Dense), Local-12 takes 1.3 x the Local-6 time (its
training steps are 1.2 to 1.3 times slower); Scaled Dense takes the Dense time; Element and Local+Global take the Local-6 time. Then one
evaluation of one structure by one checkpoint of every run adds up to 4.49 s (float32) or 5.96 s (float64).

| List | Rows | Final checkpoint of all 40 runs, float32 | float64 |
|---|---:|---:|---:|
| `split_val_monitor_5k.txt` | 5,000 | 6.2 h | 8.3 h |
| `split_val_select_10k.txt` | 10,000 | 12.5 h | 16.6 h |
| `split_test_eval_10k.txt` | 10,000 | 12.5 h | 16.6 h |
| (a list of 20,000 rows, for comparison) | 20,000 | 25.0 h | 33.1 h |
| `split_test_10.txt` (the full test set) | 399,276 | 498 h | 661 h |
| `split_val_20.txt` (the full validation set) | 796,421 | 994 h | 1,319 h |

The full procedure evaluates more than the final checkpoint:

| Step | Evaluations per row and run | 40 runs, 10,000 rows, float32 | float64 | 40 runs, 20,000 rows, float32 |
|---|---:|---:|---:|---:|
| `select`: the 8 last-half checkpoints (steps 80,000 to 150,000 at a spacing of 10,000) | 8 | 100 h | 133 h | 200 h |
| `report`: final, best, weight average, 5-member prediction average | 8 | 100 h | 133 h | 200 h |
| extra `report` on `pilot_val_5k.txt` (official `val`, mean 102 atoms) | 8 | about 100 h for 5,000 rows, if time grows with the atom count (not measured) | | |

- On a V100 (1 SU per GPU-hour) the hours are SU. With 10,000 + 10,000 rows in float32 the selection and the report cost about 200 SU for
  all 40 runs; with 20,000 + 20,000 rows, about 400 SU. One Local-6 run needs about 3.1 h for `select` and 3.1 h for `report` (float32),
  so each run fits one job.
- **Why 10,000 rows (decision).** The cost doubles with the size. The pilot curves fall at every last-half checkpoint by much more than
  the float32 noise, and the rule compares checkpoints of one run on the same rows, so the selection needs no more rows than the test
  (reasoning). The uncertainty of a mean error over n structures falls as 1 / sqrt(n): 10,000 rows give about 1.4 times the precision of
  the old 5,000 rows (reasoning; the spread of the error per structure was not measured). The full test list stays available for a later,
  larger evaluation of chosen runs.

## 8. How much of the training set a run reads

Measured with fairchem's own batch builder (`get_batches`) and batch order (`MaxAtomDistributedBatchSampler`, one GPU, 450 atoms per
batch, the run seed), on the atom counts of each list (`check`, section `sampler`). The same code gives the measured pilot epochs
exactly: 27,375, 27,402 and 27,401 batches for seeds 0, 1 and 2 of `pilot_train_200k.txt` (the epoch-end validation steps in the pilot
logs).

| Training list | Batches per epoch (seeds 0 to 4) | Distinct rows read in 150,000 steps | Passes |
|---|---|---|---:|
| `split_train_70.txt` (2,791,057 rows) | 381,036 to 381,159 | 1,097,574 to 1,099,436 (39.3 to 39.4 percent) | 0.39 |
| `long_train_1200k.txt` (1,200,000 rows) | 163,705 to 163,766 | 1,098,952 to 1,099,405 | 0.92 |

- **Doubt.** A 150,000-step run reads about 1.1 million rows of the 2.79 million training rows, each once. About 61 percent of the
  training rows are never read. The run reads the same number of distinct rows as with the 1,200,000-row list; the difference is that it
  never sees a row twice. A different seed reads a different 39 percent (reasoning: the batch order is a seeded permutation).
- So the larger training set does not give the models more data in 150,000 steps. It removes repeats and it frees the 30 percent of
  `train_4M` that the validation and test sets need (reasoning).

## 9. Files and jobs

| Item | Where |
|---|---|
| Lists and hashes | `manifests/split_*.txt`, `manifests/split_samples.json` |
| Key table of `train_4M` (composition, folder, charge, spin, elements, atoms, data source for every row) | `/ocean/projects/che260063p/shared/data/omol25/split/keys_train_4M.npz`, 91,561,294 bytes, sha256 `155f2364eaf1e9008c37d203e5bb1dda837f4609c2e5a5b16e62f3fada17180d` |
| Key table of `val` | `.../split/keys_val.npz`, 35,013,509 bytes, sha256 `9691ab0fa30f9151a6568e6f5f0f49f3b0ab978866740f0c3c8352e5c5433ba8` |
| Row -> composition -> set table (`split`, `group`, `group_keys`, `group_stratum`, `stratum_keys`) | `/ocean/projects/che260063p/shared/data/omol25/split/split_assignment.npz`, 20,932,768 bytes, sha256 `f856e67853931573d70025938450f0939581a379c565c7dda068487228af9ab8` |
| Check report | `results/data_split/checks.json` |
| Timing results | `results/data_split/timing_dense_s0_49041937.json`, `timing_local6_s0_49041936.json`; on Ocean `shared/runs/time_eval_50163685/` |

The two key tables and the row table are not in git (sizes). Ocean is not backed up: copy them with the Ocean backup (plan item F2).

| Job | What | Partition | Elapsed | SU |
|---|---|---|---|---|
| 50162814 | read the keys of all 3,986,754 `train_4M` rows (476 s) and all 2,762,021 `val` rows (229 s), 16 workers | RM-shared, 16 cores | 11 min 59 s | 3.196 (job_info) |
| 50163685 | evaluator timing, Dense and Local-6, float32 and float64, 300 rows each | GPU-small, 1 V100-32 | 7 min 20 s | 0.122 (job_info) |
| 50164159 | redraw on the node, fairchem verify (and the check that must fail), `check` with the val table | RM-shared, 8 cores | 7 min 18 s | about 1.0 (8 cores x elapsed); peak memory 11.0 GB |

## 10. Not checked

- No training run has built the 2,791,057-row training set. `verify` builds the datasets, and Section 8 computes the batches with
  fairchem's code, but the time and memory of the sampler at the start of a real run are not measured.
- Stereoisomers and conformers of one formula share a group by design. Molecules that differ only by a small change (for example one
  substituent) are different compositions and can be in different sets. The split does not hold out chemical families.
- The force and energy labels of the new rows were not checked beyond what the earlier lists showed.
- The 382,428 rows whose own data source differs from their composition's main source follow the composition. Their share per set was
  not checked separately.
