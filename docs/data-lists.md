# Data lists for the long runs and the checkpoint selection (plan item B4)

Writing style: simple English. Short sentences. One idea in each sentence.

## 1. The lists

| File | Split | Rows | sha256 | Seed | Relation |
|---|---|---:|---|---:|---|
| `manifests/long_train_1200k.txt` | `train_4M` (3,986,754 rows) | 1,200,000 | `c5a1690331676a870433b2bd851c248a9014ed54c3380ab9896473b8210f63d6` | 11787 | contains all 200,000 rows of `pilot_train_200k.txt` |
| `manifests/select_val_5k.txt` | `val` (2,762,021 rows) | 5,000 | `6174e428ac33ca8c134c4fac61797387e71f5ff1ffbb84d93d469c4b1fca3209` | 11788 | shares no row with `pilot_val_5k.txt` (the report set) |

- One row number per line, sorted, LF line ends: the same format as the pilot lists.
- `manifests/long_samples.json` holds the hashes, counts, seeds, first and last rows. It has the same layout as `pilot_samples.json`. `pilot_samples.json` is not changed: `tests/test_training_config.py` requires it to hold exactly the two pilot lists.
- The lists are the source of truth. The script only documents how they were made.

## 2. How they were made

Script: `scripts/make_long_manifests.py write`. It uses the method of `scripts/make_pilot_manifests.py`: `numpy.random.default_rng(seed).choice(..., replace=False)`, then sort. No stratification, as in the pilots.

- **Long list.** The 200,000 pilot rows, plus 1,000,000 rows drawn uniformly without replacement from the 3,786,754 `train_4M` rows that are not in the pilot list.
- **Why the union is still a uniform sample.** Let N be the split size, P the pilot list (a uniform 200,000-row draw) and Q the new draw (uniform over the rest). For any fixed set S of 1,200,000 rows, P(P u Q = S) = [C(1,200,000, 200,000) / C(N, 200,000)] x [1 / C(N - 200,000, 1,000,000)]. This does not depend on S. So the long list has the same distribution as a direct uniform draw of 1,200,000 rows, and it still contains the pilot rows. (Reasoning; the measured distributions in Section 4 agree.)
- **Selection list.** 5,000 rows drawn uniformly without replacement from the 2,757,021 `val` rows that are not in the report set. This is the pilot method on the remaining rows.
- **Check of the method.** The same code regenerates both pilot lists byte for byte (sha256 equal), on the laptop with numpy 2.4.6. The cluster has numpy 2.4.6 too. A redraw on a compute node gave the same two hashes (job 50003944).

## 3. How a run uses them

`slurm/pilot_data_overrides.sh` already reads `PILOT_TRAIN_LIST` and `PILOT_VAL_LIST`. The index files in `$OMOL_DATA_ROOT/pilot/` cover every row of each split, so no new index file and no new run of `prep_pilot_data.job` is needed.

```bash
PILOT_TRAIN_LIST=$REPO/manifests/long_train_1200k.txt source "$REPO/slurm/pilot_data_overrides.sh"
$PY scripts/check_pilot_data.py verify --samples manifests/long_samples.json \
    --train-rows "$TRAIN_LIST" --val-rows "$VAL_LIST" "${DATA_OVERRIDES[@]}"
```

- **Important for the long-run script (B2).** `pilot.sh` calls `verify` without `--samples`, so it looks the list up in `pilot_samples.json`. With the long list it stops with an error and does not train. The long-run script must pass `--samples` with a file that holds both lists of the run. `long_samples.json` holds only the two new lists. A run that trains on the long list and validates on `pilot_val_5k.txt` needs both entries in one file. Simple ways: add `--val-samples`, or a third JSON file with both entries. Not done here (part of B2).
- The selection list is for the checkpoint selection scripts (plan items C1 to C4). It is not the validation set of the training job. The report set stays `pilot_val_5k.txt`.

## 4. Checks

All values in this section are measured. Files: `results/data_lists/`.

**On the list files** (`make_long_manifests.py write`, and again in `tests/test_long_manifests.py` from the written files): counts 1,200,000 and 5,000; no duplicates; all rows inside the split; every pilot training row is in the long list; no selection row is in the report set. All pass.

**Against the real datasets** (job 50003944, `check_pilot_data.py verify`): 8 of 8 checks pass. Fairchem builds exactly the listed rows. The split sizes equal the manifest.

**Distributions.** Atoms per structure and `data_ids` come from the index files for every row. The other rows of the table come from `check_pilot_data.py stats`, over every row of each list.

| | pilot train 200k | long 1,200k | report val 5k | selection val 5k |
|---|---:|---:|---:|---:|
| Atoms per structure, mean | 55.10 | 54.92 | 102.09 | 102.79 |
| Atoms per structure, 5 / 50 / 95 percent | 9 / 43 / 141 | 9 / 42 / 141 | 17 / 95 / 223 | 18 / 96 / 222 |
| Neighbours within 6 A per atom, mean (max) | 32.6 (162) | 32.6 (183) | 35.9 (164) | 36.1 (154) |
| Force norm RMS per atom, eV/A | 2.122 | 2.123 | 1.539 | 1.537 |
| Energy residual per atom after the references, mean / RMS, eV | 0.136 / 0.445 | 0.136 / 0.442 | 0.046 / 0.251 | 0.040 / 0.276 |
| Elements present | 83 | 83 | 82 | 78 |
| Neutral structures, percent | 56.1 | 56.2 | 42.9 | 41.9 |
| Singlets, percent | 82.9 | 82.8 | 90.7 | 90.7 |

| Comparison | KS statistic, atoms per structure | Largest `data_ids` share gap | Largest element share gap | Largest charge share gap | Largest spin share gap |
|---|---:|---:|---:|---:|---:|
| long against pilot train | 0.0024 | 0.16 points | 0.09 points (S) | 0.15 points | 0.05 points |
| selection against report | 0.0118 | 0.78 points | 1.26 points (F) | 1.04 points | 0.42 points |

- No large difference. The long list matches the pilot list closely, as expected from 1,000,000 new rows.
- Selection against report: the gaps are of the size expected from two samples of 5,000 rows. For a share near 0.1, the standard error of the difference is sqrt(2 x 0.1 x 0.9 / 5,000) = 0.6 points, so 1.26 points is about 2 standard errors. That is a normal value for the largest of about 100 compared shares (reasoning). The KS statistic 0.0118 is below the 5 percent critical value 1.36 x sqrt(2 / 5,000) = 0.027 (reasoning). The four elements present only in the report set are the noble gases He, Ne, Kr and Xe, in 1 to 3 structures each.
- The largest neighbour count rises from 162 to 183 with more rows. The graph keeps up to 300 neighbours, so no neighbour is cut.

**Shared compositions and calculations.** Composition and source path come from the row data of the ASE databases.

| | Selection | Report | Shared |
|---|---:|---:|---:|
| Distinct compositions | 4,871 | 4,851 | 146 |
| Rows whose composition is also in the other set | 229 (4.6 percent) | 241 (4.8 percent) | |
| Distinct source systems (calculation folder, step number removed) | 4,993 | 4,994 | 13 |
| Source systems shared with the 1,200,000 long-list rows | 0 | 0 | |
| Source systems shared with the 200,000 pilot training rows | 0 | 0 | |

- The two validation sets share no row. They share some compositions, mostly other conformers of the same molecule. Two uniform draws from `val` always do this. The pilot design did not make validation subsets composition-disjoint. Its statistics note (`notes/pilotDecisions/A_pilotDesignStatistics.md`, Section 3.4) only asks for a bootstrap that resamples compositions, because the rows of one composition are not independent.
- No training row comes from a calculation that is in either validation set. This agrees with the OMol25 split, which holds compositions out of `val` (as the pilot note states).
- **Decision for the team (D5).** Checkpoints chosen on the selection set may favour the 4.6 percent of compositions that the report set also has. The effect on the ensemble result is expected to be small (reasoning, not measured). If the team wants no shared composition, redraw the selection set with one more rule: skip a candidate row whose composition is in the report set. This needs the compositions of the candidates only (about 5,300 reads). Not done here, because the task asked for the pilot method.

## 5. Jobs and cost

| Job | What | Partition | Elapsed | SU |
|---|---|---|---|---|
| 50003944 | redraw, verify, stats of both lists, comparison | RM-shared, 16 cores | 13 min 58 s | about 3.7 (16 cores x elapsed; job_info not posted yet) |
| 50004044 | shared source systems with all 1,200,000 long-list rows | RM-shared, 4 cores | 4 min 17 s | about 0.3 (4 cores x elapsed; job_info not posted yet) |

Files: `slurm/prep_long_data.sh` and `slurm/prep_long_data.job` (run from a copy of the code at `shared/runs/data_lists/code`). Outputs on Ocean: `/ocean/projects/che260063p/shared/runs/prep_long_50003944/` (statistics, the source path of every long-list row in `sids_long_1200k.txt`, logs). Copies in the repository: `results/data_lists/`.

## 6. Not checked

- Composition overlap between training and validation rows. The training rows were checked by source path only. Composition was read for the two validation sets only.
- Force and energy label quality of the 1,000,000 new rows, beyond the summary values in the table.
- The number of passes in 300,000 steps. Reasoning: the 200,000-row list gave 27,375 batches per epoch (pilot log), so 1,200,000 rows of the same distribution give about 6 x 27,375 = 164,250 batches, and 300,000 steps are about 1.8 passes. Not measured: the sampler batch count for the long list is known only when a run builds it.
- A training run on the long list. Only `verify` built the datasets.
