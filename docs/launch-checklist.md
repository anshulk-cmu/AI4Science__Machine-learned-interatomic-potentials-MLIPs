# Launch checklist for the final training (Wave 1 and the long runs)

Writing style: simple English. Short sentences. One idea in each sentence.

This file tells you how to check and start the final training. It does not choose the number of steps or the seeds of the long runs. You choose them. Put your values in the table of Section 2 first.

Plan: `docs/final-training-plan.md`. Rules for resumed runs: `docs/resume-test.md`, Section 6.

## 1. What is ready

| Part | Files | Checked by |
|---|---|---|
| 8 variants | `configs/variant/*.yaml` | `tests/test_training_config.py` (parameter counts of all 8), `tests/test_size_consistency_predictions.py` |
| Wave 1 job files | `slurm/wave1_<variant>.job` (8 files) | `tests/test_long_run.py`, `sbatch --test-only` |
| Long-run segment | `slurm/long_run.sh`, `slurm/long_run.job` | `tests/test_long_run.py` (stand-ins for python, nvidia-smi, sbatch) |
| Long-run chain | `slurm/submit_chain.sh` | `tests/test_long_run.py`, `--dry-run` on Bridges-2 |
| Results | `scripts/collect_final_metrics.py`, `scripts/checkpoint_tools.py`, `scripts/pilot_statistics.py`, `scripts/make_accuracy_figure.py` | their tests; they read the Wave 1 folders and the long-run folders |

## 2. Inputs you must choose

The long-run scripts have no default for these values. They stop with exit 2 if one is missing.

| Input | Option of `submit_chain.sh` | Your value | Rule |
|---|---|---|---|
| Total steps of one long run | `--steps` | `<N>` | The same for all variants of one table (plan, risk 6). |
| Seeds | `--seed` (one per chain) | `<seeds>` | Decision D9. Use 0 to k-1 for k seeds (the collector needs them). |
| Hours per segment | `--hours` | `<H>` | 1 to 48. One segment is one Slurm job. |
| Number of segments | `--segments` | `<K>` | See the formula below. |
| Validation spacing | `--eval-every` | `<E>` | At most `<N>`. |
| Checkpoint spacing | `--ckpt-every` | `<C>` | At most `<N>`. One checkpoint must fit in one segment. |

How to choose K and C (reasoning, from the measured seconds per step in Section 5):

- Steps in one segment, about: `(H x 3600 - 1800 - 900) / s`. Here `s` is the mean seconds per step, 1800 s is the stop margin and 900 s is the start (Python import, data check, data set build).
- A segment keeps only the steps up to its last checkpoint. So round the steps in one segment down to a multiple of C.
- K must be at least `ceil(N / that number)`. Add one spare segment. A spare segment that finds the run done ends in a few seconds.
- A segment repeats the steps after its last checkpoint. With `C` steps between checkpoints, a segment loses up to `C x s` seconds.
- A segment that writes no new checkpoint stops the chain (FAILED marker). So `C x s` must be well below `H x 3600 - 2700`.

## 3. Before any submission

Run these from the laptop, in Git Bash, from any folder. `B2` is the helper of the `psc-bridges2` skill.

```bash
B2="python C:/Users/worka/.claude/skills/psc-bridges2/scripts/b2.py"
export MSYS_NO_PATHCONV=1
$B2 status                       # GPU balance of che260063p, Ocean quota, your queue
```

1. Merge this work into `main`, then sync the copy that the jobs read. `syncrepo` takes the parent folder:
   ```bash
   $B2 syncrepo D:/CMU-Projects-Class/AI4Science-MLIPs /ocean/projects/che260063p/shared/repo
   ```
   It must print `0 differ or are missing`.
2. Check the GPU balance against Section 5. Other work on `che260063p` uses the same SU.
3. Do not sync while jobs run. A sync replaces every tracked file, and the next segment of a chain reads the new code.
4. Held chains read the code of `shared/repo` when they start, not when they were submitted. After a sync of the 70/20/10 code
   ([data-split.md](data-split.md)), a held chain trains on `split_train_70.txt` and validates on `split_val_monitor_5k.txt`. A run
   folder whose `settings.txt` was written with other lists refuses the new segment ("settings differ"); check with
   `ls /ocean/projects/che260063p/shared/runs/long_*/settings.txt` before you release held jobs.

## 4. Wave 1 (50,000 steps, the fixed pilot lists)

Wave 1 has fixed settings: 50,000 steps, a checkpoint and a validation every 5,000 steps, the lists `pilot_train_200k.txt` and `pilot_val_5k.txt`. The seed is the array task id.

| Job file | Variant | GPU | Seeds | Tasks |
|---|---|---|---|---|
| `wave1_dense.job` | Dense | v100-32 | 3, 4 | 2 |
| `wave1_global.job` | Global | v100-32 | 3, 4 | 2 |
| `wave1_global_frozen.job` | Frozen | v100-32 | 3, 4 | 2 |
| `wave1_dense_scaled.job` | Scaled Dense | v100-32 | 3, 4 | 2 |
| `wave1_local6.job` | Local-6 | h100-80 | 3, 4 | 2 |
| `wave1_element.job` | Element | h100-80 | 0 to 4 | 5 |
| `wave1_local12.job` | Local-12 | h100-80 | 0 to 4 | 5 |
| `wave1_local_global.job` | Local+Global | h100-80 | 0 to 4 | 5 |

Dry run (free, submits nothing), then submit after the decision:

```bash
for v in dense global global_frozen dense_scaled local6 element local12 local_global; do
    $B2 submit D:/CMU-Projects-Class/AI4Science-MLIPs/slurm/wave1_$v.job /ocean/projects/che260063p/shared/slurm
done
# after the go-ahead, the same with --yes, one file at a time
$B2 submit D:/CMU-Projects-Class/AI4Science-MLIPs/slurm/wave1_local12.job /ocean/projects/che260063p/shared/slurm --yes
```

Where the output lands:

- `/ocean/projects/che260063p/shared/runs/final_<variant>_s<seed>_<jobid>/`: `summary.txt`, `settings.txt`, `logs/`, `train/` (checkpoints, offline W&B).
- `/ocean/projects/che260063p/shared/logs/slurm-<arrayjob>_<task>.out`.

Check first: `settings.txt` says `steps=50000` and the right seed. `summary.txt` has `verify_samples` all PASS.

## 5. Long runs (Wave 2)

On a Bridges-2 login node. Fill in your values from Section 2.

```bash
ssh bridges2
C=/ocean/projects/che260063p/shared/repo/AI4Science-MLIPs/slurm/submit_chain.sh
bash $C local12 --steps <N> --seed <S> --segments <K> --hours <H> --eval-every <E> --ckpt-every <C> --dry-run
bash $C local12 --steps <N> --seed <S> --segments <K> --hours <H> --eval-every <E> --ckpt-every <C>
```

- Data ([data-split.md](data-split.md)): training on `split_train_70.txt`, validation every `--eval-every` steps on
  `split_val_monitor_5k.txt`, read from the `train_4M` folder. These are the defaults of `long_run.sh`. To use other lists, set
  `LONG_TRAIN_LIST`, `LONG_VAL_LIST` and `LONG_VAL_SPLIT` before `submit_chain.sh`; it passes them to every segment and prints them.
  The test lists are never read by a training job.
- `--dry-run` prints the folder, the GPU, the SU limit, the data lists and every `sbatch` command. It runs `sbatch --test-only` for segment 1 only.
- Without `--dry-run` it submits K jobs. Segment k+1 waits for segment k with `afterany`.
- It refuses a second chain for the same run while jobs of the first are queued. It refuses a run with a FAILED marker.
- The GPU follows the variant: v100-32 for Dense, Scaled Dense, Global, Frozen; h100-80 for Local-6, Element, Local+Global, Local-12.
- Use the seeds 0 to k-1 for k long runs of a variant. `collect_final_metrics.py --expected-seeds k` accepts only these seeds.
- When `--eval-every` divides `--steps`, the last step is validated twice: once by the spacing and once at the end of the epoch that the step limit cuts short (`docs/TRAINING.md`). In the pilots, one validation of the 5,000 report rows took at most about 6 min (Dense), 6.6 min (Global) and 9.8 min (Local-6 on an L40S). Those rows had 102 atoms on average; the monitor rows have 55, so one validation should take less (reasoning, not measured). Plan for this time in the last segment.

To stop a chain: cancelling one segment does not stop the chain. The next segment waits with `afterany`, so it starts as soon as the cancelled one ends. Cancel every job of the run by its name, which is the run folder name (`chains.txt` lists the ids):

```bash
scancel -n long_<variant>_s<S>_<N>
```

A segment cancelled while it trains ends without its END line. Its log and W&B file stay listed in `segments.txt`, and a new chain resumes from the last complete checkpoint. A training process killed long before the stop by time (exit 137, for example out of memory) writes FAILED.

What each segment does (`slurm/long_run.sh`):

1. Exit at once if `DONE` or `FAILED` is in the run folder.
2. Check the settings against `settings.txt` of the first segment, the GPU size, and the data lists (`verify` with both `--samples` files).
3. Train with the pinned run id. Stop by time `LONG_STOP_MARGIN_SEC` (1,800 s) before the limit.
4. When `checkpoints/final/.metadata` exists: run the size check at 8, 10, 15, 25 and 100 A, then write `DONE`.

Where the output lands: `/ocean/projects/che260063p/shared/runs/long_<variant>_s<seed>_<N>/`:

| File | What it holds |
|---|---|
| `settings.txt` | variant, steps, seed, data lists and their hashes, spacing |
| `summary.txt` | one block per segment: start, GPU, code hash, data check, training end, size check, RESULT |
| `segments.txt` | the training log and the W&B file of each segment, in order |
| `chains.txt` | the job ids of each submitted chain |
| `logs/train_<jobid>.log` | the full training output of one segment |
| `train/long_<variant>_s<seed>_<N>/checkpoints/` | `step_<k>` folders and `final` |
| `DONE`, `FAILED` | markers |

To restart after a fix: delete `FAILED`, then submit a new chain with the same values. It resumes from the last complete checkpoint.

### Cost per run

Measured on the pilots (50,000 steps, a validation every 5,000 steps): mean seconds per step from fairchem's own step timer, steps 20 to 49,999, validation pauses included.

| Variant | GPU | Mean s per step (3 seeds) | Measured or estimate |
|---|---|---|---|
| Dense | v100-32 | 0.630 to 0.704 | measured |
| Scaled Dense | v100-32 | 0.494 to 0.547 | measured |
| Global | v100-32 | 0.724 to 0.769 | measured |
| Frozen | v100-32 | 0.777 to 0.791 | measured |
| Local-6 | h100-80 | 0.818 to 0.942 | measured |
| Element | h100-80 | 0.629 (median 0.577) | 200-step smoke run, steps 20 to 199, no validation pause |
| Local+Global | h100-80 | 0.655 (median 0.611) | 200-step smoke run, as above |
| Local-12 | h100-80 | 1.059 (median 0.921) | 200-step smoke run, as above |

The smoke numbers come from the first 200 batches of one seed. Use them as estimates. For comparison, the Local-6 pilots had a
median of 0.70 to 0.80 s per step over 50,000 steps. So Local-12 is about 1.2 to 1.3 times Local-6, and Element and Local+Global
are not slower than Local-6 (reasoning from the medians). The pilot means include the validation pauses; the smoke means do not.

Hours of one run, about `N x s / 3600`. SU: hours x 1 (v100-32) or hours x 2 (h100-80). This is an estimate for a new N; validation every E steps changes it a little.

Disk (measured on the pilots): one checkpoint is 140 MB (Dense, Scaled Dense) or 865 MB (all MoE variants). A run keeps `N / C + 2` checkpoints.

## 6. Smoke runs (test only)

Two test jobs check the launch path on the cluster. They read the copy in `shared/repo_integration`, not the copy the real jobs read.

- `slurm/smoke_new_variants.job`: 200 steps each of Local-12, Element and Local+Global through `pilot.sh`, on an H100.
- `slurm/smoke_long_chain.job`: one Dense long run of 180 steps in three segments, on a V100.

Results (measured on 2026-11-01):

| Job | What | Time | Result |
|---|---|---|---|
| 50116336 | chain smoke, GPU-small, 1 V100 | 20 min 16 s, 0.338 SU | segment A stopped by time at step 60 (exit 124); B resumed at step 60, finished 180 steps, size check, DONE; C exited at once. `collect_final_metrics.py` read both W&B files and both logs; `discover` found steps 0 to 180. 186 optimizer steps ran in total |
| 50116348 | new variants, GPU-shared, 1 H100 | 39 min 50 s | all three: data check PASS, 200 steps, no OOM, one validation, final checkpoint, all stages passed |

| Variant | Peak GPU memory (nvidia-smi, 200 steps and one validation) | Delta at 8 / 10 / 15 / 25 / 100 A (meV) | Prediction |
|---|---|---|---|
| Element | 25,695 MiB | 3.7e-06 at every gap | zero: yes |
| Local+Global | 37,637 MiB | -50.40 at every gap | not zero and flat: yes |
| Local-12 | 37,801 MiB | 0.379 / 0.0343 / 2.8e-06 / 2.8e-06 / 2.8e-06 | not zero below 12 A, zero from 15 A: yes |

A peak over 200 steps is a lower bound. Local-6 peaked at 26 to 29 GiB in short tests and at 43 to 45 GiB over 50,000 steps.
So keep the H100 for Local-12, Element and Local+Global.

## 7. After the runs

```bash
python scripts/collect_final_metrics.py --run-manifest runs.json --merge results/pilot_final_metrics.csv --output results/final_metrics.csv
python scripts/pilot_statistics.py results/final_metrics.csv --require-final
python scripts/collect_final_metrics.py --run-manifest long_runs.json --steps <N> --expected-seeds <number of seeds> --output results/long_final_metrics.csv
python scripts/checkpoint_tools.py select --run-dir <long run folder> --variant <variant> --seed <S> --total-steps <N> --data <OMol25 train_4M folder> --device cuda --output <new folder>
```

- The long-run values that `collect_final_metrics.py` collects are the training-time validation on `split_val_monitor_5k.txt`. Its summary prints the lists from `settings.txt`. They are not test results and not comparable with the pilot values on `pilot_val_5k.txt`.
- `checkpoint_tools.py` selects on the validation list and reports on the test list, both rows of `train_4M` ([checkpoint-tools.md](checkpoint-tools.md)). Run the test report once, after every choice is fixed. The extra report on `pilot_val_5k.txt` uses `--data <val folder> --rows manifests/pilot_val_5k.txt --eval-split official_val`.
- `collect_final_metrics.py` reads `segments.txt` of a long run by itself. If the W&B file of an interrupted segment is damaged, add `--source log`.
- `--expected-seeds k` requires exactly the seeds 0 to k-1 of every variant in the table.
- Runs stopped at the training stop have no final checkpoint and no final validation. Both tools refuse them by default. Add `--allow-unfinished` to `collect_final_metrics.py` and to `checkpoint_tools.py select`: they then use the latest complete validation or the step checkpoints that exist, and every output row and the summary get `partial` and `reached_step`. Report these runs as shorter training, in a separate table. A table made this way cannot be merged later.
- W&B keeps a link `latest-run` next to each `offline-run-*` folder. A copy made with `scp -r` follows symbolic links (OpenSSH scp manual), and a tar archive unpacked on Windows can also turn the link into a real folder. The copy then holds the same `.wandb` file twice, and the collector stops with "multiple W&B histories" for a Wave 1 or pilot folder. Long runs are not affected, because `segments.txt` names the files. To avoid it, copy with `tar` on Linux, which keeps links, or delete the copied `latest-run` folders. To fix it without a new copy, add `--source log`, or name the file in a run manifest (`"wandb": ["train/.../run-<id>.wandb"]`).
- Run the checkpoint tools and the size checks in Slurm jobs, not on a login node.
