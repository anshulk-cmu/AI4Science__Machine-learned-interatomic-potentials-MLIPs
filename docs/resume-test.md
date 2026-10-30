# Resume test (plan item B1)

Writing style: simple English. Short sentences. One idea in each sentence.

## 1. Question

The long runs (300,000 steps) need several Slurm jobs of at most 48 h. Each job continues from the last checkpoint.
This is valid only if a stopped and resumed run gives the same result as a run that never stops.

Question: does a resumed run equal an uninterrupted run, and does it see the same rows in the same order?

## 2. What the code says (read before any run)

Source: fairchem-core 2.21.0 and torchtnt 0.2.4, the same versions on the laptop and in `shared/envs/mlip`.

| Item | Saved in the checkpoint? | Restored? | Where |
|---|---|---|---|
| Model weights | yes | yes | `MLIPTrainEvalUnit.state_dict`, `load_state_dict` (`units/mlip_unit/mlip_unit.py`) |
| EMA weights (incl. the `n_averaged` counter) | yes | yes | same, `set_model_state_dict(self.ema_model, ...)` |
| AdamW state | yes | yes | `get_state_dict` / `set_state_dict` |
| Learning-rate scheduler (`LambdaLR`, step count) | yes | yes | `scheduler.state_dict()` |
| Step, epoch, step in the epoch | yes | yes | torchtnt `Progress.state_dict()` |
| Sampler position | no, it is computed | yes | `on_train_start` calls `set_epoch_and_start_iteration(epoch, step_in_epoch)` after the load, before the data iterator is made |
| Random number generator state | no | no | no training step uses random numbers: the configs have no dropout |

- Data order. The sampler fixes the batches once with `numpy.random.default_rng(job.seed)`. It orders them in each epoch with `torch.randperm`, seeded by `job.seed + epoch`. The fixed lists give the rows in ascending order (`no_shuffle: true`). So the order depends only on the seed, the list, `max_atoms` and the epoch. A resume skips the batches already used in the epoch.
- Learning rate. `LambdaLR` depends only on its step count, which is restored. The schedule length is `training.steps`, so it is the same in every segment.
- How a resume starts. Each launch makes a new `job.timestamp_id` (a time stamp), and the checkpoints go to `run_dir/<timestamp_id>/checkpoints`. With `job.runner_state_path` empty, `TrainEvalRunner.load_state(None)` loads the newest folder in that directory that has a `.metadata` file. A new id means an empty folder, so the run starts again from step 0, with no error. **A chained segment must pin `+job.timestamp_id=<name>`.**
- Do not stop a segment with `runner.max_steps`. At the end of `max_steps`, torchtnt adds one to the epoch before the `final` checkpoint is saved. A resume from that checkpoint starts a new epoch with a new order.
- Stop by signal. In LOCAL mode fairchem catches SIGTERM. It links the newest complete checkpoint to `checkpoints/preemption_state` and exits with status 0. It does not save a new checkpoint, so the steps after the last checkpoint run again. A job script must not read exit 0 as "finished".
- A kill during a checkpoint write. DCP writes `.metadata` last, through a temporary file and a rename. A half-written new folder has no `.metadata` and is skipped. One window stays open: a resume first saves `step_<k>` again into the folder it just loaded. A kill during that save leaves the old `.metadata` next to a half-written data file. This takes seconds at the start of a segment.
- The distributed store file. With `distributed_init_method: FILE`, torch keeps `run_dir/<timestamp_id>/.distributed-shared-file`. A pinned id reuses this file. A killed process leaves it behind (see Section 7).

## 3. Pass criteria (fixed before the runs)

Runs: three uninterrupted references (A1, A2, A3) and resumed runs (B, C, and the epoch test EB).
The noise floor N of a number is its largest difference between two reference runs. GPU sums are not bit-exact, so N may be above zero.

A resumed run passes only if every check passes:

1. Bookkeeping. Steps 0 to 399 are each present once after the segments are merged. The run has two or more segments.
2. Exact equality with every reference at every step: the batch hash (sha256 of the structure ids, atom counts, atomic numbers and positions, in batch order), the epoch, the step in the epoch, the batch size in structures and atoms.
3. The learning rate equals the reference at every step (relative difference at most 1e-12).
4. Train loss after the resume point: the largest relative difference to any reference is at most 3 N (N over the same steps), or at most 1e-6.
5. Same state, two paths: the first segment ran on past its last checkpoint in memory. The loss of those steps from the first segment and from the resumed segment must agree within 3 N (N over the same steps), or 1e-6.
6. Final weights (model and EMA): relative L2 difference to any reference at most 3 N, or at most 1e-6.
7. Validation (energy MAE per atom, force MAE, validation loss, on the 5,000 report rows): relative difference at most 3 N, or at most 1e-6.

The factor 3 allows for the small number of reference pairs (three). The floor 1e-6 applies only if the GPU happens to be exact (N = 0).
For the epoch test, also: the batches of epoch 0 are all different, the batches of epoch 1 are batches of epoch 0, and their order is different.

## 4. Design

- Variant Dense, seed 0, `training.steps=400` (the cosine schedule is the same for all runs), a checkpoint every 100 steps, GPU `v100-32`.
- Data: the fixed pilot lists through `slurm/pilot_data_overrides.sh`. Each job first runs `scripts/check_pilot_data.py verify`, as `pilot.sh` does.
- Validation only once, at the end (`training.eval_every=1000000`; torchtnt still validates at the end of the epoch that `max_steps` cuts short). One validation on 5,000 rows takes about 5 minutes on a V100. A validation every 100 steps would add 20 minutes per run and add nothing that the final weights do not already show.
- The batch probe `training/resume_probe.py` is a test-only subclass of the production unit. It writes one CSV row per step and the validation metrics at full precision. No production config uses it.
- Job 1: A1; B stopped by SIGTERM after step 250; C stopped by SIGKILL after step 250, then a half-written `step_300` folder is left (data file cut to 1 MB, no `.metadata`); E1 and EB on a small list (below), EB killed after step 320.
- Job 2, a separate Slurm job: A2, A3, then B, C and EB resume. Then `scripts/resume_compare.py`.
- Local-6, one H100 job: L1, L2, and LB. LB is stopped by SIGTERM after step 250 and resumed by a new process in the same job.
- Epoch test: the 200,000-row list gives 27,375 batches per epoch, so 400 steps never reach a new epoch. EB uses every 100th row of the pilot list (2,000 rows, about 270 batches per epoch) and the first 200 validation rows. These small lists are test-only and are not in `pilot_samples.json`; the batch hashes check them instead.

How it was run. The code ran from a copy at `shared/runs/resume_test/code` (the shared repository copy plus the new files; the 17 files that training uses were checked equal to `main` by sha256). Job 2 was submitted after job 1 ended:

```bash
cd /ocean/projects/che260063p/shared/runs/resume_test/jobs
sbatch resume_test_1.job          # A1, B (SIGTERM at 250), C (SIGKILL at 250), E1, EB (SIGKILL at 320)
sbatch resume_test_2.job          # A2, A3, then B, C, EB resume; then the comparison
sbatch resume_test_local6.job     # L1, L2, LB (SIGTERM at 250, then resumed); then the comparison
```

GPU-shared offered a start 12 h later for every GPU type (`sbatch --test-only`, 00:05 on 29 Oct). The Dense jobs ran on `GPU-small` (one V100-32 node, v001), which started at once. That partition is not in the Bridges-2 guide's tables. `job_info` showed 0.575 SU for the 34.5 min of job 1, which is 1 SU per V100-hour, the same as GPU-shared.

## 5. Results

All numbers in this section are measured. The full tables are in `results/resume_test/` (Section 8).

**Verdict: PASS.** Every check of Section 3 passed for B, C, EB (Dense) and LB (Local-6).

| Check | B (SIGTERM) | C (SIGKILL, half-written checkpoint) | EB (resume in epoch 1) | LB (Local-6) |
|---|---|---|---|---|
| Resumed at step (epoch, step in epoch) | 200 (0, 200) | 200 (0, 200) | 300 (1, 30) | 200 (0, 200) |
| Batch hash, epoch, step in epoch, batch size: mismatches in 400 steps | 0 | 0 | 0 | 0 |
| Learning rate: largest relative difference | 0.0 | 0.0 | 0.0 | 0.0 |
| Train loss after resume: largest relative difference (noise N) | 2.5e-7 (3.5e-7) | 3.1e-7 (3.5e-7) | 1.0e-6 (6.4e-7) | 8.3e-7 (6.6e-7) |
| Same state, in memory against from disk (steps) | 1.6e-7 (200 to 250) | 1.4e-7 (200 to 251) | 1.2e-7 (300 to 321) | 1.8e-7 (200 to 250) |
| Final weights, relative L2 (N) | 9.5e-8 (1.0e-7) | 1.1e-7 (1.0e-7) | 3.7e-7 (2.4e-7) | 3.2e-7 (2.7e-7) |
| Final EMA weights, relative L2 (N) | 3.7e-8 (3.7e-8) | 3.9e-8 (3.7e-8) | 8.3e-8 (5.9e-8) | 8.7e-8 (7.6e-8) |
| Validation, largest relative difference over 4 metrics (largest N) | 1.7e-7 (2.4e-7) | 2.1e-7 (2.4e-7) | 1.0e-7 (6.1e-8) | 7.5e-8 (7.3e-8) |

References: A1, A2, A3 for B and C; E1, E2 for EB; L1, L2 for LB.

- **Noise floor.** Two uninterrupted runs with the same seed and data are not bit-identical on the GPU. Their train losses differ by at most 3.5e-7 (relative) over steps 200 to 399 (median 4e-8). Their final weights differ by 1.0e-7 (relative L2).
- **Power (negative control).** The loss of the next batch differs from the loss of the current batch by a median of 29 percent (A1, steps 200 to 399). A run that sees a wrong batch, or that loses the optimizer or the EMA state, would differ by many orders of magnitude more than the noise.
- **Epoch test.** On the 2,000-row list, epoch 0 has 270 batches, all different. The 130 batches of epoch 1 are the same batches in another order. EB resumed inside epoch 1 and saw the same order as E1 and E2.
- **Restart cases.** C resumed from `step_200` and skipped the planted half-written `step_300` (no `.metadata`). B and LB resumed with an empty `preemption_state` folder present (see Section 7). B, C, EB and LB all resumed with a stale `.distributed-shared-file` present. `init_process_group` worked each time (1 process, NCCL).
- **Time.** One 400-step Dense run with one validation took 6.5 to 13.6 min on a V100 (the first run of a job includes the cold start). One Local-6 run took 17 min on an H100.

**Limits of the margin.** For weights and validation, 3 N is below the floor of 1e-6, so the floor sets the limit. With 3 N alone, one number would fail: EB, total energy MAE, 6.9e-8 against 3 N = 4.7e-8. That N comes from one pair of runs on 200 validation rows. All differences in the table are of the size of floating-point noise (1e-8 to 1e-6).

## 6. What this means for the long runs (chain logic)

The test supports these rules. The rules are reasoning from the source and the test. They are not a tested long-run script.

1. **Pin the run folder and the id.** Each segment must use the same run folder (not one with `$SLURM_JOB_ID` in its name, as `pilot.sh` makes) and the same `+job.timestamp_id=<name>`. Then fairchem finds the last complete checkpoint by itself. Without the pin, a segment restarts at step 0 with no error.
2. **Keep `training.steps` at the full length (300,000) in every segment.** The schedule and `max_steps` come from it. Never shorten a segment with `runner.max_steps`: the `final` checkpoint then carries the next epoch number (measured: the last validation event of every run reports the epoch after the one it ran in).
3. **Done marker.** A segment is finished only when `checkpoints/final/.metadata` exists. Exit status 0 is not proof: fairchem's SIGTERM path should exit 0 (it exits 1 today, Section 7).
4. **Chain with `afterany`, not `afterok`.** A segment that reaches its time limit ends as TIMEOUT, which `afterok` treats as a failure (guide p. 20). Each segment first checks the done marker and exits at once if it is there. To stop a chain after a real crash, a segment can write a failure marker when the training exit is neither the planned stop nor a success, and the next segment can exit when it sees it.
5. **Stop before the limit.** Slurm sends SIGTERM at the limit and SIGKILL a short time later. The safest plan is to end training inside the job with `timeout --signal=TERM <limit minus 30 min>`, so the job script itself can still write its markers. All steps after the last checkpoint run again in the next segment. With a checkpoint every 10,000 steps this is up to about 2 h of Dense time per segment (0.68 s per step).
6. **Size consistency only after the last segment**, when the done marker exists. In `pilot.sh` this stage follows training in the same job; a long-run script must run it only in the segment that writes `final`.
7. **W&B.** Each segment writes a new offline folder with the same run id (`offline-run-<time>-<name>`). Upload them in order. This was not tested.

## 7. Defects found

- **fairchem 2.21 SIGTERM path (measured).** `launchers/slurm_launch.py` `graceful_shutdown` first runs `os.makedirs(preemption_state)`. `TrainEvalRunner.save_state(..., is_preemption=True)` (`components/train/train_runner.py` lines 177 to 181) then calls `os.remove` on that path, which is a folder: `IsADirectoryError`. Effects: the process exits 1, not 0. No symlink is made. The `distutils.cleanup()` call is skipped. An empty `checkpoints/preemption_state` folder stays. The resume still works, because `get_most_recent_viable_checkpoint_path` skips folders without `.metadata`. No fix is needed for correctness. A job script must not use the exit status to decide anything. Not fixed upstream; not patched here.
- **Stale store file.** A process that dies without cleanup leaves `run_dir/<id>/.distributed-shared-file` (mode 644). On the laptop, a new torch `FileStore` on such a file reused its old counter. On the cluster, `init_process_group` with 1 process worked in all 4 resumes. Two open points remain. More than one process per run was not tested. The file is not group-writable, so a teammate who resumes a run that Anshul started may not be able to open it. A long-run script can delete this file before each segment: only one segment of a run is alive at a time.
- **My tooling (fixed).** Two segments in one job write to one probe file. The first version of `resume_compare.py` read it as one segment. It now starts a new segment where the step does not increase (test added). The Local-6 comparison was rerun with the fixed script (job 50003737).

## 8. Files, jobs, cost, evidence

New files (none committed): `training/resume_probe.py`, `scripts/resume_compare.py`, `slurm/resume_test.sh`, `slurm/resume_test_1.job`, `slurm/resume_test_2.job`, `slurm/resume_test_local6.job`, `tests/test_resume_test.py` (14 tests, CPU), this file, `results/resume_test/`. No existing file was changed.

| Job | What | Partition, GPU | Elapsed | SU |
|---|---|---|---|---|
| 50002074 | job 1 (A1, B, C, E1, EB first segments) | GPU-small, V100-32 | 34 min 31 s | 0.575 (job_info) |
| 50003110 | job 2 (A2, A3, resumes, comparison) | GPU-small, V100-32 | 29 min 35 s | 0.493 (job_info) |
| 50002267 | Local-6 (L1, L2, LB) | GPU-shared, H100-80 | 45 min 22 s | 1.512 (job_info). State FAILED only because of the comparison bug above |
| 50003737 | Local-6 comparison rerun | RM-shared, 4 cores | 9 min 4 s | 0.604 RM (job_info) |

Total: 2.58 GPU SU and 0.60 RM SU.

Evidence:

- In the repository, `results/resume_test/`: `results/{dense,epoch,local6}_steps.csv` (per step and per run: epoch, step in epoch, batch sha256, loss, learning rate, gradient norm), `results/*_checks.json` (every check with value, noise and limit), `summary.txt` (the job log of every segment), and the probe files of every run.
- On Ocean, `/ocean/projects/che260063p/shared/runs/resume_test/` (19 GB, mostly checkpoints): the run folders with all checkpoints, the training logs in `logs/`, the small lists, and the code copy.

## 9. What was not tested

- Global, Frozen and Scaled Dense. They use the same runner, unit, sampler and checkpoint code as Dense and Local-6, so the same result is expected (reasoning). Element, Local-12 and Local+Global do not exist yet. If a new variant uses random numbers in training (for example `composition_dropout > 0`), the missing random-state restore matters, and this test must be repeated.
- A real Slurm time limit. The test sent SIGTERM to the training process only. At a real limit, Slurm signals the whole job, including the bash script.
- A resume on a different node. GPU-small has one node, so both Dense jobs ran on v001. The data order does not depend on the node (it is computed on the CPU from the seed), but the floating-point noise may be larger across nodes.
- Long horizons. 400 steps, with 100 or 200 steps after the resume. Over 300,000 steps a resumed run will drift from an uninterrupted one in the same way that two uninterrupted runs drift from each other.
- A kill during the re-save of the checkpoint that a segment just loaded (Section 2). The window is a few seconds at the start of each segment.
- More than one process per run, and a resume by another user.
- The full long-run script (plan item B2). Its rules are in Section 6.
