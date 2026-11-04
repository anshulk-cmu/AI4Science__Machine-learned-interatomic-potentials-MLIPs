# Two-GPU data-parallel mode (LONG_RANKS=2)

Writing style: simple English. Short sentences. One idea in each sentence. "Measured" means that a job or a script computed the
value. "Reasoning" means that it was not run.

## 1. Why

Local+Global and Local-12 need more memory than one L40S (46,068 MiB) has at 450 atoms per step: their 200-step peaks on an
H100 were 37,637 and 37,801 MiB, and the Local-6 pilots grew by about 1.55 times over 50,000 steps (`docs/launch-checklist.md`).
The memory fit of Section 4.4 agrees: the worst one-GPU batch would need about 46,000 MiB allocated (reasoning). The H100 queue was long on
2026-11-03. Two L40S in one job cost 2 SU per hour, the same as one H100.

`LONG_RANKS=2` runs one segment on two GPUs of one job with DDP. Each optimizer step uses the same 450-atom batch as a run on
one GPU. Each GPU gets about half of it.

## 2. How it works

| Part | One GPU (unchanged) | `LONG_RANKS=2` | Where |
|---|---|---|---|
| Launch | fairchem LOCAL mode, one process | fairchem LOCAL mode with `job.scheduler.ranks_per_node=2`: fairchem runs torch elastic (`elastic_launch`) in the training process, which starts two rank processes, one per GPU (NCCL) | `launchers/slurm_launch.py` `local_launch` |
| Batches | `MaxAtomDistributedBatchSampler`, 450 atoms | `training/split_sampler.py`: the batch of the one-GPU sampler (same seed, same 450 atoms, same order, same resume position), split in two parts balanced by atom count | `train_dataloader` and `eval_dataloader` `batch_sampler_fn._target_` |
| Loss and update | DDPMTLoss, AdamW, clip 10, EMA | the same (Section 3) | `modules/loss.py` |
| Validation | all batches on one GPU | each batch split over the two GPUs; a batch that both GPUs hold counts once (found by its atom content, charge and spin: OMol25 rows have an empty sid); `val/loss` reduced over the GPUs; the log gives the number of rows | `training/guarded_unit.py` |
| Out-of-memory error | clean up and retry twice, then stop | stop at once, no retry (Section 5) | `training/guarded_unit.py` |
| Log | one line per step | the same: one line per step from rank 0, with the loss of the whole step (Section 3); rank 1 logs warnings and errors only | `training/guarded_unit.py` |
| Checkpoint | DCP, one process | DCP, both ranks write; rank 0 writes `.metadata` last and the inference checkpoint | `units/mlip_unit/mlip_unit.py` `save_state` |
| Stop by time | `timeout` sends SIGTERM to the training process | the same; the elastic agent passes SIGTERM to both ranks, waits 30 s, then kills them | torch `elastic/agent/server/api.py` |
| Size check after the last segment | one process, GPU 0 | the same | `slurm/long_run.sh` |

The switch changes nothing when it is unset or 1. `tests/test_long_run.py` (`SingleGpuUnchangedTests`) runs the scripts of
commit 8ebe5fe and the new scripts with the same stand-ins and checks that every python and sbatch call, `settings.txt`,
`segments.txt`, `chains.txt` and the printed lines are the same.

### Why not 225 atoms per GPU

fairchem's sampler applies `max_atoms` to each rank's batch: it builds batches of at most `max_atoms` atoms and gives rank r
every second batch (`max_atom_distributed_sampler.py`, `get_batches` and `__iter__`). With `training.max_atoms=225` it drops
every structure with more than 225 atoms, and each step uses two batches packed separately. The training data would then not
be the data of the six variants that train on one GPU. Section 4 gives the measured share.

## 3. The update is the one-GPU update

Notation. B is the batch of one step on one GPU. B_0 and B_1 are its two parts, one per rank, so B = B_0 + B_1 with no
structure in both. For a task t (energy, forces), l_t(i) is the loss of sample i (one structure for energy, one atom for
forces) and n_t(X) is the number of samples of t in a set X. W = 2 is the number of ranks.

One GPU minimizes

    L(B) = sum_t (1 / n_t(B)) * sum_{i in B} l_t(i)                                   (1)

On rank r, DDPMTLoss (`_ddp_mean`) takes the sum over its own part, divides by the all-reduced count n_t(B_0) + n_t(B_1) =
n_t(B), and multiplies by W:

    L_r = sum_t (W / n_t(B)) * sum_{i in B_r} l_t(i)                                    (2)

DDP averages the gradients of the ranks:

    g = (1 / W) * (grad L_0 + grad L_1) = sum_t (1 / n_t(B)) * sum_{i in B} grad l_t(i) = grad L(B)     (3)

A batch with one structure cannot be split. Both ranks then hold all of B. The count is then W * n_t(B), so (2) gives
L_r = L(B) on each rank, and (3) gives g = grad L(B) again.

The all-reduce gives the same g on both ranks. Gradient clipping, AdamW and the EMA then act on the same numbers on both
ranks, as on one GPU. The learning-rate schedule counts optimizer steps, and the number of steps per epoch is the number of
one-GPU batches, so the schedule is the same.

What is not the same (reasoning):

- The sums run in another order: per rank, then over the two GPUs. The result differs at the level of float32 rounding.
  Section 4 measures it against the noise of two runs on one GPU.
- Nothing else in the update. The logs: fairchem logs L_r of (2) on each rank, the share of one rank. From (2),
  L_0 + L_1 = W L(B) for two parts, and L_0 = L_1 = L(B) for a batch that both ranks hold. So the loss of the whole step is
  the plain mean of the rank losses; no weight by atoms or structures is needed, because DDPMTLoss already divides by the
  count of the whole step. `training/guarded_unit.py` holds back fairchem's step log on rank 0, takes the mean with one
  all-reduce, and rank 0 logs `train/loss` = L(B) with the same keys (log and W&B). With one process nothing changes and no
  collective runs. Until commit c553773 the log held L_0. Rank 0 always gets the largest structure, so L_0 was off by 42 to
  78 percent on test batches with one large structure (`tests/test_ddp.py`).
- `train/atoms_per_second(approx)` is the rank-0 atom count times 2 over the step time.

## 4. Measured

### 4.1 Batches (job 50184447, CPU)

`scripts/ddp_batch_stats.py` runs fairchem's sampler and `SplitBatchSampler` on the real atom counts of `split_train_70.txt`
(2,791,057 rows, seed 0), for one epoch. File: `results/ddp/batches_split_train_70_s0.json`.

| | One GPU, 450 atoms | fairchem sampler, 2 ranks x 225 atoms | `SplitBatchSampler`, 2 ranks |
|---|---:|---:|---:|
| Steps per epoch | 381,159 | 407,556 (+6.9 percent) | 381,159 |
| Atoms per step, mean (sd) | 402.6 (43.1) | 363.4 (50.8) | 402.6 (43.1), the one-GPU batches |
| Structures dropped | 0 | 19,723 (0.71 percent of structures, 3.47 percent of atoms) | 0 |
| Steps whose structures differ from the one-GPU step | | (other batches) | 0 of 381,159; 0 of 254,106 after a resume at epoch 1, step 127,053 |
| Atoms per GPU, mean / p99 / max | 402.6 / 450 / 450 | at most 225 | 201.5 / 277 / 350 |
| Steps with one structure (both GPUs hold it) | | | 617 (0.16 percent) |
| Steps where one part has more than 225 atoms | | | 68,451 (18.0 percent) |

- So the split keeps the data, the order and the number of steps of the one-GPU run exactly. The plain sampler at 225 atoms
  would train without the 19,723 largest structures (226 to 350 atoms) and with 9.7 percent fewer atoms per step.
- The price: one GPU sometimes holds a large structure alone. The largest training structures have 225 to 350 atoms, up to
  26,308 atom pairs closer than 6 A (the message edges) and up to 93,300 closer than 12 A (the Local-12 routing graph).
  A typical 450-atom batch has about 13,000 message edges (402.6 atoms times about 33 neighbours; reasoning from the pilot
  statistics in `docs/TRAINING.md`). So the worst GPU part can have about twice the edges of a typical whole batch.
  Section 4.4 measures the memory of these structures.

The same check on the validation lists (shuffle off): `split_val_monitor_5k.txt` gives 684 steps with the split and with one GPU,
0 steps differ, and no step has one structure. At 225 atoms per rank, fairchem's sampler would drop 42 of the 5,000 rows and
repeat one batch. `pilot_val_5k.txt` gives 1,342 steps and 36 one-structure steps (counted once, Section 2).

### 4.2 Two ranks give the one-GPU training (job 50188423, 2 x V100-32)

Method: `slurm/ddp_test.sh v100`, with `slurm/resume_test.sh` and the probe of `docs/resume-test.md`. Dense, seed 0, the pilot
lists, 400 steps, a checkpoint every 100 steps, one validation at the end on the 5,000 pilot validation rows. S1 and S2 ran on
one GPU each, D1 and D2 on two ranks. The probe writes, for every step, an order-free hash of the atom content of the whole
step, the counts and the global loss (`probe/global`). `scripts/resume_compare.py --probe-dir probe/global --uninterrupted`
applies the pass criteria of `docs/resume-test.md` Section 3: exact equality of the batches, the step counters and the learning
rate, and at most max(3 N, 1e-6) for the numbers, where N is the largest difference between the two one-GPU runs.

| Check (D1 and D2 against S1 and S2) | D1 | D2 | Noise N (S1 against S2) | Result |
|---|---:|---:|---:|---|
| Structures of every step (content hash), epoch, step in epoch, counts | 0 differ | 0 differ | | PASS |
| Learning rate, largest relative difference | 0.0 | 0.0 | | PASS |
| Train loss, steps 0 to 399, largest relative difference | 5.0e-7 | 1.0e-6 | 8.4e-7 | PASS (limit 2.5e-6) |
| Final weights, relative L2 | 1.29e-7 | 1.49e-7 | 1.27e-7 | PASS (limit 1e-6) |
| Final EMA weights, relative L2 | 4.8e-8 | 5.3e-8 | 4.3e-8 | PASS |
| `val/loss` | 2.1e-8 | 1.3e-8 | 2.1e-8 | PASS |
| Energy MAE per atom / force MAE | 5.0e-8 / 1.0e-8 | 5.0e-8 / 1.0e-8 | 8.0e-8 / 1.2e-8 | PASS |

- The train loss of step 0 (same weights, same batch) is 0.8128257544 on one GPU and 0.8128257851 on two ranks. The difference,
  3.8e-8, is the other summation order. A wrong batch would change the loss by about 29 percent (`docs/resume-test.md`).
- The validation of D1 reports "5000 rows, each counted once; 1342 steps per rank". Its energy MAE per atom is 0.1146047852,
  between S1 (0.1146047794) and S2 (0.1146047886).
- Element (per-edge expert mixing, as in every Local variant), 100 steps, E2 on two ranks against E1 and E1b on one GPU:
  PASS. Train loss 1.8e-7 (N 1.7e-7), weights 4.8e-8 (N 3.9e-8), validation within 3.0e-8.
- After training, a sha256 of the model and EMA weights was the same on rank 0 and rank 1 for every two-rank run: D1, D2,
  DB, Element, Local+Global and Local-12. So every parameter got the same, synchronised update. No run raised DDP's
  unused-parameter error. (DDP already wraps the model on one GPU, with `find_unused_parameters=False`.)

### 4.3 Resume with two ranks (same job)

DB ran on two ranks, got SIGTERM after step 250 (sent to the launcher, as `timeout` does in `long_run.sh`), and a new launch
resumed it from `step_200`. The launcher passed the signal to both ranks; the process ended in 7 s with exit 1 (fairchem's
SIGTERM path, `docs/resume-test.md` Section 7). References: D1 and D2.

| Check | `probe/rank0` | `probe/rank1` | `probe/global` |
|---|---|---|---|
| Batch of every step on this rank (or of the whole step), epoch, step in epoch, counts | 0 differ | 0 differ | 0 differ |
| Learning rate | equal | equal | equal |
| Train loss after the resume (noise N) | 6.2e-7 (2.3e-7) | 4.1e-7 (7.0e-7) | 2.8e-7 (2.8e-7) |
| Final weights, EMA (N) | 1.18e-7, 4.1e-8 (9.3e-8, 3.5e-8) | same | same |
| Validation, largest difference | 9.6e-9 | same | same |
| Verdict | PASS | PASS | PASS |

The rank-0 train loss is the loss of rank 0's part only, so its noise is larger than the noise of the whole step; it is still
below the floor of 1e-6. Each rank's sampler position comes from the restored step counter: the batches of both ranks match
at every step after the resume.

`long_run.sh` with `LONG_RANKS=2` (Dense, pilot lists, 3,000 steps): segment 1 stopped by time after 890 s (limit 884 s),
training exit 124, "stopped by time", checkpoints up to step 2100. Segment 2 resumed from step 2100, finished, wrote `final`,
ran the size check (Delta 2.8e-6 meV) and wrote DONE.

### 4.4 Memory per GPU

The probe records `torch.cuda.max_memory_allocated` of every step on every rank (V100, pilot list, seed 0, first 100 steps,
200 points per variant). The peak is linear in the message edges of the rank's part:

| Variant | Peak allocated per step, MiB | Largest residual | Largest point seen |
|---|---|---:|---|
| Local+Global | 1,211 + 1.446 x edges | 96 | 24,746 (one 272-atom structure, 16,250 edges) |
| Local-12 | 1,179 + 1.454 x edges | 100 | 24,857 (the same structure) |
| Element | 1,208 + 0.945 x edges | 96 | 16,603 |

Reasoning from the fit (an extrapolation 1.6 times beyond the largest point): the training structure with the most message
edges (26,308) needs about 39,250 to 39,500 MiB allocated on one GPU. The one-GPU worst batch (that structure plus about 5,000
edges of small molecules) would need about 46,000 MiB. This agrees with the measured need for more than one L40S on one GPU.

Measured on 2 x L40S (job 50184492): two ranks trained one epoch of the 87 rows with the most atom pairs (`worst_rows.txt`,
from Section 4.1; each step held one or two of these structures, so each GPU held one). No out-of-memory error.

| Variant | Peak allocated / reserved per GPU, MiB (torch) | Structure | Fit prediction |
|---|---|---|---:|
| Local-12 | 39,364 / 39,610 | 282 atoms, 26,308 edges | 39,420 |
| Local+Global | 39,242 / 39,548 | the same | 39,247 |

`nvidia-smi` (every 5 s, with the CUDA context and NCCL buffers) peaked at 40,323 MiB on GPU 0 and 40,325 MiB on GPU 1, of
46,068 MiB: 5,743 MiB (12.5 percent) stay free. In the normal 1,000-step and 600-step runs of Section 4.5 the peaks were 27,615
MiB (Local+Global) and 26,949 MiB (Local-12).

The GPU check of `long_run.sh` for two ranks is therefore 40,000 MiB per GPU: about the measured worst allocation. It accepts
the L40S and refuses a V100-32, which ran out of memory on the same structures (Section 5).

The planned run (job 50190325, CPU): `scripts/ddp_batch_stats.py --part-steps 400000` counted the atom pairs closer than 6 A for
all 2,791,057 training rows (mean 1,791, p99 9,699, max 26,308) and went through the rank parts of the first 400,000 steps of
seed 0 (all of epoch 0 and 18,841 steps of epoch 1). The largest part is that same structure alone (282 atoms, 26,308
edges, step 320,473, rank 0); the next ones are also single structures (26,022, 25,342, 25,322 edges). So the peak per GPU of
the planned seed-0 run is the one measured above: 39,364 MiB allocated, about 40,300 MiB in `nvidia-smi`. Another seed
reorders the batches; its largest part was not computed. File: `results/ddp/parts_split_train_70_s0_400000.json`.

### 4.5 Speed on 2 x L40S (job 50184492)

`long_run.sh` with `LONG_RANKS=2` on the real lists (`split_train_70.txt`, `split_val_monitor_5k.txt`), seed 0, one segment
each. Seconds per step from fairchem's step line of rank 0: time = atoms on rank 0 x 2 / `atoms_per_second(approx)`, steps
20 onward (no validation inside these steps).

| | Local+Global | Local-12 |
|---|---:|---:|
| Steps | 1,000 | 600 |
| Seconds per step, median / mean / p90 | 0.693 / 0.742 / 0.870 | 0.607 / 0.663 / 0.757 |
| Same steps 20 to 199 (same batches), median | 0.712 | 0.610 |
| Single H100, 200-step smoke (pilot lists), median / mean | 0.611 / 0.655 | 0.921 / 1.059 |
| 400,000 steps at the median / mean, training only | 77 h / 82 h | 67 h / 74 h |
| Peak GPU memory (`nvidia-smi`, both GPUs) | 27,615 MiB | 26,949 MiB |
| Validation (5,000 monitor rows) | 684 steps per rank, all 5,000 rows once | the same |
| Size check after the run | Delta -41.76 meV at every gap | 0.0607 / 0.00675 meV at 8 / 10 A, 2.8e-6 from 15 A |
| Training process, wall time | 1,312 s | 898 s |

- Start-up: the list check took 3.7 min (Local+Global) and 0.3 min (Local-12, warm cache). Each rank built the 381,159
  batches of `split_train_70.txt` in 9.6 s.
- Local-12's 0.607 s per step may be a warm-cache figure. Local-12 cannot be cheaper than Local+Global per step; it ran
  second and read the same rows, so the page cache likely helped it (reasoning, not measured). The speed of Local-12 with a
  cold cache on 2 x L40S is not measured.
- Plan with 0.70 to 0.75 s per step for both variants on 2 x L40S: about 80 to 85 h for 400,000 steps plus 41 validations.
  For Local-12 also allow up to 1.6 times that, as on the H100, where Local-12 (run first, cold) took 1.51 times the median
  of Local+Global: up to about 1.2 s per step and 130 h, so 3 segments of 47 h plus a spare, or 4 plus a spare at the upper
  end (reasoning from the tables).
- One validation of the 5,000 monitor rows ran twice at the end of each run; the wall time of one validation was not
  separated from start-up and checkpoints (1,312 s minus 742 s of steps covers start-up, 3 checkpoints and 2 validations).

### 4.6 Logs and tools

- Rank 1 logs warnings and errors only. The training log has one line per step and one validation block, from rank 0; the
  measured logs have no repeated step lines. The torchtnt warning "Expected step function to have an annotated argument named
  data" is in one-GPU logs too; with two ranks it comes twice as often.
- W&B: fairchem starts the logger on rank 0 only (`_init_logger`), so each segment writes one `.wandb` file, and
  `long_run.sh` lists it in `segments.txt` as before.
- `collect_final_metrics.py` read the two-segment two-rank run with W&B and log (they agree) and with `--source log`.
  `training.checkpoints.discover` (used by `checkpoint_tools.py`) found steps 0 to 3000. The checkpoint holds the same
  `inference_ckpt.pt` as on one GPU (rank 0 converts the DCP files of both ranks).
- `train/loss` is the loss of the whole step (Section 3), as `make_accuracy_figure.py` and
  `size_consistency_along_training.py` expect; the validation values, including `val/loss`, are global. The runs of Section
  4.5 ran before this change and logged rank 0's share; their speed numbers do not depend on it.
- Job 50192548 (2 x V100-32, Dense, 300 steps, `slurm/ddp_test.sh loss`): the logged `train/loss` of a two-rank run against a
  one-GPU run: step 0 3.7e-8, largest 1.5e-6 over 300 steps (criterion 2.5e-6, set before the run): PASS. A two-rank run
  stopped after step 151 and resumed on one GPU from `step_100` against the one-GPU and the two-rank run: PASS on every check
  of Section 4.3 (loss 1.0e-6 against N 1.5e-6, weights 1.6e-7 against N 2.1e-7). Files: `results/ddp/loss_test_50192548.out`,
  `results/ddp/rank_change_checks.json`.

### 4.7 A real one-GPU chain with the new code (`LONG_RANKS` unset)

The live runs will read this code at their next segment. So a real chain ran through `slurm/submit_chain.sh`,
`slurm/long_run.job` and `slurm/long_run.sh` of this branch, and the same chain with the code of `main` (8ebe5fe), on GPU-small
(V100-32): Dense, the real lists, 3,200 steps, 3 segments of 1 h, a checkpoint every 100 steps, from separate code copies and
a test runs folder. A test-only `sbatch` stand-in on `PATH` changed only the partition (GPU-shared to GPU-small), the job name
(`long_*` to `chainnew_*` or `chainmain_*`) and the folder of the job output; it then called the real `sbatch`.

| | New code: jobs 50192537, 50192538, 50192539 | Main code: jobs 50192540, 50192541, 50192542 |
|---|---|---|
| Segment 1 | stopped by time after 1,558 s (limit of the 1 h job minus the 1,800 s margin); checkpoints to step 2100 | after 1,784 s; to step 3100 |
| Segment 2 (afterany, then queued for GPU-small) | resumed from step 2100, finished, final checkpoint, 2 validations, size check (Delta 1.9e-6 meV), DONE | resumed from step 3100, the same (Delta 2.8e-6 meV), DONE |
| Segment 3 | "DONE marker present", 3 s | the same |
| Final validation (log, rounded) | val/loss 0.2740, energy 0.0865, force 0.1548 | 0.2740, 0.0865, 0.1548 |

After the job ids, times, stop step, code hash and the float floor of the size check are masked, `settings.txt`,
`segments.txt`, `chains.txt` and `summary.txt` of the two chains are the same. The earlier chain smoke (job 50116336,
`docs/launch-checklist.md` Section 6) ran its three segments inside one job; this test used three Slurm jobs and the real
`afterany` dependency, and shows the same records (stop by time, resume, final, size check, DONE, then an immediate exit).
Files: `results/ddp/chain_test/`. The two segments stopped at different steps because the main-code chain ran second and
read the same rows from a warm page cache.

## 5. Out-of-memory errors with two ranks

With one GPU, the unit cleans up and runs the same batch again. With two ranks this is not safe. When one rank runs out of
memory, the other rank already waits in a collective of the same step: the all-reduce of the sample count in DDPMTLoss, or a
DDP gradient bucket. NCCL pairs collectives by their order only. A retry would pair a collective of the retried step with the
waiting one, for example a count of 1 with a count of 450, and the result would be wrong without an error.

So with more than one rank, an out-of-memory error raises at once: `OOM: train batch ran out of memory on rank r of 2 ... the
run stops`. The elastic agent sees the failed rank, sends SIGTERM to the other rank, and kills it after 30 s. `long_run.sh` then
writes FAILED ("training exited 1"), and the chain stops. No batch is skipped. The last complete checkpoint stays.
`tests/test_ddp.py` checks this with two processes on gloo: the rank that fails calls its step once and does no cleanup, and
the other rank leaves its collective with an error instead of waiting.

Measured on the GPUs: the memory test of the 87 largest structures on 2 x V100-32 ran out of memory at the first step (31.0 GiB
of 31.7 GiB). Both ranks raised the message above, torch elastic ended the run, and the process exited 1 after about 40 s.

## 6. How to use

```bash
C=/ocean/projects/che260063p/shared/repo/AI4Science-MLIPs/slurm/submit_chain.sh
LONG_RANKS=2 LONG_GPU=l40s-48 bash $C local12 --steps <N> --seed <S> --segments <K> --hours <H> --eval-every <E> --ckpt-every <C> --dry-run
```

- `submit_chain.sh` accepts `LONG_RANKS=2` only with `LONG_GPU=l40s-48` and only for Local-6, Element, Local+Global and
  Local-12. It asks for `--gpus=l40s-48:2` and prints 2 SU per hour (2 GPUs at 1 SU per GPU-hour). Every other combination stops
  with exit 2.
- `long_run.sh` checks that the job has exactly 2 GPUs and that the smaller one has at least 40,000 MiB (the V100 variants keep
  31,000 MiB per GPU). It writes `ranks=2` into `settings.txt`, so a run keeps its rank count: a segment with another count
  stops with "settings differ".
- Do not set `OMOL_GPU_MEMORY_FRACTION` with two ranks: `training/run.py` applies it in the launcher process only.
- Launch mode: fairchem's LOCAL mode is the only one that works inside our Slurm job. SLURM mode submits a new job with
  submitit and refuses to run inside a job (`_cli.py`: "SLURM DID NOT SUBMIT JOB"). `training/run.py` runs its version and
  path checks in the launcher process; the two rank processes start by spawn and import `training.run` without running
  `main` (reasoning from Python's spawn method; both ranks trained in every test).

### Change the rank count of a run (for example after an out-of-memory stop on two L40S)

A two-rank run that stops with "OOM: train batch ran out of memory on rank r of 2" meets the same batch again after every
resume on two L40S (the batch order is fixed). It can continue on one H100. `settings.txt` refuses this on purpose, so the
change is a manual override:

1. Make sure no segment of the run is queued or running: `squeue --me -n long_<variant>_s<S>_<N>`. Cancel the rest of the
   chain with `scancel -n long_<variant>_s<S>_<N>` if needed.
2. In the run folder: `cp settings.txt settings_two_ranks.txt` (keep the record), then remove the rank part from the first line:
   `sed -i '1s/ ranks=2 batch=one-GPU batches of 450 atoms split over 2 GPUs$//' settings.txt`.
   Check with `head -1 settings.txt`: the line must now end with `ckpt_every=<C>`.
3. Delete the `FAILED` marker: `rm FAILED`.
4. Submit a new chain with the same `--steps`, `--seed`, `--eval-every` and `--ckpt-every`, without `LONG_RANKS` and without
   `LONG_GPU` (so on `h100-80`). The first segment resumes from the last complete checkpoint.

Why this is safe (reasoning, and one test): the two-rank checkpoint holds the same state as a one-GPU checkpoint (model, EMA,
optimizer, scheduler and step counters are the same on both ranks, and DCP reads them at any number of ranks), and the
one-GPU sampler continues with the same batch at the same step, because the split sampler uses the one-GPU order and
position. Job 50192548 tested it with `slurm/resume_test.sh`: a two-rank run stopped after step 151 and resumed on one GPU
passed every check against one-GPU and two-rank runs (Section 4.6). The `settings.txt` edit itself was not run. The other direction (one GPU to two ranks) is the same edit in reverse: add the text
to the first line. The alternative without an override is a new run folder (another `--steps` or `LONG_RUNS_DIR`), which
starts again at step 0.

## 7. Risks and what was not verified

- **An out-of-memory stop cannot be passed on two L40S.** The batch order is fixed, so a resumed run meets the same batch
  again. For seed 0 the largest rank part of all 400,000 steps is the structure of the memory test, which left 5.7 GB free per
  GPU (Section 4.4). Allocator fragmentation over 48 h was not measured. If a run still stops this way, it can move to one
  H100 by the manual override of Section 6.
- **Long horizon.** The equivalence and resume tests ran 400 steps (Dense) and 100 steps (Element). Two runs of 400,000 steps
  drift apart from float rounding, on one GPU as on two.
- **Speed.** Measured over 600 and 1,000 steps, after a start with a warm or a cold page cache (Section 4.5). Not measured:
  the speed over a whole 48 h segment, and a node shared with other jobs.
- **Not tested on the GPUs:** a real Slurm time limit (the tests stop by `timeout`, as `long_run.sh` does), a resume on another
  node, Local-6 on two ranks (same code as Element and Local+Global), and a kill of the launcher alone (its ranks would keep
  running until the job ends; `resume_test.sh` refuses `kill:STEP` with two ranks).
- **Logs.** `train/loss` became the whole-step loss after the main GPU runs of Section 4; job 50192548 checked it on the GPUs.
- **Tests.** `tests/test_ddp.py` has 13 tests. On native Windows the 3 two-process tests skip (no gloo device); on Linux
  (job 50192700) all 13 ran and passed.
- **One seed.** The batch statistics and the largest-part check use seed 0, the seed of the planned runs.

## 8. Files, jobs and cost

| File | What |
|---|---|
| `training/split_sampler.py` | the sampler |
| `training/guarded_unit.py` | OOM stop with several ranks, validation counted once, `val/loss` reduced, rank-1 log level |
| `training/resume_probe.py`, `scripts/resume_compare.py` | probe per rank and for the whole step; `--probe-dir`, `--uninterrupted` |
| `slurm/long_run.sh`, `slurm/submit_chain.sh` | `LONG_RANKS` |
| `slurm/resume_test.sh` | `RESUME_RANKS`, `smallval`, `RESUME_SKIP_VERIFY` |
| `scripts/ddp_batch_stats.py` | batch statistics, the worst rows, the largest rank parts |
| `slurm/ddp_test.sh`, `slurm/ddp_test_{cpu,v100,l40s}.job` | the test jobs |
| `tests/test_ddp.py`, `tests/test_long_run.py`, `tests/test_resume_test.py` | tests |
| `results/ddp/` | the evidence of this page |

| Job | What | Partition, resources | Elapsed | Charge |
|---|---|---|---|---|
| 50184447 | Linux tests, batch statistics, worst rows | RM-shared, 8 cores | 33 min 18 s | about 4.4 RM SU |
| 50184478 | first V100 run, hung (bare `wait`) and Ocean errors; cancelled | GPU-small, 2 x V100-32 | 12 min 21 s | about 0.41 SU |
| 50186090, 50188427 | gloo tests on Linux (before and after the sid fix) | RM-shared, 4 cores | 8 min 38 s, 50 s | about 0.6 RM SU |
| 50186177 | V100 run with the sid defect; cancelled | GPU-small, 2 x V100-32 | about 40 min | about 1.3 SU |
| 50188423 | equivalence, resume, Element, long_run.sh, collector, OOM stop | GPU-small, 2 x V100-32 | 69 min | about 2.3 SU |
| 50184492 | Local+Global, Local-12, memory test | GPU-shared, 2 x L40S | 49 min | about 1.65 SU |
| 50190325 | largest rank parts over 400,000 steps | RM-shared, 16 cores | 7 min 52 s | about 2.1 RM SU |
| 50192546, 50192700 | test_ddp on Linux (the first found two stale test stand-ins; 13 of 13 passed in the second) | RM-shared, 4 cores | 5 min each | about 0.7 RM SU |
| 50192548 | logged `train/loss`, two ranks resumed on one GPU | GPU-small, 2 x V100-32 | 10 min 9 s | about 0.34 SU |
| 50192537 to 50192539 | one-GPU chain, new code | GPU-small, 1 x V100-32 | 30 min 11 s, 12 min 15 s, 3 s | about 0.71 SU |
| 50192540 to 50192542 | one-GPU chain, main code | GPU-small, 1 x V100-32 | 30 min 9 s, 5 min 39 s, 3 s | about 0.60 SU |

The GPU charges are elapsed time x 2 GPUs x 1 SU per GPU-hour (reasoning; see the job_info values in the run log).

Two jobs ended as FAILED in Slurm. Both are benign:

- 50184447: every step passed except my own unit test of the out-of-memory release, which timed the start-up of two
  processes (127 s) instead of the release of the waiting rank. The behaviour checks of that test passed. The test was fixed
  (commit 3d936b9), and jobs 50186090 and 50188427 passed it.
- 50188423: every check passed except the last step, the memory test on 2 x V100-32. It ran out of memory as expected (the
  worst structure needs about 39 GiB, a V100 has 31.7 GiB), and it showed the stop-together path on real GPUs (Section 5).
  The same test passed on 2 x L40S (job 50184492).
