# Training

## Environment

Use Linux or WSL2, Python 3.12, `fairchem-core==2.21.0`, and the
dependencies in `requirements.txt` and `requirements-training.txt`.
`requirements-training.txt` installs `pymatgen`, which FairChem's CPU neighbour
list (`get_neighbors_pymatgen`) needs because the data config sets `r_edges: true`.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-training.txt
python -m pip check
python -c "import torch, fairchem.core; print(torch.__version__, torch.cuda.is_available())"
```

Native Windows cannot run the training CLI, for two separate reasons:

1. `fcntl`: `fairchem.core._cli` imports `fairchem.core.launchers.api`, which imports
   `clusterscope` 0.0.18, and `clusterscope/cache.py` runs `import fcntl`. `fcntl` is a
   POSIX-only module, so the import fails with `ModuleNotFoundError: No module named 'fcntl'`.
2. NCCL: with `job.device_type: CUDA`, FairChem's `map_job_config_to_dist_config`
   (`launchers/slurm_launch.py`) picks the `nccl` process-group backend (it uses `gloo` only
   for CPU jobs), and even a single local rank calls `torch.distributed.init_process_group`
   with it. The Windows PyTorch wheels are built without NCCL
   (`torch.distributed.is_nccl_available()` returns `False`), so that call fails.

## Config

| File | What it contains |
| --- | --- |
| [`configs/train.yaml`](../configs/train.yaml) | Selects the data, model and element-reference configs and defines the job, OMol energy/force tasks, optimizer, schedule, validation, and checkpoints. |
| [`configs/data/generic.yaml`](../configs/data/generic.yaml) | Loads OMol ASE DB train/val splits with FairChem's native dataset, sampler, and collater. |
| [`configs/element_refs/omol.yaml`](../configs/element_refs/omol.yaml) | The official `omol_elem_refs` list, copied from FairChem's `uma_v1_hof_lin_refs.yaml` at tag `fairchem_core-2.21.0`. |
| [`configs/variant/dense.yaml`](../configs/variant/dense.yaml) | Uses `eSCNMDBackbone`, 4 layers, and no experts. |
| [`configs/variant/global.yaml`](../configs/variant/global.yaml) | Uses `eSCNMDMoeBackbone`, 4 layers, 8 experts, and FairChem's global composition router. |
| [`training/schedulers.py`](../training/schedulers.py) | Cosine schedule whose linear warmup is counted in optimizer steps. |

Model sizes for the current configs, measured by composing `train.yaml` and instantiating the
model on CPU (`tests/test_training_config.py` asserts them):

| Variant | Total parameters | Expert bank | Router | Active per structure |
| --- | ---: | ---: | ---: | ---: |
| Dense | 5,981,057 | none | none | 5,981,057 |
| Global | 37,652,265 | 36,179,968 (24 MOLE layers) | 17,320 | 5,998,377 |

The active count keeps one of the 8 expert weight slices in every MOLE layer, plus every
parameter outside the expert weights (the MOLE biases are shared, not per expert). Both
variants meet the 6M base budget. All variants use `num_distance_basis: 64`, as the released UMA-S-1.1 and the OMol25
eSEN-sm model do (the `eSCNMDBackbone` class default is 512, which gave 6,267,777 for Dense and 37,938,985 for Global).
The team chose 64 on 2026-10-11. Local-6 has the same parameters as Global. Each variant backbone also pins
`max_neighbors` to `training.max_neighbors`, so the graph is the same in training and in inference.

Energy targets: the energy task subtracts the official OMol element references
(`ElementReferences(torch.DoubleTensor(...))`, as in the official OMol tasks) before
normalizing. Both tasks share one normalizer, `rmsd: 1.433569`, the official allscaip
`omol_4m` value and the closest official analogue. It must be refit on `train_4M`.

Common runtime overrides:

| Config | Variable | Purpose |
| --- | --- | --- |
| `train.yaml` | `variant` | select `global`, `dense`, or another model variant. |
| `train.yaml` | `training.steps` | total optimizer steps; also the schedule length. |
| `train.yaml` | `training.warmup_steps` | linear warmup length in steps; `null` means `round(0.05 * steps)`. It does not depend on the dataloader length. |
| `train.yaml` | `training.eval_every` | steps between validation runs; defaults to `training.steps`. |
| `train.yaml` | `training.ckpt_every` | steps between checkpoints; defaults to `training.steps`. |
| `train.yaml` | `training.max_atoms` | The batch size in atoms, `450` for every variant; larger structures are dropped by the sampler. |
| `train.yaml` | `training.max_neighbors` | `300` nearest neighbours kept per atom inside the cutoff (`a2g_args.max_neigh`, both splits, and the backbone). |
| `train.yaml` | `training.max_saved_checkpoints` | step checkpoints kept on disk, `3`; the pilots keep all of them. |
| `train.yaml` | `training.cutoff` | `6.0` Å graph/model cutoff. |
| `train.yaml` | `training.normalizer_rmsd` | shared energy and force normalizer. |
| `train.yaml` | `job.run_name` | set a distinct name for each run. |
| `train.yaml` | `job.run_dir` | output directory; env `OMOL_RUN_DIR`, default `../omol25-runs`. |
| `data/generic.yaml` | `data.root` | env `OMOL_DATA_ROOT`, default `../omol25-data`; holds `train_4M` and `val`. |
| `data/generic.yaml` | `data.train_src`, `data.val_src` | ASE DB split directories. |
| `data/generic.yaml` | `train_dataset.dataset_configs.omol.splits.train.sample_n` | number of samples; set `null` to use all training rows. |
| `data/generic.yaml` | `val_dataset.dataset_configs.omol.splits.val.sample_n` | number of samples; set `null` to use all validation rows. |
| `data/generic.yaml` | `...splits.<split>.subset_to`, `metadata_path`, `no_shuffle` | a fixed list of rows instead of a random draw; see "Fixed sample lists" below. |

`training/run.py` sets `FAIRCHEM_CACHE_DIR` and `WARP_CACHE_PATH` under env `OMOL_CACHE_DIR`
(default `../omol25-cache`). Relative paths resolve from the repository root, which `run.py`
requires as the working directory. `run.py` also refuses to start unless `fairchem-core` is
exactly 2.21.0 and `fairchem.core` is imported from that installed distribution.

The split directories should contain their `.aselmdb` shards and matching
`metadata.npz`. The current tasks and label mapping are specific to OMol.

## Validation and checkpoints

- A checkpoint is written every `training.ckpt_every` steps, and also at step 0 (the callback
  runs before each step and `0 % ckpt_every == 0`) and once more at the end (`final`). With
  `save_inference_ckpt: true` each one also gets an inference checkpoint. The 3 newest step
  checkpoints are kept (`training.max_saved_checkpoints`).
- Validation runs every `training.eval_every` steps. torchtnt's `fit` also evaluates at the end
  of every epoch, including the epoch that `max_steps` cuts short, because `TrainEvalRunner`
  does not pass `evaluate_every_n_epochs` and its default is 1. When `steps` is a multiple of
  `eval_every`, the last validation therefore runs twice. FairChem offers no config hook to
  turn the epoch-end evaluation off.
- `ema_decay: 0.999` keeps an EMA copy of the weights. Validation uses the EMA model.

## Run Training

From the repository root, choose a variant and a distinct run name:

```bash
python -m training.run --mode train variant=dense job.run_name=omol_dense_check
python -m training.run --mode train variant=global job.run_name=omol_global_check
```

Larger Global run:

```bash
python -m training.run --mode train variant=global \
  job.run_name=omol_global_full training.steps=10000 \
  training.eval_every=1000 training.ckpt_every=1000 \
  train_dataset.dataset_configs.omol.splits.train.sample_n=null \
  val_dataset.dataset_configs.omol.splits.val.sample_n=5000
```

## Known gaps

- The pilots are a short regime: 50,000 steps are about 1/760 of the official eSEN run (two passes over the 200,000 training rows). They measure seed noise and test the
  pipeline, they do not give final accuracy.
- The normalizer `rmsd` is the allscaip value 1.433569, not refit. Measured on the pilot lists (job 49039106, 2026-10-11): the per-atom
  force-norm RMS is 2.12 eV/A on the 200,000 training rows (component RMS 1.23) and 1.54 on the 5,000 validation rows, so the official
  value matches neither and its origin is not known. The team keeps it for the pilots: it is the same for all variants and changes the
  output learning rate by at most a factor of about 2.
- Energy and force loss weights are 1:1 (coefficient 1.0 each) and have not been tuned.
- `training.workers` is 0, so data loading runs in the training process.
- `max_neighbors: 300` never truncates. Measured on the pilot lists: the largest neighbour count within 6 Å is 162 (training) and 164
  (validation), the mean is 32.6 and 35.9. The official cap of 30 would truncate 50.5 percent of the training atoms.
- The pilots read fixed sample lists (see "Fixed sample lists"). A run with `sample_n` draws a random subset whose ids are not recorded.

## Add a Local Router

- Implement the router as a Python `torch.nn.Module` (for example in
`training/routers.py`) and connect it through a custom backbone class (for
example `training/local_moe.py`).
- Copy `configs/variant/global.yaml` to a new
variant, change `backbone.model` to the custom class's import path, and add
its router arguments there.
- Run with `variant=<new_variant>`; the shared train
and data configs can stay the same.

## Local-6

`variant=local6` uses `training.local_moe.eSCNMDLocalMoeBackbone` (`configs/variant/local6.yaml`). It has the same
parameters as Global (37,652,265) and differs only in the router: each atom averages the element embeddings of its
neighbors within 6 A, and every message edge mixes the experts with the coefficients of its receiving atom.

- Tests: `tests/test_local_backbone.py` (13 tests, float64, CPU and CUDA). Far-apart fragments are additive for Local and
  Dense but not Global, the per-edge mixing equals fairchem's merged layers when every atom gets the structure mean, and
  the forces equal the energy gradient. Run them in Linux or WSL2.
- Memory: Local mixes the expert outputs per edge, which needs several times the memory of Global. A 20-step run with
  `training.max_atoms=320` ran out of memory on a 12 GB laptop GPU even with `checkpoint_chunk_size: 4096`
  and the mixing recomputed in the backward pass. Lower `training.max_atoms` or run on a larger GPU (Bridges-2). The peak
  memory per setting has not been measured yet.
- Not supported: a routing radius above the message cutoff (Local-12), graph parallelism, merging into one linear layer.
- fairchem builds each edge frame from Euler angles, which are singular for an edge exactly along y. Forces there differ
  from the energy gradient, for Dense as well. The tests rotate molecules off the axes for that reason.

## Weights and Biases (offline)

`train.yaml` turns the W&B logger on through the `logger: wandb` config group (`logger=none` turns it off). `training/run.py`
sets `WANDB_MODE=offline`, so nothing leaves the machine: the run files are written under the run directory
(`OMOL_RUN_DIR`, on Ocean for cluster runs) and the W&B cache and config go under `OMOL_CACHE_DIR`. Upload a run later from
a machine with internet and a W&B login:

```bash
wandb sync $OMOL_RUN_DIR/<run>/logs/wandb/offline-run-*
```

`WANDB_ENTITY` and `WANDB_PROJECT` override the defaults (`anshulk-cmu`, `mlip-local-routing`). Set `WANDB_MODE=online` to
log live. The run name is `job.run_name`.

## GPU requirement

| Variant | GPU on Bridges-2 | Why |
|---|---|---|
| Dense, Global | `v100-32` (V100, 32 GB) or larger | measured peak 9.5 GiB (Dense) and 10.5 GiB (Global) at 450 atoms per batch |
| Local-6 | `h100-80` (H100, 80 GB, 2 SU per GPU-hour), chosen by the team on 2026-10-11 for memory headroom; `l40s-48` also works | measured peak 26.3 GiB at 450 atoms in the smoke test, 27.6 GiB in the 30-step pilot test and 37.3 GiB in a longer run on an L40S (46 GiB), so the L40S had about 9 GiB free; the H100 has about 40 GiB more. The H100 has not run this environment yet |

Peaks are from the 20-step smoke test (`nvidia-smi`, so they include the allocator cache), run with the settings in the next
section (expandable segments, cache cleared every 50 steps). Without them the same runs showed 19.5 GiB (Dense) and 20.0 GiB
(Global); Local-6 stayed at 26.3 GiB. Memory is not linear in the
batch: a larger limit also admits larger molecules with many more edges. A 16 GB GPU (`v100-16`) fits none of the three
at 450 atoms, and the 12 GB laptop GPU ran out of memory on Local-6. `training/run.py` prints a warning when the GPU has
less than 31 GiB.

## Out-of-memory monitor

Every variant trains with `training.guarded_unit.GuardedTrainEvalUnit` (`train_eval_unit` in `configs/train.yaml`). It is a monitor and a
cleanup, not a filter. **No batch is ever skipped.**

- A batch that runs out of GPU memory logs a warning (`OOM: train batch (N atoms, E edges) ran out of memory, event k; cleaned up,
  attempt a of 3. GPU memory: ...`). The unit then clears the gradients, collects Python garbage, empties the CUDA cache and runs the
  same batch again. A batch that runs after a retry counts as recovered. The cleanup runs outside the `except` block, because the
  traceback of an error holds the tensors of the failed step.
- `max_oom_retries: 2`: if a batch still fails after two retries, the run stops with a message that says the batch "does not fit this
  GPU", so a real memory problem is seen and not hidden. The last checkpoint stays on disk. Validation batches are treated the same way.
- At the end the log says `OOM events: N train (R recovered), M eval (R recovered); no batch was skipped`, and W&B gets `train/oom_events`.
  The pilot `summary.txt` shows the first events and this line. Any event is worth a look; nothing blocks the run unless a batch fails
  after its retries.
- `gc_every: 50` collects Python garbage and empties the CUDA cache every 50 steps.
- `training/run.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation (override by setting it).
- Single process only.
- Drill: `OMOL_GPU_MEMORY_FRACTION=0.35` caps the GPU memory of a run so some batches cannot fit. The smoke job (`slurm/smoke_train.sh`)
  runs it for every variant and checks that every event is cleaned up and retried, and that a batch that still does not fit stops the run
  with "does not fit this GPU". The drill therefore may end with a failed run by design.

## Fixed sample lists

All pilots read the same rows, not a random draw. `manifests/pilot_train_200k.txt` (200,000 rows of `train_4M`) and
`manifests/pilot_val_5k.txt` (5,000 rows of `val`) hold one row number per line, sorted. `manifests/pilot_samples.json` has their
sha256, counts and the seed of the draw (`scripts/make_pilot_manifests.py`). The lists are the source of truth, not the script.

A run reads a list through fairchem's own `subset_to` hook: `sample_n=null`, `no_shuffle=true`, `metadata_path` set to a copy of
`metadata.npz` with an extra field `index`, and `subset_to=[{metadata_key: index, op: in, rhv: <list file>}]`
(`slurm/pilot_data_overrides.sh`). The dataset then holds the listed rows in ascending order, and the batch sampler shuffles them
with `job.seed`. `scripts/check_pilot_data.py` has three commands:

- `prep` writes the two index files under `$OMOL_DATA_ROOT/pilot/`.
- `verify` composes the training config with these overrides, builds both datasets and checks that they hold exactly the listed rows and
  that the lists match their hashes. Every pilot job runs it before training (`verify_samples` in `summary.txt`) and stops if it fails.
- `stats` measures a list: neighbours within 6 A, force and energy scales with the official element references, elements, charges, spins
  and structure ids.

Tested on 2026-10-11 on the 27,697 structures of the `neutral_val` subset (WSL, CPU): `verify` passes, and with one row removed from a
list it fails on the hash and the count.

Measured on the lists by the prep job (`slurm/prep_pilot_data.job`, job 49039106, 12 min 42 s on 16 cores; results in
`/ocean/projects/che260063p/shared/data/omol25/pilot/stats_*.json`, structure ids in `sids_*.txt`):

| | 200,000 training rows | 5,000 validation rows |
|---|---:|---:|
| Atoms per structure (mean, maximum) | 55.1, 350 | 102.1, 349 |
| Neighbours within 6 Å per atom (mean, maximum) | 32.6, 162 | 35.9, 164 |
| Atoms with more than 30 / 50 / 100 / 300 neighbours | 50.5 / 12.5 / 0.08 / 0 percent | 57.4 / 17.9 / 0.07 / 0 percent |
| Force norm RMS per atom, eV/A (component RMS) | 2.12 (1.23) | 1.54 (0.89) |
| Energy residual per atom after the official references, eV (mean, RMS) | 0.136, 0.445 | 0.046, 0.251 |
| Elements | 83, all with Z up to 83 | 82, all with Z up to 83 |
| Neutral structures, singlets | 56 percent, 83 percent | 43 percent, 91 percent |

The validation rows are larger structures than the training rows on average (102 against 55 atoms).

## Pilot jobs on Bridges-2

The team agreed on 2026-10-11, and changed the length on 2026-10-11 (later the same day): **50,000 steps**, 450 atoms per step, **three
seeds** (0, 1, 2) per model, the fixed lists above, and a checkpoint and a validation every 5,000 steps (the default, steps/10). One job
file per model in `slurm/`. Each is a Slurm array of three tasks, and the array task id is the seed (`#SBATCH --array=0-2`), so one
`sbatch` starts all three seeds. All call `slurm/pilot.sh`, which checks the lists, trains the variant on the real OMol25 data, then
checks the size consistency of the final checkpoint. Every step checkpoint is kept (`training.max_saved_checkpoints`), so the size
consistency can be measured along the training. Every job has the Bridges-2 maximum time limit of 48 h.

| Pilot | Job file | GPU | Time limit per task | Who |
|---|---|---|---|---|
| Dense | `pilot_dense.job` | `v100-32` | 48 h | Yiwei |
| Global | `pilot_global.job` | `v100-32` | 48 h | Anshul |
| Local-6 | `pilot_local6.job` | `h100-80` | 48 h | Anshul submits (Pragati can too) |

**Order.** First run `prep_pilot_data.job` once (CPU, `RM-shared`, 16 cores, 2 h limit). It writes the index files and checks the lists.
Then submit the pilots from a login node (project `che260063p`, 1 SU per GPU-hour on a V100 or an L40S, 2 SU on an H100):

```bash
cd /ocean/projects/che260063p/shared/repo/AI4Science-MLIPs/slurm
sbatch prep_pilot_data.job        # once, before the first pilot
sbatch pilot_dense.job            # or pilot_global.job, pilot_local6.job; three tasks, seeds 0 to 2
squeue --me                       # state of your jobs
```

Limits are per user, not per project: the `gpu` QOS allows 128 GPUs and 5,000 submitted jobs per user, and no per-project limit is set
(`sacctmgr show qos` and `sacctmgr show assoc`, 2026-10-11). Jobs of other projects count toward the same per-user limits, but they are
charged to their own project. The real limit is free GPUs. Bridges-2 has 24 L40S and 80 H100 GPUs in total for all users, so the three
Local-6 seeds may start one after the other. Use `-A che260063p` so the hours are charged to this project.

The job emails Anshul by default; add `--mail-user=you@andrew.cmu.edu` to `sbatch` to get the emails yourself. Output goes to
`/ocean/projects/che260063p/shared/logs/slurm-<arrayjob>_<task>.out` and everything else to
`/ocean/projects/che260063p/shared/runs/pilot_<variant>_s<seed>_<jobid>/`: `summary.txt` (one line per stage, peak GPU memory, last loss,
validation numbers, OOM events, the size-consistency Delta), `settings.txt`, `logs/` (the full output of every stage) and
`train/` (checkpoints and the offline W&B run).

**Settings.** `PILOT_STEPS` is 50000 unless set when submitting (see the test below). The seed comes from the array task id.
`PILOT_EVAL_EVERY` and `PILOT_CKPT_EVERY` default to steps/10, which is 5,000. Set `PILOT_TRAIN_SAMPLES` and `PILOT_VAL_SAMPLES` only for a
short test with a random sample. Measured at 450 atoms per batch (30-step tests and the first 2,000 steps of a longer run): Dense 735,
Global 525 and Local-6 359 atoms per second on an L40S, about 0.61, 0.86 and 1.25 s per step. The H100 speed is not measured. Each
validation reads 5,000 structures and takes at most about 6 min (Dense), 6.6 min (Global) and 9.8 min (Local-6 on an L40S). Estimate for
one seed of 50,000 steps with 11 validations: about 9.6 h (Dense), 13.1 h (Global) and 19.2 h (Local-6 at L40S speed; the H100 may be
faster). The cost of the three seeds is then about 29 SU (Dense), 39 SU (Global) and about 115 SU (Local-6 at 2 SU per hour), about 183 SU
of the 2,948. A limit is a cap, and only the hours used are charged: at the 48 h cap the worst case is 144, 144 and 288 SU, 576 SU in total.
Each seed keeps 11 step checkpoints and the final one, about 865 MB each for Global and Local-6 and 140 MB for Dense, about 70 GB for all
nine runs.
`pilot.sh` refuses a GPU that is too small (Dense and Global need 31 GB or more, Local-6 44 GB or more, so an H100 or an L40S).

**Test before the long runs.** One 30-step run per model on the real job files, the real fixed lists and seed 0 (about 3 SU in total; the
Local-6 test runs on an H100, which has not run this environment yet, so run it first).
The switches set the steps, one validation at the end (the last validation runs twice, see "Validation and checkpoints"), a checkpoint
every 15 steps, a different run folder name, one array task and a short time limit:

```bash
cd /ocean/projects/che260063p/shared/repo/AI4Science-MLIPs/slurm
sbatch --export=PILOT_STEPS=30,PILOT_EVAL_EVERY=30,PILOT_CKPT_EVERY=15,PILOT_RUN_PREFIX=pilottest --array=0 -t 2:00:00 pilot_dense.job
# the same for pilot_global.job and pilot_local6.job
```

Pass the settings with `--export`. Bridges-2 sets `SBATCH_EXPORT=NONE` in the login shell, so a variable exported before `sbatch` does not
reach the job. This went wrong on 2026-10-11: the first "test" ran as a full 20,000-step job. Check `settings.txt` in the run folder
(`steps=30`) before you wait for the result.

Check in `runs/pilottest_<variant>_s0_<jobid>/summary.txt`: `verify_samples` all PASS, no OOM event (or only recovered ones), a final checkpoint, the
size-consistency Delta, and the time of one validation (it sets the real time limits).

**Tested on 2026-10-09** (before the move to 64 basis functions, and with a random sample) with 30 steps, 2,000 training and 200 validation structures (jobs 48977565, 48977567, 48977587), all stages
passed: Dense 8 min 26 s, peak 10.5 GiB, Delta 3.7e-06 meV; Global 8 min 41 s, 11.4 GiB, Delta -31.4 meV; Local-6 12 min 20 s on an
L40S, 29.2 GiB, Delta 2.8e-06 meV. These are pipeline checks, not model quality. Most of the time of these short runs went into the
evaluations (every 3 steps) and the first import of torch from Ocean.

With `num_distance_basis: 64` the same 30-step test (jobs 48978344, 48978349 and 48978351, 2026-10-09) gave: Dense 7 min 34 s, peak
10.3 GiB, 638 atoms per second, Delta 2.8e-06 meV; Global 7 min 52 s, 11.3 GiB, 679 atoms per second, Delta -6.6 meV; Local-6 12 min 7 s on an
L40S, 28.9 GiB, 310 atoms per second, Delta 3.7e-06 meV. No batch was skipped. The new pilot scripts (array, fixed lists) are checked by
`tests/test_training_config.py` and by the list check above, and have not run a job yet.
