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
| Dense | 6,267,777 | none | none | 6,267,777 |
| Global | 37,938,985 | 36,179,968 (24 MOLE layers) | 17,320 | 6,285,097 |

The active count keeps one of the 8 expert weight slices in every MOLE layer, plus every
parameter outside the expert weights (the MOLE biases are shared, not per expert). Both
variants already meet the 6M base budget. Both use `num_distance_basis: 512`, the
`eSCNMDBackbone` class default. The released UMA configs use 64, which gives 5,981,057 for
Dense. It is left at 512 because changing it is a team decision.

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
| `train.yaml` | `training.max_neighbors` | `300` nearest neighbours kept per atom inside the cutoff (`a2g_args.max_neigh`, both splits). |
| `train.yaml` | `training.cutoff` | `6.0` Å graph/model cutoff. |
| `train.yaml` | `training.normalizer_rmsd` | shared energy and force normalizer. |
| `train.yaml` | `job.run_name` | set a distinct name for each run. |
| `train.yaml` | `job.run_dir` | output directory; env `OMOL_RUN_DIR`, default `../omol25-runs`. |
| `data/generic.yaml` | `data.root` | env `OMOL_DATA_ROOT`, default `../omol25-data`; holds `train_4M` and `val`. |
| `data/generic.yaml` | `data.train_src`, `data.val_src` | ASE DB split directories. |
| `data/generic.yaml` | `train_dataset.dataset_configs.omol.splits.train.sample_n` | number of samples; set `null` to use all training rows. |
| `data/generic.yaml` | `val_dataset.dataset_configs.omol.splits.val.sample_n` | number of samples; set `null` to use all validation rows. |

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
  checkpoints are kept.
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

- No Local router variant yet (see below).
- The normalizer `rmsd` is the allscaip value, not refit on `train_4M`.
- Energy and force loss weights are 1:1 (coefficient 1.0 each) and have not been tuned.
- `training.workers` is 0, so data loading runs in the training process.
- `max_neighbors: 300` is the `eSCNMDBackbone` default. Whether any OMol atom has more than 300
  neighbours within 6 Å, which would truncate its edges, has not been measured.
- The ids of the structures drawn by `sample_n` in pilot runs are not recorded.

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
parameters as Global (37,938,985) and differs only in the router: each atom averages the element embeddings of its
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
| Local-6 | `l40s-48` (L40S, 48 GB) | measured peak 26.3 GiB at 450 atoms (on a V100-32 and on an L40S), which leaves about 18 GiB free on the L40S and only about 5 GiB on a V100-32 |

Peaks are from the 20-step smoke test (`nvidia-smi`, so they include the allocator cache), run with the settings in the next
section (expandable segments, cache cleared every 50 steps). Without them the same runs showed 19.5 GiB (Dense) and 20.0 GiB
(Global); Local-6 stayed at 26.3 GiB. Memory is not linear in the
batch: a larger limit also admits larger molecules with many more edges. A 16 GB GPU (`v100-16`) fits none of the three
at 450 atoms, and the 12 GB laptop GPU ran out of memory on Local-6. `training/run.py` prints a warning when the GPU has
less than 31 GiB.

## Out-of-memory protection

Every variant trains with `training.guarded_unit.GuardedTrainEvalUnit` (`train_eval_unit` in `configs/train.yaml`):

- A batch that runs out of GPU memory is skipped and counted (`OOM: skipped train batch ...` in the log, with its atoms
  and edges), gradients are cleared, the cache is emptied and the learning-rate schedule still advances. A skipped
  validation batch is counted too. At the end the log says `OOM-skipped batches: N train, M eval`, and W&B gets
  `train/oom_skipped_batches`.
- `max_consecutive_oom: 5` stops the run with a clear message after five skipped batches in a row, because the batch is
  then too big for the GPU (lower `training.max_atoms` or use a larger GPU).
- `gc_every: 50` collects Python garbage and empties the CUDA cache every 50 steps.
- `training/run.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation (override by setting it).
- Single process only: skipping a batch on one rank of several would desynchronize the others.
- Drill: `OMOL_GPU_MEMORY_FRACTION=0.35` caps the GPU memory of a run so some batches cannot fit. The smoke job
  (`slurm/smoke_train.sh`) runs it for every variant and checks that the run skips batches and still finishes.

## Pilot jobs on Bridges-2

One job file per pilot in `slurm/`; each person submits one. All three call `slurm/pilot.sh`, which trains the variant on the real
OMol25 data, then checks the size consistency of the final checkpoint.

| Pilot | Job file | GPU | Time limit | Who |
|---|---|---|---|---|
| Dense | `pilot_dense.job` | `v100-32` | 6 h | Yiwei |
| Global | `pilot_global.job` | `v100-32` | 6 h | Anshul |
| Local-6 | `pilot_local6.job` | `l40s-48` | 10 h | Pragati |

Submit from a login node (project `che260063p`, 1 SU per GPU-hour):

```bash
cd /ocean/projects/che260063p/shared/repo/AI4Science-MLIPs/slurm
sbatch pilot_dense.job            # or pilot_global.job, pilot_local6.job
squeue --me                       # state of your jobs
```

The job emails Anshul by default; add `--mail-user=you@andrew.cmu.edu` to `sbatch` to get the emails yourself. Output goes to
`/ocean/projects/che260063p/shared/logs/slurm-<jobid>.out` and everything else to
`/ocean/projects/che260063p/shared/runs/pilot_<variant>_<jobid>/`: `summary.txt` (one line per stage, peak GPU memory, last loss,
validation numbers, skipped OOM batches, the size-consistency Delta), `settings.txt`, `logs/` (the full output of every stage) and
`train/` (checkpoints and the offline W&B run).

**Settings.** The four lines `PILOT_STEPS`, `PILOT_SEED`, `PILOT_TRAIN_SAMPLES` and `PILOT_VAL_SAMPLES` near the end of each job
file (10000, 0, 200000, 5000) are placeholders until the team agrees; the three job files must carry the same values. If you raise the
steps, raise the time limit (`#SBATCH -t`). Measured at 450 atoms per batch in the 30-step tests: Dense 651, Global 673 and Local-6
333 atoms per second, about 0.69, 0.67 and 1.35 s per step, so 10,000 steps take about 2, 2 and 4 hours plus the evaluations.
`pilot.sh` refuses a GPU that is too small (Dense and Global need 31 GB or more, Local-6 44 GB or more).

**Tested on 2026-10-09** with 30 steps, 2,000 training and 200 validation structures (jobs 48977565, 48977567, 48977587), all stages
passed: Dense 8 min 26 s, peak 10.5 GiB, Delta 3.7e-06 meV; Global 8 min 41 s, 11.4 GiB, Delta -31.4 meV; Local-6 12 min 20 s on an
L40S, 29.2 GiB, Delta 2.8e-06 meV. These are pipeline checks, not model quality. Most of the time of these short runs went into the
evaluations (every 3 steps) and the first import of torch from Ocean.

