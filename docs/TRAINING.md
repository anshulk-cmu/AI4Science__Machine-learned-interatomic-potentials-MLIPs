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
| `train.yaml` | `training.max_atoms` | `320` atoms per batch; larger structures are dropped by the sampler. |
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
- The W&B logger is off (`job.logger: null`).
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
