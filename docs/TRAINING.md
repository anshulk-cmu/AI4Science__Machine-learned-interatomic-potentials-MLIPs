# Training

## Environment

Use Python 3.12, `fairchem-core==2.21.0`, and the
dependencies in `requirements.txt` and `requirements-training.txt`.
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-training.txt
python -m pip check
python -c "import torch, fairchem.core; print(torch.__version__, torch.cuda.is_available())"
```
## Config

| File | What it contains |
| --- | --- |
| [`configs/train.yaml`](../configs/train.yaml) | Selects the data/model variants and defines the job, OMol energy/force tasks, optimizer, schedule, validation, and checkpoints. |
| [`configs/data/generic.yaml`](../configs/data/generic.yaml) | Loads OMol ASE DB train/val splits with FairChem's native dataset, sampler, and collater. |
| [`configs/variant/dense.yaml`](../configs/variant/dense.yaml) | Uses `eSCNMDBackbone`, 4 layers, and no experts. |
| [`configs/variant/global.yaml`](../configs/variant/global.yaml) | Uses `eSCNMDMoeBackbone`, 4 layers, 8 experts, and FairChem's global composition router. |

Model sizes for the current configs:

| Variant | Total parameters | Active parameters | Trainable parameters |
| --- | ---: | ---: | ---: |
| Dense | 6,267,777 | 6,267,777 | 6,267,521 |
| Global | 37,938,985 | 37,938,985 | 37,938,729 |


Common runtime overrides:

| Config | Variable | Purpose |
| --- | --- | --- |
| `train.yaml` | `variant` | select `global/dense` or another model variant. |
| `train.yaml` | `training.steps` | validation and checkpoint intervals also use this value. |
| `train.yaml` | `training.max_atoms` | now`320` atoms per batch; larger structures are skipped. |
| `train.yaml` | `training.cutoff` | `6.0` Å graph/model cutoff. |
| `train.yaml` | `job.run_name` | set a distinct name for each run. |
| `train.yaml` | `job.run_dir` | output directory. |
| `data/generic.yaml` | `data.train_src`, `data.val_src` | ASE DB split directories. |
| `data/generic.yaml` | `train_dataset.dataset_configs.omol.splits.train.sample_n` | number of samples; set `null` to use all training rows. |
| `data/generic.yaml` | `val_dataset.dataset_configs.omol.splits.val.sample_n` | number of samples; set `null` to use all validation rows. |

The split directories should contain their `.aselmdb` shards and matching
`metadata.npz`. The current tasks and label mapping are specific to OMol.

## Run Training

In PowerShell, choose a variant and a distinct run name:

```powershell
python -m training.run --mode train variant=dense job.run_name=omol_dense_check
python -m training.run --mode train variant=global job.run_name=omol_global_check
```

Larger Global run
```powershell
python -m training.run --mode train variant=global `
  job.run_name=omol_global_full training.steps=10000 `
  train_dataset.dataset_configs.omol.splits.train.sample_n=null `
  val_dataset.dataset_configs.omol.splits.val.sample_n=5000
```



## Add a Local Router

- Implement the router as a Python `torch.nn.Module` (for example in
`training/routers.py`) and connect it through a custom backbone class (for
example `training/local_moe.py`). 
- Copy `configs/variant/global.yaml` to a new
variant, change `backbone.model` to the custom class's import path, and add
its router arguments there. 
- Run with `variant=<new_variant>`; the shared train
and data configs can stay the same. 
