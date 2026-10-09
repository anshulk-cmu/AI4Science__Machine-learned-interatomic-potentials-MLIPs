# OMol25 data preparation for Dense

Status: downloaded and verified on 2026-10-07 (Windows) and again on Bridges-2 on 2026-10-08, where the archives match the same hashes. The reproducible audit result and fixed audit sample list are in `manifests/omol25_data_manifest.json`.

This repository's `requirements.txt` pins `fairchem-core==2.21.0`. The adjacent `fairchem` source checkout is at commit `db9b17fc6eff8bd167fc2174fddc8f43bab34629` (tag `fairchem_core-2.23.0`), so running Python from that checkout would not match the pinned package. Run the audit in the pinned package environment; its version check rejects a mismatch. The [2.21.0 source](https://github.com/facebookresearch/fairchem/blob/fairchem_core-2.21.0/src/fairchem/core/datasets/atomic_data.py) confirms `AtomicData.from_ase` uses `atoms.info.get("charge", 0)` and `atoms.info.get("spin", 0)` when those keys are requested.

## Source and scope

Use the [official OMol25 repository](https://huggingface.co/facebook/OMol25) `DATASET.md` at revision `f3fdf382be3c22e8c67d1d76bacb1961627cf8d8`. Its OMol-0 table links [train_4M.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz) and [val.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz) on Meta's public file server; the archives are **not** files in the Hugging Face repository. The [publisher's paper](https://arxiv.org/abs/2505.08762) gives 3,986,754 structures for the 4M train split and 2,762,021 for the composition-held-out validation split. The official table reports about 19 GB and 20 GB compressed, respectively. Downloaded sizes were 19,983,081,456 bytes for `train_4M.tar.gz` and 21,293,980,744 bytes for `val.tar.gz`. Their SHA-256 values are `1924f4f50128344cef731069b409757192a83eacdb47e0d0169efc3138ff9688` and `bc2f2aa70459ed029217f188d1e8ab0d90abbb90065235f6414d79246bad1ab8`, respectively. The data are ASE DB LMDB (`.aselmdb`). [FairChem documentation](https://facebookresearch.github.io/fairchem/omol25/) gives total energy in eV, forces in eV/Å, and total charge and spin multiplicity in `atoms.info`.

The `facebook/OMol25` Hugging Face repository is a **model** repository, even though it documents data. Do not substitute the ColabFit Parquet conversion or a third-party mirror: this audit uses the official ASE DB files and official split boundary. Do not download `train.tar.gz`, `val_neutral.tar.gz`, `test.tar.gz`, `checkpoints/`, or the Argonne electronic-structure / ORCA files.

## Access and disk

1. The two archives come from Meta's public file server, so no Hugging Face login is needed for the data. A Hugging Face account with access to `facebook/OMol25` is only needed for the checkpoints and the documentation page; request it at <https://huggingface.co/facebook/OMol25> (manual approval). Never put a token in this repository or in command arguments.
2. On Bridges-2 the data live in the shared project folder `/ocean/projects/che260063p/shared/data/omol25`, with `train_4M/` and `val/` (80 `.aselmdb` shards each plus `metadata.npz`). Set `OMOL_DATA_ROOT=/ocean/projects/che260063p/shared/data/omol25` for training.
3. To create them again, run the job `slurm/get_data.job` (script `slurm/get_data.sh`). It downloads both archives with resume support, checks their sizes and SHA-256 values against `manifests/omol25_data_manifest.json`, and unpacks them. Never download or unpack on a login node. Submit it from a machine with the Bridges-2 skill: `b2.py submit slurm/get_data.job /ocean/projects/che260063p/shared/slurm --yes`. Measured on 2026-10-08: 33 min 38 s on a V100-32 job, both archives matched, `train_4M` unpacked to 22 GB and `val` to 23 GB. The data and the environment use about 106 GB of the 3,715 GB Ocean quota.
4. Check the quota with `my_quotas` before a download: going over the storage quota blocks every job submission. The archives (about 41 GB) and the unpacked data (about 45 GB) coexist.

## Real-data audit

Run it inside a Slurm job on Bridges-2, from the repository folder, with the shared environment:

```bash
export PYTHONNOUSERSITE=1
export FAIRCHEM_CACHE_DIR=/ocean/projects/che260063p/shared/cache/fairchem WARP_CACHE_PATH=/ocean/projects/che260063p/shared/cache/warp
cd /ocean/projects/che260063p/shared/repo/AI4Science-MLIPs
/ocean/projects/che260063p/shared/envs/mlip/bin/python scripts/audit_omol25.py     --data-root /ocean/projects/che260063p/shared/data/omol25     --source-revision f3fdf382be3c22e8c67d1d76bacb1961627cf8d8     --output /ocean/projects/che260063p/shared/runs/omol25_manifest.json
```

The committed `manifests/omol25_data_manifest.json` was written on Yiwei's Windows machine on 2026-10-07, so its `path` fields hold Windows paths. The Bridges-2 copy is byte-identical, because `get_data.sh` checked both archives against that manifest's SHA-256 values. Rerunning the audit on Bridges-2 is optional. It hashes all 160 shards, which takes tens of minutes (hashes are cached in `sha256-cache.json` in the data root). Write it to a new file, as above, and do not overwrite the committed manifest.

The script uses `AseDBDataset` for both official splits and checks the full split row counts. With seed 11785, it selects 64 fixed indices per split (validation uses seed 11786). Each raw ASE `Atoms` object and converted FairChem sample must agree on atomic numbers, relative positions, energy, forces, total charge, and spin multiplicity. FairChem's `molecule_cell_size=120` calls `atoms.center(vacuum=120)`, which adds 120 Å of vacuum on each side of the molecule, so absolute coordinates differ by a uniform translation. Atomic numbers must lie in 1..118 and below 100, the model's `max_num_elements`. It checks source IDs, atom counts, numeric shape, and NaN/Inf. The cross-split ID check is weak: it only confirms that the sampled train and val IDs do not overlap, not that the full splits are disjoint. Both archives must be present in the data root; pass `--no-archives` to audit the extracted splits without them. It rejects missing raw charge or spin because the pinned FairChem `AtomicData.from_ase` silently uses 0 for absent values. It writes each `.aselmdb` SHA-256, archive SHA-256, actual byte sizes, sample indices and IDs, metadata key names, and summary ranges. It does not store coordinates, energy, or forces in Git. A failure leaves the previous manifest intact.

This is a fixed **audit** sample, not a complete statistical audit of every row. Unit labels come from the publisher's documentation; the script checks values and reader consistency, not the underlying DFT unit conversion, force sign, energy-reference convention, or domain meaning. These semantic questions need separate source or calculation checks before training.
