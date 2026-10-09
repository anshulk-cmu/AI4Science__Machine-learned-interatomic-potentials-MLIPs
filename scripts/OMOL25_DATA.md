# OMol25 data preparation for Dense

Status: downloaded and verified on 2026-10-07. The reproducible audit result and fixed audit sample list are in `manifests/omol25_data_manifest.json`.

This repository's `requirements.txt` pins `fairchem-core==2.21.0`. The adjacent `fairchem` source checkout is at commit `db9b17fc6eff8bd167fc2174fddc8f43bab34629` (tag `fairchem_core-2.23.0`), so running Python from that checkout would not match the pinned package. Run the audit in the pinned package environment; its version check rejects a mismatch. The [2.21.0 source](https://github.com/facebookresearch/fairchem/blob/fairchem_core-2.21.0/src/fairchem/core/datasets/atomic_data.py) confirms `AtomicData.from_ase` uses `atoms.info.get("charge", 0)` and `atoms.info.get("spin", 0)` when those keys are requested.

## Source and scope

Use the [official OMol25 repository](https://huggingface.co/facebook/OMol25) `DATASET.md` at revision `f3fdf382be3c22e8c67d1d76bacb1961627cf8d8`. Its OMol-0 table links [train_4M.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz) and [val.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz) on Meta's public file server; the archives are **not** files in the Hugging Face repository. The [publisher's paper](https://arxiv.org/abs/2505.08762) gives 3,986,754 structures for the 4M train split and 2,762,021 for the composition-held-out validation split. The official table reports about 19 GB and 20 GB compressed, respectively. Downloaded sizes were 19,983,081,456 bytes for `train_4M.tar.gz` and 21,293,980,744 bytes for `val.tar.gz`. Their SHA-256 values are `1924f4f50128344cef731069b409757192a83eacdb47e0d0169efc3138ff9688` and `bc2f2aa70459ed029217f188d1e8ab0d90abbb90065235f6414d79246bad1ab8`, respectively. The data are ASE DB LMDB (`.aselmdb`). [FairChem documentation](https://facebookresearch.github.io/fairchem/omol25/) gives total energy in eV, forces in eV/Å, and total charge and spin multiplicity in `atoms.info`.

The `facebook/OMol25` Hugging Face repository is a **model** repository, even though it documents data. Do not substitute the ColabFit Parquet conversion or a third-party mirror: this audit uses the official ASE DB files and official split boundary. Do not download `train.tar.gz`, `val_neutral.tar.gz`, `test.tar.gz`, `checkpoints/`, or the Argonne electronic-structure / ORCA files.

## Access and disk

1. Sign in at <https://huggingface.co/facebook/OMol25> and use its access form to accept the conditions. The page requires a full legal name, date of birth, and full organization name; its form says submissions cannot be edited. Wait for access approval if needed. Do not place a token in this repository or in command arguments.
2. The local Windows environment is `D:\11785 final project 2\omol25-env` (Python 3.12.10, `fairchem-core==2.21.0`, CUDA PyTorch 2.8.0). Run its `Scripts\hf.exe auth login` interactively. Credentials stay in the Hugging Face credential store outside both Git repositories.
3. Check free space before download. On 2026-10-07, D: had 1,887,190,032,384 free bytes before starting the archives. Recheck because archive and extraction coexist. The command below refuses to proceed if free space is under 100 GiB; this is an operational floor, not an official unpacked-size estimate.

```powershell
$dataRoot = 'D:\11785 final project 2\omol25-data'
$free = ([System.IO.DriveInfo]::GetDrives() | Where-Object Name -EQ 'D:\').AvailableFreeSpace
if ($free -lt 100GB) { throw "D: free space below 100 GiB: $free bytes" }
New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
```

After access, record the repository commit SHA from the official page's Files history or `HfApi().model_info('facebook/OMol25').sha`. The revision pins the download instructions; the archive URLs are independent public-file URLs. Download only these two archives, with resume support:

```powershell
$revision = 'f3fdf382be3c22e8c67d1d76bacb1961627cf8d8'
curl.exe --location --fail --retry 5 --continue-at - --output "$dataRoot\train_4M.tar.gz" 'https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz'
curl.exe --location --fail --retry 5 --continue-at - --output "$dataRoot\val.tar.gz" 'https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz'
tar -xzf "$dataRoot\train_4M.tar.gz" -C $dataRoot
tar -xzf "$dataRoot\val.tar.gz" -C $dataRoot
```

Inspect the extracted directory names before auditing. The audit expects `$dataRoot\train_4M` and `$dataRoot\val`, each containing `.aselmdb` files. Leave archives and extracted files under `$dataRoot`, which is outside `fairchem` and this repository.

## Real-data audit

Run from the root of this repository in the pinned environment. The two cache variables keep FairChem and Warp caches outside the Git repositories:

```powershell
$env:FAIRCHEM_CACHE_DIR = 'D:\11785 final project 2\omol25-cache\fairchem'
$env:WARP_CACHE_PATH = 'D:\11785 final project 2\omol25-cache\warp'
& 'D:\11785 final project 2\omol25-env\Scripts\python.exe' scripts/audit_omol25.py --data-root 'D:\11785 final project 2\omol25-data' --source-revision $revision --output manifests/omol25_data_manifest.json
```

The script uses `AseDBDataset` for both official splits and checks the full split row counts. With seed 11785, it selects 64 fixed indices per split (validation uses seed 11786). Each raw ASE `Atoms` object and converted FairChem sample must agree on atomic numbers, relative positions, energy, forces, total charge, and spin multiplicity. FairChem's `molecule_cell_size=120` calls `atoms.center(vacuum=120)`, which adds 120 Å of vacuum on each side of the molecule, so absolute coordinates differ by a uniform translation. Atomic numbers must lie in 1..118 and below 100, the model's `max_num_elements`. It checks source IDs, atom counts, numeric shape, and NaN/Inf. The cross-split ID check is weak: it only confirms that the sampled train and val IDs do not overlap, not that the full splits are disjoint. Both archives must be present in the data root; pass `--no-archives` to audit the extracted splits without them. It rejects missing raw charge or spin because the pinned FairChem `AtomicData.from_ase` silently uses 0 for absent values. It writes each `.aselmdb` SHA-256, archive SHA-256, actual byte sizes, sample indices and IDs, metadata key names, and summary ranges. It does not store coordinates, energy, or forces in Git. A failure leaves the previous manifest intact.

This is a fixed **audit** sample, not a complete statistical audit of every row. Unit labels come from the publisher's documentation; the script checks values and reader consistency, not the underlying DFT unit conversion, force sign, energy-reference convention, or domain meaning. These semantic questions need separate source or calculation checks before training.
