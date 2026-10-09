# Audit code

Run everything from this folder (`audit/`). Results land in `../results/`.

## Environments

- `IDLF26-mlip` (UMA, ORB, analysis) and `IDLF26-mlip-mace` (MACE). Setup steps are in `../requirements.txt` and
  `../requirements-mace.txt`.
- UMA weights are gated on Hugging Face. Log in and accept the license first. Never put a token in a file.

## Commands

```powershell
$py1 = "C:\Users\<you>\anaconda3\envs\IDLF26-mlip\python.exe"
$py2 = "C:\Users\<you>\anaconda3\envs\IDLF26-mlip-mace\python.exe"
cd audit

& $py1 geometry.py                                               # geometry self-test
& $py1 preflight.py                                              # router mean check, self-test

# MACE first: it must give zero, or the pipeline is wrong
& $py2 nulls.py     --family mace --model medium
& $py2 run_audit.py --family mace --model medium
& $py1 nulls.py     --family orb  --model orb-v3
& $py1 run_audit.py --family orb  --model orb-v3
foreach ($m in "uma-s-1p1", "uma-s-1p2") {
  foreach ($t in "omat", "omol") {
    & $py1 nulls.py     --family uma --model $m --task $t
    & $py1 run_audit.py --family uma --model $m --task $t
  }
}
# UMA-M needs a 24 GB GPU and was run in float32 on a rented machine, on both tasks:
#   nulls.py / run_audit.py --family uma --model uma-m-1p1 --task omat --dtype float32
#   nulls.py / run_audit.py --family uma --model uma-m-1p1 --task omol --dtype float32

# router version test (5 shaken runs), both directions and both tasks:
foreach ($t in "omat", "omol") {
  & $py1 nulls.py     --family uma --model uma-s-1p2 --task $t --model-version 1.1
  & $py1 run_audit.py --family uma --model uma-s-1p2 --task $t --model-version 1.1 --runs 5
  & $py1 nulls.py     --family uma --model uma-s-1p1 --task $t --model-version 1.0
  & $py1 run_audit.py --family uma --model uma-s-1p1 --task $t --model-version 1.0 --runs 5
}

& $py1 analyze.py       # statistics, decision rule, plots
& $py1 report.py        # results.md and the overview plot

# neon probe (1 ideal + 20 shaken runs)
& $py2 neon_probe.py --family mace --model medium
& $py1 neon_probe.py --family orb  --model orb-v3
foreach ($m in "uma-s-1p1", "uma-s-1p2") {
  foreach ($t in "omat", "omol") { & $py1 neon_probe.py --family uma --model $m --task $t }
}

# size-consistency check of trained checkpoints (the pilot jobs call this)
& $py1 check_checkpoint.py <path to inference_ckpt.pt>

# forcing test F1, UMA only; tau comes from the null tests of the same label
& $py1 f1_forcing.py --model uma-s-1p1 --task omat
& $py1 f1_forcing.py --model uma-s-1p2 --task omat
& $py1 f1_forcing.py --model uma-s-1p1 --task omat --model-version 1.0 --runs 2
& $py1 f1_forcing.py --model uma-s-1p2 --task omat --model-version 1.1 --runs 2
& $py1 f1_forcing.py --model uma-s-1p2 --task omat --no-balance --label uma-s-1p2-nobal --tau-label uma-s-1p2 --runs 2

# upload the tables, plots and result files to W&B (needs a W&B login)
& $py1 log_wandb.py
```

## Notes

- Run one job at a time on a laptop GPU.
- `run_audit.py` and `neon_probe.py` resume after a crash. A run counts as done only when all its rows are saved (18
  per run for the audit, 2 for the neon probe); the rows of an unfinished run are dropped and the run is repeated.
- `f1_forcing.py --no-balance` changes the model, so the Delta it gives is not a result. It switches off only the
  charge balancing and leaves the spin channels untouched.
- Never use the fairchem `turbo` setting. It merges the experts once, for the first structure. A later structure of
  different composition makes `merge_mole` raise an error, so turbo would only silently hide the effect in
  same-composition cases.
- `tests/test_audit.py` checks this code on CPU with every model replaced by a stub. Run it from the repository root
  with `python -m unittest tests.test_audit`.
- ORB needs `compile=False` on Windows because Triton is not installed. The loaders already set it.

## Rerunning the null tests and the neon probe on Bridges-2

`slurm/rerun_nulls_neon.sh` (a V100-32 job) runs the null tests of the nine MACE and UMA labels and the neon probe of MACE
and UMA-S, and `slurm/rerun_orb.sh` (an L40S job: ORB's graph builder, NVIDIA Warp with CUDA 13, needs a GPU of compute
capability 7.5 or higher) runs ORB-v3. Both write the new files to `shared/runs/<job>/work/results`, not into the repository, so
a later sync cannot overwrite them. The UMA weights are read from a private folder (the model is gated). The run of
2026-10-09 (jobs 48975723 and 48975738) took 7 min 41 s and 5 min 30 s; its numbers are in `status-wip/README.md`.
