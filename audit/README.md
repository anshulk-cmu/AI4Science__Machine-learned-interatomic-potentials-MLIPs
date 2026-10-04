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
# UMA-M needs a 24 GB GPU and was run in float32 on a rented machine:
#   nulls.py / run_audit.py --family uma --model uma-m-1p1 --task omat --dtype float32

# router version test (5 shaken runs):
& $py1 nulls.py     --family uma --model uma-s-1p2 --task omat --model-version 1.1
& $py1 run_audit.py --family uma --model uma-s-1p2 --task omat --model-version 1.1 --runs 5

& $py1 analyze.py       # statistics, decision rule, plots
& $py1 report.py        # results.md and the overview plot
```

## Notes

- Run one job at a time on a laptop GPU.
- `run_audit.py` resumes after a crash, since it skips runs already saved.
- Never use the fairchem `turbo` setting: it merges the experts from the first structure and hides the effect.
- ORB needs `compile=False` on Windows because Triton is not installed. The loaders already set it.
