# Extra seeds (1 to 4) for the 8 variants

Status: started on 2026-11-16 on CMU Babel. These runs are a backup. They go into the report only when a whole variant (all 4 seeds) is finished and checked.

## Rules

- Same settings as the seed 0 runs: 400,000 steps, the 70 percent training list, the 5,000 row monitor list, the same code and configs. Only `job.seed` changes (1, 2, 3, 4).
- Checkpoints are saved every 2,000 steps and validation runs every 10,000 steps, as in the Babel part of the seed 0 runs.
- Run folders: `/data/user_data/anshulk/mlip/runs_seeds/long_<variant>_s<seed>_400000` (separate from the seed 0 folders in `runs/`, which are never touched).
- Two model types at a time, four seeds each (8 runs at once for the one-GPU types). The next pair starts when every run of the current pair has a `DONE` file.
- Launch: `bash slurm/submit_chain.sh <variant> --steps 400000 --seed S --segments 6 --hours 12 --eval-every 10000 --ckpt-every 2000` with `LONG_RUNS_DIR`, `LONG_CPUS=1` (see `docs/babel.md`); two-GPU types add `LONG_RANKS=2 NCCL_P2P_DISABLE=1`.

## Order (cheap first)

| Pair | Types | GPUs at once | Estimated GPU-hours per run (reasoning, from seed 0 speeds) |
|---|---|---|---|
| 1 | Scaled Dense, Dense | 8 | 34 and 40 |
| 2 | Global, Frozen | 8 | 60 and 71 |
| 3 | Local-6, Local+Global | 16 (2 per run) | about 96 each |
| 4 | Local-12, Element | 12 | about 100 and 105 |

About 600 GPU-hours per seed, about 2,400 for the four seeds.

## Before the numbers are used

Run the same checks as for seed 0: `DONE` file, data check, size-consistency check, full test list with composition cluster SEs (`scripts/cluster_se.py`), and add the seeds to the statistics only for a complete variant.
