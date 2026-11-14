# Running the final training on CMU Babel

Status: the 8 long runs (seed 0, 400,000 steps) were finished on Babel between 7 and 14 November 2026. This file records how they were run, what went wrong, and the exact commands, so another person can repeat or resume a run.

## Where things are

| Item | Place |
|---|---|
| Code used by the jobs | `/data/user_data/anshulk/mlip/code-babel/AI4Science-MLIPs` (branch `babel`, now `main`) |
| Python environment | `/home/anshulk/mlipEnv` (Python 3.12, torch 2.8.0+cu128, fairchem-core 2.21.0; `pip freeze` equals the Bridges-2 freeze) |
| Run folders | `/data/user_data/anshulk/mlip/runs/long_<variant>_s0_400000` |
| Data and split keys | `/data/user_data/anshulk/mlip/data/omol25` |
| Old Bridges-2 archive | `/data/user_data/anshulk/mlip/archive` (the Bridges-2 system went offline on 9 November) |
| Copy of the finished runs | laptop, `finalRuns/` (not in git: checkpoints and logs) |

`/data` is visible only on compute nodes while you have a job there. Use `ssh babel-<node>` while a job runs on that node.

## How a run is launched

Account `bapoczos`, partition `preempt`, QoS `preempt_qos`. Jobs can be preempted at any time. A run is a chain of 12 hour segments joined with `afterany`; each segment resumes from the last complete checkpoint (`slurm/long_run.sh`, `slurm/submit_chain.sh`).

```bash
export LONG_SHARED=/data/user_data/anshulk/mlip LONG_PY=/home/anshulk/mlipEnv/bin/python
export LONG_REPO=$LONG_SHARED/code-babel/AI4Science-MLIPs
export LONG_CPUS=1                 # 1 CPU per GPU (see lessons)
export LONG_FEAT='VRAM_48GB|VRAM_80GB|VRAM_96GB|VRAM_141GB'
cd $LONG_REPO
# two-GPU runs (Local-6, Local-12, Local+Global): add LONG_RANKS=2 and, if a node hangs, NCCL_P2P_DISABLE=1 and LONG_EXCLUDE=<nodes>
bash slurm/submit_chain.sh local6 --steps 400000 --seed 0 --segments 5 --hours 12 --eval-every 10000 --ckpt-every 2000 --dry-run
```

`--steps`, `--seed`, `--eval-every`, `--ckpt-every` and the rank count must equal line 1 of the run's `settings.txt`. To stop a chain: `scancel -n long_<variant>_s0_400000`. Never cancel one segment of a chain that is running: the later segments would start at once.

The first 8 runs started on Bridges-2 with a checkpoint every 10,000 steps. On Babel the spacing was changed to 2,000 steps (line 1 of `settings.txt`, backup `settings.txt.bak10000`) so that a preemption loses less work.

## Lessons from the runs (what failed and what was changed)

1. **Resume from the wrong step.** After the copy to Babel the newest folder by date was `step_90000`, and fairchem resumes from the newest folder. `long_run.sh` now touches the highest complete step first.
2. **A preemption must not write `FAILED`.** The stop signal reaches the batch shell and the training process together. The script now traps USR1 and TERM and exits 0, so Slurm requeues the job.
3. **Jobs waited for days with 4 CPUs per GPU.** Many nodes had free GPUs but no free CPUs. With 1 CPU per GPU (`LONG_CPUS=1`) the first segments started in minutes to hours. A 200 step test gave the same speed within about 1 percent (different GPU types, so a rough check).
4. **NCCL hangs on some 2-GPU nodes.** On babel-o5-20, babel-q5-20 and babel-o5-28 the first all-reduce never finished (30 minute watchdog, then a crash). On babel-x9-16 one Local-12 segment hung after 9 hours. `long_run.sh` now treats "Watchdog caught collective operation timeout" as a node hang: no `FAILED`, the next segment resumes. The cause is not proven; `NCCL_P2P_DISABLE=1` and `LONG_EXCLUDE` were used as precautions.
5. **Slurm holds a job after about 5 requeues** ("launch failure limit exceeded"). The held segment never runs, so its chain stalls. Cancel the chain by name and submit it again.
6. **A code update during a running segment** cut a smoke job short. The script is now read into memory when it starts.

## Speed, measured on Babel

| Run | GPU | Seconds per step |
|---|---|---|
| Dense | L40S | 0.35 to 0.40 |
| Scaled Dense | L40S | 0.31 |
| Global | L40S | 0.54 to 0.57 |
| Frozen | A6000, L40S | 0.64 |
| Element | L40S | 0.95 to 0.98 |
| Local-6, Local-12 (2 GPUs) | A100 80 GB (smoke test) | 0.43 |

## Checks

All 8 runs passed the data check, wrote a final checkpoint and a `DONE` file, and passed the size-consistency check of `audit/check_checkpoint.py` (it reports the gap; a passed check does not mean the gap is zero). The archive on Babel was compared with the Bridges-2 copy by sha256 for all 125,805 files.
