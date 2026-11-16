# Limits and provenance of the 8 long runs

Status: 2026-11-16. "Measured" means a script or a log gave the value. "Reasoning" means it was not run. Results of the runs:
[final-evaluation.md](final-evaluation.md). This page lists what the numbers do not cover and which code trained each run.

## 1. What the standard errors cover, and what they do not

- **They cover the sampling of test structures, by composition.** The test list was split by composition
  ([data-split.md](data-split.md)), so rows of one composition are correlated. The SEs of `results/final_test_cluster_se.csv` and
  `results/final_test_paired_cluster.csv` treat each composition as one sampling unit (`scripts/cluster_se.py`). Measured: 110,864
  compositions in the 399,276 test rows; the largest holds 9,907 rows (`C4H7N1O1`). The earlier row SEs treated each structure as
  independent. They are too small: by 5.1 to 7.4 times for the energy MAE of one model, 2.8 to 2.9 times for its force MAE, 1.9 to
  7.0 times for a paired energy difference and 1.19 to 1.69 times for a paired force difference (measured). A cluster bootstrap with
  1,000 resamples gives the same SEs within 4 percent (measured).
- **They do not cover the training seed.** Each variant has one run (seed 0). The spread between seeds at 400,000 steps was not
  measured. For scale only: in the 50,000-step pilots the seed standard deviation of the force MAE was 0.26 meV/A for Dense (3 seeds,
  `scripts/pilot_statistics.py`). That is a different, shorter setup, so it does not transfer (reasoning).
- **They do not cover run-to-run noise of the GPU path.** The training log records `deterministic: False`. Measured from the logged
  `train/loss` of steps that were trained twice after a resume (same checkpoint, same batch):

  | Run | Overlap | Repeated steps | Relative loss difference, first step | Median over the overlap |
  |---|---|---:|---:|---:|
  | Global | step 240,000, V100 (Bridges-2) against L40S (Babel) | 8,090 | 2.1e-7 | 2.3e-2 |
  | Global | step 322,000, L40S against L40S (Babel) | 1,182 | 1.9e-7 | 4.0e-7 |
  | Local-6 | step 220,000, 2 x L40S Bridges-2 against 2 x L40S Babel | 2,192 | 2.1e-7 | 1.2e-2 |
  | Local-6 | 11 overlaps, both on Babel | 63 to 2,000 | 2.0e-9 to 1.3e-6 | 2.3e-7 to 4.9e-5 |
  | Element | 33 overlaps on Babel and 1 Bridges-2 against Babel | 42 to 1,681 | 6.4e-9 to 8.4e-7 | 1.8e-7 to 8.7e-7 |

  The first repeated step never matches exactly (measured). Over up to 2,000 steps on one cluster the copies stay within about 5e-5;
  the two long overlaps that crossed from Bridges-2 to Babel drift apart by 1 to 3 percent of the loss (measured). The cross-cluster
  overlaps also changed the code version and, for Global, the GPU type, so this table cannot say how much of the drift comes from
  nondeterminism alone (reasoning). So the path is not bit-for-bit reproducible, and the same seed would not give the same final weights (reasoning). How
  large a spread in the final MAEs this causes was not measured. Force differences near 0.1 meV/A (1e-4 eV/A), such as Local-6 minus
  Global (+0.007 meV/A) or Local-12 minus Global (+0.077 meV/A), cannot be separated from it (reasoning).
- **The two-atom rows.** 804 two-atom structures (0.2 percent of the rows, 18 compositions) carry most of the energy differences
  between models: `S2` alone (358 rows) carries 95 percent of the summed Local-6 minus Global energy difference and 102 percent, with
  the opposite sign, of Global minus Dense (measured). The table without them (`results/final_test_by_size.csv`) was made after this
  was seen, so its z values are descriptive, not a planned test. With 18 compositions the cluster SEs of the two-atom subset are rough.
- **Many pairs, no correction.** 14 paired differences are reported, with no correction for multiple comparisons.
- **float32.** The test list was evaluated in float32. On 300 test rows of two pilot checkpoints, float32 and float64 gave the same
  MAEs within 4e-8 eV/atom and 8e-7 eV/A ([data-split.md](data-split.md), Section 7; measured on pilots, not on the long runs).
- **Final checkpoint.** The `final` checkpoint is reported; no checkpoint selection or averaging was run.

## 2. Training length and data order

- 400,000 steps are 1.049 epochs. Epoch 1 starts at step 381,159 (`train/epoch` is 1.0 at that step in the logs of Dense, Global and
  Local-6; measured), so the last 18,841 steps (4.7 percent) are a second pass over training rows. The batch order is the same for all
  8 runs: same seed and sampler, and the 2-rank runs split the one-GPU batches (`docs/ddp.md`: 0 of 381,159 steps differ; measured
  there).
- The validation at the epoch boundary (step 381,159) is not in `results/long_validation_curves.csv`: `scripts/long_curves.py` keeps
  multiples of 10,000 steps only (measured: no row at that step).
- `results/long_final_metrics.csv` and `results/long_runs_validation.csv` hold the training-log validation values, rounded to 4
  decimals by fairchem's log. Rankings and differences use the full-precision files instead: `results/final_eval/val5k_metrics.csv`
  (same checkpoints and rows, evaluated again) and `results/final_test_metrics.csv`.

## 3. Code versions

`slurm/long_run.sh` writes the first 16 hex digits of the sha256 of
`cat slurm/long_run.sh configs/train.yaml configs/variant/*.yaml training/*.py local_routing/*.py` (C collation) at every segment
start. `settings.txt` records no commit. The table below comes from the `segment start` lines of each `summary.txt`; the step range is
from the `complete checkpoints up to step` lines (measured).

| Run | Code sha256 (first 8) and steps trained |
|---|---|
| Dense | fd4826bf 0 to 300,000 (V100) ; 7ae0287e 300,000 to 400,000 |
| Scaled Dense | fd4826bf 0 to 300,000 (V100) ; 7ae0287e 300,000 to 400,000 |
| Frozen | fd4826bf 0 to 250,000 (V100) ; 7ae0287e 250,000 to 400,000 |
| Global | fd4826bf 0 to 240,000 (V100) ; 7ae0287e 240,000 to 400,000 |
| Element | 7addfa24 0 to 170,000 (L40S) ; 7ae0287e 170,000 to 400,000 |
| Local-6 | f0ec157a 0 to 220,000 (2 x L40S) ; 7ae0287e 220,000 to 268,000 ; 74d57c6e 268,000 to 400,000 |
| Local+Global | f0ec157a 0 to 240,000 (2 x L40S) ; 7ae0287e 240,000 to 364,000 ; 74d57c6e 364,000 to 400,000 |
| Local-12 | f0ec157a 0 to 240,000 (2 x L40S) ; 7ae0287e 240,000 to 390,000 ; 74d57c6e 390,000 to 400,000 |

The first version of each run ran on Bridges-2, the others on Babel. So each run used two or three code versions, five in all.

**Hash to commit.** The same hash was recomputed from `git cat-file` for all 187 commits of the repository (measured):

| Hash | Commits that give it | Reproduced |
|---|---|---|
| fd4826bf47746deb | `d4f7116` to `8ebe5fe` (7 commits, 3 November 01:19 to 14:14) | yes |
| 7addfa24807d2457 | none | no |
| f0ec157a9aecb223 | none | no |
| 7ae0287e3b779d2e | `d4ddae7`, `39d7aef`, `3e95e39` (7 to 9 November) | yes |
| 74d57c6ebe65210d | `cdb792b` (11 November) and every later commit, including the current one | yes |

- 7addfa24 (Element) started at 14:59 on 3 November, after `8ebe5fe` (14:14) and before the next commits (15:08). f0ec157a (the three
  2-rank runs) started at 02:16 to 02:56 on 4 November; the last commit before was `54eecfe` (22:08 on 3 November), and it does not give
  f0ec157a. So the code on Bridges-2 held edits that were not committed in that form. Their exact content cannot be recovered
  (measured: no match; the cause is reasoning).
- **What changed between the reproduced versions** (`git diff` on the hashed files; measured):
  - fd4826bf -> 7ae0287e (`8ebe5fe` -> `d4ddae7`): `slurm/long_run.sh` (segment bookkeeping on Babel), `training/guarded_unit.py`
    (new code runs only when the world size is above 1: the 2-rank step log, an out-of-memory stop, and counting a shared validation
    batch once; the one-rank path calls the same fairchem `train_step` and `eval_step`), `training/split_sampler.py` (new, used by
    2-rank runs only) and `training/resume_probe.py` (a test-only unit, used by no production config). No file of the model, the
    variant configs or the loss changed.
  - 7ae0287e -> 74d57c6e (`3e95e39` -> `cdb792b`): 5 lines of `slurm/long_run.sh`, so that an NCCL collective timeout ends a segment
    without the FAILED marker.
  - Reasoning, not measured: the two unreproduced versions lie in time between `8ebe5fe` and `d4ddae7`, whose model and config files
    are identical. A temporary edit to a model file that was later undone cannot be ruled out.

## 4. Hardware

Each run resumed many times after preemptions, on V100, L40S, RTX A6000, RTX 6000 Ada and RTX PRO 6000 Blackwell GPUs (the GPU names
are in each `summary.txt`). Global's first 240,000 steps ran on V100 and the rest on L40S; Local-6 ran on L40S, RTX A6000 and RTX 6000
Ada. So training speed is not comparable between runs, and the GPU type changes inside a run (measured from `summary.txt`).

## 5. What the results support

- Force ordering on the test list, with composition cluster SEs (all |z| above 6 except Local-6 minus Global).
- Local-6 and Global have the same force error within the error (+0.007 meV/A, cluster SE 0.011).
- Size consistency: Local-6 is additive to the float64 floor at all 52 checkpoints checked along training, and Dense, Scaled Dense,
  Frozen and Element at their final checkpoint (21 geometries); Global is not additive at any of its 73 checkpoints, step 0 included
  ([size-consistency-long-runs.md](size-consistency-long-runs.md)).
- The sampling z of a force difference does not include run-to-run noise (Section 1). Local-12 minus Global (+0.077 meV/A,
  cluster z +6.4) is small enough that it may not survive another seed (reasoning).
- Not supported: an energy difference between Dense, Global and Local-6 on the full test list. See
  [final-evaluation.md](final-evaluation.md), Section 3.
