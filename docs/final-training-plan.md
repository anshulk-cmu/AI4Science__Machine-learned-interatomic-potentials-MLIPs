# Final training plan, 5 seeds, long runs, checkpoint ensemble

Status: PLAN ONLY. Nothing in this file has been run or submitted.
Written: Wed 28 Oct 2026. Owner of the plan: Anshul. Work owners are named in Section 8.
Dates are proposals. The team must confirm them.

Writing style: simple English (ASD-STE100 rules at about 80 percent). Short sentences. One idea in each sentence.

---

## 1. Goal

**Question.** Does Local-6 (a router that looks only at the atoms within 6 A of each atom) keep the accuracy gain of Global routing (UMA style) over Dense, and stay size consistent?

**What the midterm showed (3 seeds, 50,000 steps).**
- Global and Local-6 reduce force error by about 4 percent against Dense. Both reach 0.0372 eV/A. Dense reaches 0.0388 eV/A.
- Local-6 keeps the force gain and the energy gain. The lower bounds are 0.84 (force) and 0.72 (energy). Our threshold is 0.5.
- The energy bound from the Student-t method is 0.50, which is on the threshold. We need more seeds.
- Global breaks additivity (the error is -5.1, -6.6 and +2.4 eV). Dense and Local-6 stay at the float64 floor.

**What the final project must add.**
1. Five seeds for every variant, so the intervals are narrower.
2. All six model types of the proposal, plus the two controls.
3. Longer training (300,000 steps) for all 8 variants, to test if the gain holds with more training.
4. A final model that uses several saved checkpoints together.
5. Everything finished before the Bridges-2 outage.

---

## 2. Dates that cannot move

| Date | Event |
|---|---|
| Wed 4 Nov | Last day of training. Cancel all unfinished jobs. |
| Fri 6 Nov | Ocean backup due (set by Anshul on 28 Oct). |
| Sun 1 Nov | Earlier note: "backup hard limit". Confirm with the PSC notice. If it is real, the backup date is too late. |
| Mon 9 Nov | Bridges-2 migration. Ocean is not backed up by PSC. |

---

## 3. Model types

| Id | Name | What the router reads | Config file | Status |
|---|---|---|---|---|
| 1 | Dense | no router | `configs/variant/dense.yaml` | exists, 3 seeds done |
| 2 | Global | mean element embedding of the whole structure (UMA) | `global.yaml` | exists, 3 seeds done |
| 3 | Element | element of the atom only (radius 0) | `element.yaml` | NEW |
| 4 | Local-6 | neighbours within 6 A | `local6.yaml` | exists, 3 seeds done |
| 5 | Local-12 | neighbours within 12 A | `local12.yaml` | NEW, needs new code |
| 6 | Local+Global | both: the local composition and the global composition, joined | `local_global.yaml` | NEW, needs new code |
| C1 | Frozen | Global with a fixed router (alpha 0.13 for each expert) | `global_frozen.yaml` | exists, 3 seeds done |
| C2 | Scaled Dense | Dense with a larger learning rate for some weights | `dense_scaled.yaml` | exists, 3 seeds done |

Types 1 to 6 are the six matched configurations of the proposal. C1 and C2 are controls that we added. They test if routing matters at all.

**Predicted size consistency (from `project/literature/buildProjectIdea.md`, Section 8).**
- Dense, Element, Local-6: Delta is zero at gaps of 8 A and more.
- Local-12: Delta is NOT zero at gaps of 8 and 10 A. It is zero from 15 A.
- Global, Local+Global: Delta is not zero and does not decay.

---

## 4. Runs

### Wave 1: 50,000 steps, same data as the pilots, 5 seeds (seed 0 to 4)

| Group | Runs | Seeds to run |
|---|---|---|
| Dense, Global, Local-6, Frozen, Scaled Dense | 10 | seeds 3 and 4 only (0 to 2 exist) |
| Element, Local-12, Local+Global | 15 | seeds 0 to 4 |
| Total | 25 | |

### Wave 2: 300,000 steps, long runs, with resumed jobs

All 8 variants get a 300,000-step run (decision made by Anshul on 28 Oct).

| Group | Runs |
|---|---|
| All 8 variants, 2 seeds each (recommended, decision D9) | 16 |

The number of seeds for the long runs is open (decision D9). The budget allows 2 seeds for 8 variants, together with Wave 1. It does not allow 5 seeds. See the cost table.

With 2 seeds we cannot give a confidence interval. We report Wave 2 as a description. We do not call it a test. The 5-seed statistics come from Wave 1.

### Estimated cost (reasoning, from the pilot run times)

| Item | Basis | SU |
|---|---|---|
| Wave 1, extra seeds of old variants | V100 1 SU/h, Local-6 on H100 2 SU/h | about 130 |
| Wave 1, Element and Local+Global (10 runs) | assumed the same memory as Local-6, so H100 | about 240 |
| Wave 1, Local-12 (5 runs) | assumed 30 SU each | about 150 |
| Wave 2, one seed of all 8 variants at 300,000 steps | 6 times the cost of a 50,000-step run | about 850 |

The 850 SU for one seed of the long runs is made of: Dense 56, Global 63, Frozen 63, Scaled Dense 56, Local-6 144, Element 144, Local+Global 144, Local-12 180.

| Option for the long runs | Long-run SU | With Wave 1 (about 520 SU) | Fits in 2,705 SU? |
|---|---|---|---|
| 2 seeds each (16 runs), recommended | about 1,700 | about 2,220 | yes, about 485 SU left for failed runs |
| 3 seeds each (24 runs), no Wave 1 | about 2,550 | about 2,550 | barely, about 155 SU left, no room for a failed run |
| 5 seeds each (40 runs) | about 4,250 | about 4,770 | no |

The first chat estimate of Wave 1 (450 SU) used V100 cost for the new variants. This table corrects it.
Frozen, Scaled Dense, Element, Local+Global and Local-12 times are guesses. A short test must measure them.

### Time for one long run

- Dense and Scaled Dense: about 55 to 65 h. Global and Frozen: about 60 to 65 h. Local-6, Element and Local+Global: about 70 to 80 h (assumed for the new ones). Local-12: not known, we assume about 90 h.
- The limit of one job is 48 h. So each long run needs 2 chained jobs (3 if there is a delay or a failure).
- Each job saves checkpoints. The next job starts from the last saved checkpoint.

---

## 5. Data for the long runs (decision D1)

- The pilots read a fixed list of 200,000 training rows. 50,000 steps are about 2 passes over this list.
- 300,000 steps on 200,000 rows are about 12 passes. The model may over-fit.
- Recommended: a new fixed list of 1,200,000 rows from `train_4M`. 300,000 steps are then about 2 passes, as in the pilots.
- The validation list stays `pilot_val_5k.txt`, so Wave 1 and Wave 2 use the same report set.
- Alternative: use all 4M rows. It costs more data preparation and gives fewer passes.
- Wave 1 keeps the 200,000-row list, so the new seeds stay comparable to the pilots.

---

## 6. Checkpoint ensemble (the final model)

**Idea.** Save many checkpoints. Select the best ones. Use them together as the final model.

**Two methods. We test both. The team picks one (decision D4).**
1. **Weight average.** Average the weights of K checkpoints from one run. The result is one model. Cost at inference: the same as one model.
2. **Prediction average.** Run K checkpoints. Average the energy and the forces. Cost at inference: K times.

**Why this keeps the physics.**
- Local-6 and Element with averaged weights stay local. Local-6 with averaged predictions stays additive. The average of additive energies is additive.
- Global with averaged weights or predictions stays non-additive.
- Forces from averaged predictions are still the gradient of the averaged energy.
- We must run the size-consistency test on the final combined model. We must not assume it.

**Anti-leakage rules (we must write them before we look at results).**
1. The rule that selects checkpoints is fixed first. It is the same for every variant.
2. We select checkpoints on a selection set. We report on a different set. The two sets share no rows.
   - Report set: `pilot_val_5k.txt` (already used for all pilot numbers).
   - Selection set (NEW): 5,000 rows from the `val` split that are not in the report set (decision D5).
3. Proposed rule: take the K = 5 checkpoints with the lowest force error on the selection set. Use only checkpoints from the last half of the run.
4. We report the single best checkpoint and the final checkpoint beside the ensemble, so the reader sees the gain from the ensemble.
5. We never choose K or the rule after we see the report-set result.

---

## 7. New code

Each item lists the owner, the files, what the code does, and when it is done. "Test" means a unit test in `tests/` that does not need a GPU, unless the item says otherwise. All 152 existing tests must still pass.

### A. Model code

**A1. Element router.** Owner: Yiwei.
- Add `configs/variant/element.yaml` with `routing_radius: 0`.
- `local_routing/router.py` already returns the plain embedding for radius 0. Check that `training/local_moe.py` accepts radius 0 and gives the same router output for two atoms of the same element in different places.
- Test: coefficients depend only on the element, the charge, the spin and the task.
- Done when: a 200-step run finishes and the test passes. Add the parameter count to `tests/test_training_config.py`.

**A2. Local-12.** Owner: Pragati.
- Problem: `training/local_moe.py` raises `NotImplementedError` when `routing_radius` is larger than the message cutoff (6 A). The router needs its own neighbour list at 12 A.
- Keep the message graph at 6 A. Build a second graph at 12 A for the router only. It must handle periodic images and keep gradients on distances.
- Two ways: build the second graph on the GPU inside the backbone, or build it in the data loader. Pragati chooses and writes the reason.
- Expected size: about 8 times more neighbours (12 A radius). Measure the memory and the speed. The router uses only indexes and distances, so we expect a moderate cost. This is not measured.
- Test: for a pair of fragments 10 A apart, the router output changes when the other fragment is added. At 15 A it does not change. Also test that the 6 A path gives the same numbers as before.
- Add `configs/variant/local12.yaml` and the GPU minimum in `slurm/pilot.sh`.
- Done when: the tests pass and a 200-step run finishes with a memory report.

**A3. Local+Global router.** Owner: Yiwei.
- Router input = the local composition `c_i` joined with the global composition `c(S)` (mean over the whole structure, same convention as UMA version 1.1).
- The first layer of the routing network gets a larger input. Create a new backbone class in `training/` and a config `local_global.yaml`.
- The parameter count rises a little. Report the new count in the docs and in `tests/test_training_config.py`.
- Test: the output shape is right. The global part equals the mean embedding. Delta for two fragments 100 A apart is not zero (as predicted).
- Done when: the tests pass and a 200-step run finishes.

**A4. Docs page for each new variant.** Owner: the person who wrote the variant. One short page in `docs/`, like `docs/frozen-router.md`. It says what the variant does, the parameter count, the test result and the GPU needed.

### B. Training infrastructure

**B1. Resume test.** Owner: Anshul. This is the first task, because Wave 2 depends on it.
- Check how fairchem 2.21 restores a run (it has `load_state` and a saved resume config). Our wrapper `training/run.py` and `slurm/pilot.sh` do not use it yet.
- The test: train 200 steps in one job. Train 100 steps, stop, resume, train 100 more. The two loss curves must match.
- Check that these are restored: weights, EMA weights, optimizer, learning-rate step, step count, and the position in the data. If the data order restarts, the run sees some rows twice. We must fix this before Wave 2.
- Done when: the curves match, or the difference is known and accepted by the team.

**B2. Long-run job script.** Owner: Anshul.
- New `slurm/long_run.sh` and job files. A segment runs until its time limit or until the last step. It then writes a marker file.
- Chain the segments with Slurm dependencies, so segment 2 is already in the queue when segment 1 ends. If a segment fails, the chain stops and sends the usual email.
- The size-consistency check runs only after the last segment.
- Keep the two email lines in every job file (standing rule).

**B3. Job files for Wave 1.** Owner: Anshul.
- New seeds for old variants: array 3 to 4. New variants: array 0 to 4. Use a different run prefix, so the pilot folders stay clean.
- `slurm/pilot.sh` has a list of known variants and GPU minimums. Add Element, Local-12 and Local+Global.

**B4. Data lists for long runs and the selection set.** Owner: Anshul.
- Extend `scripts/make_pilot_manifests.py` to make `manifests/long_train_1200k.txt` (if D1 is accepted) and `manifests/select_val_5k.txt`.
- Extend `slurm/prep_pilot_data.job` to build the matching index files. Check that the two val lists share no rows.

**B5. Checkpoint plan and disk use.** Owner: Anshul.
- Proposed: save every 10,000 steps in the long runs (30 checkpoints plus step 0 and final). Set `training.max_saved_checkpoints` high enough to keep them all.
- Measure the size of one pilot checkpoint folder on Ocean before we decide. Then multiply by the number of checkpoints and runs. We have not measured this.
- Validation every 10,000 steps takes 5 to 8 minutes each. For 30 evaluations this is a few hours of each run.

**B6. Learning-rate schedule (decision D3).** The schedule follows `training.steps`, so 300,000 steps give one cosine curve over 300,000 steps. The default warmup is 5 percent, which is 15,000 steps. Proposed: keep 5 percent, so the rule stays the same as the pilots.

### C. Checkpoint ensemble tools

**C1. Selection script.** Owner: Pragati. Reads all checkpoints of a run. Computes force and energy error on the selection set. Writes a table. Applies the fixed rule and writes the list of chosen checkpoints.

**C2. Weight-average script.** Owner: Pragati. Averages the weights of the chosen checkpoints and writes one new inference checkpoint. First check which weights the inference checkpoint holds (raw or EMA). Test on three pilot checkpoints.

**C3. Prediction-average wrapper.** Owner: Pragati. A calculator that loads K checkpoints and returns the mean energy and the mean forces. It must work in the audit code (`audit/models.py`).

**C4. Report-set evaluation.** Owner: Pragati. Computes energy per atom and force MAE of the single best, the final, and the ensemble models on the report set.

**C5. Size-consistency test of the final models.** Owner: Pragati, with A2. See Section 7D.

All of C1 to C5 can be built and tested now on the 99 pilot checkpoints. They do not need to wait for the new runs.

### D. Size-consistency tools

**D1. Guard and gaps for any routing radius.** Owner: Pragati.
- `audit/check_checkpoint.py` uses the gaps 8 and 100 A by default. The guard in `audit/geometry.py` uses the message cutoff (6 A).
- For Local-12 the guard must use the larger of the message cutoff and the routing radius. The gaps must be 8, 10, 15, 25 and 100 A.
- Test: the guard stops a pair that is closer than the routing radius plus 1 A.

**D2. Size consistency along training for new variants.** Owner: Pragati. `scripts/size_consistency_along_training.py` must accept the new variants. Run it on a thinned set of checkpoints (for example every 50,000 steps), to save GPU time.

### E. Statistics and figures

**E1. Final-metrics table from the new runs.** Owner: Yiwei. A script that reads the last validation values of each run from its log or its W&B file and writes one CSV row for each run. Use the same columns as `results/pilot_final_metrics.csv`. Add the check that each variant has exactly 5 rows.

**E2. Statistics for 5 seeds and 8 variants.** Owner: Yiwei.
- Implemented in `scripts/pilot_statistics.py`: infer the seed count from the data, check equal seed sets, and support all eight variants. See [the statistics guide](final-statistics.md) for usage and assumptions.
- For 5 seeds the full exact resample has 5 to the power 15 (30,517,578,125) combinations. Multinomial weights compress these into 126 to the power 3 (2,000,376) group triples without changing the bootstrap distribution. The Student-t method uses 4 degrees of freedom.
- Add all variants to the list. Compute the retention share against Dense and Global for Element, Local-12, Local+Global, Frozen and Scaled Dense.
- Keep the decision rule fixed before the results: Local-6 passes if the 5 percent lower bound of retention is at least 0.5 (the proposal threshold).
- Test: the new code gives the same numbers as the old code on the existing 3-seed file.

**E3. Figures.** Owner: Yiwei. Implemented: `scripts/make_accuracy_figure.py` infers seed counts, supports all eight variants with the Okabe-Ito palette, accepts explicit curve/final-metric CSVs, and offers endpoint-only and descriptive long-run figures. `scripts/make_ensemble_figure.py` compares report-set final, selected-best and ensemble evaluations. See [the figure guide](final-figures.md) for commands and the ensemble CSV contract. Real new-run and ensemble figures still require their measured inputs.

**E4. Accuracy by charge and spin group.** Owner: Yiwei. The ACCESS justification promised results for neutral closed-shell, charged and open-shell structures. No script exists for this. Add one that groups the report-set errors by these classes. Low priority. Do it after E1 to E3.

### F. Closing tasks

**F1. W&B upload of all new runs.** Owner: Anshul. Same method as before. The key is set only in the ssh session. It is never in a file, a command line or the repository.

**F2. Ocean backup.** Owner: Anshul, Fri 6 Nov. Choose what to keep: final and selected checkpoints, all logs, all result files. Use Globus or the data transfer node. Check the copy with file sizes and hashes. Do not rely on the PSC backup.

**F3. Repository, report and docs.** Owner: all. Update `README.md` (status, variants, tests), the proposal numbers (24 runs becomes 31 runs, 5 seeds each), and the final report. Record each step in the run log at the time it happens.

---

## 8. Who does what (proposal)

| Person | Items | Why this person |
|---|---|---|
| Anshul | B1 to B6, F1, F2, submission and monitoring | owns the cluster scripts and the pilot jobs |
| Pragati | A2, C1 to C5, D1, D2 | wrote the standalone local routing code and the molecular checks |
| Yiwei | A1, A3, A4, E1 to E4 | wrote the config draft and the pilot statistics notes |

Work balance: Pragati has the hardest model item (A2). Yiwei has two small model items and the statistics. If Yiwei finishes early, she takes C1 and C4 from Pragati. The team may change this table.

Rules for everyone:
- Work on a branch. Open a pull request into `main`. Run the full test suite before the pull request.
- Do not submit Slurm jobs. Anshul submits them after the dry-run check.
- The shared allocation is `che260063p`. Every job needs `-A che260063p` and the two email lines.
- Do not put keys or passwords in any file.

---

## 9. Schedule (proposal)

| Date | Target |
|---|---|
| Wed 28 Oct | Team confirms this plan and the decisions in Section 10. Branches created. |
| Thu 29 Oct | Anshul: B1 resume test, B4 data lists. Pragati: A2 and D1 start. Yiwei: A1 done, A3 start. |
| Fri 30 Oct | Submit the long runs of the 5 old variants (Dense, Global, Local-6, Frozen, Scaled Dense) and the Wave 1 extra seeds of old variants (10 runs). Needs B1 and B2 done. |
| Sat 31 Oct | A1, A3 and A2 merged. 200-step smoke runs for the three new variants. |
| Sun 1 Nov | Submit Wave 1 for Element, Local+Global and Local-12 (15 runs) and the long runs of the three new variants. See the risk about the long runs of the new variants. |
| Mon 2 Nov | Check all jobs. Fix failures. Last day to restart a failed 50,000-step run. |
| Tue 3 Nov | Check the long runs. C1 to C5 and E1 to E3 ready, tested on pilot data. |
| Wed 4 Nov | Training stop. Unfinished long runs: use the best saved checkpoint and report them as shorter training. |
| Wed 4 to Thu 5 Nov | Select checkpoints. Build the final models. Run statistics, figures and size-consistency tests. Upload to W&B. |
| Fri 6 Nov | Ocean backup. Push all code to GitHub. |
| Sat 7 to Sun 8 Nov | Final checks. No new jobs. |
| Mon 9 Nov | Bridges-2 migration. |

If A2 (Local-12) is late, we drop Local-12 first. We report it as not trained.

---

## 10. Decisions the team must make

| Id | Question | Proposed answer |
|---|---|---|
| D1 | Training rows for the 300,000-step runs | a new list of 1,200,000 rows |
| D2 | Which variants get a 300,000-step run | decided: all 8 |
| D9 | Seeds for the long runs, and what to do if the long runs of the new variants cannot finish | 2 seeds each. If a new variant cannot finish, use 200,000 steps for it or use its best saved checkpoint and say so |
| D3 | Warmup of the long runs | 5 percent (15,000 steps) |
| D4 | Ensemble method, K, and which steps | test both methods, K = 5, last half of the run, one run at a time |
| D5 | Selection set for the checkpoints | 5,000 `val` rows that are not in the report set |
| D6 | Ocean backup date | Fri 6 Nov. Confirm against the PSC notice (the earlier note says 1 Nov) |
| D7 | How many checkpoints to keep, and how often to save them | every 10,000 steps. Decide after we measure one checkpoint |
| D8 | Ensemble across seeds | not in this plan. Each run is combined alone |

---

## 11. Risks

1. **Time.** 300,000 steps leave almost no slack before the training stop. A fallback is a shorter long run (200,000 steps).
   - The long runs of Element, Local+Global and Local-12 can start only after their code is merged and tested. In the schedule this is Sun 1 Nov. A run of 70 to 90 h, plus waits between chained jobs, ends after the training stop (Wed 4 Nov) unless the code is ready sooner.
   - To reduce the risk: A1, A3 and A2 must be merged as early as possible, and the smoke test must be short.
2. **Queue.** 41 runs (25 in Wave 1 and 16 in Wave 2) and the chained jobs can wait a long time. Check the job limits in the dry run. Submit the long runs first.
3. **Resume.** If resume repeats or loses data, the long runs are not valid. B1 must pass before any submission.
4. **Local-12 memory and speed.** Not measured. A2 must report both.
5. **Storage.** Many checkpoints use much Ocean space and make the backup larger. B5 must measure it.
6. **Fair comparison.** All variants in one table must have the same steps, data and selection rule.
7. **Not comparable.** Wave 2 results are not directly comparable to the 50,000-step pilots. Report them in a separate table.
8. **Leakage.** Never choose checkpoints on the report set (Section 6).
9. **Shared allocation.** Other work on `che260063p` also uses SU. Check the balance before each wave.
10. **Unmeasured costs.** Run times and SU for Element, Local+Global, Local-12, Frozen and Scaled Dense are estimates.

---

## 12. Not in this plan

- OMol25 distance-scaling and protein-ligand evaluation tasks (listed in `buildProjectIdea.md`). Add them only if time allows after 5 Nov.
- NVE energy-conservation runs.
- The stretch variant of Objective O3 (correction only for unusual atoms).
- Any use of Vera, Kaggle or other clusters. The compute rule stays: local GPU, Bridges-2, Babel, Colab.
- Training on more than one dataset.
