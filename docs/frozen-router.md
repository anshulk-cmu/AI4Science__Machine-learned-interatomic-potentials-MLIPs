# Frozen router: does routing help learning?

Date: 2026-10-18. Branch `frozen-router`. Author: Anshul Kumar (11-785 Team 6).

## 1. Question

The router check (`docs/router-check.md`) showed one fact at inference. A uniform router or a fixed mean router hurts every trained MoLE model. This fact has two explanations.

- Routing helps learning. The router finds mixtures that a fixed mixture cannot give.
- Co-adaptation. The network learned to work with its own router. Any other alpha breaks that fit, even if a fixed alpha could have trained as well from the start.

Section 6 of the router check calls the second case the strongest counterargument. Section 8, item 1, names the only direct test: train a model with a constant router from the first step. This document describes that run.

The question is simple. Does a model trained with one constant alpha reach the accuracy of Global and Local-6?

The reference numbers are the seed 0 pilots, final validation, EMA weights. The full-precision values come from the offline W&B files, as recorded in `notes/pilotResults50k.md` and quoted in the router check (section 3). We did not reread the W&B files for this document.

| Model | Seed | Force MAE (eV/A) | Energy MAE per atom (eV) |
|---|---:|---:|---:|
| Global | 0 | 0.037028264 | 0.013105457 |
| Local-6 | 0 | 0.037045170 | 0.013100062 |
| Dense | 0 | 0.0387 | 0.0141 |

The Dense row has 4 decimals. It comes from `summary.txt` of run `pilot_dense_s0_49041937`.

## 2. Design

### 2.1 The variant

The new variant is `global_frozen` (`configs/variant/global_frozen.yaml`). It loads `global.yaml` through the Hydra defaults list and changes two keys:

```yaml
defaults:
  - global
  - _self_

backbone:
  model: training.frozen_moe.eSCNMDFrozenMoeBackbone
  fixed_alpha: 0.13
```

So every other setting is the Global setting. A change to `global.yaml` also reaches `global_frozen`.

### 2.2 The constant

We use alpha = 0.13 for each of the 8 experts. This is the vector that `scripts/router_check.py` calls `uniform`. fairchem computes alpha = softmax(z) + 0.005. For equal logits z, softmax gives 1/8 = 0.125 per expert, so alpha = 0.13 and the sum is 1.04. A trained router always gives this sum.

We chose this constant for four reasons.

- It is the router's own uniform point. At step 0 the Global router is near uniform (entropy 2.0739 against ln 8 = 2.0794, router check table 4.1). So the frozen model starts almost where Global starts.
- It keeps the scale of the effective weight. A MoLE layer uses W(alpha) = sum_k alpha_k W_k. With 0.13 for every expert, this is the same mixture that the router check measured at step 0 (0.368 times the norm of one expert, section 5.3). So the frozen model and Global share the initial scale. The run does not mix the question of routing with a change of scale.
- It uses no information from a trained run. The trained mean alpha of Global seed 0 favours expert 0 because of the random initial weights of that seed. A constant taken from it would carry knowledge from a routed run into the frozen run.
- It is the vector of the router check ablation. So the frozen model equals the `uniform` condition of that ablation, but trained from the start.

### 2.3 Same recipe as the pilots

Everything else is the Global pilot recipe:

- seed 0 (and seeds 1 and 2, see section 6);
- the fixed lists `manifests/pilot_train_200k.txt` and `pilot_val_5k.txt`, checked by the `verify_samples` stage before training;
- 50,000 steps, the cosine schedule with 5 percent warmup, AdamW with lr 2e-4 and weight decay 0.001, gradient clipping at 10, EMA 0.999;
- batches of at most 450 atoms;
- float32: `bf16: false` in `configs/train.yaml`, so autocast is off, as in the pilots; the router code runs with autocast off in any case;
- validation and a checkpoint every 5,000 steps, all checkpoints kept;
- the OOM monitor (`training/guarded_unit.py`) and offline W&B logging;
- the size-consistency stage on the final checkpoint;
- one V100 with 32 GB on GPU-shared, a 48 h limit.

The run folders are `frozen_global_s<seed>_<jobid>` under `/ocean/projects/che260063p/shared/runs/`. The new variable `PILOT_RUN_TAG` sets only the middle part of the folder name. `settings.txt` still records `variant=global_frozen`.

## 3. What is frozen and how

### 3.1 The hook

`training/frozen_moe.py` defines `eSCNMDFrozenMoeBackbone`, a subclass of fairchem's `eSCNMDMoeBackbone`. It does two things in its constructor and nothing else.

1. It replaces `mole_expert_coefficient_norm` with a function that returns a tensor full of `fixed_alpha`, with the shape of the router logits. This is the hook that `router_check.py` (`fixed_alpha`) uses at inference. fairchem calls this norm in `set_MOLE_coefficients` in every forward pass: in training, in validation and in inference. So the constant is active in all three. No fairchem file changes.
2. It sets `requires_grad = False` on the 7 router tensors.

The constant is a model config value. fairchem writes the model config into every inference checkpoint and builds the model from it on load. So a loaded checkpoint has the same constant router.

### 3.2 The parameters that stay untrained

Only the router uses these parameters:

| Tensor | Shape | Parameters |
|---|---|---:|
| `backbone.routing_mlp.0.weight`, `.bias` | 16 x 256, 16 | 4,112 |
| `backbone.routing_mlp.2.weight`, `.bias` | 16 x 16, 16 | 272 |
| `backbone.routing_mlp.4.weight`, `.bias` | 8 x 16, 8 | 136 |
| `backbone.composition_embedding.weight` | 100 x 128 | 12,800 |
| Total | | 17,320 |

They stay at their random initial values for the whole run. Three facts keep them fixed.

- The constant does not depend on the logits, so no gradient flows back into the router.
- `requires_grad` is False, so autograd does not build a gradient for them. DDP wraps the model even for one rank (`train_strategy` DDP in fairchem). DDP ignores parameters with `requires_grad` False. It would otherwise wait for gradients that never come.
- AdamW skips every parameter whose gradient is None. So it applies no step and no weight decay to them. The EMA copy of a constant parameter is the same constant.

The charge, spin and dataset embeddings feed the router and the node features. They stay as in Global: `cs_emb_grad: false` and `dataset_emb_grad: false` already freeze their 256 parameters in every variant, and `mix_csd` trains as in Global.

The model keeps all 37,652,265 parameters. The state dict has the same keys as Global. With the same seed, the initial weights are identical to the Global weights.

### 3.3 What the frozen model can still learn (reasoning)

With a constant alpha, every expert of a layer gets the same gradient, 0.13 times the gradient of W(alpha). AdamW normalises each element by the same moments, so the 8 experts get the same update. The experts then differ only by their random initial values, for the whole run. The effective matrix W(alpha) moves by 8 x 0.13 = 1.04 times one AdamW step.

So the frozen model is a Dense model in a new parameterisation. Its 24 MoLE layers start at 0.368 times the norm of one expert and train with about 1.04 times the learning rate. This is also item 2 of the router check section 8 (Dense with the SO(2) weights scaled by 0.368). One run therefore tests two things: the learned mixture and the initial scale. We did not measure this equivalence on the full model. The test in section 4 checks the equal gradients on a small model.

The size consistency follows from the same fact. The frozen model applies the same matrix to every structure. So two far-apart fragments are additive, as in Dense. We expect Delta near 0, not the Global value.

## 4. Tests

`tests/test_frozen_router.py` adds 8 tests. They run on small float64 models built from the real configs.

1. `test_existing_variants_compose_as_on_main`: the composed configs of Dense, Global and Local-6 have the SHA-256 that we measured on main (9817f01), with the job sections removed (they hold machine paths and timestamps). The full-size parameter counts are 5,981,057 (Dense) and 37,652,265 (Global, Local-6 and `global_frozen`).
2. `test_frozen_variant_is_global_with_a_constant_router`: the model config equals Global except the two keys. The parameter counts (total, active, bank, MoLE layers, router) equal Global. The newly frozen parameters are exactly the router tensors, 17,320 in total. With the same seed the initial state dict equals Global.
3. `test_job_file_carries_the_pilot_recipe`: account, partition, GPU, 48 h, the two email lines, an absolute output path, the defaults (50,000 steps, seed 0, evaluation and checkpoint every 5,000, prefix `frozen`, `repo_frozen`), no array; `pilot.sh` accepts the variant and runs the router check before the size check.
4. `test_val_curve_parser_labels_each_validation_with_its_step`: the analysis script labels each validation with its step and keeps the last value of a repeated step.
5. `test_router_parameters_are_frozen_flags`: the 7 router tensors have `requires_grad` False.
6. `test_router_gets_no_gradient_and_stays_fixed_over_optimizer_steps`: four AdamW steps (weight decay 0.001, clipping at 10) in training mode. The router gradients are None. All 8 experts of each MoLE layer get the same gradient. The coefficients are 0.13 for every structure. After the steps the router tensors are bit-for-bit unchanged, and every MoLE weight changed.
7. `test_output_equals_the_uniform_router_of_router_check`: on the same weights, the frozen model gives the same energy and forces as Global under `router_check.fixed_alpha(uniform)` (difference at most 1e-12). The router-only pass gives 0.13 for each expert of each structure.
8. `test_inference_checkpoint_rebuilds_the_frozen_model`: a checkpoint in fairchem's inference format loads through `MLIPPredictUnit` (the path of the size check and of the router check). The backbone is the frozen class with alpha 0.13 and frozen router tensors. `predict` uses 0.13. The loaded EMA model gives the same energy as the saved model (1e-12). `scripts/check_frozen_router.py` passes on a trained copy and fails twice (model and EMA) when one router weight moves by 1e-6.

Results, `python -m unittest discover -s tests` in `IDLF26-mlip` on Windows CPU:

- Before, on main (9817f01): 123 tests, OK.
- After, on this branch (4711714): 131 tests, OK (the 123 old tests and the 8 new ones), in 54.8 s.

Deliberate break. We removed each half of the freeze once and ran the new tests:

- Without the constant (the router computes alpha again): 3 tests failed (6, 7 and 8).
- Without `requires_grad` False: 3 tests failed (2, 5 and 8). Test 6 still passed, because the constant alone already blocks the gradient. Test 5 is there for DDP.

We restored the file and all tests passed again. The new tests also passed on the laptop GPU (`ROUTER_TEST_DEVICE=cuda`, RTX 5070 Ti) together with the router check tests: 11 tests OK.

Laptop check. The tiny-model tests above ran on the laptop GPU. We also started a full-size check: the 37.65 M model, float32, five AdamW steps on a 40-molecule batch. It did not finish. The laptop GPU (12 GB) was full, because another agent ran an inference timing job on it at the same time. Our process grew to about 30 GB of RAM, and we stopped it after about 8 minutes. It printed no result. The project owner then limited local work to tiny models. So the full-size check is the Bridges-2 smoke test in section 5, not a laptop run. The tiny-model GPU tests ran at about 22:38, during the timing job, so they may have disturbed its timings.

## 5. Smoke test

Job 49442923, 2026-10-18, a 30-step run like the earlier `pilottest` jobs: the real job file, the real fixed lists, seed 0, a validation at step 30 and a checkpoint every 15 steps, a 1 h limit. The code on Bridges-2 was `/ocean/projects/che260063p/shared/repo_frozen/AI4Science-MLIPs`, synced with `b2.py syncrepo` (185 tracked files, all identical by SHA-256). We submitted it with every setting on the command line:

```bash
cd /ocean/projects/che260063p/shared/logs
sbatch -t 1:00:00 --job-name=frozentest --export=PILOT_SEED=0,PILOT_STEPS=30,PILOT_EVAL_EVERY=30,PILOT_CKPT_EVERY=15,PILOT_RUN_PREFIX=frozentest,PILOT_RUN_TAG=global,PILOT_REPO=/ocean/projects/che260063p/shared/repo_frozen/AI4Science-MLIPs /ocean/projects/che260063p/shared/repo_frozen/AI4Science-MLIPs/slurm/frozen_global.job
```

Measured, from `runs/frozentest_global_s0_49442923/summary.txt`, `settings.txt`, `logs/train.log` and `sacct`:

| Item | Value |
|---|---|
| settings.txt | `variant=global_frozen steps=30 seed=0`, both fixed lists with their hashes, `repo=.../repo_frozen/AI4Science-MLIPs` |
| GPU | Tesla V100-SXM2-32GB, node v010 |
| verify_samples | 8 of 8 PASS |
| train | exit 0, 811 s, peak GPU memory 10,718 MiB, 562.6 atoms per second at the last step |
| Router in the log | `Expert mean: 1.30e-01` for all 8 experts |
| Validation at step 30 | loss 0.7967, energy MAE per atom 0.1415 eV, force MAE 0.5041 eV/A |
| OOM | 0 events (no `OOM:` line; `OOM events: 0 train (0 recovered), 0 eval (0 recovered)`) |
| router_frozen | 5 of 5 PASS: in the model and the EMA state dicts, the 7 router tensors changed by exactly 0 and 123 of 129 other float tensors changed; the config names the frozen backbone with alpha 0.13 |
| size_consistency | Delta 1.86265e-06 meV at 8 A and at 100 A |
| Result | `RESULT: all stages passed`; sacct COMPLETED, exit 0:0, elapsed 18 min 25 s |
| Cost | about 0.31 SU (1 V100 for 18 min 25 s at 1 SU per GPU-hour; our calculation from sacct) |

The training loss over 30 steps does not show a clear fall: the mean of the first 10 steps is 0.8662 and of the last 10 is 0.9127. The loss of single batches varies much more than 30 steps of training can move it, and the learning rate is still near its start. The earlier 30-step Global test with the same seed and the same batches (`pilottest_global_s0_49041065`) shows the same pattern (0.8661 and 0.9126). Its per-step losses agree with the frozen run to about 0.001 at every step. Its validation at step 30 was 0.1409 eV and 0.5039 eV/A. This agreement is expected: at step 0 the Global router is close to uniform, so for 30 steps both models are nearly the same function. The first validation of the long run, at step 5,000, is the real check that the loss falls.

Interpretation, kept apart from the measurements above:

- The freeze works in the real pipeline: the router tensors of the final checkpoint equal those of step 0, in both state dicts.
- The Delta of 1.9e-06 meV is the Dense class (Dense seed 0: 2.79e-06 meV; Local-6 seed 0: 1.86e-06 meV), not the Global class (the 30-step Global test gave -6.05 meV). This is what section 3.3 predicts: one constant mixture is one fixed matrix, so the model is size consistent.
- We listed the 6 other float tensors that did not change (a separate read of the two checkpoints on Bridges-2): `backbone.Jd_0`, `Jd_1` and `Jd_2` (constant Wigner matrices, buffers) and the charge, spin and dataset embeddings, which every variant freezes. So every trained tensor changed, and only the router and the tensors that are fixed in every variant stayed fixed.

## 6. The jobs

Three single-task jobs, one per seed, all from `slurm/frozen_global.job` and the code in `repo_frozen` (synced at commit 4711714, 186 tracked files identical by SHA-256). The login shell sets `SBATCH_EXPORT=NONE`, so we passed every setting with `--export` on the command line (guide B9 row 18). The job file holds the same values as defaults. The seed 0 command was:

```bash
cd /ocean/projects/che260063p/shared/logs
J=/ocean/projects/che260063p/shared/repo_frozen/AI4Science-MLIPs/slurm/frozen_global.job
sbatch --export=PILOT_SEED=0,PILOT_STEPS=50000,PILOT_EVAL_EVERY=5000,PILOT_CKPT_EVERY=5000,PILOT_RUN_PREFIX=frozen,PILOT_RUN_TAG=global,PILOT_REPO=/ocean/projects/che260063p/shared/repo_frozen/AI4Science-MLIPs $J
```

Seeds 1 and 2 used the same command with `PILOT_SEED=1` and `PILOT_SEED=2`. Each command passed `sbatch --test-only` first. The owner asked for seeds 1 and 2 after the plan for seed 0. We submitted them only after the first validation of seed 0 looked sane.

| Seed | Job | Submitted | Node | Run folder |
|---:|---:|---|---|---|
| 0 | 49444920 | 2026-10-18 23:08 | v018 | `runs/frozen_global_s0_49444920` |
| 1 | 49450911 | 2026-10-19 00:30 | v004 | `runs/frozen_global_s1_49450911` |
| 2 | 49450922 | 2026-10-19 00:30 | v013 | `runs/frozen_global_s2_49450922` |

All three: account `che260063p`, GPU-shared, one `v100-32`, limit 48 h, emails for all events and at 80 percent of the limit, output in `/ocean/projects/che260063p/shared/logs/slurm-<jobid>.out`.

settings.txt, read after each job started (measured):

- seed 0: `variant=global_frozen steps=50000 seed=0`, `pilot_train_200k.txt sha256 de75c47ff4dae8da`, `pilot_val_5k.txt sha256 7ecd0a5457118d8e`, `eval_every=5000 ckpt_every=5000`, `array_task=none`, `repo=/ocean/projects/che260063p/shared/repo_frozen/AI4Science-MLIPs`.
- seeds 1 and 2: the same, with `seed=1` and `seed=2`.

The list hashes equal those of the Global pilot `pilot_global_s0_49041939`. verify_samples of seed 0 passed 8 of 8 checks.

First validation of seed 0, step 5,000 (measured, `logs/train.log`, 4 decimals):

| Run | Force MAE (eV/A) | Energy MAE per atom (eV) |
|---|---:|---:|
| Frozen, seed 0 | 0.1104 | 0.0503 |
| Global, seed 0 | 0.1099 | 0.0503 |
| Local-6, seed 0 | 0.1104 | 0.0509 |
| Dense, seed 0 | 0.1063 | 0.0479 |

The validation loss was 0.1870. The mean training loss fell from 0.5819 (steps 1 to 1,000) to 0.2731 (steps 4,001 to 5,000). The log has no `OOM:` line and no NaN. The `OOM events` summary line appears only at the end of training. The log prints `Expert mean: 1.30e-01` for all experts and `Expert variance: 0.00e+00`. The checkpoints `step_0` and `step_5000` exist. Training ran at about 0.75 s per step on the V100 (steps 2,733 to 4,398 in 1,254 s).

At step 5,000 the four models are still close. In the pilots the gap between Dense and the MoLE models opened only after step 10,000. So this first point cannot separate the cases of section 7 yet.

Expected end (reasoning from the Global pilots, which took 10 h 28 min to 11 h 5 min per seed on the same GPU type): seed 0 about 09:40 to 10:20 on 2026-10-19, seeds 1 and 2 about 11:00 to 11:40 on 2026-10-19.

Cost estimate: about 10.5 to 11.1 SU per seed, so about 32 to 33 SU for the three seeds, plus 0.31 SU for the smoke test. The cap is 48 SU per seed (144 SU in the worst case). The project had 2,792 of 2,950 GPU SU before these jobs (`projects`, 2026-10-18).

Results will appear in each run folder: `summary.txt` (stages, last validation, OOM line, `router_frozen` lines, Delta), `logs/train.log`, the checkpoints every 5,000 steps under `train/*/checkpoints/`, and the offline W&B run under `train/*/logs/wandb/`.

## 7. What the outcomes would mean

Let F be the final force MAE of the frozen run, G that of Global (0.0370) and D that of Dense (0.0387), all seed 0. The gap D - G is about 0.0017 eV/A. The same holds for the energy (0.0131 against 0.0141).

Case 1. F matches Global (F near 0.0370). A fixed mixture trains as well as the router. Then the router adds nothing that a constant could not, and H3 is the answer: the gain over Dense comes from the parameterisation (the bank, the initial scale, the learning rate per layer), not from routing. The inference damage of the router check is then co-adaptation damage. With section 3.3, the likely cause is then the smaller initial scale of the MoLE layers (or the 1.04 learning-rate factor). Item 2 of the router check section 8 (Dense with the SO(2) weights scaled by 0.368) would test this directly.

Case 2. F stays near Dense (F near 0.0387). Routing helps learning. A constant alpha with the same initial scale does not give the gain. Then the router does real work, and the scale alone is not the mechanism. The Local-6 result still keeps the gain while it stays size consistent, so local routing would keep a real benefit.

Case 3. F lies in between. Both parts matter. Report the share s = (D - F) / (D - G). The share of the gain that a fixed mixture explains is s. The router explains the rest.

How large a difference is real? The seed spread of the pilots is the scale. The three Global seeds end at 0.037028, 0.037456 and 0.037124 eV/A in the router check reproduction (table 4.4), a range of about 0.0004. A single frozen seed inside about 0.0004 of Global counts as Case 1. A value above Global plus 0.0004 and below Dense minus 0.0004 counts as Case 3. One seed is a pilot, not proof. Seeds 1 and 2 give the spread of the frozen model itself.

## 8. How to analyse

Compare the validation curves at every 5,000 steps with Global and Local-6 of the same seed, and with Dense. `scripts/compare_val_curves.py` reads `logs/train.log` of each run. It labels each validation with its step and prints the share of the Dense to Global gain at each step. train.log prints 4 decimals. The full precision is in the offline W&B files (`train/*/logs/wandb/offline-run-*`).

Run it on Bridges-2 when the jobs end (a few seconds of log parsing, so a login node is enough):

```bash
cd /ocean/projects/che260063p/shared/repo_frozen/AI4Science-MLIPs
R=/ocean/projects/che260063p/shared/runs
PY=/ocean/projects/che260063p/shared/envs/mlip/bin/python
$PY -I scripts/compare_val_curves.py $R/pilot_dense_s0_49041937 $R/pilot_global_s0_49041939 $R/pilot_local6_s0_49041936 $R/frozen_global_s0_49444920
$PY -I scripts/compare_val_curves.py $R/pilot_dense_s1_49041938 $R/pilot_global_s1_49041940 $R/pilot_local6_s1_49041974 $R/frozen_global_s1_49450911
$PY -I scripts/compare_val_curves.py $R/pilot_dense_s2_49041928 $R/pilot_global_s2_49041929 $R/pilot_local6_s2_49041927 $R/frozen_global_s2_49450922
grep -a -E 'router_frozen|PASS frozen|FAIL|Delta|OOM events|val/|RESULT' $R/frozen_global_s*_494*/summary.txt
```

Also read `summary.txt` of each frozen run: the `router_frozen` lines must all be PASS, and the size-consistency Delta should be near 0.

Then check three things.

- The final values against the table in section 1 and the cases in section 7.
- The shape of the curve. If the frozen run lags early and catches up late, the router helps the speed of learning, not the end point.
- The other two seeds against Global seeds 1 and 2 (`pilot_global_s1_49041940`, `pilot_global_s2_49041929`).

## 9. Caveats

- Short training. 50,000 steps are about 1/760 of the official eSEN run. A difference can appear or vanish later.
- One constant. We test the uniform point only. A different constant, for example the trained mean, starts from a different effective matrix.
- The frozen model cannot specialise at all. So Case 2 says that a learned mixture helps, not that per-structure routing helps. The difference between a learned constant and a learned per-structure mixture needs one more run: a trainable alpha that does not depend on the input.
- The equivalence to a rescaled Dense model (section 3.3) is reasoning on the update rule, checked on a small model only. Gradient clipping acts on the total norm, which differs between the two parameterisations.

## 10. Files

- `configs/variant/global_frozen.yaml`: the variant.
- `training/frozen_moe.py`: the backbone with the constant router.
- `scripts/check_frozen_router.py`: checks that the router of the final checkpoint equals step 0 (stage `router_frozen`).
- `scripts/compare_val_curves.py`: the analysis.
- `slurm/frozen_global.job`: the job; `slurm/pilot.sh`: accepts `global_frozen`, runs the router check, and takes `PILOT_RUN_TAG`.
- `tests/test_frozen_router.py`: the tests.
