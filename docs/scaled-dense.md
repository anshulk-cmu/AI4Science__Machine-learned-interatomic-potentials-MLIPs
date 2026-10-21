# Scaled Dense: is the frozen-router gain a matter of scale and step size?

Date: 2026-10-21. Branch `dense-scaled`. Author: Anshul Kumar (11-785 Team 6).

## 1. Question

The frozen-router model (`docs/frozen-router.md`) keeps alpha = 0.13 for each of the 8 experts in every MoLE layer, for every structure and for the whole run. Its three seeds end at these values (final validation, EMA weights, `summary.txt` of each run, 4 decimals):

| Model | Force MAE (eV/A), seeds 0, 1, 2 | Energy MAE per atom (eV), seeds 0, 1, 2 | Mean force | Mean energy |
|---|---|---|---:|---:|
| Frozen | 0.0374, 0.0376, 0.0374 | 0.0130, 0.0135, 0.0129 | 0.0375 | 0.0131 |
| Global | not reread | not reread | 0.0372 | 0.0131 |
| Dense | not reread | not reread | 0.0388 | 0.0142 |

The Global and Dense means are the 3-seed means quoted in the summary of `docs/router-check.md`. We did not reread their files for this document.

So the frozen model keeps most of the gain of Global over Dense, although its router does no work. Section 3.3 of the frozen-router document gives a reason. With a constant alpha, all experts of a layer get the same gradient and the same AdamW update. The layer then acts as one Dense matrix with a smaller initial scale and a slightly larger step. This document describes a direct test of that reason. We train a plain Dense model whose 24 SO(2) linear weights start and move exactly as the effective matrix of the frozen model does.

The question is simple. Does Dense with the frozen model's initial scale and step size reach the frozen result?

## 2. The effective-scale argument

### 2.1 Notation

- K = 8 is the number of experts, and a = 0.13 is the constant coefficient of each expert.
- W_k (k = 1 to K) is the weight matrix of expert k in one MoLE layer, with fan-in n.
- W_eff = a (W_1 + ... + W_K) is the matrix that the layer applies.
- L is the loss and G = dL/dW_eff is its gradient with respect to W_eff.
- eta is the learning rate of a step, lambda the weight decay, eps the Adam epsilon (1e-8), and beta_1, beta_2 the Adam decay rates.
- m and v are the Adam moment estimates of a gradient sequence; a hat marks the bias-corrected value.

### 2.2 Gradient and update

The layer output depends on W_k only through W_eff, so

    (1)  dL/dW_k = a G        for every k.

All experts get the same gradient. Adam keeps elementwise moments, so for expert k

    (2)  m_k = a m(G),   v_k = a^2 v(G),

and the bias corrections are the same for every expert. The Adam direction of expert k is

    (3)  u_k = m_k_hat / (sqrt(v_k_hat) + eps) = m_hat / (sqrt(v_hat) + eps / a).

So all experts get the same direction u, and its epsilon is eps / a. torch AdamW first applies the decoupled decay and then the step:

    (4)  W_k <- W_k (1 - eta lambda) - eta u.

We sum (4) over k and multiply by a:

    (5)  W_eff <- W_eff (1 - eta lambda) - K a eta u.

Equation (5) is the update of one Dense matrix with three properties:

- the step is K a eta = 8 x 0.13 = 1.04 eta;
- the decay per step is eta lambda, the Dense value, because decay is linear and acts on each expert in the same way;
- the Adam epsilon is eps / a = 7.69e-8 instead of 1e-8.

### 2.3 Matching a Dense group

A Dense weight in an AdamW group with learning rate eta', decay lambda' and epsilon eps' moves by

    (6)  W <- W (1 - eta' lambda') - eta' m_hat / (sqrt(v_hat) + eps').

Equation (6) equals (5) when

    (7)  eta' = 1.04 eta,   lambda' = lambda / 1.04,   eps' = eps / a.

The weight-decay detail is the second condition. If we kept lambda' = lambda, the decay per step would be 1.04 eta lambda, 4 percent stronger than in the frozen model. With the pilot values (eta at most 2e-4, lambda 1e-3), the decay per step is at most 2e-7, so the difference is small. The tiny-model check in section 5 measures it.

### 2.4 Initial scale

fairchem's `init_linear` draws each expert from U(-k, k) with k = 1 / sqrt(n). The entries of W_eff are sums of 8 independent such draws times a, so their variance is a^2 K k^2 / 3. Their standard deviation is

    (8)  a sqrt(K) = 0.13 x sqrt(8) = 0.368

times that of one expert. The Dense layers differ from one expert in one detail. Dense's `fc_m0` draws U(-k, k), but every `SO2_m_Conv.fc` (m = 1 and m = 2) multiplies its draw by 1 / sqrt(2) at construction. A MoLE expert does not. So, relative to the Dense initial weights, the frozen W_eff is 0.368 times larger for the 8 `fc_m0` weights and 0.368 x sqrt(2) = 0.520 times larger for the 16 `so2_m_conv` weights. Section 5.3 of the router check compared W_eff with one expert. Its statement that the MoLE layer starts at 0.368 times the Dense layer holds for the 8 `fc_m0` weights only.

One difference remains. W_eff at step 0 is a sum of 8 uniform draws, so its entries are close to normal. The scaled Dense weight is one uniform draw times a constant. Both have the same variance but not the same distribution.

## 3. Design

### 3.1 What is scaled

The scaled weights are exactly the weights that MoLE replaces in Global. fairchem's `recursive_replace_so2_linear` replaces `fc_m0` and every `so2_m_conv[i].fc` of each `SO2_Convolution` in the 4 blocks. Each block has two such convolutions, so there are 4 x 2 x 3 = 24 weight tensors with 4,521,984 parameters. Their biases are not scaled. MoLE keeps one shared bias per layer, which is not mixed by alpha and which starts as a Dense bias, so it trains at the Dense rate in the frozen model too.

| Weight (each block, 4 blocks) | Count | Initial factor relative to Dense | Learning rate | Weight decay | Adam eps |
|---|---:|---:|---:|---:|---:|
| `so2_conv_1.fc_m0`, `so2_conv_2.fc_m0` | 8 | 0.368 | 1.04 x | 1/1.04 x | 1/0.13 x |
| `so2_conv_{1,2}.so2_m_conv.{0,1}.fc` | 16 | 0.520 | 1.04 x | 1/1.04 x | 1/0.13 x |
| all other parameters | | 1 (bit-identical to Dense) | Dense | Dense | Dense |

### 3.2 The hook

The variant is `configs/variant/dense_scaled.yaml`. It loads `dense.yaml` through the Hydra defaults list and changes three keys:

```yaml
defaults:
  - dense
  - _self_

backbone:
  model: training.dense_scaled.eSCNMDScaledBackbone
  frozen_alpha: 0.13
  frozen_experts: 8
```

`training/dense_scaled.py` holds two small pieces.

1. `eSCNMDScaledBackbone`, a subclass of fairchem's Dense `eSCNMDBackbone`. After the Dense constructor it multiplies the 24 weights by 0.368 (`fc_m0`) or 0.520 (`so2_m_conv`) and tags each of them with `lr_scale = 1.04` and `eps_scale = 1 / 0.13`. It multiplies the Dense draws and draws no new random numbers, so every other parameter equals Dense for the same seed. The three factors (0.368, 1.04 and 1 / 0.13) are derived from `frozen_alpha` and `frozen_experts` in one place.
2. `lr_scaled`, a wrapper around the optimizer partial. It puts the tagged parameters into a second AdamW group with lr x 1.04, weight decay / 1.04 and eps / 0.13, and leaves the first group as in Dense. A model without tags gets exactly the call it got before. The training unit (`training/guarded_unit.py`) applies the wrapper to `optimizer_fn` in its constructor, which is one added line. The cosine scheduler is a `LambdaLR`, so it multiplies both groups by the same factor and keeps the ratio 1.04 at every step. fairchem logs `train/lr` from group 0, so the logged rate is still the Dense rate.

Why a training-unit hook and not a config key? The optimizer lives in `configs/train.yaml`, which every variant shares. A change there would change the composed config of every variant. The variant file can only change the model. So the model tags its parameters and the unit reads the tags. fairchem calls `optimizer_fn(params=model.parameters())` for our model, because `HydraModel` has no `no_weight_decay` method.

Everything else is the Dense pilot recipe: the fixed lists `manifests/pilot_train_200k.txt` and `pilot_val_5k.txt`, 50,000 steps, the cosine schedule with 5 percent warmup, AdamW with lr 2e-4 and weight decay 0.001, clipping at 10, EMA 0.999, batches of at most 450 atoms, float32, validation and a checkpoint every 5,000 steps, the OOM monitor, offline W&B, the size-consistency stage, and one V100 with 32 GB.

## 4. Tests

`tests/test_dense_scaled.py` adds 9 tests on small float64 models built from the real configs.

1. `test_existing_variants_compose_as_on_main`: the composed configs of Dense, Global, Local-6 and `global_frozen` have the SHA-256 that we measured on main (4cbd05f), without the job sections. The full-size parameter counts are 5,981,057 (Dense and `dense_scaled`) and 37,652,265 (Global, Local-6 and `global_frozen`).
2. `test_variant_is_dense_with_scaled_mole_layers`: the model config equals Dense except the three keys. The full-size model scales 24 tensors, and their names are exactly the names of the 24 MoLE layers of Global.
3. `test_initial_scale_and_other_parameters_bit_identical`: with the same seed, every untagged parameter is bit-identical to Dense. Each tagged weight equals the Dense weight times 0.368 or 0.520 (relative tolerance 1e-6, the float32 rounding of the init). The norm of each tagged weight and of the frozen model's W_eff lies within 10 percent of the expected norm sqrt(numel / 3) k x 0.368.
4. `test_optimizer_group_has_the_lr_multiplier`: two groups; group 0 has lr 2e-4 and weight decay 1e-3; the ratio of learning rates is 1.04 (12 decimals); the decay per step is equal; the eps ratio is 1 / 0.13; the ratio stays 1.04 for 30 scheduler steps. For Dense, Global, Local-6 and `global_frozen` the wrapper gives one group with the same parameters in the same order as before.
5. `test_training_unit_wraps_the_optimizer`: the training unit passes the wrapped partial to fairchem.
6. `test_forward_is_a_dense_output`: finite energy and forces of the right shapes; the state dict loads into a plain Dense model with `strict=True` and gives the same output.
7. `test_inference_checkpoint_loads`: a checkpoint in fairchem's inference format loads through `MLIPPredictUnit`, rebuilds the scaled backbone and gives the same energy (1e-12).
8. `test_job_file_carries_the_pilot_recipe`: account, partition, GPU, 48 h, the two email lines, an absolute output path, the defaults, no array, `repo_dscaled`.
9. `test_follows_the_frozen_model_over_adamw_steps`: the tiny-model equivalence check of section 5.

Results, `python -m unittest discover -s tests` in `IDLF26-mlip` on the laptop CPU:

- Before, on main (4cbd05f): 138 tests, OK, in 36.9 s.
- After, on this branch: 147 tests, OK (the 138 old tests and the 9 new ones), in 51.5 s.

Deliberate break. We replaced the scaling factor by 1.0 in `eSCNMDScaledBackbone`. Test 3 failed and the other 8 passed. Test 9 still passed, because it copies the frozen weights into the scaled model before the steps. We restored the file (byte-identical, checked with `cmp`) and all 9 tests passed again.

## 5. Tiny-model equivalence check

We built the tiny `global_frozen` and `dense_scaled` models (2 blocks, 16 channels, lmax 1, float64), set each scaled Dense weight to W_eff = 0.13 x sum_k W_k of the frozen model and copied every other shared tensor, so both start as the same function. Then we ran 5 AdamW steps on both with the pilot values (lr 2e-4, weight decay 1e-3, clipping at 10), one small molecule per step (CH3OH, H2O, CH4, CH3OH, H2O), loss |E| + sum |F|^2. After the steps we compared the scaled weights with 0.13 x sum_k W_k, the other parameters, and energy and forces on ethanol, which the steps did not see. The largest change of a scaled weight over the 5 steps was 1.04e-3 (measured).

Largest absolute difference after 5 steps (measured, `trajectory_gap` in the test file):

| Setting | Scaled weights | Other parameters | Energy | Forces |
|---|---:|---:|---:|---:|
| As built (eps / 0.13, decay / 1.04) | 6.9e-17 | 5.6e-17 | 2.2e-16 | 2.4e-17 |
| Same, without clipping | 6.9e-17 | 5.6e-17 | 2.2e-16 | 2.4e-17 |
| Dense eps (1e-8) in the scaled group | 3.0e-4 | 5.1e-6 | 2.0e-6 | 2.4e-7 |
| Weight decay not divided by 1.04 | 7.9e-9 | 1.2e-10 | 6.4e-9 | 3.1e-10 |

Interpretation, kept apart from the table:

- As built, the two models follow the same trajectory up to float64 rounding. This confirms equations (5) to (7) on this model.
- Clipping did not act in these steps (the rows with and without it are equal), so this check does not test the clipping difference of section 10.
- The Adam epsilon matters most on the tiny model: with Dense's eps the scaled weights drift by up to 3.0e-4 in 5 steps, about 28 percent of the largest step. Entries with a very small gradient feel the epsilon. This is why we match eps too. How much it matters on the full model depends on how many gradient entries fall near 1e-7, which we did not measure.
- The weight-decay choice changes the scaled weights by 7.9e-9 in 5 steps. It is small but not zero, and with the division the match is exact.

## 6. Smoke test

Job 49532915, 2026-10-21, a 30-step run like the earlier `pilottest` and `frozentest` jobs: the real job file, the real fixed lists, seed 0, a validation at step 30 and a checkpoint every 15 steps, a 1 h limit. The code on Bridges-2 was `/ocean/projects/che260063p/shared/repo_dscaled/AI4Science-MLIPs`, synced from this branch with `b2.py syncrepo` (202 tracked files, all identical by SHA-256, 0 extra).

`b2.py submit` uploads a job file and runs a plain `sbatch` on it, with no way to add `--export`. The login shell sets `SBATCH_EXPORT=NONE`, so no variable from the shell reaches the job, and the job file defaults are the settings (guide B9 row 18 allows settings in the job file). So each job file holds its own settings: the committed `slurm/dense_scaled.job` is the long run for seed 0, and the smoke file and the per-seed files are copies made with `sed` that change only these lines:

```bash
# smoke test
sed -e 's/^#SBATCH -t 48:00:00/#SBATCH -t 1:00:00/' -e 's/^#SBATCH --job-name=dense_scaled/#SBATCH --job-name=dscaledtest/' \
    -e 's/PILOT_STEPS:-50000/PILOT_STEPS:-30/' -e 's/PILOT_EVAL_EVERY:-5000/PILOT_EVAL_EVERY:-30/' \
    -e 's/PILOT_CKPT_EVERY:-5000/PILOT_CKPT_EVERY:-15/' -e 's/PILOT_RUN_PREFIX:-dense}/PILOT_RUN_PREFIX:-dscaledtest}/' \
    slurm/dense_scaled.job > dscaledtest.job
# seed N of the long run (N = 0, 1, 2)
sed -e "s/PILOT_SEED:-0}/PILOT_SEED:-N}/" -e "s/^#SBATCH --job-name=dense_scaled/#SBATCH --job-name=dense_scaled_sN/" \
    slurm/dense_scaled.job > dense_scaled_sN.job
```

Each file went first through `b2.py submit FILE /ocean/projects/che260063p/shared/slurm` (upload, `bash -n`, `sbatch --test-only`, no job) and then through the same command with `--yes`. After each job started we read its `settings.txt`.

Measured, from `runs/dscaledtest_scaled_s0_49532915/summary.txt`, `settings.txt`, `logs/train.log`, the final training checkpoint and `sacct`:

| Item | Value |
|---|---|
| settings.txt | `variant=dense_scaled steps=30 seed=0`, `pilot_train_200k.txt sha256 de75c47ff4dae8da`, `pilot_val_5k.txt sha256 7ecd0a5457118d8e`, `eval_every=30 ckpt_every=15`, `array_task=none`, `repo=.../repo_dscaled/AI4Science-MLIPs` |
| GPU | Tesla V100-SXM2-32GB, node v003 |
| verify_samples | 8 of 8 PASS |
| train | exit 0, 736 s, peak GPU memory 9,896 MiB, 669.3 atoms per second at the last step |
| Validation at step 30 | loss 0.7783, energy MAE per atom 0.1386 eV, force MAE 0.4897 eV/A |
| OOM | 0 events (no `OOM:` line; `OOM events: 0 train (0 recovered), 0 eval (0 recovered)`); no `nan` in the log |
| Optimizer groups, final checkpoint | group 0: 102 tensors, initial lr 2e-4, weight decay 1e-3, eps 1e-8; group 1: 24 tensors, initial lr 2.08e-4, weight decay 9.615e-4, eps 7.692e-8 |
| size_consistency | Delta 1.86265e-06 meV at 8 A and at 100 A |
| Result | `RESULT: all stages passed`; sacct COMPLETED, exit 0:0, elapsed 18 min 24 s |
| Cost | about 0.31 SU (1 V100 for 18 min 24 s at 1 SU per GPU-hour; our calculation from sacct) |

We read the optimizer groups from the torch distributed checkpoint (`checkpoints/final`, keys `unit_state.optim.param_groups.*`) on a login node. They show that the hook acts in the real training pipeline, not only in the unit tests.

The training loss over 30 steps (mean of the first 10 and of the last 10 values of `train/loss` in train.log), compared with the earlier 30-step tests of the same seed and the same batches:

| Run | Steps 1 to 10 | Steps 21 to 30 | Val force MAE | Val energy MAE per atom |
|---|---:|---:|---:|---:|
| Scaled Dense, `dscaledtest_scaled_s0_49532915` | 0.8469 | 0.8979 | 0.4897 | 0.1386 |
| Dense, `pilottest_dense_s0_49041064` | 0.8609 | 0.9002 | 0.4982 | 0.1321 |
| Frozen, `frozentest_global_s0_49442923` | 0.8662 | 0.9127 | 0.5041 | 0.1415 |

Interpretation, kept apart from the measurements:

- The loss behaves as in the earlier 30-step tests. In all three, the variation between batches is larger than what 30 steps at the warmup learning rate can change, so the last 10 steps are not lower than the first 10. The validation at step 5,000 of the long run is the real check that the loss falls.
- The scaled model starts as a different function from Dense (smaller SO(2) weights), so its step-30 values differ from Dense by about as much as the frozen values do. This small test cannot rank the models.
- The Delta of 1.9e-06 meV is the Dense class, as expected for a Dense model.

## 7. The jobs

Three single-task jobs, one per seed, all from `slurm/dense_scaled.job` (seeds 1 and 2 through the `sed` line of section 6) and the code in `repo_dscaled`. Each was submitted with `b2.py submit FILE /ocean/projects/che260063p/shared/slurm`, first as a dry run and then with `--yes`. Seed 0 went first. We submitted seeds 1 and 2 only after the first validation of seed 0 looked sane.

| Seed | Job | Submitted | Started | Node | Run folder |
|---:|---:|---|---|---|---|
| 0 | 49533514 | 2026-10-21 01:19 | 01:22 | v009 | `runs/dense_scaled_s0_49533514` |
| 1 | 49535801 | 2026-10-21 02:31 | 02:46 | v009 | `runs/dense_scaled_s1_49535801` |
| 2 | 49535804 | 2026-10-21 02:31 | 02:46 | v009 | `runs/dense_scaled_s2_49535804` |

All three: account `che260063p`, GPU-shared, one `v100-32`, limit 48 h, emails for all events and at 80 percent of the limit, output in `/ocean/projects/che260063p/shared/logs/slurm-<jobid>.out`.

settings.txt, read after each job started (measured):

- seed 0: `variant=dense_scaled steps=50000 seed=0`, `pilot_train_200k.txt sha256 de75c47ff4dae8da`, `pilot_val_5k.txt sha256 7ecd0a5457118d8e`, `eval_every=5000 ckpt_every=5000`, `array_task=none`, `repo=/ocean/projects/che260063p/shared/repo_dscaled/AI4Science-MLIPs`.
- seeds 1 and 2: the same, with `seed=1` and `seed=2`; both also ran on v009, so all three jobs share one node (one GPU each).
- verify_samples passed 8 of 8 checks in each of the three jobs; training started at 01:26:56 (seed 0) and 02:50:14 (seeds 1 and 2).

First validation of seed 0, step 5,000 (measured, `logs/train.log`, 4 decimals). The other rows are the step-5,000 values of the same seed quoted in `docs/frozen-router.md` (section 6):

| Run | Force MAE (eV/A) | Energy MAE per atom (eV) |
|---|---:|---:|
| Scaled Dense, seed 0 | 0.1097 | 0.0490 |
| Frozen, seed 0 | 0.1104 | 0.0503 |
| Global, seed 0 | 0.1099 | 0.0503 |
| Local-6, seed 0 | 0.1104 | 0.0509 |
| Dense, seed 0 | 0.1063 | 0.0479 |

The validation loss was 0.1852. The mean training loss fell from 0.5804 (steps 1 to 1,000) to 0.2691 (steps 4,001 to 5,000). The log has no `OOM:` line and no `nan`. The checkpoints `step_0` and `step_5000` exist. Training reached step 5,162 at 02:30:27, 63.5 minutes after the train stage started at 01:26:56, so about 0.74 s per step including the startup and the first validation.

At step 5,000 the models are still close, and in the pilots the gap between Dense and the MoLE models opened only after step 10,000 (`docs/frozen-router.md`, section 6). So this first point cannot separate the outcomes of section 8.

Expected end (reasoning from the rate above: 50,000 steps at about 0.74 s per step is about 10.3 h, plus validations and the size check): seed 0 about 12:00 on 2026-10-21. Seeds 1 and 2 started at 02:46, so they should end about 13:30 to 14:00 on 2026-10-21, if three jobs on one node run at the speed of one.

Cost estimate: about 10.5 to 11 SU per seed, so about 32 to 33 SU for the three seeds, plus 0.31 SU for the smoke test. The cap is 48 SU per seed (144 SU in the worst case). The project had 2,755 of 2,950 GPU SU before these jobs (`projects`, 2026-10-21, before the smoke test).

Results will appear in each run folder: `summary.txt` (stages, last validation, OOM line, Delta), `logs/train.log`, the checkpoints every 5,000 steps under `train/*/checkpoints/`, and the offline W&B run under `train/*/logs/wandb/`.

## 8. What the outcomes would mean

Let S be the 3-seed mean final force MAE of the scaled Dense model, F = 0.0375 the frozen mean, G = 0.0372 the Global mean and D = 0.0388 the Dense mean. The gap D - F is 0.0013 eV/A. The same reading holds for the energy (frozen 0.0131, Dense 0.0142).

Outcome A. S is close to F. Then Dense with the frozen model's initial scale and step size trains as well as the frozen model. The gain of the frozen model over Dense comes from the initial scale and the step size of the 24 SO(2) weights, not from the larger expert bank. The bank, under a constant router, is only a costly way to set these two numbers. Together with the frozen result, the gain of Global over Dense would then also be mostly a matter of parameterisation, and the router would add only the small difference between Global and the frozen model.

Outcome B. S stays close to D. Then the scale and the step size alone do not give the gain, although equations (5) to (7) say that they reproduce the update of W_eff. What remains different is the distribution of the initial weights (a sum of 8 uniform draws against one scaled uniform draw), the different random draws, and gradient clipping, which acts on a different total norm. Then one of these details matters, or the equivalence fails on the full model in a way the tiny model did not show. The expert bank would then matter even with a constant router.

An intermediate S is reported as the retained share r = (D - S) / (D - F), the share of the frozen model's gain over Dense that the scaled Dense model keeps. Both outcomes carry the same weight in the report.

How large a difference is real? The seed spread is the scale. The frozen seeds end at 0.0374, 0.0376 and 0.0374 eV/A (range 0.0002). With 3 seeds per model, a Welch t test has about 2 to 4 degrees of freedom, so only a large and consistent difference reaches p below 0.05.

## 9. How to analyse

Compare the validation curves at every 5,000 steps, seed by seed, with Dense, Global, Local-6 and the frozen model of the same seed. `scripts/compare_val_curves.py` reads `logs/train.log` of each run and labels each validation with its step. It computes the share only for the frozen run, so we compute r for the scaled runs with the short script below. train.log prints 4 decimals. The full precision is in the offline W&B files (`train/*/logs/wandb/offline-run-*`).

On Bridges-2 when the jobs end (a few seconds of log parsing, so a login node is enough):

```bash
cd /ocean/projects/che260063p/shared/repo_dscaled/AI4Science-MLIPs
R=/ocean/projects/che260063p/shared/runs
PY=/ocean/projects/che260063p/shared/envs/mlip/bin/python
$PY -I scripts/compare_val_curves.py $R/pilot_dense_s0_49041937 $R/pilot_global_s0_49041939 $R/pilot_local6_s0_49041936 $R/frozen_global_s0_49444920 $R/dense_scaled_s0_49533514
$PY -I scripts/compare_val_curves.py $R/pilot_dense_s1_49041938 $R/pilot_global_s1_49041940 $R/pilot_local6_s1_49041974 $R/frozen_global_s1_49450911 $R/dense_scaled_s1_49535801
$PY -I scripts/compare_val_curves.py $R/pilot_dense_s2_49041928 $R/pilot_global_s2_49041929 $R/pilot_local6_s2_49041927 $R/frozen_global_s2_49450922 $R/dense_scaled_s2_49535804
grep -a -E 'Delta|OOM events|val/|RESULT' $R/dense_scaled_s*/summary.txt
```

Final values, Welch tests and the retained share (run from the same folder):

```bash
$PY -I - <<'EOF'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
from scipy import stats
from compare_val_curves import curve
R = Path("/ocean/projects/che260063p/shared/runs")
runs = {"dense": ["pilot_dense_s0_49041937", "pilot_dense_s1_49041938", "pilot_dense_s2_49041928"],
        "global": ["pilot_global_s0_49041939", "pilot_global_s1_49041940", "pilot_global_s2_49041929"],
        "local6": ["pilot_local6_s0_49041936", "pilot_local6_s1_49041974", "pilot_local6_s2_49041927"],
        "frozen": ["frozen_global_s0_49444920", "frozen_global_s1_49450911", "frozen_global_s2_49450922"],
        "scaled": ["dense_scaled_s0_49533514", "dense_scaled_s1_49535801", "dense_scaled_s2_49535804"]}
final = {m: {k: [curve(R / r / "logs" / "train.log")[50000][k] for r in rs] for k in ("force", "energy")} for m, rs in runs.items()}
for k in ("force", "energy"):
    mean = {m: sum(v[k]) / 3 for m, v in final.items()}
    print(k, {m: round(x, 5) for m, x in mean.items()})
    for other in ("dense", "frozen", "global", "local6"):
        t = stats.ttest_ind(final["scaled"][k], final[other][k], equal_var=False)
        print(f"  scaled vs {other}: t {t.statistic:.2f}, p {t.pvalue:.3f}")
    print(f"  retained share (D - S) / (D - F): {(mean['dense'] - mean['scaled']) / (mean['dense'] - mean['frozen']):.2f}")
EOF
```

Then check three things:

- the final values against section 8;
- the shape of the curve: if the scaled run lags early and catches up late, the effect is on the speed of learning, not the end point;
- the size-consistency Delta in `summary.txt`, which should be near 0, because the model is a Dense model.

## 10. Caveats

- Three seeds per model. A Welch test with 3 against 3 runs has little power, and the frozen and Global means differ by only 0.0003 eV/A.
- The equivalence holds up to the initial distribution and gradient clipping. Equations (5) to (7) match the update rule exactly, including weight decay and the Adam epsilon. They do not match the shape of the initial weights (section 2.4) or the clipping, which acts on the total gradient norm. The 24 MoLE layers contribute 0.368 times their W_eff gradient norm to that total in the frozen model (8 experts, each with a times the gradient), and the full gradient norm in the scaled Dense model. When clipping acts, the two models take differently scaled steps. The tiny-model check did not exercise clipping.
- One design. We test one parameterisation (scale and step of the uniform frozen point). Other constants, or a scale without the step change, would need more runs.
- Short training. 50,000 steps are about 1/760 of the official eSEN run.
- The different random draws. For the same seed, the scaled Dense model shares every untagged initial weight with Dense. With the frozen model it shares most backbone weights, because fairchem builds the Dense backbone first and adds the experts afterwards. On the tiny model, 54 of the 64 shared tensors were equal; the `fc_m0` biases and the energy head differed, because the expert draws come before them (measured). The 24 weights themselves are different draws. So a seed-for-seed comparison with the frozen model compares different initial functions.

## 11. Files

- `configs/variant/dense_scaled.yaml`: the variant.
- `training/dense_scaled.py`: the backbone and the optimizer-group wrapper.
- `training/guarded_unit.py`: one line that applies the wrapper.
- `slurm/dense_scaled.job`: the job; `slurm/pilot.sh` accepts `dense_scaled`.
- `tests/test_dense_scaled.py`: the tests and the equivalence check.
