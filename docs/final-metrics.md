# Collect final validation metrics

Script: `scripts/collect_final_metrics.py`. Tests: `tests/test_collect_final_metrics.py`.

This script reads local files only. It does not load model checkpoints, train,
query W&B, authenticate or upload anything. Log/CSV processing uses the Python
standard library. Binary W&B records require the installed `wandb` protobufs;
the optional canonical-config fallback uses PyYAML. Both are in requirements.txt.

## Output and strict defaults

The CSV has exactly the existing `results/pilot_final_metrics.csv` columns:

```text
variant,seed,energy_mae_per_atom_eV,force_mae_eV_per_A,val_loss
```

Defaults require all eight variants (`dense`, `global`, `element`, `local6`,
`local12`, `local_global`, `frozen`, `dense_scaled`), exactly seeds 0..4 for each,
and 50,000 completed training steps: 40 rows. `global_frozen` is normalized to
`frozen`. The metric values are not rounded when writing the CSV.

Beside `final_metrics.csv`, the script writes `final_metrics.provenance.json`.
It records completeness, missing variant/seed pairs, run paths, ordered source
files, metadata source, target steps, data identifiers and metric precision.
It includes a SHA-256 of the CSV, checked when that table is merged later.

Duplicate variant/seed pairs always fail, including reruns and CSV/run overlap.
Explicitly select which run to include. Missing pairs fail unless
`--allow-partial` is supplied. Partial mode never permits invalid metric values,
failed training stages, wrong step budgets, ambiguous sources or duplicate rows.
All validation finishes before the output files are replaced. An invalid input
leaves an existing output untouched. Output paths cannot overwrite input files.

## Select runs explicitly

Run from the repository root. Supply each run directory, not the parent folder
holding every experiment. No directory-name pattern is used to infer identity.
For example, to inspect one completed run before all 40 runs are available:

```bash
python scripts/collect_final_metrics.py /path/to/run \
  --output results/final_metrics_partial.csv --allow-partial
```

Each run normally has `settings.txt`, written by `slurm/pilot.sh`, with
`variant=... steps=... seed=... data=...`. If absent, the script requires exactly
one resolved `canonical_config.yaml` and reads the saved backbone, job seed and
runner max_steps. Unsupported or unresolved configs are rejected. A final
nonzero `END train exit ...` in summary.txt is also rejected. Failure of a later
audit stage does not by itself invalidate completed training metrics.

For many runs, create a JSON manifest, e.g. `runs.json`:

```json
[
  {"path": "/path/to/wave1_dense_s3"},
  {"path": "/path/to/wave1_element_s0"},
  {"path": "/path/to/wave1_local_global_s0"}
]
```

Relative run paths resolve against the manifest's directory. Supply all 25 new
runs in the real Wave 1 manifest, then merge the 15 existing pilot rows:

```bash
python scripts/collect_final_metrics.py --run-manifest runs.json \
  --merge results/pilot_final_metrics.csv \
  --output results/final_metrics.csv
```

`--merge` is repeatable. A legacy CSV without this script's sidecar cannot prove
its step budget or data identity. Its rows are marked `legacy_csv_unverified`
and `step_verified: false`, and the CLI reports their count. By including that
CSV, the caller asserts it belongs to the selected reporting cohort. The
existing pilot CSV is documented as the 50,000-step cohort. Tables with sidecars
are checked for step-budget conflicts and hash mismatches.

Known data identifiers must agree across the table. Missing identifiers are
reported, never invented. Comparisons are strict: settings.txt uses its data
description; the canonical-config fallback uses serialized dataset configs.
Equivalent data expressed differently may therefore need consistent metadata
before merging. This script does not verify dataset file contents or hashes
against the actual data on disk.

## Source selection and final-event semantics

`--source auto` is the default:

1. Prefer a local `.wandb` history. If none exists, try `wandb-history.jsonl`.
2. If no history exists, parse `logs/train.log` and mark its rows
   `rounded_train_log`. Native fairchem log output has four decimal places.
3. When history and logs both exist, validate both final events and compare
   their metrics with a 0.00005001 absolute rounding tolerance.

Corrupt or ambiguous history is an error, not a silent fallback. Use
`--source log` explicitly to collect from logs only, or `--source wandb` to use
history only. Those choices are visible in the provenance source files.
`wandb-summary.json` is deliberately not a source: its last value for each key
can come from different evaluations and has no reliable validation boundary.

The reader uses `train/step + 1` to obtain completed optimizer steps. It also
accepts an explicit `val/step` in history, interpreted as completed steps.
W&B `_step` and `val/epoch` are never treated as training steps. A history that
only has W&B `_step` is insufficient to identify the final training step.

The final row comes from the latest validation event after the latest training
step. A later incomplete validation fails even if an earlier event was complete.
Duplicate final evaluations use the last one; the best MAE is never selected.
All three metrics must occur within that event and be finite and nonnegative.
The script verifies observed steps and validation completeness, not checkpoint
existence or successful completion of every external job stage.

The binary reader is read-only, checks CRCs and record boundaries, and supports
fragmented version-0 W&B files. It uses the format documented by
[W&B's datastore implementation](https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/internal/datastore.py).
It does not depend on the old DataStore Python API, which is absent from the
locally installed W&B 0.30.0 package. Unsupported formats and truncated records
are rejected.

## Resumed and long runs

An appended train.log may contain several attempts. A later training row
invalidates preceding validation values, including when its step counter goes
backward on resume. Multiple log/history files require explicit segment order:

```json
[
  {
    "path": "/path/to/resumed_run",
    "logs": ["logs/segment1.log", "logs/segment2.log"],
    "wandb": ["train/segment1/run-a.wandb", "train/segment2/run-b.wandb"]
  }
]
```

Source paths resolve relative to that run. Do not order segments by modification
time or filesystem glob order. Only the last listed segment can provide the
final event; a stale final event in an earlier segment is not accepted. Each
segment must carry its own training-step evidence or explicit validation step.
If auto-discovery finds multiple histories, it asks for this ordered manifest.

Collect long-run results in a separate table, with explicit settings:

```bash
python scripts/collect_final_metrics.py --run-manifest long_runs.json \
  --steps 300000 --expected-seeds 2 --output results/long_final_metrics.csv
```

`--variants element local_global` can limit a deliberate subset report.
`--expected-seeds N` requires exactly seeds 0..N-1 within that report.
Do not combine the 50,000-step and 300,000-step cohorts.

## Validation

```bash
python -B -m unittest discover -s tests -p 'test_collect_final_metrics.py' -v
```

Tests cover log event boundaries, scientific notation, missing/invalid metrics,
resume segments, aliases, duplicate/missing seeds, binary CRCs and fragmentation,
full-precision history selection, source disagreement, legacy CSV merging,
provenance hashes and output preservation on error. A fixture combines the real
15-row pilot CSV with 25 synthetic new runs to check the 40-row contract; these
synthetic results are not experimental measurements.

On 2026-11-01, all 30 collector tests passed, and the complete regression suite
passed 218 tests in 71.617 seconds. CLI help and `git diff --check` passed too.
Binary tests use generated W&B protobuf records with real version-0 framing,
including multi-block records and deliberate corruption. An additional attempt
to create a live offline run with the local W&B 0.30.0 SDK did not finish startup
and was stopped; it is not counted as a successful integration check.

No new-run metrics have been collected from the cluster by this implementation
task. The existing pilot CSV was not modified.
