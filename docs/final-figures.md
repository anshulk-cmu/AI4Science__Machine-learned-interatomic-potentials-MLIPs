# Accuracy, long-training and ensemble figures

The plotting entry points are `scripts/make_accuracy_figure.py` and
`scripts/make_ensemble_figure.py`. Both export PDF, PNG (300 dpi) or SVG according
to the output filename. Output directories are created when needed. Shared
styles, validation and seed summaries live in `scripts/figure_common.py`.

## Five-seed final accuracy

After collecting final metrics, run from the repository root:

```bash
python scripts/make_accuracy_figure.py results/figures/final_accuracy.pdf --metrics results/final_metrics.csv --metrics-only --require-final
```

This produces two panels: energy MAE per atom (eV) and force MAE (eV/A).
`--require-final` requires all eight variants and seeds 0 through 4, exactly as
in the statistics script. The accepted columns are `variant`, `seed`,
`energy_mae_per_atom_eV` and `force_mae_eV_per_A`; additional columns such as
`val_loss` are ignored. The collector's output is directly compatible.

To include training and validation curves:

```bash
python scripts/make_accuracy_figure.py results/figures/final_curves.pdf --metrics results/final_metrics.csv --curves results/final_validation_curves.csv --require-final
```

This produces the original four panels: training loss, validation force MAE,
gain over Dense and final force MAE. The curve CSV follows
`results/pilot_validation_curves.csv`, requiring:

```text
variant,seed,step,val_force_mae_eV_per_A,train_loss_mean_preceding_500_steps
```

Training loss must be the mean over the preceding 500 steps, consistent with
the existing CSV definition. Other columns are allowed. The final-metrics
collector does not collect curves; provide a separately extracted curve CSV or
use `--metrics-only`. Custom curves and metrics must be supplied together to
avoid accidentally mixing new runs with the default pilot files.

Curves show the seed mean and shaded minimum-to-maximum seed range, **not a
confidence band**. The gain panel retains the pilot definition: Dense's seed
mean minus each model's individual seed value; it is not a paired seed
difference. It starts at the first available evaluation at or beyond 30% of
the final step (15,000 for the original 50,000-step pilots).

Endpoint panels show each seed at a deterministic horizontal offset, a mean
marker and a 95% Student-t interval when at least three seeds are present.
The half-width is `t.ppf(0.975, n-1) * sd / sqrt(n)`, using sample sd. Five seeds
use t(4); the existing three-seed results use t(2). One- and two-seed tables
are descriptive: points and mean only. Intervals are not clipped at zero.

## Long training

```bash
python scripts/make_accuracy_figure.py results/figures/long_accuracy.pdf --metrics results/long_final_metrics.csv --curves results/long_validation_curves.csv --long-run
```

`--long-run` requires explicit inputs, labels the figure as descriptive and
omits confidence intervals regardless of seed count, following the training
plan. Step ticks and the endpoint title come from the supplied curves, so
300,000-step runs are not mislabeled as 50,000-step runs. The old pilots and
long runs must be kept in separate files.

For unequal training budgets, use an endpoint-only comparison:

```bash
python scripts/make_accuracy_figure.py results/figures/long_accuracy.pdf --metrics results/long_final_metrics.csv --metrics-only --long-run
```

An optional integer `step` column in the metrics CSV adds actual endpoint steps
under each variant's tick label. Multiple endpoints within a variant are shown
separated by `/`. Add these values from verified run metadata if the runs
stopped at different budgets. Without this column, the endpoint-only figure
makes no claim about the final training step. With curves and `step` supplied,
the script verifies that endpoint steps equal the last curve step.

## Ensemble comparison

```bash
python scripts/make_ensemble_figure.py results/ensemble_report_metrics.csv results/figures/ensemble_accuracy.pdf
```

This produces energy and force panels comparing methods within each variant.
Colours identify variants and marker shapes identify methods. The input is a
new report-evaluation table (one row per variant, seed and method), requiring:

```text
variant,seed,method,eval_split,energy_mae_per_atom_eV,force_mae_eV_per_A
```

Allowed `method` values:

| Value | Meaning |
|---|---|
| `final` | Final checkpoint from the run |
| `best` | Single checkpoint selected on the separate selection set |
| `weight_average` | Evaluation of the checkpoint weight-average model |
| `prediction_average` | Evaluation of averaged checkpoint predictions |

Both `final` and `best` are required, together with at least one of the two
ensemble methods. Every method must cover the same variant/seed pairs. A table
may contain selected variants, but all present variants must have the same
seed IDs. `eval_split` must be `report` on every row; do not substitute the
selection-set scores. This is a schema check, not proof of dataset provenance.
Use the recorded report set and selection rule from the evaluation pipeline.

The figure consumes evaluations of the actual ensemble: it does **not** average
checkpoint MAEs to estimate ensemble performance, select checkpoints or create
an ensemble. The C4 report-evaluation tool and real ensemble results are still
separate work. No synthetic metrics are installed in `results/` as new results.

For fewer than three seeds, this figure automatically omits confidence
intervals. Add `--descriptive` for a long-run comparison with more seeds. All
entry points accept `--title` to identify a dataset or cohort explicitly.

## Validation, styles and compatibility

The eight variants use the complete Okabe-Ito palette. Original colours are
preserved; Element uses black, Local-12 sky blue and Local+Global yellow, with
distinct markers and dark marker edges to keep the yellow points visible.
`global_frozen` is accepted as an alias for `frozen`.

Empty tables, unknown variants, duplicate keys, inconsistent seed sets,
noninteger/negative seeds or steps, and missing/nonfinite/negative metrics are
rejected. Curve force errors must be positive for the logarithmic axis. Every
curve run must have the same evaluation-step grid and must match the metrics
table's variant/seed pairs. Missing curve rows fail rather than shrinking the
seed range silently. Use endpoint-only plots for unequal or unavailable curves.
Plotting validates tables; training completion and compatible datasets must be
checked using the metrics collector's provenance records.

The legacy command still reads the original pilot CSVs:

```bash
python scripts/make_accuracy_figure.py results/figures/pilot_accuracy.pdf
```

The figure layout is wider to accommodate eight variants. Numerical summaries
retain the old definitions, but the layout is not pixel-identical to the old
midterm image. The separate `make_midterm_figures.py` audit, size-consistency
and router-ablation figures are unchanged by this E3 accuracy task.

## Tests

```bash
python -m unittest discover -s tests -p test_final_figures.py -v
python -m unittest discover -s tests -p test_report_scripts.py -v
```

Tests inspect plotted means, seed offsets, confidence interval endpoints,
curve means/ranges, Dense gains, all eight variants, long-run axes, ensemble
method coverage, malformed inputs and legacy/explicit CLI exports in PDF,
PNG and SVG. Rendered pilot and clearly labeled synthetic five-seed, long-run
and ensemble figures are also visually checked during development.
