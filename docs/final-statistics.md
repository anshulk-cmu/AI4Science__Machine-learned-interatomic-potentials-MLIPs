# Final validation statistics

`scripts/pilot_statistics.py` accepts the CSV written by
`scripts/collect_final_metrics.py`. It supports Dense, Global, Element, Local-6,
Local-12, Local+Global, Frozen and Scaled Dense, in that display order. The
`global_frozen` spelling is normalized to `frozen`.

## Run

From the repository root, after collecting the new runs:

```bash
python scripts/pilot_statistics.py results/final_metrics.csv --require-final > results/final_statistics.txt
```

`--require-final` requires exactly eight variants with seeds 0 through 4 (40
rows). The input path is explicit: this command does not create missing metrics
or launch training. Use a separate CSV for each training budget and dataset.
Check the collector's provenance file before interpreting results; this statistics
script validates the numeric table, not run provenance or training completion.

The existing pilot report remains reproducible:

```bash
python scripts/pilot_statistics.py results/pilot_final_metrics.csv
```

Without `--require-final`, the script analyzes the variants present, requiring
Dense and Global and at least two seeds per model. Every variant must have the
same seed IDs. Unknown variants, duplicate variant/seed rows, fractional or
negative seeds, missing metric columns, and nonfinite or negative MAEs fail
before any statistics are printed. Row order does not affect the result.

`--draws` defaults to 1,000,000 and `--rng-seed` to 0. These control the normal
and Student-t Monte Carlo calculations; exact resampling does not use an RNG.

## Report and methods

For energy MAE per atom (eV) and force MAE (eV/A), the script prints:

- Sample count, mean, sample standard deviation and two-sided 95% Student-t
  interval: mean +/- t(0.975, n-1) times sd / sqrt(n).
- Percentage error reduction relative to Dense.
- Welch tests for every pair (28 pairs per metric for eight variants).
- Global-gain retention for each of the six non-baseline variants:
  `(mean(Dense) - mean(X)) / (mean(Dense) - mean(Global))`, plus descriptive
  ratios matched by seed ID.
- Fifth percentiles of the retention distribution from normal draws, exact
  seed bootstrap and Student-t draws, with accepted draw fractions.
- A separate Local-6 decision for each metric: the normal fifth percentile
  must be at least 0.5. Exact and Student-t results are sensitivity checks;
  they do not replace this fixed decision rule.

The analysis preserves the pilot's **independent model draws**. Normal draws
use each mean and sd / sqrt(n); Student-t draws use the same scale and n-1
degrees of freedom (4 for five seeds, 2 for three). These are the report's
uncertainty propagation methods, not a paired bootstrap or a guarantee of
frequentist coverage for a ratio. Matching seed IDs in the descriptive ratios
does not change the independence assumption used in the bounds or Welch tests.
Welch p-values are unadjusted for multiple comparisons.

All three retention bounds condition on Dense error > Global error. The
accepted fraction makes this filtering visible; a small fraction indicates
that the conditional bound is not strong evidence of a positive Global gain.
Filtering happens before division. If no draws satisfy the condition, the bound
is `nan` and the accepted fraction is zero. If the observed mean Global gain is
nonpositive, the reported ratio is `nan` and the Local-6 decision is
`UNAVAILABLE`. Individual seeds with nonpositive gains are shown as `undefined`.
Zero Dense mean makes the percentage reduction undefined; two zero sample
variances make the Welch test undefined. These cases are printed as `nan`.

The formal final analysis uses five seeds. Although smaller exploratory tables
can be processed, the two-seed long-run follow-up remains descriptive under the
training plan; do not treat its printed intervals or threshold output as a
confirmatory result. Do not mix short and long runs in one table.

## Exact resampling without expanding billions of samples

Each model resamples n seeds with replacement n times. A retention calculation
uses three independently resampled models, so the full space is n^(3n):
19,683 combinations for n=3 and **30,517,578,125** for n=5.

The script groups resamples by seed counts. For five seeds there are
choose(9, 5) = 126 groups per model and 126^3 = 2,000,376 group triples.
Each group has multiplicity n! / product(count_i!). Triple weights multiply.
The fifth percentile uses the sorted weighted values and interpolates the two
adjacent ranks exactly as NumPy's default linear percentile would on the
expanded samples, up to floating-point roundoff. Taking an unweighted
percentile over group means would be incorrect.

The allocation guard rejects more than 5,000,000 compressed triples before
resampling. For larger seed counts, `--skip-exact` explicitly omits this method;
normal and Student-t calculations still use the observed n. There is no silent
replacement of exact resampling with a Monte Carlo approximation.

## Verification

```bash
python -m unittest discover -s tests -p test_final_statistics.py -v
python -m unittest discover -s tests -p test_report_scripts.py -v
```

Tests cover the synthetic 40-row analysis, validation failures, dynamic t
degrees of freedom, seed alignment, exact multiplicity and percentile weights,
brute-force bootstrap agreement, undefined gains and the allocation guard.
The existing report tests retain the original three-seed numerical checks.
Synthetic fixtures test the software only and are not experimental results.
