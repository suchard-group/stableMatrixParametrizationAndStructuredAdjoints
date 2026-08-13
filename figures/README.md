# Benchmark figures

`make_benchmark_figures.qmd` reads CSVs from `../benchmarks/branch/results/`
and writes to `output/` (created on render, not committed):

```bash
quarto render make_benchmark_figures.qmd --to html
```

Only two of this document's figures are actually included in the compiled
manuscript:

| Figure | Included in |
|---|---|
| `ou_exp_adjoint_cached_benchmarks.pdf` | `main.tex` |
| `ou_ssbp_vs_lyapunov_param_stability_summary.pdf` | `Supplement.tex` |

The qmd computes several other benchmark comparisons (a standalone kernel
figure, dt-stress, numerical-stability, lyapunov-param-stability, and
matched-param-stability figures, a gradient-stress figure/table, and
time-series/tree comparisons against other software) that are not
referenced by any compiled manuscript file — some because they were
consolidated into the two figures above, others because they were cut
entirely (one, the gradient-stress pair, only ever existed in a
fully-commented-out section of a draft long-form supplement). Their
chunks are marked `eval=FALSE` rather than deleted, since several of them
compute intermediate data frames (`boundary_summary`, `conditioning_panel_data`,
`lyapunov_param_plot_data`) that the two active figures still depend on —
disabling only each chunk's own plot/save step was the safe way to stop
generating unused output without risking the shared computation. The three
comparison-method benchmark folders those disabled chunks would have read
from (`benchmarks/timeseries/`, `benchmarks/tree/`) and two more disabled
CSVs (`matched_param_stability_results.csv`, `smbp_gradient_stress_results.csv`)
have been removed from this repository entirely, along with
`make_ou_stability_summary_one_row.R` (an alternate one-row layout of the
second figure above that is commented out in `Supplement.tex`).

`stableMatricesTable.tex` and `stableMatricesWithAdjointsTable.tex`
(hand-authored qualitative comparison tables) and `smbp_gradient_stress_table.tex`
(generated from the disabled gradient-stress chunk) are likewise not
referenced by any compiled manuscript file and have been removed.
