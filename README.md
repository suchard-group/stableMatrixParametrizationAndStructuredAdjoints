# Stable matrix parametrization — companion code

Code and data to reproduce the results in "A stable matrix parametrization
for multivariate Ornstein-Uhlenbeck processes" (H-SSBP: Hurwitz Smooth
Spectral Block Parametrization). Message-passing likelihood and
gradient calculations are implemented in a branch of
[BEAST X](https://github.com/beast-dev/beast-mcmc); this repository holds
everything else: benchmark code, simulation/application configs, compact
outputs used for the paper, and the analysis/plotting scripts that turn them
into every figure and table in the manuscript.

## Layout

```
environment/    how to build BEAST, Python/R/Julia dependencies
benchmarks/     Julia kernel/stability benchmarks feeding the 2 benchmark figures below
figures/        turns benchmarks/ CSVs into the manuscript's benchmark figures
simulations/    K=5 MAP simulation study: generators, summarizers, and compact results
applications/   BitMEX trade-data example and the Anolis phylogenetic example
run_local_beast.sh   minimal single-machine runner: java -jar beast.jar ... on any XML here
```

This repository was trimmed to what the compiled manuscript and supplement
(`main.tex` / `Supplement.tex`) actually cite — see "Excluded from this
repository" below for what was cut and why, and each subdirectory's own
README for details specific to it.

Every subdirectory has its own README with the run settings, BEAST commit,
compact manuscript inputs, and regeneration commands.

## Quickstart

```bash
# 1. Build BEAST (see environment/BEAST.md)
git clone https://github.com/beast-dev/beast-mcmc.git && cd beast-mcmc
git checkout 88a7b1e6a14f06778693581b527984d05509b814 && ant build && cd ..

# 2. Python dependencies for simulation generation/summarization helpers
#    (R/Quarto figure dependencies are listed in environment/README.md)
pip install -r environment/requirements.txt

# 3. Regenerate a figure from the results already shipped here, e.g. the
#    benchmark figures (fast — no BEAST/Julia run needed, just the R figures
#    pipeline over the checked-in CSVs):
cd figures && quarto render make_benchmark_figures.qmd --to html
```

Where raw BEAST outputs are shipped, the corresponding README records the XML,
seed, and regeneration command. The final simulation study ships compact
processed summaries plus scripts that regenerate the XMLs and raw outputs.
To reproduce a run from scratch, use `run_local_beast.sh` with the BEAST jar
you built and the XML you want to rerun.

## Manuscript figure/table -> path in this repo

Checked directly against `\includegraphics`/`\input` in `main.tex` and
`Supplement.tex` (not just file presence in `manuscript/figures/` —
several files there turned out to be unreferenced drafts; see below):

| Manuscript output | Path | Used in |
|---|---|---|
| `ou_exp_adjoint_cached_benchmarks` | `figures/make_benchmark_figures.qmd` (+ `benchmarks/branch/results/*.csv`) | `main.tex` |
| `ou_ssbp_vs_lyapunov_param_stability_summary` | same | `Supplement.tex` |
| `supp_simulations_combined_rmse_final_battery_lambda0p1_r25_best_likelihood` | `simulations/` - see `simulations/README.md` | `main.tex` |
| `bitmex_jan2022_j20_trade_7h_cartesian_map_row` | `applications/bitmex/` | `Supplement.tex`; summarized in `main.tex` |
| `anole_phylogeny_drift_matrix_row_10m_fresh_2m` | `applications/anole/` | `main.tex` |

## Excluded from this repository

While curating this repository, every figure/table file under
`manuscript/figures/` and `manuscript/tables/` was checked against actual
`\includegraphics`/`\input` usage in the compiled manuscript sources, not
just its presence in those folders. Several turned out to be unreferenced —
either superseded by a later consolidated figure, or cut drafts — and their
supporting code/data/runs were removed accordingly:

- **Benchmarks**: a standalone kernel figure, dt-stress, numerical-stability,
  lyapunov-param-stability, and matched-param-stability figures, a
  gradient-stress figure/table, a one-row alternate layout, and comparisons
  against other software's time-series and tree likelihoods
  (`ou_kernel_benchmarks`, `ou_dt_stress_benchmarks`,
  `ou_numerical_stability_benchmarks`, `ou_lyapunov_param_stability_benchmarks`,
  `ou_matched_param_stability_benchmarks`, `ou_gradient_stress_benchmarks`,
  `smbp_gradient_stress_table`, `ou_ssbp_vs_lyapunov_param_stability_summary_one_row`,
  `ou_time_series_benchmarks`, `ou_tree_benchmarks`) are not referenced by
  any compiled manuscript file. See `figures/README.md` for exactly what was
  removed vs. disabled and why.
- Two hand-authored qualitative comparison tables (`stableMatricesTable`,
  `stableMatricesWithAdjointsTable`) are likewise unreferenced.
- The earlier BitMEX 30-minute/60-minute exploratory runs and old row/three-panel
  figures have been archived locally under the ignored
  `applications/bitmex/legacy_30min_work/` folder. The active public BitMEX
  bundle is the January 2022 13:00--20:00 UTC post-shock analysis (see
  `applications/bitmex/README.md`).
- The earlier 200,000-state Anolis run and standalone selection-strength
  heatmaps are unreferenced by the active manuscript. Those local draft
  artifacts are archived under the ignored `applications/anole/legacy_200k_work/`
  folder. The active public Anolis bundle is the 2,000,000-state common-priors
  summary used by `main.tex` (see `applications/anole/README.md`).
- Two additional real-data analyses (an *Aquilegia* pollination dataset and
  a PhysioNet Challenge 2012 time-series analysis) were explored during
  development but do not appear in the manuscript at all, so their private
  code/data are not included here.
- Personal batch-system submission scripts are not included;
  `run_local_beast.sh` and the scripts under `simulations/scripts/` cover
  running generated XMLs on a single machine or any scheduler-neutral task
  runner.

## License

MIT — see `LICENSE`.
