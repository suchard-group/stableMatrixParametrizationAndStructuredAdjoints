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

## License

MIT — see `LICENSE`.
