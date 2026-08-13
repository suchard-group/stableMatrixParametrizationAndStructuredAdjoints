# Environment

## BEAST

See `BEAST.md` — clone `beast-dev/beast-mcmc`, check out branch
`ou-time-series`, build with `ant`. Required for every `*.xml` run in
`simulations/` and `applications/`.

## Julia

Julia 1.10+ with ILP64 OpenBLAS/LAPACK (as used for the manuscript's
timings). All benchmark scripts under `benchmarks/branch/code/` use only the
standard library (`LinearAlgebra`, `Printf`, `Random`, `Statistics`) — no
`Project.toml`/package installation is needed.

## Python

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib
```

Used by the simulations/BitMEX/Anolis data-generation, analysis, and
plotting scripts.

## R

Three independent sets of R packages, none needed unless you intend to run
that part:

- **Figures** (`figures/make_benchmark_figures.qmd`): `dplyr`, `forcats`,
  `fs`, `ggplot2`, `patchwork`, `readr`, `scales`, `showtext`, `stringr`,
  `sysfonts`, `tibble`, `tidyr`. All from CRAN.
- **Simulation figure** (`simulations/scripts/plot_simulation_figure.R`):
  `dplyr`, `ggplot2`, `patchwork`, `readr`, `scales`. All from CRAN.
- **Anolis phylogeny** (`applications/anole/scripts/plot_anole_phylogeny.R`):
  `ape`, `ggplot2` (CRAN) and `ggtree` (Bioconductor —
  `BiocManager::install("ggtree")`).

CMU Serif and Latin Modern Math fonts give the closest visual match to the
manuscript's rendered figures (see the font-loading block in
`figures/make_benchmark_figures.qmd`); both are optional and fall back to a
system serif font if not installed.
