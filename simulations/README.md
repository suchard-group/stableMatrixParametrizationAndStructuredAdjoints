# Simulation study

This folder contains the reproducible code and compact result summaries for the
K=5 MAP simulation study reported in the paper.

## Design

The study uses one irregularly sampled trajectory with 800 observation times
per simulated data set.
All five coordinates are observed at every time point.
Inter-observation gaps are independent Uniform(0, 0.1) draws, the equilibrium
mean is fixed at zero, and the initial state is drawn from the stationary
distribution.

Three grids are used:

| Panel | Grid |
|---|---|
| Real/complex boundary | omega / omega* = 0.5, 0.75, 0.95, 1, 1.05, 1.25, 1.5 |
| Non-orthogonal shear | shear multiplier = 3, 4, 5, 6, 7, 8 |
| Jordan(4,1) coupling | coupling = 0.25, 0.5, 0.75, 1, 1.25, 1.5 |

For each grid point there are 25 independently simulated data sets.
For each data set and fit family there are five small-random starts.
The fit families are:

- orthogonal H-SSBP;
- dense-R H-SSBP with global normalization and shrinkage prior
  log p(R) = C + lambda log |det R| - lambda ||R||_F^2 / 2, lambda = 0.1.

Each MAP optimization uses 5000 L-BFGS iterations.
The reported estimate for a data set and fit family is the completed start with
the highest final likelihood, with posterior used only as a tie-breaker.
The stabilization diagnostic is the relative log-posterior improvement between
the last two recorded optimizer evaluations; the manuscript figure uses
tolerance 1e-6.

## Layout

```
scripts/generate_inputs.py       writes BEAST XML tasks and simulated data
scripts/run_map_task.py          runs one generated XML task with a BEAST jar
scripts/summarize_maps.py        summarizes completed MAP outputs
scripts/plot_simulation_figure.R regenerates the three-panel RMSE figure
results/processed/               compact summaries used for the paper figure
```

Generated XMLs and raw BEAST outputs are intentionally not checked in.
They are written under `runs/` and `results/raw/`, which are ignored because
they can be regenerated from the scripts above.

## Reproduce from scratch

From the repository root:

```bash
python3 simulations/scripts/generate_inputs.py --force
```

This writes `simulations/runs/final_battery/inputs/task_manifest.tsv` and one
XML file per MAP start.
To run a task:

```bash
python3 simulations/scripts/run_map_task.py \
  --run-dir simulations/runs/final_battery \
  --task 1 \
  --beast-jar /path/to/beast.jar
```

The script is deliberately scheduler-neutral.
On a multi-core machine or batch system, dispatch the task IDs listed in the
manifest and call the same command for each task.

After all tasks finish:

```bash
python3 simulations/scripts/summarize_maps.py \
  --run-dir simulations/runs/final_battery \
  --out-dir simulations/results/raw/final_battery \
  --write-raw-by-start
```

The processed summary used by the figure is
`simulations/results/raw/final_battery/summary_for_plot_best_likelihood.csv`.
The checked-in copy under `results/processed/` records the final run used for
the manuscript.

## Recreate the figure from checked-in summaries

```bash
Rscript simulations/scripts/plot_simulation_figure.R
```

The output is written to
`simulations/figures/output/supp_simulations_combined_rmse_final_battery_lambda0p1_r25_best_likelihood.pdf`.
