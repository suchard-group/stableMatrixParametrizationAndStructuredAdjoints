# BitMEX Trade-Data Example

This directory contains the public reproducibility bundle for the January 24,
2022 BitMEX post-shock analysis used in the manuscript supplement.
The analysis fits the five-instrument panel
`XBTUSDT, ETHUSDT, LTCUSDT, BCHUSDT, DOGEUSDT` on the trade-level recovery
window starting at 13:00 UTC.

## Active Bundle

```text
data/jan2022_J20_trade_7h/
  events.csv                 exact processed analysis input
  data_availability.csv      observation counts and window diagnostics
  metadata.json              analysis settings, priors, best start

data/jan2022_J20_trade_7h_context/
  events.csv                 plotting data with 08:00--13:00 UTC pre-window
                             context and the analyzed 13:00--20:00 UTC window
  metadata.json              context/source metadata and separator location

runs/jan2022_J2_4h/
  lambda0p1_random0*/input.xml
                             earlier deterministic MAP XMLs retained for the
                             four-hour public bundle

results/compact/jan2022_J20_trade_7h/
  map_runs_lambda0p1.csv     five-start posterior/likelihood/diagnostic table
  best_by_window.csv         highest-posterior row, `lambda0p1_random02`
  matrices/                  drift, selection, reconstruction, and finite-time
                             response matrices for the plotted MAP start
```

The MAP fits use normalized Cartesian H-SSBP block coordinates with the
change-of-basis matrix parameterized directly by the raw entries of `R`, not by
a globally normalized auxiliary matrix.
They use ordered effective block rates, observation variance `1e-5`, fixed
dense-R shrinkage `lambda=0.1`, log-normal priors on rates and skew terms, and
uniform `(-1,1)` priors on block ratios with a logit transform.
The dense-basis prior is
`log p(R) = C + lambda log|det R| - lambda * dim(R) * ||R||_F^2 / 2`.
The XMLs here are regenerated from the original run generator with
`nIterations=1000`; the raw cluster stdout/log files are not shipped, but the
compact summaries and matrices are.

The older 30-minute exploratory BitMEX work has been moved under
`legacy_30min_work/`, which is intentionally ignored by Git.

## Reproduce the Figure

From this directory:

```bash
Rscript scripts/plot_bitmex_trade_drifts.R
```

This writes `output/bitmex_jan2022_j20_trade_7h_cartesian_map_row.pdf` and
`.png`.
The default plot uses the highest-posterior MAP start
`lambda0p1_random02` and shows the manuscript-scale drift matrix, including
diagonal entries.
The time-series panel starts at 08:00 UTC, ends at 20:00 UTC, and marks the
13:00 UTC start of the analyzed recovery window with a dashed vertical line.

The explicit equivalent command is:

```bash
Rscript scripts/plot_bitmex_trade_drifts.R \
  --events data/jan2022_J20_trade_7h_context/events.csv \
  --metadata data/jan2022_J20_trade_7h_context/metadata.json \
  --matrix-csv results/compact/jan2022_J20_trade_7h/matrices/jan2022_J20_trade_7h_lambda0p1_random02_manuscript_drift_matrix.csv \
  --matrix-kind drift \
  --show-diagonal true \
  --layout paper-row \
  --out-prefix bitmex_jan2022_j20_trade_7h_cartesian_map_row
```

## Rerun the MAPs

After building the pinned BEAST jar described in `../../environment/BEAST.md`,
run any start from the repository root, for example:

```bash
./run_local_beast.sh /path/to/beast.jar \
  applications/bitmex/runs/jan2022_J2_4h/lambda0p1_random04/input.xml
```

Repeat for `lambda0p1_random01` through `lambda0p1_random05` to reproduce the
five-start MAP battery.
The XML file names are relative-safe and write logs/checkpoints next to the
current working directory used by BEAST.
