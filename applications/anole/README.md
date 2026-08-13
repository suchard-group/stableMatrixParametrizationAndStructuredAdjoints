# Anolis phylogenetic comparative example

Multivariate phylogenetic OU model fit to the *Anolis* comparative dataset
distributed with the R package `phytools` (manuscript Section "Diffusion over
phylogenetic trees"): a fixed 100-tip phylogeny and six centered, unit-variance
morphological traits (snout-vent length, head length, hindlimb length,
forelimb length, lamella number, tail length).

## Active manuscript bundle

```
data/
  Anolis.tre, anole.data.csv

xml/
  anole_6char_orthogonal_common_priors_2m.xml

results/
  anole_selection_strength_posterior_median_10m_fresh_2m.tsv
  anole_drift_entry_hpd_reported_10m_fresh_2m.tsv
  anole_block_regime_reported_10m_fresh_2m.tsv
  anole_run_summary_10m_fresh_2m.tsv

scripts/
  plot_anole_combined_row.R
  plot_anole_phylogeny.R
  plot_anole_drift_matrix_heatmap.R
```

The active manuscript figure is
`anole_phylogeny_drift_matrix_row_10m_fresh_2m.pdf`.
The matrix TSV stores the positive BEAST selection-strength matrix; the plotting
scripts negate it to the manuscript Hurwitz drift convention, so rows are target
traits and columns are source traits.

The long-chain summary used in `main.tex` is a 2,000,000-state orthogonal
H-SSBP chain, logged every 100 states with 20% burn-in, leaving 16,000
post-burn-in logged samples. The XML records the common-priors setup used for
this analysis: ordered block rates, `blockTheta ~ N(0, 0.5^2)`,
`blockT ~ HN(0, 0.5)`, `blockRho ~ HN(0, 1.5)`,
`rotation.angle ~ N(0, 0.1^2)`, and diffusion marginal scales
`HN(0, 0.1)`.

The compact reported TSVs are the direct public source for the manuscript prose:
the four off-diagonal HPD intervals excluding zero, the three complex-block
proportions and approximate switching counts, and the run-length/ESS summary.
The earlier 200,000-state draft run is retained locally under
`legacy_200k_work/`, which is ignored by Git.

## Regenerate the figure

From this directory:

```bash
Rscript scripts/plot_anole_combined_row.R
```

This writes `output/anole_phylogeny_drift_matrix_row_10m_fresh_2m.pdf` and
`.png`. The component scripts can also be run separately to write the phylogeny
and heatmap panels to `output/`.

## Rerun the chain

After building the pinned BEAST jar described in `../../environment/BEAST.md`,
run from this directory, for example:

```bash
../../run_local_beast.sh /path/to/beast.jar \
  xml/anole_6char_orthogonal_common_priors_2m.xml \
  202607171
```

The checked-in compact summaries are the exact manuscript inputs; rerunning the
MCMC produces a new stochastic trace to compare against them.
