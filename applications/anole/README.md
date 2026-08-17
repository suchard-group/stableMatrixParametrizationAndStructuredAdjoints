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
  chains/
    anole_6char_orthogonal_common_priors_2m_chain01.xml
    anole_6char_orthogonal_common_priors_2m_chain02.xml
    anole_6char_orthogonal_common_priors_2m_chain03.xml
    anole_6char_orthogonal_common_priors_2m_chain04.xml

results/
  anole_selection_strength_posterior_median_10m_fresh_2m.tsv
  anole_drift_entry_hpd_reported_10m_fresh_2m.tsv
  anole_block_regime_reported_10m_fresh_2m.tsv
  anole_block_regime_by_chain_10m_fresh_2m.tsv
  anole_chain_details_10m_fresh_2m.tsv
  anole_rank_diagnostics_10m_fresh_2m.tsv
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

The long-chain summary used in `main.tex` pools four 2,000,000-state
orthogonal H-SSBP chains, logged every 100 states with 20% burn-in, yielding
64,000 retained draws. The chain-details table records two starts initialized
at the posterior-coordinate median and two starts initialized from the paper
XML values. The chain-specific XMLs record the exact starting values used for
the four chains. They match the reported Anolis result:
normalized-Cartesian stable blocks with ordered
`blockRate`, `blockRatio ~ Uniform(-1,1)`, `blockT ~ LogNormal(0,1)`,
`rotation.angle ~ N(0, 0.25^2)`, `anole.meanParameter ~ N(0,1)`, LKJ(1)
correlations, and `Gamma(0.5,0.5)` diffusion marginal variances.

The compact reported TSVs are the direct public source for the manuscript prose:
the pooled posterior median drift heat map, the four off-diagonal HPD intervals
excluding zero, the three complex-block proportions and approximate switching
ranges, the chain-level block-regime counts, and the rank-normalized
`\widehat R`/ESS summaries.
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
  xml/chains/anole_6char_orthogonal_common_priors_2m_chain01.xml \
  2026081521
```

Repeat with the XML paths and seeds in
`results/anole_chain_details_10m_fresh_2m.tsv` to recreate the four-chain
design. The checked-in compact summaries are the exact manuscript inputs;
rerunning the MCMC produces new stochastic traces to compare against them.
