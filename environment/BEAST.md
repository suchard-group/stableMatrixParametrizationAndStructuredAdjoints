# Building BEAST X (`ou-time-series` branch)

The Hurwitz Smooth Spectral Block Parametrization (H-SSBP) message-passing
and gradient code used by every BEAST-driven analysis in this repository
lives on a public branch of the official BEAST X repository, not in this
repository. Clone and build it separately:

```bash
git clone https://github.com/beast-dev/beast-mcmc.git
cd beast-mcmc
git checkout ou-time-series
git checkout 88a7b1e6a14f06778693581b527984d05509b814   # commit used for all results in the manuscript
ant build
```

This produces `build/dist/beast.jar`, which every `*.xml` run in this
repository (`simulations/`, `applications/bitmex/`, `applications/anole/`)
is designed to be run with:

```bash
java -jar build/dist/beast.jar -seed <seed> path/to/run.xml
```

## BEAGLE

None of the runs shipped here require BEAGLE (the time-series and tree
likelihoods used are the native Java Kalman/message-passing implementations
added on this branch). If you want to cross-check against a BEAGLE-backed
build for unrelated substitution-model work, see BEAGLE's own build
instructions at <https://github.com/beagle-dev/beagle-lib>; it is not a
dependency for reproducing anything in this repository.

## Seeds

Simulation and BitMEX run folders record the seeds used to generate the checked-in
inputs and compact summaries. The Anolis folder ships the active long-chain XML
and compact manuscript summaries; rerunning that XML produces a new stochastic
MCMC trace to compare against the shipped summary tables.
