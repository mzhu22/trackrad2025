# Notebooks

Statistical analysis and figures for the manuscript.

# Installation
Requires Python and R

```console
uv sync
```

Then run the Jupyter notebooks:

- `stats_glmm.ipynb`: normality tests, Kruskal-Wallis tests, the beta/gamma GLMMs (R `glmmTMB`/`lme4` via `rpy2`, so R must be installed) with Bonferroni correction, and the accuracy and runtime figures.
- `stats.ipynb`: the same metrics loading plus the residual Q-Q plots.

To regenerate the metrics themselves, see the reproduction steps in the top-level README.
