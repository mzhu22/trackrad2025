# Notebooks

Statistical analysis and figures for the manuscript.

# Installation
Requires [uv](https://docs.astral.sh/uv/) and, for `stats_glmm.ipynb`, R with the `glmmTMB` and `lme4` packages:

```console
uv sync
Rscript -e 'install.packages(c("glmmTMB", "lme4"), repos = "https://cloud.r-project.org")'
```

To regenerate the metrics themselves, see the reproduction steps in the top-level README.
