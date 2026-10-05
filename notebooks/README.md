# Notebooks

Statistical analysis and figures for the manuscript. They read the per-case evaluation metrics in `metrics/` (one `<model>_<training set>.json` per combination: 5 models [`tiny`, `small`, `base_plus`, `large`, `medsam2`] × `zero_shot`, `manual`, `semiauto`, `combined` = 20 files) and write figures to `figures/`.

```console
cd notebooks
uv sync
uv run jupyter lab   # or open the notebooks in VS Code
```

- `stats_glmm.ipynb`: normality tests, Kruskal-Wallis tests, the beta/gamma GLMMs (R `glmmTMB`/`lme4` via `rpy2`, so R must be installed) with Bonferroni correction, and the accuracy and runtime figures.
- `stats.ipynb`: the same metrics loading plus the residual Q-Q plots.

To regenerate the metrics themselves, see the reproduction steps in the top-level README.
