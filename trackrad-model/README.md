# Model

Inference pipeline and training/evaluation code for tracking with SAM2 and MedSAM2.

- `inference.py`, `model.py`: Grand Challenge inference entrypoint and the tracking algorithm (see `Dockerfile`).
- `propagate_labels.py`: propagates the crowd-sourced first-frame masks through each unlabeled sequence to build the semi-automatic training set.
- `evaluate.py`, `helpers.py`, `monai_metrics.py`, `minimal_mha_simpleitk.py`: TrackRAD2025 scoring code.
- `scripts/`: data preparation, fine-tuning and evaluation scripts. See `scripts/README.md`.

- `resources/`: scripts that download the SAM2.1 and MedSAM2 checkpoints (the `.pt` files are git-ignored).

See the top-level README for how to download the data and checkpoints and the order to run things in.
