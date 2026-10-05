# Model

Inference pipeline and training/evaluation code for tracking with SAM2 and MedSAM2.

- `inference.py`, `model.py`: Grand Challenge inference entrypoint and the tracking algorithm (see `Dockerfile`).
- `vos_inference.py`: propagates the crowd-sourced first-frame masks through each unlabeled sequence to build the semi-automatic training set.
- `evaluate.py`, `helpers.py`, `monai_metrics.py`, `minimal_mha_simpleitk.py`: TrackRAD2025 scoring code.
- `scripts/`: data preparation, fine-tuning and evaluation scripts. See `scripts/README.md`.

Data and checkpoints are downloaded separately. See the top-level README for where to get them and the order to run things in.
