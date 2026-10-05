# Semi-Automated Image Labeling and Object Tracking for MR-Guided Radiotherapy Using a Foundation Vision Model

The paper studies the use of [Segment Anything Model 2](https://github.com/facebookresearch/sam2) (SAM2) and MedSAM2, foundation models for video object segmentation. We also developed a web application for image annotation to generate training data for fine-tuning them.

We tested on [TrackRAD2025](https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025), an open dataset of 2D sagittal cine-MR sequences from MR-linacs, released for the [TrackRAD2025 challenge](https://trackrad2025.grand-challenge.org/) on real-time tumor tracking.

Directory structure:

-   `/trackrad-model`: Models used for the inference loop, along with scripts for fine-tuning and evaluation
-   `/labeling-app`: Web application for semi-automated data annotation
-   `/notebooks`: Statistical analysis of model performance

See additional README files within each directory for more information.

## Data and checkpoints

Download these separately (all paths relative to the repo root):

| What | Where | Put it in |
| --- | --- | --- |
| Original TrackRAD2025 data (labeled training/testing and unlabeled sequences) | https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025 | `data/` (`trackrad2025_labeled_training_data/`, `trackrad2025_labeled_testing_data/`, `trackrad2025_unlabeled_training_data/`) |
| Semi-automatic labels (first-frame masks drawn with the labeling app) | https://huggingface.co/datasets/mzhu22/bouncing-target | Downloaded automatically by `vos_inference.py` |
| SAM2.1 checkpoints | `trackrad-model/sam2/checkpoints/download_ckpts.sh` | `trackrad-model/resources/` |
| MedSAM2 checkpoint (`MedSAM2_latest.pt`) | https://github.com/bowang-lab/MedSAM2 | `trackrad-model/resources/` |

## Reproducing the paper

Run from `trackrad-model/` after `uv sync`.

1. Convert the manually labeled training data: `uv run python scripts/prepare_sam2_finetune_data.py`
2. Propagate the first-frame labels through the unlabeled sequences (needs a GPU and `resources/sam2.1_hiera_small.pt`): `uv run python vos_inference.py`
3. Rerun step 1 to write the `manual`, `semiauto` and `combined` file lists.
4. Generate the 15 fine-tuning configs: `uv run python scripts/make_finetune_configs.py`
5. Fine-tune all 15 models (the paper used two 80 GB A100s): `bash scripts/sam2-finetune-launch-all.sh`
6. Evaluate each checkpoint, and the five zero-shot models, on the labeled test split with `scripts/eval_sam2_only.py`, writing JSONs into `notebooks/metrics/` (see the script's docstring).
7. Run the statistics and figures in `notebooks/` (see `notebooks/README.md`).

The metrics JSONs behind the paper's tables and figures are already checked in to `notebooks/metrics/`, so step 7 works without running steps 1-6.
