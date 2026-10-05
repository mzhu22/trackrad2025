# Semi-automated labeling and object tracking for MRgRT using SAM2

Supporting code and data for Semi-Automated Image Labeling and Object Tracking for MR-Guided Radiotherapy Using a Foundation Vision Model.

The paper studies the use of [Segment Anything Model 2](https://github.com/facebookresearch/sam2) (SAM2) and MedSAM2, foundation models for video object segmentation. We also developed a web application for image annotation to generate training data for fine-tuning them.

We tested on [TrackRAD2025](https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025), an open dataset of 2D sagittal cine-MR sequences from MR-linacs, released for the [TrackRAD2025 challenge](https://trackrad2025.grand-challenge.org/) on real-time tumor tracking.

## Code
Directory structure:

-   `/trackrad-model`: Models used for the inference loop, along with scripts for fine-tuning and evaluation
-   `/labeling-app`: Web application for semi-automated data annotation
-   `/notebooks`: Statistical analysis of model performance

See additional README files within each directory for more information.

## Data and checkpoints

All data comes from public Hugging Face datasets, and all downloads are scripted. Run these from `trackrad-model/` after `uv sync`:

```console
# Checkpoints -> trackrad-model/resources/
bash resources/download_sam2_checkpoints.sh      # SAM2.1 Tiny/Small/Base+/Large
bash resources/download_medsam2_checkpoint.sh    # MedSAM2_latest.pt

# Data -> data/ (repo root, git-ignored)
uv run python scripts/download_data.py
```

`scripts/download_data.py` fetches:

| What | Source | Lands in |
| --- | --- | --- |
| Labeled training data (50 sequences, manual labels used for fine-tuning) | [TrackRAD2025](https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025) | `data/trackrad2025_labeled_training_data/` |
| Labeled testing + pre-testing data (30 + 8 = 38 test sequences) | TrackRAD2025 | `data/trackrad2025_labeled_{testing,pre-testing}_data/`, linked together into `data/trackrad2025_labeled_test_data/` |
| First-frame masks drawn with the labeling app (200 sequences) | [mzhu22/bouncing-target](https://huggingface.co/datasets/mzhu22/bouncing-target), pinned to a commit | `data/bouncing-target/` |
| Unlabeled sequences for those 200 labels (just those, not the full 2.8M-frame unlabeled set) | TrackRAD2025 | `data/trackrad2025_unlabeled_training_data/` |

Use `--skip-labeled` or `--skip-semiauto` to download only part of it.

## Reproducing the paper

Run from `trackrad-model/` after downloading the data and checkpoints above.

1. Convert the manually labeled training data: `uv run python scripts/prepare_sam2_finetune_data.py`
2. Propagate the first-frame labels through the unlabeled sequences with SAM2.1 Small (needs a GPU): `uv run python propagate_labels.py`
3. Rerun step 1 to write the `manual`, `semiauto` and `combined` file lists.
4. Generate the 15 fine-tuning configs: `uv run python scripts/make_finetune_configs.py`
5. Fine-tune all 15 models (the paper used two 80 GB A100s): `bash scripts/sam2-finetune-launch-all.sh`
6. Evaluate each checkpoint, and the five zero-shot models, on `data/trackrad2025_labeled_test_data` with `scripts/eval_sam2_only.py`, writing JSONs into `notebooks/metrics/` (see the script's docstring).
7. Run the statistics and figures in `notebooks/` (see `notebooks/README.md`).

The metrics JSONs behind the paper's tables and figures are already checked in to `notebooks/metrics/`, so step 7 works without running steps 1-6.
