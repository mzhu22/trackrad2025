# Semi-automated labeling and object tracking for MRgRT using SAM2

Supporting code and data for Semi-Automated Image Labeling and Object Tracking for MR-Guided Radiotherapy Using a Foundation Vision Model.

The paper studies the use of [Segment Anything Model 2](https://github.com/facebookresearch/sam2) (SAM2) and MedSAM2, foundation models for video object segmentation. We also developed a web application for SAM2-assisted image annotation to generate training data.

We tested on [TrackRAD2025](https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025), an open dataset of 2D sagittal cine-MR sequences from MR-linacs, released for the [TrackRAD2025 challenge](https://trackrad2025.grand-challenge.org/) on real-time tumor tracking.

## Code
Directory structure:

-   `/trackrad-model`: Models used for the inference loop, along with scripts for fine-tuning and evaluation
-   `/labeling-app`: Web application for semi-automated data annotation
-   `/notebooks`: Statistical analysis of model performance

See additional README files within each directory for more information.

## Data and checkpoints
Run these from `trackrad-model/` (the data script needs `uv sync` first):

```console
# Checkpoints -> trackrad-model/resources/
bash resources/download_sam2_checkpoints.sh      # SAM2.1 Tiny/Small/Base+/Large
bash resources/download_medsam2_checkpoint.sh    # MedSAM2_latest.pt

# Data -> data/ (repo root, git-ignored)
uv run python scripts/download_data.py
```

`scripts/download_data.py` fetches:

| What | Source | Lands in | Size |
| --- | --- | --- | --- |
| TrackRAD2025 datatset | [TrackRAD2025](https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025) | `data/trackrad2025_labeled_*_data/` | ~1.0 GB |
| First-frame masks drawn with the labeling app (200 sequences) | [mzhu22/bouncing-target](https://huggingface.co/datasets/mzhu22/bouncing-target) | `data/bouncing-target/` | ~0.3 MB |
| Unlabeled sequences for those 200 labels | TrackRAD2025 | `data/trackrad2025_unlabeled_training_data/` | ~9.6 GB |

## Reproducing the paper

Run from `trackrad-model/` after downloading the data and checkpoints above.

1. Propagate the first-frame labels through the unlabeled sequences (recommend a GPU): `uv run python propagate_labels.py`
2. Convert the manually labeled training data and write the `manual`, `semiauto` and `combined` file lists: `uv run python scripts/prepare_sam2_finetune_data.py`
3. Fine-tune all 15 models (recommend GPU, the paper used two A100s): `bash scripts/sam2-finetune-launch-all.sh`. Each run writes its checkpoint to `sam2/sam2_logs/configs/sam2.1_training/<config>.yaml/checkpoints/checkpoint.pt`, where `<config>` is `<model>_<dataset>_finetune` (e.g. `sam2.1_hiera_t_manual_finetune`; datasets are `manual`, `semiauto`, `combined`).
4. Evaluate all 20 model/training-set combinations (5 models × zero-shot, manual, semi-auto, combined) on the 38 test sequences, one run each:

   ```console
   uv run python scripts/eval_sam2_only.py --variant tiny --training-set manual
   ```

   - `--variant` is one of `tiny`, `small`, `base_plus`, `large`, `medsam2`
   - `--training-set` is one of `zero_shot`, `manual`, `semiauto`, `combined`
   - The script picks the checkpoint (the original one in `resources/` for `zero_shot`, otherwise the fine-tuned one from step 3) and writes `notebooks/metrics/<variant>_<training set>.json` (e.g. `tiny_manual.json`), which the notebooks pick up. The test data is read from `data/trackrad2025_labeled_test_data` by default (`--data-dir` to override).

5. Run notebooks to compute statistics in `notebooks/` (see `notebooks/README.md`).

The metrics JSONs behind the paper's tables and figures are already checked in to `notebooks/metrics/`, so step 5 works without running steps 1-4.
