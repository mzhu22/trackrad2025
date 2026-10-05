# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is Team YouBetcha's (Mayo Clinic Radiation Oncology) submission to the [TrackRAD2025 Grand Challenge](https://trackrad2025.grand-challenge.org/): real-time tumor tracking in 2D MRI-linac video. It also contains the code and metrics behind the accompanying SAM2/MedSAM2 fine-tuning manuscript — see "Important caveats" below.

Three independent sub-projects, each with its own `pyproject.toml`/`uv.lock` and no shared dependencies between them (see `trackrad2025.code-workspace` for the VS Code multi-root layout):

- **`trackrad-model/`** — the inference pipeline: SAM2 (video object segmentation). This is what's packaged into the Docker container submitted to Grand Challenge.
- **`labeling-app/`** — a Gradio web app ("Bouncing Target", deployed to HuggingFace Spaces) used to crowdsource bounding-box annotations, which SAM2 turns into segmentation masks for training data.
- **`notebooks/`** — statistics/analysis notebooks (R via `rpy2`, GLMM significance testing) and figures for the writeup, decoupled from the other two projects.

`writeups/` contains the project plan, poster, and final manuscript PDFs — read these for the scientific rationale/methods if you need background beyond the code.

## Important caveats

- Data and checkpoints are not checked in. See the top-level `README.md` for where to download them (public HF datasets `LMUK-RADONC-PHYS-RES/TrackRAD2025` and `mzhu22/bouncing-target`) and the order in which to run the preparation, fine-tuning and evaluation scripts. The Docker image additionally expects a `resources/` directory with checkpoints.
- `trackrad-model/sam2/` is a vendored copy of the upstream [facebookresearch/sam2](https://github.com/facebookresearch/sam2) repo (installed as a local path dependency via `uv`, see `trackrad-model/pyproject.toml`'s `[tool.uv.sources]`). Don't assume changes here are TrackRAD-specific — check upstream before modifying.
- `labeling-app` works locally, but "Submit"/save actions push to the HuggingFace dataset `mzhu22/bouncing-target` using a `HF_TOKEN` env var (write access), so submission won't work without credentials. Reading the dataset is public.

## Development commands

Each sub-project uses [uv](https://docs.astral.sh/uv/) for dependency management. Run `uv sync` inside the relevant directory before working in it.

### labeling-app (runnable locally)

```console
cd labeling-app
uv sync
uv run gradio app.py
```

### trackrad-model (Docker image; fine-tuning/eval scripts need a GPU plus downloaded data and checkpoints)

```console
cd trackrad-model
docker build -t trackrad-model .
```

The Dockerfile installs with the `cu124` extra (`uv sync --extra cu124`) and expects a `resources/` directory (SAM2 checkpoints) that is not checked into this repo. The container entrypoint is `inference.py`, reading from `/input` and writing to `/output` per the Grand Challenge algorithm interface.

There is no test suite, linter config, or CI in this repo.

## Architecture: trackrad-model inference pipeline

The pipeline processes one "case" (an MRI-linac video + a target mask on frame 0) at a time:

1. **`inference.py`** — entrypoint (`python inference.py`, run as the Docker `ENTRYPOINT`). Reads Grand Challenge–formatted inputs from `/input` (frame-rate/field-strength/scanned-region JSON, `.mha`/`.tiff` image series via SimpleITK), builds the SAM2 video predictor (`setup_sam2`), calls `run_algorithm` from `model.py`, and writes the resulting per-frame mask series back out as `.mha`. Not meant to be modified — algorithm changes go in `model.py`.
2. **`model.py`** — `run_algorithm` is the actual tracking algorithm:
   - Dumps the input frame series to disk as JPEGs (SAM2's video predictor reads from a directory of frames, not in-memory arrays).
   - Seeds SAM2 with the frame-0 ground-truth mask (`add_new_mask`) and propagates the mask forward through the video (`propagate_in_video`) — this is the core object-tracking step. Note the comment that SAM2 doesn't use future frames despite being given the whole video upfront, so this could be adapted to true real-time/streaming use.
   - Fills holes in the final masks (`binary_fill_holes`) to patch known segmentation gaps.
3. **`evaluate.py`** — the Grand Challenge evaluation container's scoring script (not run by contestants directly). Computes Dice, 95th-percentile Hausdorff distance, average surface distance, 2D center-of-mass error, and a custom dosimetric metric (`DoseMetric`, via a shifted-point-cloud DVH approximation) per case, using a vendored `monai_metrics.py` reimplementation and `minimal_mha_simpleitk.py` (a dependency-light `.mha` reader/writer) instead of importing MONAI/SimpleITK directly in the eval container.
4. **`helpers.py`** — `run_prediction_processing` parallelizes `evaluate.py`'s per-case scoring across processes, capped by `GRAND_CHALLENGE_MAX_WORKERS`.
5. **`postprocessing.py`** — a MONAI-based `UNet`/`AttentionUnet` mask-refinement model and preprocessing transform pipeline; an earlier/alternate approach to mask refinement not used in the final `model.py` pipeline.
6. **`propagate_labels.py`** — a standalone batch-labeling script that propagates the labeling-app's first-frame masks (downloaded beforehand into `data/bouncing-target/`) through the matching *unlabeled* TrackRAD2025 sequences to produce the semi-automatic training set — this is the "AI-assisted" half of the labeling-app workflow, run offline rather than interactively.
7. **`resources/`** — checkpoint download scripts (SAM2.1 and MedSAM2); the `.pt` files are git-ignored.
8. **`scripts/`** — `download_data.py` (all data into `data/`), data preparation (`prepare_sam2_finetune_data.py`), fine-tuning config generation (`make_finetune_configs.py` renders `finetune_template.yaml` into 15 git-ignored configs under `sam2/sam2/configs/sam2.1_training/`), the local launcher (`sam2-finetune-launch-all.sh`), and `eval_sam2_only.py`, which writes the metrics JSONs consumed by `notebooks/`.

## Architecture: labeling-app

A single-page Gradio `Blocks` app (`app.py`) built around a `gr.State`-driven pipeline:

1. `get_completed_and_todo` (`hf_datasets.py`) diffs the unlabeled-images HF dataset (`LMUK-RADONC-PHYS-RES/TrackRAD2025`) against the masks-output HF dataset (`mzhu22/bouncing-target`) to compute which `(patient_id, sequence_number)` pairs (the `Frames` type in `common.py`) still need annotation.
2. `next_sequence` samples a random *patient* first, then a sequence, to avoid over-representing patients with many sequences.
3. On sequence selection: `download_images` pulls the `.mha` + metadata JSON from HF, `load_image`/`load_video` render the first frame (for box-drawing, normalized to the 99th percentile to avoid artifact clipping) and a preview video (`video.py`, OpenCV-based).
4. The user draws bounding boxes on the first frame (`gradio_image_annotation`); clicking "Get Segmentation Masks" calls `predict.py`'s `predict_gpu`/`predict_cpu` (SAM2 image predictor, not the video predictor used in `trackrad-model`) to turn boxes into masks.
5. "Submit" (`save_annotation` → `hf_datasets.save_masks`) uploads the mask PNG (DAVIS-palette-indexed, one color per object) + confidence scores to the `mzhu22/bouncing-target` HF dataset; "Report Bad Image" (`save_bad_image_report`) logs unusable sequences instead.

Both `model.py` (trackrad-model) and `hf_datasets.py` (labeling-app) independently define the same `DAVIS_PALETTE` byte string and `save_ann_png`-style logic — these are not shared, so keep them in sync manually if the mask-encoding format changes.

## Architecture: notebooks

`stats.ipynb` and `stats_glmm.ipynb` compute significance statistics (GLMM-based, via `rpy2` calling into R) over metrics JSON files in `notebooks/metrics/` — these are the pre-computed evaluation outputs (Dice, Hausdorff, etc., named by model variant/checkpoint date, e.g. `02_28_l_trackrad_labeled_training.yaml.json`) referenced in the manuscript's results tables and figures under `notebooks/figures/`.
