# Scripts

Utilities for preparing data, generating configs, fine-tuning SAM2/MedSAM2, and evaluating checkpoints. Run from `trackrad-model/` unless noted. See the top-level README for the full reproduction order.

| File | Purpose |
| --- | --- |
| `download_data.py` | Downloads the labeled TrackRAD2025 splits, the semi-automatic first-frame labels, and the unlabeled sequences those labels need into `../data/`. |
| `prepare_sam2_finetune_data.py` | Converts the manually labeled TrackRAD2025 training split into the JPEG/PNG layout SAM2 trains on, and writes the `manual`, `semiauto` and `combined` file lists. |
| `sam2-finetune-launch-all.sh` | Trains all 15 configs sequentially on the local machine (`NUM_GPUS`, default 2). |
| `eval_sam2_only.py` | Runs a checkpoint over the labeled test split and writes a metrics JSON (as in `../notebooks/metrics/`). Use `../data/trackrad2025_labeled_test_data` for the paper's 38 test sequences.. |
| `setup_vm.sh` | Environment setup for a fresh Linux GPU VM. |
