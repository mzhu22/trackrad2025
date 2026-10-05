"""SAM2-only evaluation driver for a fine-tuned checkpoint.

Runs the SAM2 video predictor (no nnU-Net refinement) over every case in a
ground-truth directory, then scores the results with evaluate.py. Runs
non-interactively against a given checkpoint/data directory and writes the
resulting metrics.json (e.g. into ../notebooks/metrics/).

The checkpoint and output file are derived from --variant and --training-set:
the output is ../notebooks/metrics/<variant>_<training set>.json.

Usage:
    uv run python scripts/eval_sam2_only.py \
        --variant {tiny,small,base_plus,large,medsam2} \
        --training-set {zero_shot,manual,semiauto,combined}
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import inference
import torch
from model import run_algorithm
from sam2.build_sam import build_sam2_video_predictor

# inference.setup_sam2() hardcodes the "small" architecture (the variant used
# by the deployed pipeline), which only matches checkpoints fine-tuned from
# sam2.1_hiera_small. Fine-tuning ablations produce other architectures, so
# pick the right base config + image_size (matching each finetune config's
# `scratch.resolution`) per variant instead of reusing that helper.
VARIANT_MODEL_CFG = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", 1024),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", 1024),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", 1024),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", 1024),
    "medsam2": ("configs/sam2.1/sam2.1_hiera_t.yaml", 512),
}

TRAINING_SETS = ["zero_shot", "manual", "semiauto", "combined"]

# Per variant: original (zero-shot) checkpoint in resources/ and the prefix of
# the fine-tuning config name (`<prefix>_<training set>_finetune`).
VARIANT_CHECKPOINTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t"),
    "small": ("sam2.1_hiera_small.pt", "sam2.1_hiera_s"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+"),
    "large": ("sam2.1_hiera_large.pt", "sam2.1_hiera_l"),
    "medsam2": ("MedSAM2_latest.pt", "sam2.1_medsam2"),
}


def checkpoint_path(variant: str, training_set: str) -> Path:
    zero_shot_ckpt, config_prefix = VARIANT_CHECKPOINTS[variant]
    if training_set == "zero_shot":
        return ROOT / "resources" / zero_shot_ckpt
    config = f"{config_prefix}_{training_set}_finetune"
    return (
        ROOT
        / "sam2/sam2_logs/configs/sam2.1_training"
        / f"{config}.yaml/checkpoints/checkpoint.pt"
    )


def setup_sam2_for_variant(checkpoint: Path, variant: str):
    model_cfg, image_size = VARIANT_MODEL_CFG[variant]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.autocast("cuda", dtype=dtype).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    return build_sam2_video_predictor(
        model_cfg,
        checkpoint.as_posix(),
        device=device.type,
        vos_optimized=False,
        hydra_overrides_extra=[f"++model.image_size={image_size}"],
    )


def prediction_json(
    ground_truth_directory: Path,
    input_directory: Path,
    output_directory: Path,
    job_id: str,
    case_id: str,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
    start_time: str,
    end_time: str,
) -> str:
    return json.dumps(
        {
            "pk": job_id,
            "ground_truth_directory": str(ground_truth_directory),
            "input_directory": str(input_directory),
            "output_directory": str(output_directory),
            "inputs": [
                {"value": frame_rate, "interface": {"slug": "frame-rate"}},
                {
                    "value": magnetic_field_strength,
                    "interface": {"slug": "magnetic-field-strength"},
                },
                {"value": scanned_region, "interface": {"slug": "scanned-region"}},
                {
                    "image": {"name": "mri-linac-target.mha"},
                    "interface": {
                        "slug": "mri-linac-target",
                        "relative_path": "images/mri-linac-target",
                    },
                },
                {
                    "image": {"name": f"{case_id}.mha"},
                    "interface": {
                        "slug": "mri-linac-series",
                        "relative_path": "images/mri-linacs",
                    },
                },
            ],
            "status": "Succeeded",
            "outputs": [
                {
                    "image": {"name": "output.mha"},
                    "interface": {
                        "slug": "mri-linac-series-targets",
                        "relative_path": "images/mri-linac-series-targets",
                    },
                }
            ],
            "started_at": start_time,
            "completed_at": end_time,
        }
    )


def run_eval(checkpoint: Path, variant: str, data_dir: Path, work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)

    predictions_dir = work_dir / "predictions"
    output_dir = work_dir / "output"
    predictions_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    cases = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    print(f"Evaluating {len(cases)} cases against checkpoint {checkpoint} ({variant})")

    predictor = setup_sam2_for_variant(checkpoint, variant)

    predictions = []
    for case_id in cases:
        case_dir = data_dir / case_id
        print(f"  case {case_id}")

        frames = sitk.GetArrayFromImage(
            sitk.ReadImage(str(case_dir / "images" / f"{case_id}_frames.mha"))
        )
        target = sitk.GetArrayFromImage(
            sitk.ReadImage(str(case_dir / "targets" / f"{case_id}_first_label.mha"))
        )
        frame_rate = json.loads((case_dir / "frame-rate.json").read_text())
        b_field_strength = json.loads(
            (case_dir / "b-field-strength.json").read_text()
        )
        scanned_region = json.loads((case_dir / "scanned-region.json").read_text())

        job_id = str(uuid.uuid4())
        output_path = predictions_dir / job_id / "output"

        start_time = datetime.now().isoformat()
        output_mask = run_algorithm(
            predictor,
            None,
            case_id=case_id,
            frames=frames,
            target=target,
            frame_rate=frame_rate,
            magnetic_field_strength=b_field_strength,
            scanned_region=scanned_region,
            refinement_lookback_frames=0,
            do_refinement=False,
            save_annotations=False,
        ).astype(np.uint8)
        end_time = datetime.now().isoformat()

        inference.write_array_as_image_file(
            location=output_path / "images/mri-linac-series-targets",
            array=output_mask,
        )

        predictions.append(
            json.loads(
                prediction_json(
                    ground_truth_directory=data_dir,
                    input_directory=predictions_dir,
                    output_directory=output_dir,
                    job_id=job_id,
                    case_id=case_id,
                    frame_rate=frame_rate,
                    magnetic_field_strength=b_field_strength,
                    scanned_region=scanned_region,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        )

    (predictions_dir / "predictions.json").write_text(json.dumps(predictions))

    # Run scoring in a fresh subprocess: evaluate.main() parallelizes across
    # cases with a forking ProcessPoolExecutor (helpers.run_prediction_processing),
    # and forking this process would deadlock since it already holds an
    # initialized CUDA context from the SAM2 predictor above.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[3]); "
            "import evaluate; "
            "evaluate.main(input_directory=Path(sys.argv[1]), output_directory=Path(sys.argv[2]))",
            str(predictions_dir),
            str(output_dir),
            str(ROOT),
        ],
        check=True,
    )
    return output_dir / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANT_MODEL_CFG), required=True)
    parser.add_argument("--training-set", choices=TRAINING_SETS, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT.parent / "data/trackrad2025_labeled_test_data",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("./tmp/eval_sam2_only")
    )
    args = parser.parse_args()

    checkpoint = checkpoint_path(args.variant, args.training_set)
    out = ROOT.parent / "notebooks/metrics" / f"{args.variant}_{args.training_set}.json"

    metrics_path = run_eval(
        checkpoint=checkpoint,
        variant=args.variant,
        data_dir=args.data_dir,
        work_dir=args.work_dir,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(metrics_path, out)

    metrics = json.loads(metrics_path.read_text())
    print("\nAggregates:")
    print(json.dumps(metrics["aggregates"], indent=2))
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()
