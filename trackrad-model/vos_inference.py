import argparse
import itertools
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import SimpleITK
import submitit
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

HF_TOKEN = "REDACTED"
REPO_NAME = "mzhu22/bouncing-target"
TRACKRAD_DATASET_PATH = Path(
    "/rodata/mnradonc_dev/m299164/trackrad/datasets/trackrad2025/trackrad2025_unlabeled_training_data"
)
current_date = date.today()
output_dataset = Path(
    f"/rodata/mnradonc_dev/m299164/trackrad/datasets/bouncing-target-labeled-{current_date.isoformat()}"
)
output_dataset.mkdir(parents=True, exist_ok=True)

prev_date = date(2025, 7, 1)
prev_dataset = Path(
    f"/rodata/mnradonc_dev/m299164/trackrad/datasets/bouncing-target-labeled-{prev_date.isoformat()}"
)


def save_mri_series_as_jpegs(input_mri_linac_series, jpegs_path: Path) -> Path:
    if jpegs_path.exists():
        shutil.rmtree(jpegs_path)
    jpegs_path.mkdir(parents=True, exist_ok=True)

    for i in range(input_mri_linac_series.shape[2]):
        frame = input_mri_linac_series[:, :, i]
        frame = frame.astype(np.float32)
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(
            np.uint8
        )

        img = Image.fromarray(frame)
        img.convert("L").save(
            jpegs_path / f"{i:05d}.jpg", "JPEG", quality=95, subsampling=0
        )
    return jpegs_path


# From trackrad-model/sam2/tools/vos_inference.py
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path: Path) -> np.ndarray:
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(str(path))
    mask = np.array(mask).astype(np.uint8)
    return mask


def save_ann_png(path: Path, mask: np.ndarray):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(DAVIS_PALETTE)
    output_mask.save(str(path))


def get_per_obj_mask(mask: np.ndarray):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask: dict[int, np.ndarray]) -> np.ndarray:
    """Combine per-object masks into a single mask."""
    first_mask = next(iter(per_obj_mask.values()))
    height, width = np.squeeze(first_mask).shape
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def save_ann_pngs(
    path: Path,
    scores: list[float],
    per_frame_inferred_masks: dict[int, dict[int, np.ndarray]],
):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    for frame_idx, per_obj_mask in per_frame_inferred_masks.items():
        mask = put_per_obj_mask(per_obj_mask)
        save_ann_png(path / f"{frame_idx:05d}.png", mask)


SCORE_THRESHOLD = 0


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(predictor: SAM2VideoPredictor, jpegs_dir: Path, ann_png_path: Path):
    annotations = load_ann_png(ann_png_path)
    per_obj_mask = get_per_obj_mask(annotations)
    inference_state = predictor.init_state(
        video_path=str(jpegs_dir), async_loading_frames=False
    )
    for object_id, mask in per_obj_mask.items():
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            mask=mask,
        )

    per_frame_inferred_masks = {}
    model_out = predictor.propagate_in_video(inference_state)
    for out_frame_idx, out_obj_ids, out_mask_logits in model_out:
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > SCORE_THRESHOLD).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        per_frame_inferred_masks[out_frame_idx] = per_obj_output_mask

    return per_frame_inferred_masks


def main():
    sam2_checkpoint = "./resources/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    parser = argparse.ArgumentParser(
        description="Download labeled images from HF and propagate masks across MR sequences"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    predictor = build_sam2_video_predictor(
        config_file=model_cfg,
        ckpt_path=sam2_checkpoint,
        device="cuda",
    )

    files = list_repo_files(
        REPO_NAME,
        repo_type="dataset",
        token=HF_TOKEN,
        revision="81adb9d0ea934edbe82eeca0403d06e3662a4875",
    )
    png_files = [f for f in files if f.endswith(".png")]

    processed_paths = [
        path
        for path in itertools.chain(
            prev_dataset.glob("JPEGImages/*"),
            # For idempotency if the script fails on the current dataset
            output_dataset.glob("JPEGImages/*"),
        )
        if path.is_dir()
    ]
    processed_sequences = [tuple(path.name.split("-")) for path in processed_paths]

    for f in png_files:
        ann_png = hf_hub_download(
            repo_id=REPO_NAME,
            filename=f,
            repo_type="dataset",
            token=HF_TOKEN,
        )

        folder = ann_png.rsplit("/", 2)[-2]
        slug, _ = folder.rsplit("_", 1)
        patient, sequence, frame, _ = slug.split("-")

        scores_file = hf_hub_download(
            repo_id=REPO_NAME,
            filename=f.replace("masks.png", "scores.json"),
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(scores_file) as f:
            scores_json = json.load(f)

        scores = (
            [scores_json[0]]
            if len(scores_json) == 1
            else [score[0] for score in scores_json]
        )
        if any(s < 0.8 for s in scores):
            print(patient, sequence, scores)

        if (patient, sequence) in processed_sequences:
            print(f"Skipping {patient}-{sequence} as it is already processed.")
            continue

        print(patient, sequence)

        sequence_idx = sequence if sequence != "1" else ""
        sequence_path = (
            TRACKRAD_DATASET_PATH
            / patient
            / "images"
            / f"{patient}_frames{sequence_idx}.mha"
        )
        sequence_image = SimpleITK.ReadImage(str(sequence_path))
        sequence_array = SimpleITK.GetArrayFromImage(sequence_image)

        # Clip sequences
        sequence_array = sequence_array[:, :, :200]

        jpegs_dir = output_dataset / "JPEGImages" / f"{patient}-{sequence}"
        ann_dir = output_dataset / "Annotations" / f"{patient}-{sequence}"

        save_mri_series_as_jpegs(sequence_array, jpegs_dir)
        per_frame_inferred_masks = vos_inference(predictor, jpegs_dir, Path(ann_png))
        save_ann_pngs(ann_dir, scores, per_frame_inferred_masks)


if __name__ == "__main__":
    main()

    # executor = submitit.AutoExecutor(folder="submitit_logs")
    # executor.update_parameters(
    #     gpus_per_node=4,
    #     num_nodes=1,
    #     timeout_min=60 * 24,  # 24 hours
    #     slurm_partition="gen-a100p",
    #     account="m299164",
    # )
    # job = executor.submit(main)
    # print(job.job_id)

    # output = job.result()
    # print(output)
