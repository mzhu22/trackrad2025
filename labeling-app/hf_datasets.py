import json
import re
import shutil
import uuid
from pathlib import Path
from typing import TypedDict

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files, upload_folder
from PIL import Image

from common import TOKEN, Frames
from predict import PredictResult

OUT_DIR = Path("tmp/output")

IMAGES_REPO = "LMUK-RADONC-PHYS-RES/TrackRAD2025"
UNLABELED_FOLDER = "trackrad2025_unlabeled_training_data"
MASKS_REPO = "mzhu22/bouncing-target"
DATASET_REPO_TYPE = "dataset"


def download_image_files(patient: str, idx: int) -> tuple[str, str, str, str]:
    frames_idx = "" if idx == 1 else str(idx)

    frames_file = hf_hub_download(
        repo_id=IMAGES_REPO,
        repo_type=DATASET_REPO_TYPE,
        filename=f"{UNLABELED_FOLDER}/{patient}/images/{patient}_frames{frames_idx}.mha",
    )
    field_strength_file = hf_hub_download(
        repo_id=IMAGES_REPO,
        repo_type=DATASET_REPO_TYPE,
        filename=f"{UNLABELED_FOLDER}/{patient}/b-field-strength.json",
    )
    scanned_region_file = hf_hub_download(
        repo_id=IMAGES_REPO,
        repo_type=DATASET_REPO_TYPE,
        filename=f"{UNLABELED_FOLDER}/{patient}/scanned-region{frames_idx}.json",
    )
    frame_rate_file = hf_hub_download(
        repo_id=IMAGES_REPO,
        repo_type=DATASET_REPO_TYPE,
        filename=f"{UNLABELED_FOLDER}/{patient}/frame-rate{frames_idx}.json",
    )
    return frames_file, field_strength_file, scanned_region_file, frame_rate_file


MASKS_PNG_FILE = "masks.png"
SCORES_FILE = "scores.json"
BAD_IMAGE_FILE = "bad-image.txt"

# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def save_masks(
    username: str, masks_and_scores: PredictResult, sequence: Frames, frame: int
) -> None:
    """Masks == (N, X, Y) tensor where N == number of objects"""
    masks = masks_and_scores["masks"]
    scores = masks_and_scores["scores"]

    masks = np.flip(masks, axis=1)  # Flip over X axis to match the original orientation

    patient, sequence_idx = sequence

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / SCORES_FILE, "w") as f:
        json.dump(scores.tolist(), f)

    # Save masks as a single PNG where each pixel value is the mask index (0 = background)
    index_map = np.zeros_like(masks[0], dtype=np.uint8)
    for idx, mask in enumerate(masks, start=1):
        index_map[mask > 0] = idx

    output_mask = Image.fromarray(index_map)
    output_mask.putpalette(DAVIS_PALETTE)
    output_mask.save(str(OUT_DIR / MASKS_PNG_FILE))

    path_in_repo = f"{patient}-{sequence_idx}-{frame}-{username}_{uuid.uuid4()}"
    upload_folder(
        repo_id=MASKS_REPO,
        repo_type=DATASET_REPO_TYPE,
        folder_path=str(OUT_DIR),
        path_in_repo=str(path_in_repo),
        token=TOKEN,
    )

    # Remove the temporary output directory after upload
    shutil.rmtree(OUT_DIR)


def save_bad_image_report(
    username: str, sequence: Frames, frame: int, comments: str
) -> None:
    """Save a bad image report to the masks directory."""
    patient, idx = sequence
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / BAD_IMAGE_FILE, "w") as f:
        f.write(comments)

    path_in_repo = f"{patient}-{idx}-{frame}-{username}_{uuid.uuid4()}"
    upload_folder(
        repo_id=MASKS_REPO,
        repo_type=DATASET_REPO_TYPE,
        folder_path=str(OUT_DIR),
        path_in_repo=str(path_in_repo),
        token=TOKEN,
    )

    # Remove the temporary output directory after upload
    shutil.rmtree(OUT_DIR)


def parse_output_fpath(path: str) -> Frames:
    folder_prefix, _ = path.split("/")
    parts = folder_prefix.split("-")
    patient = parts[0]
    sequence = parts[1]
    return patient, int(sequence)


def parse_image_path(path: str) -> Frames:
    """
    Parse the image path to extract the patient ID and sequence.
    Example: "trackrad2025_unlabeled_training_data/A_033/images/A_033_frames.mha" -> ("A_003, 1")
    Returns a tuple (patient_id, sequence).
    """
    # Example path: "trackrad2025_unlabeled_training_data/A_033/images/A_033_frames44.mha"
    _, patient, _, fname = path.split("/")

    # Get any digits at the end of the filename before the extension
    digits = re.findall(r"\d+$", fname.split(".")[0])
    sequence = digits[0] if digits else "1"
    return patient, int(sequence)


class SequenceStatus(TypedDict):
    all: list[Frames]
    completed: list[Frames]
    todo: list[Frames]
    completed_patients: set[str]
    all_patients: set[str]


def get_completed_and_todo() -> SequenceStatus:
    images_repo_files = list_repo_files(
        repo_id=IMAGES_REPO,
        repo_type=DATASET_REPO_TYPE,
    )
    image_files = [
        fname
        for fname in images_repo_files
        if fname.startswith(UNLABELED_FOLDER) and fname.endswith(".mha")
    ]
    all_sequences = [parse_image_path(fname) for fname in image_files]

    masks_repo_files = list_repo_files(
        repo_id=MASKS_REPO, repo_type=DATASET_REPO_TYPE, token=TOKEN
    )

    mask_files = [f for f in masks_repo_files if f.endswith(MASKS_PNG_FILE)]
    bad_image_files = [f for f in masks_repo_files if f.endswith(BAD_IMAGE_FILE)]

    annotated = [parse_output_fpath(fname) for fname in mask_files]
    bad_images = [parse_output_fpath(fname) for fname in bad_image_files]

    completed = annotated + bad_images

    todo = sorted(list(set(all_sequences) - set(completed)))

    # Optionally, you can add type hints for patient IDs if you have a specific type

    return SequenceStatus(
        all=all_sequences,
        completed=completed,
        todo=todo,
        completed_patients=set(seq[0] for seq in completed),
        all_patients=set(seq[0] for seq in all_sequences),
    )
