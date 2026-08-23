"""Convert the labeled TrackRAD2025 data/ splits into a DAVIS-style JPEG/PNG
video dataset for SAM2 VOS fine-tuning (training/dataset/vos_raw_dataset.py's
PNGRawDataset), plus file-list manifests for the "alldata" and "notest"
training subsets.

Run with: uv run python scripts/prepare_sam2_finetune_data.py
(from trackrad-model/)
"""

from pathlib import Path

import numpy as np
import SimpleITK
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = DATA_ROOT / "sam2_finetune"
JPEG_ROOT = OUT_ROOT / "JPEGImages"
ANN_ROOT = OUT_ROOT / "Annotations"
FILE_LIST_ROOT = OUT_ROOT / "file_lists"

TRAINING_SPLIT = "trackrad2025_labeled_training_data"
TESTING_SPLIT = "trackrad2025_labeled_testing_data"
PRETEST_SPLIT = "trackrad2025_labeled_pre-testing_data"

# From trackrad-model/sam2/tools/vos_inference.py (kept identical for
# PNGRawDataset(is_palette=True) compatibility).
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def save_mri_series_as_jpegs(frames: np.ndarray, jpegs_dir: Path) -> None:
    jpegs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames.shape[2]):
        frame = frames[:, :, i].astype(np.float32)
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(
            np.uint8
        )
        img = Image.fromarray(frame)
        # quality=100 (vs. 95 elsewhere in the repo) to preserve as much
        # signal as possible for training.
        img.convert("L").save(
            jpegs_dir / f"{i:05d}.jpg", "JPEG", quality=100, subsampling=0
        )


def save_ann_pngs(masks: np.ndarray, ann_dir: Path) -> None:
    ann_dir.mkdir(parents=True, exist_ok=True)
    for i in range(masks.shape[2]):
        mask = masks[:, :, i].astype(np.uint8)
        output_mask = Image.fromarray(mask)
        output_mask.putpalette(DAVIS_PALETTE)
        output_mask.save(ann_dir / f"{i:05d}.png")


def is_already_converted(case_id: str, expected_num_frames: int) -> bool:
    jpegs_dir = JPEG_ROOT / case_id
    ann_dir = ANN_ROOT / case_id
    return (
        jpegs_dir.is_dir()
        and len(list(jpegs_dir.glob("*.jpg"))) == expected_num_frames
        and ann_dir.is_dir()
        and len(list(ann_dir.glob("*.png"))) == expected_num_frames
    )


def convert_case(case_dir: Path) -> str:
    case_id = case_dir.name
    frames_path = case_dir / "images" / f"{case_id}_frames.mha"
    labels_path = case_dir / "targets" / f"{case_id}_labels.mha"

    frames = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(frames_path)))
    labels = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(labels_path)))
    assert frames.shape == labels.shape, (
        f"{case_id}: frame series shape {frames.shape} != label series shape {labels.shape}"
    )

    if is_already_converted(case_id, frames.shape[2]):
        print(f"skip {case_id} (already converted)")
        return case_id

    print(f"convert {case_id} ({frames.shape[2]} frames)")
    save_mri_series_as_jpegs(frames, JPEG_ROOT / case_id)
    save_ann_pngs(labels, ANN_ROOT / case_id)
    return case_id


def write_file_list(path: Path, case_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(case_ids)) + "\n")


def main() -> None:
    training_cases = sorted((DATA_ROOT / TRAINING_SPLIT).iterdir())
    testing_cases = sorted((DATA_ROOT / TESTING_SPLIT).iterdir())
    pretest_cases = sorted((DATA_ROOT / PRETEST_SPLIT).iterdir())

    all_case_dirs = training_cases + testing_cases + pretest_cases
    for case_dir in all_case_dirs:
        convert_case(case_dir)

    alldata_ids = [d.name for d in all_case_dirs]
    notest_ids = [d.name for d in training_cases + pretest_cases]

    write_file_list(FILE_LIST_ROOT / "alldata.txt", alldata_ids)
    write_file_list(FILE_LIST_ROOT / "notest.txt", notest_ids)
    print(f"alldata.txt: {len(alldata_ids)} cases")
    print(f"notest.txt: {len(notest_ids)} cases")


if __name__ == "__main__":
    main()
