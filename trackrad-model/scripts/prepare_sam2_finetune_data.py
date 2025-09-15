"""Convert the labeled TrackRAD2025 data/ splits into a DAVIS-style JPEG/PNG
video dataset for SAM2 VOS fine-tuning (training/dataset/vos_raw_dataset.py's
PNGRawDataset), plus file-list manifests for the "manual", "semiauto" and
"combined" training subsets.

Converts the manually labeled training split, then writes the manual/semiauto/
combined file lists. Run it after propagate_labels.py so the file lists pick up
the semi-automatic sequences it writes to the same folders.

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


def write_file_lists() -> None:
    """Write manual/semiauto/combined file lists from the converted case folders.

    Manual cases are TrackRAD2025 ids (e.g. A_017); semi-auto cases are named
    `<patient>-<sequence>` by propagate_labels.py, so the "-" tells them apart.
    """
    case_ids = sorted(d.name for d in ANN_ROOT.iterdir() if d.is_dir())
    manual_ids = [c for c in case_ids if "-" not in c]
    semiauto_ids = [c for c in case_ids if "-" in c]
    write_file_list(FILE_LIST_ROOT / "manual.txt", manual_ids)
    write_file_list(FILE_LIST_ROOT / "semiauto.txt", semiauto_ids)
    write_file_list(FILE_LIST_ROOT / "combined.txt", manual_ids + semiauto_ids)
    print(
        f"manual.txt: {len(manual_ids)}, semiauto.txt: {len(semiauto_ids)}, "
        f"combined.txt: {len(manual_ids) + len(semiauto_ids)} cases"
    )


def main() -> None:
    training_cases = sorted((DATA_ROOT / TRAINING_SPLIT).iterdir())

    for case_dir in training_cases:
        convert_case(case_dir)

    # Run again after propagate_labels.py to include the semi-automatic sequences
    write_file_lists()


if __name__ == "__main__":
    main()
