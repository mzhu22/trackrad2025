"""Download the data needed to reproduce the paper into <repo>/data/.

Everything comes from public Hugging Face datasets:
  * TrackRAD2025 (https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025)
      - trackrad2025_labeled_training_data/   manual labels for fine-tuning
      - trackrad2025_labeled_testing_data/ and trackrad2025_labeled_pre-testing_data/
        the 38 test sequences (30 + 8); linked together into
        trackrad2025_labeled_test_data/ for scripts/eval_sam2_only.py
      - trackrad2025_unlabeled_training_data/ only the sequences that have a
        semi-automatic label (the full unlabeled set is not needed)
  * mzhu22/bouncing-target (pinned revision): first-frame masks drawn with the
    labeling app, saved to data/bouncing-target/

Run with: uv run python scripts/download_data.py
(from trackrad-model/)
"""

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

IMAGES_REPO = "LMUK-RADONC-PHYS-RES/TrackRAD2025"
LABELS_REPO = "mzhu22/bouncing-target"
# Latest commit as of writing: all 200 labeled sequences used in the paper
LABELS_REVISION = "a2bc81cbd17231a0382a340dd4e0981d7bab025c"

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
LABELED_SPLITS = [
    "trackrad2025_labeled_training_data",
    "trackrad2025_labeled_testing_data",
    "trackrad2025_labeled_pre-testing_data",
]
UNLABELED_FOLDER = "trackrad2025_unlabeled_training_data"
TEST_DIR = DATA_ROOT / "trackrad2025_labeled_test_data"


def download_labeled() -> None:
    snapshot_download(
        repo_id=IMAGES_REPO,
        repo_type="dataset",
        allow_patterns=[f"{split}/*" for split in LABELED_SPLITS],
        local_dir=DATA_ROOT,
    )
    # evaluate.py wants a single directory of cases
    TEST_DIR.mkdir(exist_ok=True)
    for split in LABELED_SPLITS[1:]:
        for case_dir in sorted((DATA_ROOT / split).iterdir()):
            link = TEST_DIR / case_dir.name
            if not link.exists():
                link.symlink_to(case_dir.resolve(), target_is_directory=True)
    print(f"{len(list(TEST_DIR.iterdir()))} test cases in {TEST_DIR}")


def download_semiauto_labels() -> Path:
    labels_dir = DATA_ROOT / "bouncing-target"
    snapshot_download(
        repo_id=LABELS_REPO,
        repo_type="dataset",
        revision=LABELS_REVISION,
        local_dir=labels_dir,
    )
    return labels_dir


def download_unlabeled_for(labels_dir: Path) -> None:
    """Fetch only the unlabeled sequences that have a semi-automatic label."""
    sequences = set()
    for masks in labels_dir.glob("*/masks.png"):
        patient, sequence, _ = masks.parent.name.split("-", 2)
        sequences.add((patient, sequence))

    for patient, sequence in sorted(sequences):
        idx = "" if sequence == "1" else sequence
        folder = f"{UNLABELED_FOLDER}/{patient}"
        for filename in [
            f"{folder}/images/{patient}_frames{idx}.mha",
            f"{folder}/b-field-strength.json",
            f"{folder}/frame-rate{idx}.json",
            f"{folder}/scanned-region{idx}.json",
        ]:
            hf_hub_download(
                repo_id=IMAGES_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=DATA_ROOT,
            )
    print(f"{len(sequences)} unlabeled sequences downloaded")


def main() -> None:
    DATA_ROOT.mkdir(exist_ok=True)
    download_labeled()
    labels_dir = download_semiauto_labels()
    download_unlabeled_for(labels_dir)


if __name__ == "__main__":
    main()
