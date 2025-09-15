"""
Edit this file to implement your algorithm.

The file must contain a function called `run_algorithm` that takes two arguments:
- `frames` (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
- `target` (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    from sam2.sam2_video_predictor import SAM2VideoPredictor  # type:ignore


def save_mri_series_as_jpegs(
    path: Path,
    input_mri_linac_series: np.ndarray,
) -> Path:
    jpegs_dir = path / "jpegs"

    if jpegs_dir.exists():
        shutil.rmtree(jpegs_dir)
    jpegs_dir.mkdir(parents=True, exist_ok=True)

    for i in range(input_mri_linac_series.shape[2]):
        frame = input_mri_linac_series[:, :, i]
        frame = frame.astype(np.float32)
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(
            np.uint8
        )

        img = Image.fromarray(frame)
        img.convert("L").save(
            jpegs_dir / f"{i:03d}.jpg", "JPEG", quality=95, subsampling=0
        )
    return jpegs_dir


DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def save_ann_png(path: Path, mask: np.ndarray, frame_idx: int):
    """Save a mask as a PNG file with the given palette."""
    ann_dir = path / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_png = ann_dir / f"{frame_idx:03d}.png"

    mask = mask.astype(np.uint8)  # Ensure mask is in uint8 format
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(DAVIS_PALETTE)
    output_mask.save(str(ann_png))


LOGIT_THRESHOLD = 0.0


def run_algorithm(
    predictor: SAM2VideoPredictor,
    refiner: nnUNetPredictor,
    case_id: str,  # unique ID for the case to disambiguate filepaths for saved images
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
    refinement_lookback_frames: int,  # sliding window length for nnUNet input
    do_refinement: bool,  # whether to perform refinement using nnUNet
    save_annotations: bool,  # whether to save annotation PNGs for visualization
) -> np.ndarray:
    """Implement your algorithm here.

    Args:
    - frames (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
    - target (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
    - frame_rate (float): The frame rate of the MRI linac series.
    - magnetic_field_strength (float): The magnetic field strength of the MRI linac series.
    - scanned_region (str): The scanned region of the MRI linac series.
    """

    path = Path(f"./tmp/{case_id}")
    jpeg_path = save_mri_series_as_jpegs(path, frames)

    # NOTE: Despite needing to initialize with the full video, SAM2 does not use future frames for inference
    # Using SAM2 in real-time is possible with modifications, e.g., https://github.com/Gy920/segment-anything-2-real-time
    inference_state = predictor.init_state(video_path=str(jpeg_path))
    first_frame_mask = target[:, :, 0]
    obj_id = 0
    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        mask=first_frame_mask,
    )

    # video_segments = {frame_idx: {obj_id: mask}}
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > LOGIT_THRESHOLD).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Collect masks in order of out_frame_idx
    ordered_indices = sorted(video_segments.keys())
    ordered_masks = []
    for frame_idx in ordered_indices:
        # Assuming only one object per frame; adjust if multiple objects per frame
        mask = video_segments[frame_idx][obj_id]
        mask = mask.squeeze()
        ordered_masks.append(mask)

    masks_array = np.stack(ordered_masks, axis=-1)

    if do_refinement:
        # Mask refinement using nnUNet
        assert refiner.list_of_parameters
        # Store raw image and mask in two channels
        #
        # C = 2 * refinement_lookback_frames
        # (C, 1, W, H, T) array
        combined_array = np.vstack(
            [
                # nnUNet expects (C, 1, W, H) format
                frames[np.newaxis, np.newaxis],
                masks_array[np.newaxis, np.newaxis],
            ],
            dtype=np.float32,
        )

        # length T list of (C, 1, W, H) arrays
        combined_images = []
        for i in range(combined_array.shape[-1]):
            lookback_frames = []
            for j in range(refinement_lookback_frames):
                frame_idx = i - (refinement_lookback_frames - 1) + j
                if frame_idx < 0:
                    frame_idx = 0
                lookback_frames.append(combined_array[:, :, :, :, frame_idx])
            lookback_array = np.concatenate(lookback_frames, axis=0)
            combined_images.append(lookback_array)

        # From nnUNet/nnunetv2/imageio/natural_image_reader_writer.py
        natural_image_props = {"spacing": (999, 1, 1)}

        # Apparently this is needed for ensembling
        save_or_return_probabilities = len(refiner.list_of_parameters) > 1

        image_iter = tqdm(combined_images, "Refining masks with nnUNet")
        refined_masks = [
            refiner.predict_single_npy_array(
                img,
                natural_image_props,
                save_or_return_probabilities=save_or_return_probabilities,
            )
            for img in image_iter
        ]
        if save_or_return_probabilities:
            refined_masks = cast(list[tuple[np.ndarray, np.ndarray]], refined_masks)
            refined_masks = [rf[0] for rf in refined_masks]
        else:
            refined_masks = cast(list[np.ndarray], refined_masks)
        # Convert list of torch tensors to a single numpy 3D array (W, H, T)
        masks_array = np.stack(refined_masks, axis=-1).squeeze()

    # We've seen some cases where nnUNet produces incomplete masks
    # So fill holes if they exist
    for i in range(masks_array.shape[-1]):
        masks_array[:, :, i] = binary_fill_holes(masks_array[:, :, i]).astype(  # type: ignore  # binary_fill_holes should return np.ndarray
            masks_array.dtype
        )

    if save_annotations:
        for i in range(masks_array.shape[-1]):
            save_ann_png(path, masks_array[:, :, i], i)

    return masks_array
