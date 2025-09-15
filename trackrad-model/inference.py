"""
The following is an entrypoint script for an algorithm.

It load the input data, runs the algorithm and saves the output data.

The actual algorithm is implemented in the model.py file.

You should not need to modify this file.

"""

from __future__ import annotations

import json
import time
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from sam2.build_sam import build_sam2_video_predictor

if TYPE_CHECKING:
    from sam2.sam2_video_predictor import SAM2VideoPredictor  # type:ignore

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

# Defaults when running in the Docker container
SAM2_CHECKPOINT = Path("./resources/sam2.1_hiera_small_trackrad_07_21.pt")
NN_UNET_MODEL_FOLDER = Path("./resources/nnunet-model")
NN_UNET_FOLDS = ("all",)
NN_UNET_CHECKPOINT = "checkpoint_final.pth"
NN_UNET_TILE_STEP_SIZE = 0.05
NN_UNET_REFINEMENT_LOOKBACK_FRAMES = 5


# Add runtime params for easier testing
def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    sam2_checkpoint: Path = SAM2_CHECKPOINT,
    nnunet_model_folder: Path = NN_UNET_MODEL_FOLDER,
    nnunet_folds: tuple[str] | tuple[int, ...] = NN_UNET_FOLDS,
    nnunet_checkpoint_name: str = NN_UNET_CHECKPOINT,
    nnunet_tile_step_size: float = NN_UNET_TILE_STEP_SIZE,
    do_refinement: bool = True,
) -> int:
    loading_start_time = time.perf_counter()

    # Read the inputs
    input_frame_rate = load_json_file(
        location=input_path / "frame-rate.json",
    )
    input_magnetic_field_strength = load_json_file(
        location=input_path / "b-field-strength.json",
    )
    input_scanned_region = load_json_file(
        location=input_path / "scanned-region.json",
    )

    input_mri_linac_series = load_image_file_as_array(
        location=input_path / "images/mri-linacs",
    )
    input_mri_linac_target = load_image_file_as_array(
        location=input_path / "images/mri-linac-target",
    )

    ## BEGIN Additional setup
    # 1. Load SAM2 for object tracking
    predictor = setup_sam2(sam2_checkpoint)
    # 2. Load nnUNet for macrodata refinement
    refiner = nnUNetPredictor(
        # Change from defaults to improve inference speed
        tile_step_size=nnunet_tile_step_size,  # Between 0 and 1
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    refiner.initialize_from_trained_model_folder(
        nnunet_model_folder.as_posix(),
        use_folds=nnunet_folds,  # type: ignore  # use_folds type annotation is wrong
        checkpoint_name=nnunet_checkpoint_name,
    )
    ## END Additional setup

    print(f"Runtime loading:   {time.perf_counter() - loading_start_time:.5f} s")

    from model import run_algorithm

    algo_start_time = time.perf_counter()

    output_mri_linac_series_targets = run_algorithm(
        predictor,
        refiner,
        case_id=input_path.name,
        frames=input_mri_linac_series,
        target=input_mri_linac_target,
        frame_rate=input_frame_rate,
        magnetic_field_strength=input_magnetic_field_strength,
        scanned_region=input_scanned_region,
        refinement_lookback_frames=NN_UNET_REFINEMENT_LOOKBACK_FRAMES,
        do_refinement=do_refinement,
        save_annotations=False,
    )

    # Enforce uint8 as output dtype
    output_mri_linac_series_targets = output_mri_linac_series_targets.astype(np.uint8)

    print(f"Runtime algorithm: {time.perf_counter() - algo_start_time:.5f} s")

    writing_start_time = time.perf_counter()

    # Save the output
    write_array_as_image_file(
        location=output_path / "images/mri-linac-series-targets",
        array=output_mri_linac_series_targets,
    )
    print(f"Runtime writing:   {time.perf_counter() - writing_start_time:.5f} s")

    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def setup_sam2(checkpoint_path: Path) -> SAM2VideoPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.autocast("cuda", dtype=dtype).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    predictor = build_sam2_video_predictor(
        model_cfg,
        checkpoint_path.as_posix(),
        device=device.type,
        # vos_optimized=True is supposed to make inference faster but I couldn't make it work
        vos_optimized=False,
    )
    return predictor


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
