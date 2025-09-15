from typing import Any, Dict, TypedDict

import cv2
import gradio as gr
import numpy as np
import spaces  # type: ignore
import torch
from sam2.build_sam import build_sam2  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT = "./assets/sam2.1_hiera_base_plus.pt"


# ZeroGPU decorator
# Default duration is 60s, setting a lower number apparently gives better queue priority
# http://huggingface.co/docs/hub/en/spaces-zerogpu
@spaces.GPU(duration=10)
@torch.inference_mode()
def predict_gpu(annotations: Dict[str, Any]):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model = build_sam2(CONFIG, CHECKPOINT, device=torch.device("cuda"))  # type: ignore
        return predict(model, annotations)


def predict_cpu(annotations: Dict[str, Any]):
    model = build_sam2(CONFIG, CHECKPOINT, device=torch.device("cpu"))  # type: ignore
    return predict(model, annotations)


class PredictResult(TypedDict):
    masks: np.ndarray
    scores: np.ndarray


def predict(model, annotations: dict[str, Any]) -> tuple[Any, PredictResult]:
    predictor = SAM2ImagePredictor(model)
    if not annotations["boxes"]:
        raise gr.Error(
            "Please draw at least one bounding box on the image before proceeding."
        )
    predictor.set_image(annotations["image"])
    coordinates = []
    for i in range(len(annotations["boxes"])):
        coordinate = [
            int(annotations["boxes"][i]["xmin"]),
            int(annotations["boxes"][i]["ymin"]),
            int(annotations["boxes"][i]["xmax"]),
            int(annotations["boxes"][i]["ymax"]),
        ]
        coordinates.append(coordinate)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(coordinates),
        multimask_output=False,
    )

    # Convert image to grayscale (HWC to single channel)
    gray_image = cv2.cvtColor(annotations["image"], cv2.COLOR_BGR2GRAY)
    masks = masks.squeeze()

    # Overlay masks onto the grayscale image
    if masks.ndim == 2:
        masks = np.expand_dims(masks, axis=0)
    gray_overlay = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for mask in masks:
        color_mask = np.zeros_like(gray_overlay)
        color_mask[mask > 0] = [0, 255, 0]  # Green overlay
        gray_overlay = cv2.addWeighted(gray_overlay, 1.0, color_mask, 0.5, 0)

    return {"image": gray_overlay}, {"masks": masks, "scores": scores}
