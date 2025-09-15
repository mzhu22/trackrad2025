import random
from typing import TypedDict

import cv2
import gradio as gr
import numpy as np
import SimpleITK as sitk
import torch
from gradio_image_annotation import image_annotator  # type: ignore

from common import Frames
from hf_datasets import (
    SequenceStatus,
    download_image_files,
    get_completed_and_todo,
    save_bad_image_report,
    save_masks,
)
from predict import PredictResult, predict_cpu, predict_gpu
from video import numpy_to_video_opencv


class ImageData(TypedDict):
    patient: str
    sequence_idx: int
    frames_file: str
    field_strength: str
    scanned_region: str
    fps: int


def download_images(sequence: Frames) -> ImageData:
    if not sequence:
        return None
    patient, idx = sequence
    frames_file, field_strength_file, scanned_region_file, fps_file = (
        download_image_files(patient, idx)
    )
    with open(field_strength_file, "r") as f:
        field_strength = f.read().strip()
    with open(scanned_region_file, "r") as f:
        scanned_region = f.read().strip()
    with open(fps_file, "r") as f:
        fps_raw = f.read().strip()
        fps = int(float(fps_raw))

    return {
        "patient": patient,
        "sequence_idx": idx,
        "frames_file": frames_file,
        "field_strength": field_strength,
        "scanned_region": scanned_region,
        "fps": fps,
    }


def load_image(img: ImageData):
    frames = sitk.ReadImage(img["frames_file"])
    frames_array = sitk.GetArrayFromImage(frames)
    frames_array = np.flip(frames_array, axis=0)
    first_frame = frames_array[:, :, 0]

    # Convert 16-bit grayscale to 8-bit grayscale for display/annotation
    # Normalize to p99 instead of max bc some images have bright artifacts
    p99: float = np.percentile(first_frame, 99)  # type: ignore
    first_frame_8bit = cv2.convertScaleAbs(first_frame, alpha=(255.0 / p99))
    first_frame = first_frame_8bit
    image_state = first_frame.copy()

    return {"image": image_state, "boxes": []}


def image_md(img: ImageData) -> str:
    return f"Patient {img['patient']} | Sequence {img['sequence_idx']} | Region {img['scanned_region']} | Field Strength {img['field_strength']}T | Frame Rate {img['fps']}Hz"


def load_video(img: ImageData) -> str:
    frames = sitk.ReadImage(img["frames_file"])
    frames_array = sitk.GetArrayFromImage(frames)
    frames_array = np.flip(frames_array, axis=0)
    output_path = numpy_to_video_opencv(
        frames_array,
        f"patient_{img['patient']}_sequence_{img['sequence_idx']}",
        fps=img["fps"],
    )
    return output_path


def next_sequence(progress: SequenceStatus | None) -> Frames | None:
    """
    Sample uniformly from the remaining patients, to avoid biasing towards patients with more MR sequences.
    """
    if not progress:
        return None
    todo = progress["todo"]
    todo_patients = [seq[0] for seq in todo]
    patient = random.choice(todo_patients) if todo_patients else None
    if not patient:
        return None

    return next(seq for seq in todo if seq[0] == patient)


with gr.Blocks() as demo:
    progress_state = gr.State()
    demo.load(
        fn=get_completed_and_todo,
        outputs=[progress_state],
    )

    sequence_state = gr.State()
    progress_state.change(
        fn=next_sequence,
        inputs=[progress_state],
        outputs=[sequence_state],
    )

    gr.Markdown("""
        # AI-assisted MR Object Annotation
    """)
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ## FAQ
                **Q. What do I annotate?**
                
                **A.** Pretty much any object you see (up to 5 objects per frame). The pre-labeled training data seems to include both tumors and organs.
                [See examples here](https://huggingface.co/spaces/mayo-radonc/bouncing-target-reference)
                """
            )
        with gr.Column():
            gr.Markdown(
                """
                ## Progress
                ### Deadline: August 1, 2025
                """
            )
            progress_md = gr.Markdown("Loading...")
    gr.LoginButton("""[OPTIONAL] Sign in to attribute annotations to your account.""")

    progress_state.change(
        fn=lambda progress: (
            f"""
            - {len(progress["completed"])} / {len(progress["all"])} MR sequences annotated
            - {len(progress["completed_patients"])} / {len(progress["all_patients"])} patients with at least one sequence annotated
            """
        ),
        inputs=[progress_state],
        outputs=[progress_md],
    )

    with gr.Accordion("Looking for instructions or more info? Click here", open=False):
        gr.Markdown(
            """
            # Background
            This is a tool for semi-automated annotation of MR images. By drawing bounding boxes around objects, you guide an AI model to infer the 
            exact borders, or segmentation masks, of the objects. The goal is to create a high-quality dataset to train another model that will be 
            used for tumor tracking. By providing accurate annotations, you will improve the model's ability to identify anatomical structures in medical images.
            Thanks for the help!

            More info about the dataset: https://trackrad2025.grand-challenge.org/trackrad2025/

            Questions, feature requests, or bug reports? Reach out to Mike Zhu at [zhu.henian@mayo.edu](mailto:zhu.henian@mayo.edu).
            """
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    # Instructions
                    1. Draw boxes around as many objects of interest as you can find. E.g. organs, lesions, tumors, etc. 
                        - DO NOT rotate the image. The code can't handle rotation and I can't disable the buttons
                        - Not sure what to annotate? Draw boxes around *any* object
                        - We want the AI to improve its understanding of the medical images *in general*
                    2. Click the **Get Segmentation Masks** button to infer masks for the objects.
                    3. Repeat 1 and 2 until you are satisfied with the masks.
                    4. Click **Submit** when complete. The page will automatically reload.
                    """
                )
            with gr.Column():
                gr.Markdown(
                    """
                    # Tips
                    - You can zoom in/out using the mouse
                    - To draw a new box, click the rectangle icon, then click and drag on the image
                    - Use the hand icon to edit boxes. You can move and resize them
                    - You can also use the hand icon to pan the image
                    - To delete a box, click the hand icon, click the box, then click the trash can

                    # Troubleshooting
                    - Error? Refresh the page and try again
                    """
                )

    patient_info_md = gr.Textbox(label="Sequence info", interactive=False)
    scores_state = gr.State()
    gr.Markdown(
        "DO NOT rotate the image! The code can't handle rotation and I can't disable the buttons."
    )
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(
                autoplay=True,
                loop=True,
                label="MR Sequence (up to 10 seconds)",
            )
        with gr.Column(scale=2):
            annotator = image_annotator(
                value=None,
                sources=[],
                disable_edit_boxes=True,
                label="First frame (Draw bounding boxes!)",
                show_share_button=False,
                show_clear_button=False,
            )
        with gr.Column(scale=2):
            masked_image = image_annotator(
                label="AI-generated masks (Don't draw on this!)",
                sources=[],
                show_share_button=False,
                show_clear_button=False,
            )

    image_state = gr.State()
    sequence_state.change(
        fn=download_images,
        inputs=[sequence_state],
        outputs=[image_state],
    )
    image_state.change(
        fn=load_image,
        inputs=[image_state],
        outputs=[annotator],
    )
    image_state.change(
        fn=image_md,
        inputs=[image_state],
        outputs=[patient_info_md],
    )
    image_state.change(
        fn=load_video,
        inputs=[image_state],
        outputs=[video],
    )

    def to_dropdown_value(sequence: Frames, progress: SequenceStatus) -> str:
        """
        Convert the sequence (patient ID, sequence number) to a dropdown value.
        Example: ("A_033", 1) -> "A_033-frames1"
        """
        patient, frames = sequence
        sequence_label = f"{patient}-frames{frames}"
        if sequence in progress["completed"]:
            return f"{sequence_label} ✅"
        return sequence_label

    def from_dropdown_value(dropdown_value: str) -> Frames | None:
        """
        Parse the dropdown value to extract the patient ID and sequence number.
        Example: "A_033-frames1" -> ("A_033", 1)
        """
        if dropdown_value is None:
            return None
        sequence_label = dropdown_value.split(" ")[0]  # Remove any ✅ or other symbols
        patient, frames = sequence_label.split("-frames")
        return patient, int(frames)

    with gr.Accordion("Problem with the image? Click here", open=False):
        bad_image_comments = gr.Textbox(
            label="Explain the issue(s)",
            placeholder="e.g. too dark, blurry, too many artifacts, etc.",
            lines=3,
        )
        bad_image_btn = gr.Button("Report Bad Image and Skip", variant="primary")

    with gr.Row():
        get_masks_btn = gr.Button("Get Segmentation Masks")
        save_btn = gr.Button("Submit", interactive=False, variant="primary")
    with gr.Accordion("Advanced Controls", open=False):
        patient_dropdown = gr.Dropdown(
            choices=[],
            label="Select specific MR sequence",
            value=None,
            interactive=True,
        )
        sequence_state.change(
            fn=lambda sequence, progress: to_dropdown_value(sequence, progress)
            if sequence
            else None,
            inputs=[sequence_state, progress_state],
            outputs=[patient_dropdown],
        )
        progress_state.change(
            fn=lambda progress: gr.update(
                patient_dropdown.elem_id,
                choices=[to_dropdown_value(v, progress) for v in progress["all"]],
            ),
            inputs=[progress_state],
            outputs=[patient_dropdown],
        )

    patient_dropdown.change(
        fn=from_dropdown_value,
        inputs=[patient_dropdown],
        outputs=[sequence_state],
    )

    def change_btn_state_after_masks(
        save_btn_id, bad_image_btn_id, bad_image_comments_id
    ):
        return (
            gr.update(save_btn_id, interactive=True),
            gr.update(bad_image_btn_id, interactive=False),
            gr.update(bad_image_comments_id, interactive=False),
        )

    masks_state = gr.State()
    get_masks_btn.click(
        fn=(
            lambda annotator: predict_gpu(annotator)
            if torch.cuda.is_available()
            else predict_cpu(annotator)
        ),
        inputs=[annotator],
        outputs=[masked_image, masks_state],
    )
    masks_state.change(
        fn=change_btn_state_after_masks,
        inputs=[save_btn, bad_image_btn, bad_image_comments],
        outputs=[save_btn, bad_image_btn, bad_image_comments],
    )

    def save_annotation(
        masks_and_scores: PredictResult,
        sequence: Frames,
        profile: gr.OAuthProfile | None,
    ):
        gr.Info("Saving... Page will reload when done.")
        save_masks(
            username=profile.username if profile else "anonymous",
            masks_and_scores=masks_and_scores,
            sequence=sequence,
            frame=0,  # TEMP
        )
        return "Done"

    def save_bad_image(
        sequence: Frames, comments: str, profile: gr.OAuthProfile | None
    ):
        gr.Info("Saving... Page will reload when done.")
        save_bad_image_report(
            username=profile.username if profile else "anonymous",
            sequence=sequence,
            frame=0,  # TEMP
            comments=comments,
        )
        return "Done"

    output_state = gr.State()
    save_btn.click(
        fn=save_annotation,
        inputs=[masks_state, sequence_state],
        outputs=[output_state],
    )
    bad_image_btn.click(
        fn=save_bad_image,
        inputs=[sequence_state, bad_image_comments],
        outputs=[output_state],
    )
    output_state.change(fn=None, inputs=output_state, js="window.location.reload()")


demo.launch()
