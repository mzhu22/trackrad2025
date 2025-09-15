from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("tmp/videos")


def numpy_to_video_opencv(array: np.ndarray, output_prefix: str, fps: int) -> str:
    limit = 10 * fps
    array_clip = array[:, :, :limit]  # 10s of video
    p99: float = np.percentile(array_clip, 99)  # type: ignore
    array_clip_normalized = cv2.convertScaleAbs(array_clip, alpha=(255.0 / p99))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str((OUT_DIR / output_prefix).with_suffix(".webm"))

    # Define codec and create VideoWriter
    # VP90 is supported by browsers and is available in the pip-installed opencv
    fourcc = cv2.VideoWriter.fourcc(*"VP90")
    X, Y, T = array_clip.shape
    out = cv2.VideoWriter(output_path, fourcc, fps, (X, Y), isColor=False)
    # Write frames
    for t in range(T):
        frame = array_clip_normalized[:, :, t]
        # OpenCV expects frames in BGR format, but for grayscale we can use as-is
        out.write(frame)

    out.release()
    return output_path
