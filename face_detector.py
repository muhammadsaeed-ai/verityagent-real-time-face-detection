"""MediaPipe Tasks-based face detection and bounding-box utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FaceBox:
    """Bounding box in pixel coordinates, clipped to the image dimensions."""
    x1: int
    y1: int
    x2: int
    y2: int


def normalized_box_to_pixels(
    origin_x: float, origin_y: float, width: float, height: float,
    image_width: int, image_height: int,
) -> FaceBox:
    """Convert a normalized box to a clipped pixel-coordinate box."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    values = (origin_x, origin_y, width, height)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Bounding-box values must be finite")
    left = max(0, min(image_width, int(np.floor(origin_x * image_width))))
    top = max(0, min(image_height, int(np.floor(origin_y * image_height))))
    right = max(0, min(image_width, int(np.ceil((origin_x + width) * image_width))))
    bottom = max(0, min(image_height, int(np.ceil((origin_y + height) * image_height))))
    return FaceBox(left, top, max(left, right), max(top, bottom))


def detection_boxes(result: Any, image_width: int, image_height: int) -> list[FaceBox]:
    """Extract valid pixel boxes from a MediaPipe FaceDetectorResult-like object."""
    boxes: list[FaceBox] = []
    for detection in getattr(result, "detections", None) or []:
        bbox = getattr(detection, "bounding_box", None)
        if bbox is None:
            continue
        # MediaPipe Tasks Detection.bounding_box values are already image pixels.
        left = max(0, min(image_width, int(bbox.origin_x)))
        top = max(0, min(image_height, int(bbox.origin_y)))
        right = max(0, min(image_width, int(bbox.origin_x + bbox.width)))
        bottom = max(0, min(image_height, int(bbox.origin_y + bbox.height)))
        box = FaceBox(left, top, max(left, right), max(top, bottom))
        if box.x2 > box.x1 and box.y2 > box.y1:
            boxes.append(box)
    return boxes


class FaceDetector:
    """Thin wrapper around the supported MediaPipe Tasks FaceDetector API."""
    def __init__(self, model_path: str | Path, min_detection_confidence: float = 0.5) -> None:
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Face detector model not found: {path}. Follow README instructions to download it."
            )
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe Tasks is unavailable. Install the pinned dependencies from requirements.txt."
            ) from exc

        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self._mp = __import__("mediapipe")
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        """Detect faces in a BGR OpenCV frame and return pixel-coordinate boxes."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a non-empty HxWx3 BGR image")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        return detection_boxes(result, frame_bgr.shape[1], frame_bgr.shape[0])

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()
