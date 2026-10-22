"""Validated runtime configuration for VerityAgent face detection."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera_index: int = 0
    min_detection_confidence: float = 0.5
    show_fps: bool = True
    model_path: str = "models/blaze_face_short_range.tflite"
    camera_width: int = 1280
    camera_height: int = 720
    window_name: str = "VerityAgent | Real-time Face Detection"

    def __post_init__(self) -> None:
        if isinstance(self.camera_index, bool) or not isinstance(self.camera_index, int) or self.camera_index < 0:
            raise ValueError("camera_index must be a non-negative integer")
        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("camera_width and camera_height must be positive integers")
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")
