from types import SimpleNamespace

import pytest

from config import AppConfig
from face_detector import FaceBox, detection_boxes, normalized_box_to_pixels


def test_normalized_box_to_pixels_rounds_and_clips() -> None:
    box = normalized_box_to_pixels(-0.1, 0.2, 0.5, 0.9, 200, 100)
    assert box == FaceBox(0, 20, 80, 100)


def test_detection_boxes_counts_valid_detections() -> None:
    detections = [
        SimpleNamespace(bounding_box=SimpleNamespace(origin_x=10, origin_y=10, width=40, height=30)),
        SimpleNamespace(bounding_box=SimpleNamespace(origin_x=0, origin_y=0, width=0, height=10)),
        SimpleNamespace(bounding_box=None),
    ]
    result = SimpleNamespace(detections=detections)
    assert detection_boxes(result, 100, 80) == [FaceBox(10, 10, 50, 40)]


def test_detection_boxes_counts_pixel_coordinate_detections() -> None:
    detections = [
        SimpleNamespace(bounding_box=SimpleNamespace(origin_x=20, origin_y=20, width=60, height=40)),
        SimpleNamespace(bounding_box=SimpleNamespace(origin_x=100, origin_y=10, width=40, height=20)),
    ]
    boxes = detection_boxes(SimpleNamespace(detections=detections), 200, 100)
    assert len(boxes) == 2
    assert boxes[0] == FaceBox(20, 20, 80, 60)
    assert boxes[1] == FaceBox(100, 10, 140, 30)


def test_config_accepts_valid_values() -> None:
    config = AppConfig(camera_index=2, min_detection_confidence=0.75, show_fps=False)
    assert config.camera_index == 2
    assert config.min_detection_confidence == 0.75
    assert config.show_fps is False


@pytest.mark.parametrize("kwargs", [
    {"camera_index": -1},
    {"camera_index": True},
    {"min_detection_confidence": -0.1},
    {"min_detection_confidence": 1.1},
    {"camera_width": 0},
    {"camera_height": -1},
    {"model_path": "   "},
])
def test_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        AppConfig(**kwargs)


def test_normalized_box_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        normalized_box_to_pixels(0, 0, 1, 1, 0, 10)
