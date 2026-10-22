"""Run the VerityAgent real-time face detector."""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from config import AppConfig
from face_detector import FaceBox, FaceDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face detection (no identification).")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection threshold from 0.0 to 1.0")
    parser.add_argument("--model", default="models/blaze_face_short_range.tflite", help="MediaPipe .tflite model path")
    parser.add_argument("--no-fps", action="store_true", help="Hide the FPS counter")
    parser.add_argument("--headless", action="store_true", help="Run one inference pass on a blank test frame; no webcam or display")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    return parser.parse_args()


def draw_overlay(frame: np.ndarray, boxes: list[FaceBox], fps: float | None = None) -> None:
    """Draw detection rectangles and the live face count on a frame."""
    for box in boxes:
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (40, 220, 120), 2)
    cv2.rectangle(frame, (0, 0), (300, 78 if fps is not None else 48), (16, 22, 35), -1)
    cv2.putText(frame, f"Faces detected: {len(boxes)}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (14, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2, cv2.LINE_AA)


def run_headless(detector: FaceDetector) -> int:
    """Smoke-test inference on a synthetic frame without a camera or GUI."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector.detect(frame)
    print(f"Headless inference completed. Faces detected in blank frame: {len(boxes)}")
    return 0


def run_camera(config: AppConfig, detector: FaceDetector) -> int:
    camera = cv2.VideoCapture(config.camera_index)
    try:
        if not camera.isOpened():
            print(f"ERROR: Could not open camera index {config.camera_index}. Check connection and OS permissions.")
            return 2
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        previous = time.perf_counter()
        fps = 0.0
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("ERROR: Failed to read a frame from the webcam.")
                return 3
            boxes = detector.detect(frame)
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            if elapsed > 0:
                instantaneous = 1.0 / elapsed
                fps = instantaneous if fps == 0 else 0.85 * fps + 0.15 * instantaneous
            draw_overlay(frame, boxes, fps if config.show_fps else None)
            cv2.imshow(config.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        config = AppConfig(
            camera_index=args.camera,
            min_detection_confidence=args.confidence,
            show_fps=not args.no_fps,
            model_path=args.model,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
        )
        detector = FaceDetector(config.model_path, config.min_detection_confidence)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        if args.headless:
            return run_headless(detector)
        return run_camera(config, detector)
    finally:
        detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
