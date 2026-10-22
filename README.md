<div align="center">

# VerityAgent · Real-Time Face Detection

**Local-first computer vision with Python, OpenCV, and MediaPipe Tasks**

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-0097A7)
![Tests](https://img.shields.io/badge/tests-pytest-informational)

*VerityAgent Python AI Mini Projects*

</div>

## Overview

A real-time **face detection** application that reads webcam frames, runs MediaPipe's supported **Tasks FaceDetector API**, draws bounding boxes, and overlays the number of detected faces and optional FPS. Inference runs locally after the model is downloaded; no API key or hosted inference service is required.

> **Privacy boundary:** detects face locations only. It does not identify people, compare identities, or infer demographic attributes. Frames are not uploaded by this application.

## Example output

```text
┌──────────────────────────────────────────┐
│ Faces detected: 2              FPS: 28.4 │
│                                          │
│       ┌──────────┐       ┌─────────┐      │
│       │  FACE 1  │       │ FACE 2  │      │
│       └──────────┘       └─────────┘      │
│             Live webcam frame             │
└──────────────────────────────────────────┘
```

## Requirements

- Python **3.10–3.12** (matches the pinned dependency target).
- Webcam for interactive mode; not required for unit tests or `--headless`.
- Camera permission for the terminal/Python process.
- Internet access to install dependencies and download the model. Inference is local.

## Installation

### 1. Clone and enter the repository

```bash
git clone https://github.com/muhammadsaeed-ai/verityagent-real-time-face-detection.git
cd verityagent-real-time-face-detection
```

### 2. Create an environment and install dependencies

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the official MediaPipe model

The model is intentionally excluded from Git. From the repository root, run:

```bash
python -c "from pathlib import Path; from urllib.request import urlretrieve; p=Path('models/blaze_face_short_range.tflite'); p.parent.mkdir(parents=True, exist_ok=True); urlretrieve('https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite', p); print(f'Model saved to {p} ({p.stat().st_size:,} bytes)')"
```

Use only a trusted MediaPipe-compatible `.tflite` model asset.

## Run

```bash
python app.py
```

Press **Q** in the video window to quit. Camera release and OpenCV window cleanup are handled in a `finally` block.

### Configuration

```bash
python app.py --camera 1                 # choose camera index
python app.py --confidence 0.7            # threshold from 0.0 to 1.0
python app.py --no-fps                    # hide FPS
python app.py --model models/custom.tflite
python app.py --headless                  # synthetic frame; no webcam or GUI
python app.py --camera-width 640 --camera-height 480
```

Higher confidence thresholds generally reject more low-confidence detections. The default camera resolution is 1280×720.

## Tests

```bash
pytest -q
```

The unit tests use mocked detection objects and cover normalized-to-pixel conversion/clipping, detection counting, and configuration validation. They do not require a webcam or model inference.

## Project structure

- `app.py` — CLI, camera lifecycle, FPS smoothing, and visualization.
- `face_detector.py` — MediaPipe Tasks setup, BGR→RGB conversion, and bounding-box helpers.
- `config.py` — immutable runtime configuration and validation.
- `tests/test_face_detector.py` — deterministic unit tests.
- `models/` — downloaded model location; model binary is git-ignored.

## API and dependency compatibility

Pinned dependency target: `mediapipe==0.10.21`, `opencv-python==4.10.0.84`, and `numpy==1.26.4`. The implementation uses `mediapipe.tasks.python.vision.FaceDetector`, `FaceDetectorOptions`, `BaseOptions`, `RunningMode.IMAGE`, and `mp.Image`; it does **not** use the legacy `mp.solutions.face_detection` API. Use Python 3.10–3.12 for this pinned environment.

## Troubleshooting

- **Model not found:** download the model from the command above or pass the correct `--model` path.
- **Camera unavailable:** reconnect the camera, close other apps using it, try `--camera 1`, and grant OS camera permission.
- **Frame read failure:** check camera drivers and try a lower resolution.
- **Install errors:** create a clean Python 3.10–3.12 virtual environment and reinstall requirements.
- **No GUI available:** interactive mode needs a display and GUI-capable OpenCV; use `--headless` for a non-GUI smoke run.
- **Low FPS:** lower resolution and close CPU-heavy applications.

## Limitations

- BlazeFace short-range is designed for relatively nearby faces; small, occluded, angled, or poorly lit faces may be missed.
- FPS and detection quality vary by hardware, lighting, resolution, and camera drivers.
- Headless mode validates the inference path on a synthetic blank frame; it is not an accuracy benchmark.
- Unit tests do not replace a real webcam test on the target OS.

---

Part of **VerityAgent Python AI Mini Projects** — a maintainable, auditable computer-vision baseline for experimentation.
