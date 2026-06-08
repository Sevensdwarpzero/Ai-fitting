import os
import subprocess
import sys
import urllib.request

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")

if os.path.exists(VENV_PYTHON):
    current_python = os.path.normcase(os.path.abspath(sys.executable))
    project_python = os.path.normcase(os.path.abspath(VENV_PYTHON))

    if current_python != project_python:
        print("Wrong Python detected.")
        print(f"Current Python: {sys.executable}")
        print(f"Project Python: {VENV_PYTHON}")
        print("Restarting with the project Python...")
        result = subprocess.run(
            [VENV_PYTHON, os.path.abspath(__file__), *sys.argv[1:]],
            cwd=PROJECT_DIR,
        )
        raise SystemExit(result.returncode)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = os.path.join(PROJECT_DIR, "pose_landmarker_full.task")
PERSON_IMAGE_PATH = os.path.join(PROJECT_DIR, "person.jpg")
SHIRT_IMAGE_PATH = os.path.join(PROJECT_DIR, "shirt.png")
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_DIR, "output_fit.jpg")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def alpha_blend(background, overlay, x_offset, y_offset):
    h, w, _ = background.shape
    overlay_h, overlay_w, _ = overlay.shape

    x1, y1 = max(x_offset, 0), max(y_offset, 0)
    x2, y2 = min(x_offset + overlay_w, w), min(y_offset + overlay_h, h)

    if x1 >= x2 or y1 >= y2:
        return background

    sx1 = x1 - x_offset
    sy1 = y1 - y_offset
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    overlay_crop = overlay[sy1:sy2, sx1:sx2]
    alpha = overlay_crop[:, :, 3] / 255.0

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            alpha * overlay_crop[:, :, c]
            + (1 - alpha) * background[y1:y2, x1:x2, c]
        )

    return background


if not os.path.exists(MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

image_cv = cv2.imread(PERSON_IMAGE_PATH)
shirt = cv2.imread(SHIRT_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

if image_cv is None:
    raise FileNotFoundError("person.jpg not found")
if shirt is None:
    raise FileNotFoundError("shirt.png not found")

if shirt.ndim == 2:
    shirt = cv2.cvtColor(shirt, cv2.COLOR_GRAY2BGRA)
elif shirt.shape[2] == 3:
    shirt = cv2.cvtColor(shirt, cv2.COLOR_BGR2BGRA)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    image_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB),
    )
    results = landmarker.detect(image_mp)

if results.pose_landmarks:
    h, w, _ = image_cv.shape
    landmarks = results.pose_landmarks[0]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    lx, ly = int(left_shoulder.x * w), int(left_shoulder.y * h)
    rx, ry = int(right_shoulder.x * w), int(right_shoulder.y * h)

    shoulder_width = abs(rx - lx)
    shirt_w = int(shoulder_width * 2.5)
    ratio = shirt_w / shirt.shape[1]
    shirt_h = int(shirt.shape[0] * ratio)
    shirt_resized = cv2.resize(shirt, (shirt_w, shirt_h))

    x_offset = rx - (shirt_w - shoulder_width) // 2
    y_offset = min(ly, ry) - int(shirt_h * 0.15)

    image_cv = alpha_blend(image_cv, shirt_resized, x_offset, y_offset)
else:
    print("No pose detected. Saving the original image.")

cv2.imwrite(OUTPUT_IMAGE_PATH, image_cv)
print(f"Virtual fitting result saved: {OUTPUT_IMAGE_PATH}")
