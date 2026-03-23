"""Configuration constants for the mini project analyzer."""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

# Blink detection defaults.
DEFAULT_EAR_THRESHOLD = 0.21
DEFAULT_CONSEC_FRAMES = 2
DEFAULT_FRAME_STEP = 1

# Approximate average human iris diameter used for scale estimation.
IRIS_DIAMETER_MM = 11.8

# FaceMesh landmark indices.
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Iris landmarks (requires refine_landmarks=True).
RIGHT_IRIS_HORIZONTAL = (469, 471)
RIGHT_IRIS_VERTICAL = (470, 472)
LEFT_IRIS_HORIZONTAL = (474, 476)
LEFT_IRIS_VERTICAL = (475, 477)

# Facial measurements.
FACE_WIDTH_POINTS = (234, 454)
FACE_HEIGHT_POINTS = (10, 152)
NOSE_WIDTH_POINTS = (129, 358)
NOSE_HEIGHT_POINTS = (168, 2)
MOUTH_WIDTH_POINTS = (61, 291)
MOUTH_HEIGHT_POINTS = (13, 14)
LEFT_EYE_WIDTH_POINTS = (362, 263)
LEFT_EYE_HEIGHT_POINTS = (386, 374)
RIGHT_EYE_WIDTH_POINTS = (33, 133)
RIGHT_EYE_HEIGHT_POINTS = (159, 145)
