"""Video analysis pipeline for blink-rate and facial dimension estimation."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config import (
    DEFAULT_CONSEC_FRAMES,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_FRAME_STEP,
    FACE_HEIGHT_POINTS,
    FACE_WIDTH_POINTS,
    LEFT_EYE_HEIGHT_POINTS,
    LEFT_EYE_WIDTH_POINTS,
    LEFT_IRIS_HORIZONTAL,
    LEFT_IRIS_VERTICAL,
    MOUTH_HEIGHT_POINTS,
    MOUTH_WIDTH_POINTS,
    NOSE_HEIGHT_POINTS,
    NOSE_WIDTH_POINTS,
    RIGHT_EYE_HEIGHT_POINTS,
    RIGHT_EYE_WIDTH_POINTS,
    RIGHT_IRIS_HORIZONTAL,
    RIGHT_IRIS_VERTICAL,
    SCRIPT_DIR,
)
from metrics import facial_dimensions_px
from scaling import convert_dimensions_to_mm, estimate_mm_per_px, smooth_scale


DIMENSION_PAIRS = {
    "left_eye_width": LEFT_EYE_WIDTH_POINTS,
    "left_eye_height": LEFT_EYE_HEIGHT_POINTS,
    "right_eye_width": RIGHT_EYE_WIDTH_POINTS,
    "right_eye_height": RIGHT_EYE_HEIGHT_POINTS,
    "face_width": FACE_WIDTH_POINTS,
    "face_height": FACE_HEIGHT_POINTS,
    "nose_width": NOSE_WIDTH_POINTS,
    "nose_height": NOSE_HEIGHT_POINTS,
    "mouth_width": MOUTH_WIDTH_POINTS,
    "mouth_height": MOUTH_HEIGHT_POINTS,
}

IRIS_PAIRS = [
    (RIGHT_IRIS_HORIZONTAL, RIGHT_IRIS_VERTICAL),
    (LEFT_IRIS_HORIZONTAL, LEFT_IRIS_VERTICAL),
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")


def ensure_model_downloaded() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face_landmarker_task Model to {MODEL_PATH} ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")


@dataclass
class AnalysisSettings:
    ear_threshold: float = DEFAULT_EAR_THRESHOLD
    consec_frames: int = DEFAULT_CONSEC_FRAMES
    frame_step: int = DEFAULT_FRAME_STEP


def list_video_files(videos_dir: str, extensions: set[str]) -> List[str]:
    """Return sorted list of supported video files."""
    if not os.path.isdir(videos_dir):
        return []
    video_paths = []
    for name in sorted(os.listdir(videos_dir)):
        full_path = os.path.join(videos_dir, name)
        _, ext = os.path.splitext(name)
        if os.path.isfile(full_path) and ext in extensions:
            video_paths.append(full_path)
    return video_paths


def _empty_second_bucket() -> Dict:
    return {
        "frames": 0,
        "valid_face_frames": 0,
        "blink_score_sum": 0.0,
        "blinks": 0,
        "scale_sum": 0.0,
        "scale_count": 0,
        "dimension_sums_mm": {key: 0.0 for key in DIMENSION_PAIRS},
        "dimension_counts": {key: 0 for key in DIMENSION_PAIRS},
    }


def analyze_video(video_path: str, settings: AnalysisSettings) -> Dict:
    """Analyze one video and return detailed per-second rows plus summary."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps if frame_count > 0 else 0.0

    total_processed_frames = 0
    total_face_frames = 0
    total_blinks = 0
    low_ear_counter = 0

    scale_history: List[float] = []

    dimension_sum_mm = {key: 0.0 for key in DIMENSION_PAIRS}
    dimension_count = {key: 0 for key in DIMENSION_PAIRS}

    second_buckets: Dict[int, Dict] = {}

    ensure_model_downloaded()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if settings.frame_step > 1 and frame_idx % settings.frame_step != 0:
            continue

        total_processed_frames += 1
        frame_height, frame_width = frame.shape[:2]

        timestamp_sec = frame_idx / fps
        sec_key = int(timestamp_sec)
        if sec_key not in second_buckets:
            second_buckets[sec_key] = _empty_second_bucket()
        second_buckets[sec_key]["frames"] += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection_result = detector.detect(mp_image)

        if not detection_result.face_landmarks:
            if low_ear_counter >= settings.consec_frames:
                total_blinks += 1
                second_buckets[sec_key]["blinks"] += 1
            low_ear_counter = 0
            continue

        face = detection_result.face_landmarks[0]
        blendshapes = detection_result.face_blendshapes[0]

        total_face_frames += 1
        second_buckets[sec_key]["valid_face_frames"] += 1

        left_blink_score = 0.0
        right_blink_score = 0.0
        for blendshape in blendshapes:
            if blendshape.category_name == "eyeBlinkLeft":
                left_blink_score = blendshape.score
            elif blendshape.category_name == "eyeBlinkRight":
                right_blink_score = blendshape.score

        avg_blink_score = (left_blink_score + right_blink_score) / 2.0
        second_buckets[sec_key]["blink_score_sum"] += avg_blink_score

        is_blinking = avg_blink_score > settings.ear_threshold

        if is_blinking:
            low_ear_counter += 1
        else:
            if low_ear_counter >= settings.consec_frames:
                total_blinks += 1
                second_buckets[sec_key]["blinks"] += 1
            low_ear_counter = 0

        dimensions_px = facial_dimensions_px(face, frame_width, frame_height, DIMENSION_PAIRS)
        instant_scale = estimate_mm_per_px(face, frame_width, frame_height, IRIS_PAIRS)
        
        # Jitter filtering for mm/px scale estimation
        if instant_scale is not None and 0.05 < instant_scale < 2.0:
            mm_per_px = smooth_scale(scale_history, instant_scale, window_size=60)
        else:
            mm_per_px = smooth_scale(scale_history, None, window_size=60)

        if mm_per_px is not None:
            second_buckets[sec_key]["scale_sum"] += mm_per_px
            second_buckets[sec_key]["scale_count"] += 1

        dimensions_mm = convert_dimensions_to_mm(dimensions_px, mm_per_px)
        for key, value in dimensions_mm.items():
            if value is None:
                continue
            dimension_sum_mm[key] += value
            dimension_count[key] += 1
            second_buckets[sec_key]["dimension_sums_mm"][key] += value
            second_buckets[sec_key]["dimension_counts"][key] += 1

    cap.release()
    detector.close()

    if low_ear_counter >= settings.consec_frames:
        total_blinks += 1
        if second_buckets:
            second_buckets[max(second_buckets)]["blinks"] += 1

    per_second_rows: List[Dict] = []
    for sec in sorted(second_buckets.keys()):
        bucket = second_buckets[sec]
        valid = bucket["valid_face_frames"]
        row = {
            "video_file": os.path.basename(video_path),
            "second": sec,
            "frames_processed": bucket["frames"],
            "valid_face_frames": valid,
            "coverage_ratio": (valid / bucket["frames"]) if bucket["frames"] > 0 else 0.0,
            "avg_blink_score": (bucket["blink_score_sum"] / valid) if valid > 0 else 0.0,
            "blinks_this_second": bucket["blinks"],
            "avg_mm_per_px": (bucket["scale_sum"] / bucket["scale_count"]) if bucket["scale_count"] > 0 else "",
        }
        for key in DIMENSION_PAIRS:
            count = bucket["dimension_counts"][key]
            row[f"avg_{key}_mm"] = (bucket["dimension_sums_mm"][key] / count) if count > 0 else ""
        per_second_rows.append(row)

    avg_dimensions_mm = {
        key: (dimension_sum_mm[key] / dimension_count[key]) if dimension_count[key] > 0 else np.nan
        for key in DIMENSION_PAIRS
    }

    coverage_ratio = (total_face_frames / total_processed_frames) if total_processed_frames > 0 else 0.0
    blinks_per_second = (total_blinks / duration_seconds) if duration_seconds > 0 else 0.0

    summary = {
        "video_file": os.path.basename(video_path),
        "duration_sec": duration_seconds,
        "frames_total": frame_count,
        "frames_processed": total_processed_frames,
        "valid_face_frames": total_face_frames,
        "coverage_ratio": coverage_ratio,
        "total_blinks": total_blinks,
        "blinks_per_second": blinks_per_second,
    }

    for key, value in avg_dimensions_mm.items():
        summary[f"avg_{key}_mm"] = "" if np.isnan(value) else float(value)
        summary[f"avg_{key}_cm"] = "" if np.isnan(value) else float(value) / 10.0

    return {
        "summary": summary,
        "per_second_rows": per_second_rows,
    }


def aggregate_global_summaries(video_summaries: List[Dict]) -> Dict[str, float]:
    """Compute global means for dimension metrics across videos."""
    if not video_summaries:
        return {}

    keys = [key for key in video_summaries[0].keys() if key.startswith("avg_") and key.endswith("_mm")]
    global_means: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in video_summaries if row.get(key) not in ("", None)]
        if values:
            global_means[key.replace("avg_", "").replace("_mm", "")] = float(np.mean(values))
    return global_means
