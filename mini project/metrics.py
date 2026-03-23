"""Geometry helpers for blink and facial measurements."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


Point = Tuple[float, float]


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Return Euclidean distance between 2D points."""
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def get_landmark_xy(face_landmarks, index: int, frame_width: int, frame_height: int) -> Point:
    """Convert normalized face landmark coordinates to pixel coordinates."""
    lm = face_landmarks[index]
    return float(lm.x * frame_width), float(lm.y * frame_height)


def pair_distance(face_landmarks, pair: Tuple[int, int], frame_width: int, frame_height: int) -> float:
    """Distance between two face landmarks in pixels."""
    p1 = get_landmark_xy(face_landmarks, pair[0], frame_width, frame_height)
    p2 = get_landmark_xy(face_landmarks, pair[1], frame_width, frame_height)
    return euclidean_distance(p1, p2)


def facial_dimensions_px(face_landmarks, frame_width: int, frame_height: int, indices: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    """Compute facial dimensions in pixels for configured landmark pairs."""
    return {
        key: pair_distance(face_landmarks, pair, frame_width, frame_height)
        for key, pair in indices.items()
    }
