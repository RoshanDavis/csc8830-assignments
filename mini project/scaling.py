"""Scale estimation utilities using iris landmark diameter."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import IRIS_DIAMETER_MM
from metrics import pair_distance


IrisPairSet = Tuple[Tuple[int, int], Tuple[int, int]]


def estimate_mm_per_px(face_landmarks, frame_width: int, frame_height: int, iris_pairs: Iterable[IrisPairSet]) -> Optional[float]:
    """Estimate millimeters per pixel from iris diameter landmarks.

    Each iris uses horizontal and vertical diameters; we average all valid diameters.
    """
    diameters_px: List[float] = []
    for horizontal_pair, vertical_pair in iris_pairs:
        h = pair_distance(face_landmarks, horizontal_pair, frame_width, frame_height)
        v = pair_distance(face_landmarks, vertical_pair, frame_width, frame_height)
        for value in (h, v):
            if value > 0:
                diameters_px.append(value)

    if not diameters_px:
        return None

    mean_diameter_px = float(np.mean(diameters_px))
    if mean_diameter_px <= 0:
        return None

    return IRIS_DIAMETER_MM / mean_diameter_px


def smooth_scale(scales: List[float], new_scale: Optional[float], window_size: int = 60) -> Optional[float]:
    """Maintain a rolling average mm/px scale to reduce frame-level jitter."""
    if new_scale is not None and new_scale > 0:
        scales.append(new_scale)
    if len(scales) > window_size:
        del scales[: len(scales) - window_size]
    if not scales:
        return None
    return float(np.mean(scales))


def convert_dimensions_to_mm(dimensions_px: Dict[str, float], mm_per_px: Optional[float]) -> Dict[str, Optional[float]]:
    """Convert dimension dictionary from pixels to millimeters."""
    output: Dict[str, Optional[float]] = {}
    for key, value_px in dimensions_px.items():
        output[key] = None if mm_per_px is None else float(value_px * mm_per_px)
    return output
