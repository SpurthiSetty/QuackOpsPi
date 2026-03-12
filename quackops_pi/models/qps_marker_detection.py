"""
qps_marker_detection.py

Holds a single ArUco detection result.
Produced by: qpsMarkerDetector
Consumed by: qpsLandingController
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class qpsMarkerDetection:
    """Immutable result of detecting one ArUco marker in a frame."""

    marker_id: int
    corners: np.ndarray  # shape (4, 2)
    center_px: Tuple[float, float]
    confidence: float

    # Populated after estimate_pose() call
    tvec: Optional[Tuple[float, float, float]] = None
    rvec: Optional[Tuple[float, float, float]] = None
    distance_m: Optional[float] = None