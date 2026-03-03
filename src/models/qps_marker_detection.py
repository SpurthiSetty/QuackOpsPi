from dataclasses import dataclass

import numpy


@dataclass
class qpsMarkerDetection:
    """Represents a single detected ArUco marker with pose estimation data."""

    marker_id: int
    corners: numpy.ndarray
    center: tuple[float, float]
    distance_m: float
    rotation_vec: numpy.ndarray
    translation_vec: numpy.ndarray
    confidence: float
