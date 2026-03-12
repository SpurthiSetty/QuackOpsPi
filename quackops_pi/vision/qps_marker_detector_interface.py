"""
qps_marker_detector_interface.py

Defines the contract for detecting and localizing ArUco markers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from quackops_pi.models import qpsMarkerDetection


class qpsMarkerDetectorInterface(ABC):
    """Contract for ArUco marker detection and pose estimation."""

    @abstractmethod
    async def detect(self, frame: np.ndarray) -> List[qpsMarkerDetection]:
        ...

    @abstractmethod
    async def estimate_pose(self, detection: qpsMarkerDetection) -> qpsMarkerDetection:
        ...