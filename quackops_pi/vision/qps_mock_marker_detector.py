from __future__ import annotations

from typing import List, Optional

import numpy

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_marker_detector_interface import qpsMarkerDetectorInterface
from quackops_pi.models.qps_marker_detection import qpsMarkerDetection


class qpsMockMarkerDetector(qpsMarkerDetectorInterface):
    """Mock marker detector that returns scripted detection results.

    Supports per-call scripted sequences or a constant detection for
    every call.
    """

    def __init__(self, config: qpsConfig) -> None:
        self.config: qpsConfig = config
        self.scripted_detections: list[list[qpsMarkerDetection]] = []
        self.call_index: int = 0
        self._constant_detection: Optional[qpsMarkerDetection] = None

    async def detect(self, frame: numpy.ndarray) -> List[qpsMarkerDetection]:
        if self._constant_detection is not None:
            return [self._constant_detection]
        if self.call_index < len(self.scripted_detections):
            result = self.scripted_detections[self.call_index]
            self.call_index += 1
            return result
        return []

    async def estimate_pose(self, detection: qpsMarkerDetection) -> qpsMarkerDetection:
        return detection

    # ── Test-helper methods ───────────────────────────────────────────────────

    def set_scripted_detections(
        self, detections: list[list[qpsMarkerDetection]]
    ) -> None:
        self.scripted_detections = list(detections)
        self.call_index = 0
        self._constant_detection = None

    def set_constant_detection(self, detection: qpsMarkerDetection) -> None:
        self._constant_detection = detection
        self.scripted_detections = []
        self.call_index = 0

    def set_no_detection(self) -> None:
        self._constant_detection = None
        self.scripted_detections = []
        self.call_index = 0
