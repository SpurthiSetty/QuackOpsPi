import numpy

from src.config.qps_config import qpsConfig
from src.interfaces.qps_i_marker_detector import qpsIMarkerDetector
from src.models.qps_marker_detection import qpsMarkerDetection


class qpsMockMarkerDetector(qpsIMarkerDetector):
    """Mock marker detector that returns scripted detection results.

    Supports per-call scripted sequences or a constant detection for
    every call.
    """

    def __init__(self, config: qpsConfig) -> None:
        """Initialise the mock marker detector.

        Args:
            config: Application configuration (stored but largely unused).
        """
        self.config: qpsConfig = config
        self.scripted_detections: list[list[qpsMarkerDetection]] = []
        self.call_index: int = 0
        self._constant_detection: qpsMarkerDetection | None = None

    def detect(self, frame: numpy.ndarray) -> list[qpsMarkerDetection]:
        """Return the next scripted detection result.

        If a constant detection is set, it is returned every time.
        If scripted detections are provided, they are returned in order.
        Otherwise, an empty list is returned.

        Args:
            frame: Input frame (ignored).

        Returns:
            list[qpsMarkerDetection]: Scripted detection results.
        """
        if self._constant_detection is not None:
            return [self._constant_detection]

        if self.call_index < len(self.scripted_detections):
            result = self.scripted_detections[self.call_index]
            self.call_index += 1
            return result

        return []

    def estimate_pose(self, detection: qpsMarkerDetection) -> qpsMarkerDetection:
        """Return the detection unchanged (pose data is pre-populated).

        Args:
            detection: The detection to return.

        Returns:
            qpsMarkerDetection: Same detection, unmodified.
        """
        return detection

    def get_confidence(self, detection: qpsMarkerDetection) -> float:
        """Return the confidence value already stored on the detection.

        Args:
            detection: The detection to evaluate.

        Returns:
            float: detection.confidence.
        """
        return detection.confidence

    # ------------------------------------------------------------------
    # Test-helper methods
    # ------------------------------------------------------------------

    def set_scripted_detections(
        self, detections: list[list[qpsMarkerDetection]]
    ) -> None:
        """Set a per-call sequence of detection results.

        Args:
            detections: List of detection lists, one per detect() call.
        """
        self.scripted_detections = list(detections)
        self.call_index = 0
        self._constant_detection = None

    def set_constant_detection(self, detection: qpsMarkerDetection) -> None:
        """Set a single detection to return on every detect() call.

        Args:
            detection: The detection to return repeatedly.
        """
        self._constant_detection = detection
        self.scripted_detections = []
        self.call_index = 0

    def set_no_detection(self) -> None:
        """Configure the detector to return no detections."""
        self._constant_detection = None
        self.scripted_detections = []
        self.call_index = 0
