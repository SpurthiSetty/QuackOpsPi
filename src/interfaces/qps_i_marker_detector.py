from abc import ABC, abstractmethod

import numpy

from src.models.qps_marker_detection import qpsMarkerDetection


class qpsIMarkerDetector(ABC):
    """Abstract interface for ArUco marker detection and pose estimation."""

    @abstractmethod
    def detect(self, frame: numpy.ndarray) -> list[qpsMarkerDetection]:
        """Detect all ArUco markers visible in a camera frame.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            list[qpsMarkerDetection]: All markers detected in the frame.
        """
        ...

    @abstractmethod
    def estimate_pose(self, detection: qpsMarkerDetection) -> qpsMarkerDetection:
        """Estimate the 3-D pose of a detected marker relative to the camera.

        Args:
            detection: A marker detection whose pose fields will be populated.

        Returns:
            qpsMarkerDetection: The same detection with updated rotation_vec,
                translation_vec, and distance_m fields.
        """
        ...

    @abstractmethod
    def get_confidence(self, detection: qpsMarkerDetection) -> float:
        """Compute a confidence score for a detection.

        Args:
            detection: The marker detection to evaluate.

        Returns:
            float: A confidence value between 0.0 and 1.0.
        """
        ...
