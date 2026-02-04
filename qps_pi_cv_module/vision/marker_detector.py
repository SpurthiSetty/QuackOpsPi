"""
ArUco Marker Detector implementation using OpenCV.

This module provides real-time detection and pose estimation
of ArUco markers for precision landing operations.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..core.interfaces import IMarkerDetector, MarkerDetection
from ..core.config import VisionConfig

logger = logging.getLogger(__name__)


# Mapping of string names to OpenCV ArUco dictionary constants
ARUCO_DICT_MAP: Dict[str, int] = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_5X5_1000": aruco.DICT_5X5_1000,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_6X6_1000": aruco.DICT_6X6_1000,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,
}


class MarkerDetector(IMarkerDetector):
    """
    ArUco marker detector using OpenCV's aruco module.
    
    This class provides methods to detect ArUco markers in video frames
    and estimate their 3D pose relative to the camera. It is designed
    for use in precision landing scenarios where the drone needs to
    locate and descend onto a marked landing pad.
    
    Attributes:
        config: Vision configuration settings
        aruco_dict: The ArUco dictionary being used
        detector: The ArUco detector instance
        
    Example:
        >>> detector = MarkerDetector(config)
        >>> detections = detector.detect(frame)
        >>> for d in detections:
        ...     print(f"Marker {d.marker_id} at ({d.center_x}, {d.center_y})")
    """
    
    def __init__(self, config: VisionConfig):
        """
        Initialize the marker detector.
        
        Args:
            config: Vision configuration settings
        """
        self.config = config
        self._initialize_detector()
        
        # Performance tracking
        self._detection_count = 0
        self._total_detection_time = 0.0
        
        logger.info(
            f"MarkerDetector initialized with dictionary: {config.aruco_dictionary}"
        )
    
    def _initialize_detector(self) -> None:
        """Initialize the ArUco detector with configured dictionary."""
        dict_id = ARUCO_DICT_MAP.get(self.config.aruco_dictionary)
        if dict_id is None:
            raise ValueError(
                f"Unknown ArUco dictionary: {self.config.aruco_dictionary}. "
                f"Valid options: {list(ARUCO_DICT_MAP.keys())}"
            )
        
        self.aruco_dict = aruco.getPredefinedDictionary(dict_id)
        
        # Configure detector parameters
        self.detector_params = aruco.DetectorParameters()
        
        # Adjust parameters for better detection in various conditions
        self.detector_params.adaptiveThreshConstant = 7
        self.detector_params.minMarkerPerimeterRate = 0.03
        self.detector_params.maxMarkerPerimeterRate = 4.0
        self.detector_params.polygonalApproxAccuracyRate = 0.03
        self.detector_params.minCornerDistanceRate = 0.05
        
        # Create the detector (OpenCV 4.7+ API)
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.detector_params)
    
    def detect(self, frame: np.ndarray) -> List[MarkerDetection]:
        """
        Detect ArUco markers in the given frame.
        
        Args:
            frame: BGR image as numpy array (HxWx3)
            
        Returns:
            List of MarkerDetection objects for each detected marker
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided for detection")
            return []
        
        start_time = time.time()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        detection_time = time.time() - start_time
        self._detection_count += 1
        self._total_detection_time += detection_time
        
        if ids is None:
            logger.debug(f"No markers detected (took {detection_time*1000:.1f}ms)")
            return []
        
        # Build detection objects
        detections = []
        timestamp = time.time()
        
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0]  # Shape: (4, 2)
            
            # Calculate center point
            center_x = float(np.mean(marker_corners[:, 0]))
            center_y = float(np.mean(marker_corners[:, 1]))
            
            # Calculate marker area (using Shoelace formula)
            area = self._calculate_polygon_area(marker_corners)
            
            # Calculate confidence based on area and shape
            confidence = self._calculate_confidence(marker_corners, area)
            
            detection = MarkerDetection(
                marker_id=int(marker_id),
                corners=marker_corners,
                center_x=center_x,
                center_y=center_y,
                area=area,
                confidence=confidence,
                timestamp=timestamp,
            )
            
            detections.append(detection)
        
        logger.debug(
            f"Detected {len(detections)} markers in {detection_time*1000:.1f}ms"
        )
        
        return detections
    
    def estimate_pose(
        self,
        detection: MarkerDetection,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> MarkerDetection:
        """
        Estimate the 3D pose of a detected marker.
        
        Uses solvePnP to estimate the rotation and translation vectors
        that describe the marker's position relative to the camera.
        
        Args:
            detection: The marker detection to estimate pose for
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            
        Returns:
            Updated detection with pose information (rvec, tvec, distance)
        """
        if not self.config.pose_estimation_enabled:
            return detection
        
        # Define 3D object points for the marker (in marker coordinate frame)
        # Marker is assumed to be centered at origin, lying in XY plane
        half_size = self.config.marker_size_cm / 2.0
        obj_points = np.array([
            [-half_size, half_size, 0],   # Top-left
            [half_size, half_size, 0],    # Top-right
            [half_size, -half_size, 0],   # Bottom-right
            [-half_size, -half_size, 0],  # Bottom-left
        ], dtype=np.float32)
        
        # Get image points from detection
        img_points = detection.corners.astype(np.float32)
        
        # Solve PnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if not success:
            logger.warning(f"Pose estimation failed for marker {detection.marker_id}")
            return detection
        
        # Calculate distance from camera to marker center
        distance_cm = float(np.linalg.norm(tvec))
        
        # Update detection with pose data
        detection.rotation_vector = rvec
        detection.translation_vector = tvec
        detection.distance_cm = distance_cm
        
        logger.debug(
            f"Marker {detection.marker_id}: distance={distance_cm:.1f}cm, "
            f"tvec=({tvec[0][0]:.1f}, {tvec[1][0]:.1f}, {tvec[2][0]:.1f})"
        )
        
        return detection
    
    def detect_with_pose(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> List[MarkerDetection]:
        """
        Convenience method to detect markers and estimate poses in one call.
        
        Args:
            frame: BGR image
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            
        Returns:
            List of detections with pose information
        """
        detections = self.detect(frame)
        
        for i, detection in enumerate(detections):
            detections[i] = self.estimate_pose(detection, camera_matrix, dist_coeffs)
        
        return detections
    
    def find_target_marker(
        self,
        detections: List[MarkerDetection],
        target_id: int
    ) -> Optional[MarkerDetection]:
        """
        Find a specific marker by ID from a list of detections.
        
        Args:
            detections: List of marker detections
            target_id: The marker ID to find
            
        Returns:
            The matching detection, or None if not found
        """
        for detection in detections:
            if detection.marker_id == target_id:
                return detection
        return None
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[MarkerDetection],
        draw_pose: bool = True,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw detected markers and their information on a frame.
        
        Args:
            frame: BGR image to draw on
            detections: List of marker detections
            draw_pose: Whether to draw pose axes
            camera_matrix: Required if draw_pose is True
            dist_coeffs: Required if draw_pose is True
            
        Returns:
            Frame with drawings
        """
        output = frame.copy()
        
        for detection in detections:
            # Draw marker outline
            corners = detection.corners.astype(np.int32)
            cv2.polylines(output, [corners], True, (0, 255, 0), 2)
            
            # Draw center point
            center = (int(detection.center_x), int(detection.center_y))
            cv2.circle(output, center, 5, (0, 0, 255), -1)
            
            # Draw marker ID
            cv2.putText(
                output,
                f"ID: {detection.marker_id}",
                (corners[0][0], corners[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Draw distance if available
            if detection.distance_cm is not None:
                cv2.putText(
                    output,
                    f"{detection.distance_cm:.0f}cm",
                    (corners[0][0], corners[0][1] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )
            
            # Draw pose axes
            if (draw_pose and detection.is_pose_available 
                    and camera_matrix is not None and dist_coeffs is not None):
                axis_length = self.config.marker_size_cm / 2
                cv2.drawFrameAxes(
                    output,
                    camera_matrix,
                    dist_coeffs,
                    detection.rotation_vector,
                    detection.translation_vector,
                    axis_length
                )
        
        return output
    
    def _calculate_polygon_area(self, corners: np.ndarray) -> float:
        """Calculate area using the Shoelace formula."""
        n = len(corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        return abs(area) / 2.0
    
    def _calculate_confidence(self, corners: np.ndarray, area: float) -> float:
        """
        Calculate detection confidence based on marker properties.
        
        Factors considered:
        - Area (larger is more confident)
        - Shape regularity (square-ness)
        """
        # Check if area is within expected range
        if area < 100:  # Too small
            return 0.3
        
        # Calculate shape regularity (compare side lengths)
        side_lengths = []
        for i in range(4):
            j = (i + 1) % 4
            length = np.linalg.norm(corners[j] - corners[i])
            side_lengths.append(length)
        
        mean_side = np.mean(side_lengths)
        side_variance = np.var(side_lengths) / (mean_side ** 2) if mean_side > 0 else 1.0
        
        # Lower variance = more square-like = higher confidence
        shape_confidence = max(0.0, 1.0 - side_variance * 10)
        
        # Area confidence (saturates at certain size)
        area_confidence = min(1.0, area / 10000)
        
        return 0.5 * shape_confidence + 0.5 * area_confidence
    
    @property
    def average_detection_time_ms(self) -> float:
        """Get average detection time in milliseconds."""
        if self._detection_count == 0:
            return 0.0
        return (self._total_detection_time / self._detection_count) * 1000
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._detection_count = 0
        self._total_detection_time = 0.0
