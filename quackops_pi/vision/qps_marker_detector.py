"""
qps_marker_detector.py

Production ArUco marker detector using OpenCV's aruco module.

detect() offloads the synchronous cv2 work to a thread via asyncio.to_thread()
so the event loop is never blocked.  estimate_pose() is a no-op stub for the
demo — only "was marker seen" matters for the SITL hover-search flow.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

import cv2
import numpy as np

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_marker_detector_interface import qpsMarkerDetectorInterface
from quackops_pi.models.qps_marker_detection import qpsMarkerDetection

logger = logging.getLogger("qps.marker_detector")


class qpsMarkerDetector(qpsMarkerDetectorInterface):
    """Detects ArUco markers in BGR frames using cv2.aruco.detectMarkers.

    Only detect() is fully implemented.  estimate_pose() returns the detection
    unchanged — pose estimation is not needed for the hover-search demo because
    qpsHoverSearchController uses the drone's current GPS position rather than
    projecting the marker position from a tvec.
    """

    def __init__(self, config: qpsConfig) -> None:
        """Initialise the detector with the configured ArUco dictionary.

        Args:
            config: Application configuration.  config.aruco_dictionary must be
                    a valid cv2.aruco constant name (e.g. "DICT_4X4_50").
        """
        self._config = config
        dict_id = getattr(cv2.aruco, config.aruco_dictionary)
        self._dictionary: cv2.aruco.Dictionary = (
            cv2.aruco.getPredefinedDictionary(dict_id)
        )
        self._detector_params: cv2.aruco.DetectorParameters = (
            cv2.aruco.DetectorParameters()
        )
        # OpenCV 4.7+ uses ArucoDetector instead of the standalone detectMarkers()
        self._detector: cv2.aruco.ArucoDetector = cv2.aruco.ArucoDetector(
            self._dictionary, self._detector_params
        )

    # ── qpsMarkerDetectorInterface ────────────────────────────────────

    async def detect(self, frame: np.ndarray) -> List[qpsMarkerDetection]:
        """Detect all ArUco markers in a BGR frame.

        Offloads cv2 work to a thread pool so the event loop is not blocked.

        Args:
            frame: BGR image as a numpy ndarray (e.g. from qpsCVCameraManager).

        Returns:
            List of qpsMarkerDetection, one per detected marker.  Empty if none.
        """
        return await asyncio.to_thread(self._detect_sync, frame)

    async def estimate_pose(
        self, detection: qpsMarkerDetection
    ) -> qpsMarkerDetection:
        """Return the detection unchanged.

        Full solvePnP pose estimation is not required for the SITL demo.
        qpsHoverSearchController reports the drone's current GPS as the landing
        position, so tvec/rvec are not needed.

        Args:
            detection: A previously detected marker.

        Returns:
            The same detection object, unmodified.
        """
        return detection

    # ── Synchronous detection ─────────────────────────────────────────

    def _detect_sync(self, frame: np.ndarray) -> List[qpsMarkerDetection]:
        """Run cv2.aruco.detectMarkers on a grayscale-converted frame.

        Called via asyncio.to_thread() from detect().
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            return []

        results: List[qpsMarkerDetection] = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            # marker_corners shape: (1, 4, 2) — squeeze to (4, 2)
            pts: np.ndarray = marker_corners[0]
            center_px = (
                float(np.mean(pts[:, 0])),
                float(np.mean(pts[:, 1])),
            )
            results.append(
                qpsMarkerDetection(
                    marker_id=int(marker_id),
                    corners=pts,
                    center_px=center_px,
                    confidence=1.0,
                )
            )
            logger.debug(
                "Detected marker ID=%d  center=(%.1f, %.1f)",
                marker_id, center_px[0], center_px[1],
            )

        if results:
            logger.info(
                "Detected %d marker(s): IDs=%s",
                len(results), [d.marker_id for d in results],
            )

        return results
