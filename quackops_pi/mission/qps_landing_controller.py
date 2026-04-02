"""
qps_landing_controller.py

Executes the marker search phase during the orbit mission.
Returns a qpsLandingResult to qpsMissionController — does NOT
command landing itself.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Optional

from quackops_pi.vision import qpsCameraManagerInterface, qpsMarkerDetectorInterface
from quackops_pi.config import qpsConfig
from quackops_pi.models import (
    qpsGPSPosition,
    qpsLandingOutcome,
    qpsLandingResult,
    qpsMarkerDetection,
)
from quackops_pi.telemetry import qpsTelemetryMonitor

logger = logging.getLogger("qps.landing_controller")

_EARTH_RADIUS_M = 6_371_000.0


class qpsLandingController:
    """Runs the visual marker search during an orbit mission.

    Lifecycle:
        1. Mission controller calls execute_marker_search(target_marker_id).
        2. Starts camera, enters frame-detect loop.
        3. On detection → computes marker GPS, stops camera, returns success.
        4. On timeout  → captures fallback GPS, stops camera, returns timeout.
        5. On abort    → stops camera immediately, returns aborted.
    """

    def __init__(
        self,
        camera_manager: qpsCameraManagerInterface,
        marker_detector: qpsMarkerDetectorInterface,
        telemetry_monitor: qpsTelemetryMonitor,
        config: qpsConfig,
    ) -> None:
        self._camera = camera_manager
        self._detector = marker_detector
        self._telemetry = telemetry_monitor
        self._config = config

        self._abort_requested: bool = False
        self._searching: bool = False

    # ── Public API ────────────────────────────────────────────────────

    async def execute_marker_search(self, target_marker_id: int) -> qpsLandingResult:
        """Run the marker search loop."""
        self._abort_requested = False
        self._searching = True
        frames_searched = 0
        start_time = time.monotonic()

        logger.info(
            "Starting marker search for ID=%d (timeout=%.1fs)",
            target_marker_id,
            self._config.search_timeout_s,
        )

        try:
            await self._camera.start()
        except Exception:
            logger.exception("Camera failed to start")
            self._searching = False
            return qpsLandingResult(
                outcome=qpsLandingOutcome.CAMERA_FAILURE,
                search_duration_s=0.0,
                frames_searched=0,
                target_marker_id=target_marker_id,
            )

        try:
            return await self._search_loop(target_marker_id, start_time, frames_searched)
        finally:
            await self._safe_stop_camera()
            self._searching = False

    def abort(self) -> None:
        """Signal the search loop to stop at the next iteration."""
        if self._searching:
            logger.warning("Abort requested")
            self._abort_requested = True

    @property
    def is_searching(self) -> bool:
        return self._searching

    # ── Core search loop ──────────────────────────────────────────────

    async def _search_loop(
        self,
        target_marker_id: int,
        start_time: float,
        frames_searched: int,
    ) -> qpsLandingResult:
        while True:
            elapsed = time.monotonic() - start_time

            # Check abort
            if self._abort_requested:
                logger.info("Search aborted after %.1fs (%d frames)", elapsed, frames_searched)
                return qpsLandingResult(
                    outcome=qpsLandingOutcome.ABORTED,
                    fallback_gps=self._telemetry.get_gps_position(),
                    search_duration_s=elapsed,
                    frames_searched=frames_searched,
                    target_marker_id=target_marker_id,
                )

            # Check timeout
            if elapsed >= self._config.search_timeout_s:
                logger.warning("Search timeout after %.1fs (%d frames)", elapsed, frames_searched)
                return qpsLandingResult(
                    outcome=qpsLandingOutcome.SEARCH_TIMEOUT,
                    fallback_gps=self._telemetry.get_gps_position(),
                    search_duration_s=elapsed,
                    frames_searched=frames_searched,
                    target_marker_id=target_marker_id,
                )

            # Capture frame
            frame = await self._camera.get_frame()
            if frame is None:
                logger.error("Camera returned None frame")
                return qpsLandingResult(
                    outcome=qpsLandingOutcome.CAMERA_FAILURE,
                    fallback_gps=self._telemetry.get_gps_position(),
                    search_duration_s=elapsed,
                    frames_searched=frames_searched,
                    target_marker_id=target_marker_id,
                )
            frames_searched += 1

            # Detect markers
            detections = await self._detector.detect(frame)
            target = self._find_target(detections, target_marker_id)

            if target is None:
                await asyncio.sleep(0)
                continue

            # Marker found — estimate pose and compute GPS
            logger.info(
                "Marker ID=%d detected (confidence=%.2f) at frame %d",
                target_marker_id, target.confidence, frames_searched,
            )

            pose_detection = await self._detector.estimate_pose(target)
            drone_state = self._telemetry.get_drone_state()
            marker_gps = self._compute_marker_gps(pose_detection, drone_state)

            if marker_gps is None:
                logger.warning("GPS computation failed — continuing search")
                await asyncio.sleep(0)
                continue

            logger.info(
                "Marker GPS computed: lat=%.7f, lon=%.7f (%.1fs, %d frames)",
                marker_gps.latitude_deg, marker_gps.longitude_deg,
                time.monotonic() - start_time, frames_searched,
            )

            return qpsLandingResult(
                outcome=qpsLandingOutcome.MARKER_FOUND,
                marker_gps=marker_gps,
                search_duration_s=time.monotonic() - start_time,
                frames_searched=frames_searched,
                target_marker_id=target_marker_id,
            )

    # ── Marker GPS computation ────────────────────────────────────────

    def _compute_marker_gps(
        self,
        detection: qpsMarkerDetection,
        drone_state,  # qpsDroneState
    ) -> Optional[qpsGPSPosition]:
        """Project marker pose from camera frame into world GPS.

        Camera mounted pointing straight down:
            camera_x → body_east,  camera_-y → body_north
        Rotate by drone heading to get NED, then convert to lat/lon.
        """
        if detection.tvec is None or detection.distance_m is None:
            return None

        gps = drone_state.gps_position if drone_state else None
        if gps is None:
            return None

        tx, ty, tz = detection.tvec
        body_east = tx
        body_north = -ty

        heading_rad = math.radians(gps.heading_deg)
        ned_north = body_north * math.cos(heading_rad) - body_east * math.sin(heading_rad)
        ned_east = body_north * math.sin(heading_rad) + body_east * math.cos(heading_rad)

        delta_lat = ned_north / _EARTH_RADIUS_M * (180.0 / math.pi)
        delta_lon = ned_east / (
            _EARTH_RADIUS_M * math.cos(math.radians(gps.latitude_deg))
        ) * (180.0 / math.pi)

        return qpsGPSPosition(
            latitude_deg=gps.latitude_deg + delta_lat,
            longitude_deg=gps.longitude_deg + delta_lon,
            altitude_m=0.0,
            heading_deg=gps.heading_deg,
            speed_m_s=0.0,
            timestamp=gps.timestamp,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _find_target(
        detections: list[qpsMarkerDetection],
        target_id: int,
    ) -> Optional[qpsMarkerDetection]:
        matches = [d for d in detections if d.marker_id == target_id]
        if not matches:
            return None
        return max(matches, key=lambda d: d.confidence)

    async def _safe_stop_camera(self) -> None:
        try:
            await self._camera.stop()
        except Exception:
            logger.exception("Error stopping camera during cleanup")