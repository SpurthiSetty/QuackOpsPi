"""
qps_hover_search_controller.py

Simplified landing controller for the SITL delivery demo.

Replaces the full orbit-based qpsLandingController for demos where the drone
hovers in place (GUIDED hold) and the tester holds up a printed ArUco marker
in front of the laptop webcam.

Returns the same qpsLandingResult type as qpsLandingController so
qpsMissionControllerImpl can accept either interchangeably via duck typing.

Key difference from qpsLandingController:
    - No orbit waypoint upload — drone is already hovering when search starts.
    - marker_gps in the result = drone's current GPS (not a projected position).
      Good enough for demo: we know we're already over the delivery location.
"""

from __future__ import annotations

import asyncio
import logging
import time

import cv2
import numpy as np

from quackops_pi.vision.qps_camera_manager_interface import qpsCameraManagerInterface
from quackops_pi.vision.qps_marker_detector_interface import qpsMarkerDetectorInterface
from quackops_pi.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor
from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.models.qps_landing_result import qpsLandingResult, qpsLandingOutcome

logger = logging.getLogger("qps.hover_search_controller")


class qpsHoverSearchController:
    """Hover-in-place ArUco marker search for the delivery demo.

    Lifecycle:
        1. MissionController calls execute_marker_search(target_marker_id).
        2. Camera is started (if manage_camera_lifecycle=True); frame-detect
           loop begins.
        3. Marker found  → camera stopped, returns MARKER_FOUND result.
        4. Timeout       → camera stopped, returns SEARCH_TIMEOUT result.
        5. Camera error  → returns CAMERA_FAILURE result immediately.
        6. abort()       → next iteration exits, returns ABORTED result.

    When manage_camera_lifecycle=False, the caller is responsible for
    camera start/stop. The controller becomes a pure frame consumer — it
    reads frames from whatever the camera is already producing and never
    calls camera.start() or camera.stop(). Use this when the camera is
    owned by a recording stack (e.g. qpsStreamServer) that must keep
    running through the landing phase.
    """

    def __init__(
        self,
        camera_manager: qpsCameraManagerInterface,
        marker_detector: qpsMarkerDetectorInterface,
        telemetry_monitor: qpsTelemetryMonitor,
        config: qpsConfig,
        manage_camera_lifecycle: bool = True,
    ) -> None:
        self._camera = camera_manager
        self._detector = marker_detector
        self._telemetry = telemetry_monitor
        self._config = config
        self._manage_camera_lifecycle = manage_camera_lifecycle

        self._abort_requested: bool = False
        self._searching: bool = False

    # ── Public API ────────────────────────────────────────────────────

    async def execute_marker_search(self, target_marker_id: int) -> qpsLandingResult:
        """Run the hover marker search loop.

        Starts the camera, searches frames for target_marker_id, and returns
        a qpsLandingResult describing what was found (or why search ended).

        Args:
            target_marker_id: ArUco marker ID to search for.

        Returns:
            qpsLandingResult with outcome MARKER_FOUND, SEARCH_TIMEOUT,
            CAMERA_FAILURE, or ABORTED.
        """
        self._abort_requested = False
        self._searching = True
        frames_searched = 0
        start_time = time.monotonic()

        logger.info(
            "Hover search started — marker ID=%d  timeout=%.1fs",
            target_marker_id,
            self._config.search_timeout_s,
        )

        # Start camera (only if this controller owns the lifecycle)
        if self._manage_camera_lifecycle:
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
            if self._manage_camera_lifecycle:
                await self._safe_stop_camera()
            else:
                cv2.destroyAllWindows()
            self._searching = False

    def abort(self) -> None:
        """Signal the search loop to exit at the next iteration."""
        if self._searching:
            logger.warning("Hover search abort requested")
            self._abort_requested = True

    @property
    def is_searching(self) -> bool:
        """True while execute_marker_search is running."""
        return self._searching

    # ── Search loop ───────────────────────────────────────────────────

    async def _search_loop(
        self,
        target_marker_id: int,
        start_time: float,
        frames_searched: int,
    ) -> qpsLandingResult:
        while True:
            elapsed = time.monotonic() - start_time

            # ── Abort ─────────────────────────────────────────────
            if self._abort_requested:
                logger.info(
                    "Search aborted after %.1fs (%d frames)",
                    elapsed, frames_searched,
                )
                return qpsLandingResult(
                    outcome=qpsLandingOutcome.ABORTED,
                    fallback_gps=self._telemetry.get_gps_position(),
                    search_duration_s=elapsed,
                    frames_searched=frames_searched,
                    target_marker_id=target_marker_id,
                )

            # ── Timeout ───────────────────────────────────────────
            if elapsed >= self._config.search_timeout_s:
                logger.warning(
                    "Search timeout after %.1fs (%d frames)",
                    elapsed, frames_searched,
                )
                return qpsLandingResult(
                    outcome=qpsLandingOutcome.SEARCH_TIMEOUT,
                    fallback_gps=self._telemetry.get_gps_position(),
                    search_duration_s=elapsed,
                    frames_searched=frames_searched,
                    target_marker_id=target_marker_id,
                )

            # ── Get frame ─────────────────────────────────────────
            frame = await self._camera.get_frame()
            if frame is None:
                # Camera not yet producing frames — yield and retry
                await asyncio.sleep(0.05)
                continue

            frames_searched += 1

            # ── Detect markers ────────────────────────────────────
            detections = await self._detector.detect(frame)
            target = next(
                (d for d in detections if d.marker_id == target_marker_id),
                None,
            )

            # ── OpenCV preview window ─────────────────────────────
            self._show_preview(frame, detections, target_marker_id, elapsed, frames_searched)

            if target is not None:
                logger.info(
                    "Marker ID=%d detected! (frame=%d  elapsed=%.1fs)",
                    target_marker_id, frames_searched, elapsed,
                )
                # Use current drone GPS as landing position.
                # We are already hovering over the delivery zone, so the
                # drone position is a good enough proxy for marker position.
                return qpsLandingResult(
                    outcome=qpsLandingOutcome.MARKER_FOUND,
                    marker_gps=self._telemetry.get_gps_position(),
                    search_duration_s=elapsed,
                    frames_searched=frames_searched,
                    target_marker_id=target_marker_id,
                )

            # Yield to the event loop between frames
            await asyncio.sleep(0)

    # ── Helpers ───────────────────────────────────────────────────────

    # ── OpenCV preview ────────────────────────────────────────────────

    def _show_preview(
        self,
        frame: np.ndarray,
        detections: list,
        target_marker_id: int,
        elapsed: float,
        frames_searched: int,
    ) -> None:
        """Draw detection overlays and show the frame in an OpenCV window.

        Draws all detected markers (even non-target ones) so the tester can
        see the camera is working.  Adds a status line at the top of the frame.
        Calls cv2.waitKey(1) to pump the window event loop.
        """
        annotated = frame.copy()

        # Draw any detected markers
        if detections:
            corners = [np.array([d.corners], dtype=np.float32) for d in detections]
            ids = np.array([[d.marker_id] for d in detections], dtype=np.int32)
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        # Status text
        target_found = any(d.marker_id == target_marker_id for d in detections)
        if target_found:
            status_text = f"MARKER FOUND!  ID={target_marker_id}"
            colour = (0, 255, 0)    # green
        else:
            status_text = f"SEARCHING FOR MARKER ID={target_marker_id}"
            colour = (0, 200, 255)  # amber

        cv2.putText(
            annotated, status_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Elapsed: {elapsed:.1f}s   Frames: {frames_searched}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

        cv2.imshow("QuackOps Marker Search", annotated)
        cv2.waitKey(1)

    async def _safe_stop_camera(self) -> None:
        try:
            await self._camera.stop()
        except Exception:
            logger.exception("Error stopping camera during cleanup")
        cv2.destroyAllWindows()
