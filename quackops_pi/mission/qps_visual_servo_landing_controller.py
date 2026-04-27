"""
qps_visual_servo_landing_controller.py

Landing controller that uses proportional control to center the drone over
a detected marker, then commands land once centered for N consecutive frames.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager_interface import qpsFlightManagerInterface
from quackops_pi.mission.qps_landing_controller_interface import qpsLandingControllerInterface
from quackops_pi.models.qps_landing_result import qpsLandingOutcome, qpsLandingResult
from quackops_pi.vision.qps_camera_manager_interface import qpsCameraManagerInterface
from quackops_pi.vision.qps_marker_detector_interface import qpsMarkerDetectorInterface

logger = logging.getLogger("qps.servo_landing")


class qpsVisualServoLandingController(qpsLandingControllerInterface):
    """Proportional-control landing controller.

    Computes the pixel offset of the detected marker from frame center and
    sends NED velocity corrections to drive the offset toward zero. Once the
    marker has been within center_tolerance_px for N consecutive frames,
    commands land.

    Pitch channel: nose-down = forward = north (negative pitch offset).
    Roll channel:  right = east (positive roll offset).
    Camera Y-axis is inverted relative to NED north.
    """

    def __init__(
        self,
        camera_manager: qpsCameraManagerInterface,
        marker_detector: qpsMarkerDetectorInterface,
        flight_manager: qpsFlightManagerInterface,
        config: qpsConfig,
    ) -> None:
        self._camera = camera_manager
        self._detector = marker_detector
        self._flight = flight_manager
        self._config = config
        self._abort_requested: bool = False
        self._active: bool = False

    # ── Interface implementation ──────────────────────────────────────

    async def execute_landing(self, target_marker_id: int) -> qpsLandingResult:
        """Center over marker using proportional control, then command land.

        Loops until the marker is centered for N consecutive frames, timeout,
        camera failure, or abort. On every frame without centering, sends a
        velocity correction. On abort/timeout, sends hover setpoint first to
        stop any in-progress drift.
        """
        self._active = True
        self._abort_requested = False
        consecutive_centered = 0
        frames_searched = 0
        start_time = time.monotonic()

        # Obtain frame dimensions from first valid frame
        frame_width: Optional[float] = None
        frame_height: Optional[float] = None

        try:
            while True:
                # ── Abort check ───────────────────────────────────
                if self._abort_requested:
                    await self._flight.send_hover_setpoint()
                    logger.info("Landing aborted after %d frames", frames_searched)
                    return qpsLandingResult(
                        outcome=qpsLandingOutcome.ABORTED,
                        search_duration_s=time.monotonic() - start_time,
                        frames_searched=frames_searched,
                        target_marker_id=target_marker_id,
                    )

                # ── Timeout check ─────────────────────────────────
                elapsed = time.monotonic() - start_time
                if elapsed > self._config.search_timeout_s:
                    await self._flight.send_hover_setpoint()
                    logger.warning(
                        "Landing search timed out after %.1fs (%d frames)",
                        elapsed, frames_searched,
                    )
                    return qpsLandingResult(
                        outcome=qpsLandingOutcome.SEARCH_TIMEOUT,
                        search_duration_s=elapsed,
                        frames_searched=frames_searched,
                        target_marker_id=target_marker_id,
                    )

                # ── Grab frame ────────────────────────────────────
                frame = await self._grab_frame()
                if frame is None:
                    logger.error("Camera returned None frame — CAMERA_FAILURE")
                    return qpsLandingResult(
                        outcome=qpsLandingOutcome.CAMERA_FAILURE,
                        search_duration_s=time.monotonic() - start_time,
                        frames_searched=frames_searched,
                        target_marker_id=target_marker_id,
                    )

                frames_searched += 1

                # Latch frame dimensions on first valid frame
                if frame_width is None:
                    frame_height, frame_width = frame.shape[:2]

                # ── Detect ────────────────────────────────────────
                detections = await self._detector.detect(frame)
                target = self._find_target(detections, target_marker_id)

                if target is not None:
                    offset_x = target.center_px[0] - (frame_width / 2)
                    offset_y = target.center_px[1] - (frame_height / 2)
                    tol = self._config.center_tolerance_px

                    if abs(offset_x) <= tol and abs(offset_y) <= tol:
                        # Within tolerance — count toward lock
                        consecutive_centered += 1
                        await self._flight.send_hover_setpoint()
                        logger.info(
                            "Marker centered (offset=%.0f,%.0f) — locked=%d/%d",
                            offset_x, offset_y,
                            consecutive_centered, self._config.lock_frame_count,
                        )
                        if consecutive_centered >= self._config.lock_frame_count:
                            logger.info("Centered and locked — commanding land")
                            await self._flight.land()
                            return qpsLandingResult(
                                outcome=qpsLandingOutcome.MARKER_FOUND,
                                search_duration_s=time.monotonic() - start_time,
                                frames_searched=frames_searched,
                                target_marker_id=target_marker_id,
                            )
                    else:
                        # Outside tolerance — send proportional correction
                        consecutive_centered = 0
                        Kp = self._config.proportional_gain
                        max_vel = self._config.max_correction_velocity

                        vel_east = max(-max_vel, min(max_vel, Kp * offset_x))
                        vel_north = -max(-max_vel, min(max_vel, Kp * offset_y))

                        logger.debug(
                            "Correcting: offset=(%.0f,%.0f) → N=%.3f E=%.3f m/s",
                            offset_x, offset_y, vel_north, vel_east,
                        )
                        await self._flight.send_velocity_ned(vel_north, vel_east, 0.0)

                else:
                    # Marker lost — stop correcting, hold position
                    consecutive_centered = 0
                    await self._flight.send_hover_setpoint()
                    logger.debug("Marker %d not found — hovering", target_marker_id)

                await asyncio.sleep(0)  # yield to event loop

        finally:
            self._active = False

    def abort(self) -> None:
        """Signal the landing loop to exit at the next iteration."""
        logger.info("Abort requested")
        self._abort_requested = True

    @property
    def is_active(self) -> bool:
        """Whether a landing sequence is currently in progress."""
        return self._active

    # ── Helpers ───────────────────────────────────────────────────────

    async def _grab_frame(self):
        """Get a frame from the camera manager, handling sync or async impls."""
        result = self._camera.get_frame()
        if asyncio.iscoroutine(result):
            return await result
        return result

    @staticmethod
    def _find_target(detections: list, target_id: int):
        """Return the highest-confidence detection matching target_id, or None."""
        matches = [d for d in detections if d.marker_id == target_id]
        if not matches:
            return None
        return max(matches, key=lambda d: d.confidence)
