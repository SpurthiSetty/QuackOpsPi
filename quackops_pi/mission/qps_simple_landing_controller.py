"""
qps_simple_landing_controller.py

Landing controller that confirms a marker with N consecutive detections
and then commands land. No position correction — the drone lands wherever
it is when the lock is acquired.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager_interface import qpsFlightManagerInterface
from quackops_pi.mission.qps_landing_controller_interface import qpsLandingControllerInterface
from quackops_pi.models.qps_landing_result import qpsLandingOutcome, qpsLandingResult
from quackops_pi.vision.qps_camera_manager_interface import qpsCameraManagerInterface
from quackops_pi.vision.qps_marker_detector_interface import qpsMarkerDetectorInterface

logger = logging.getLogger("qps.simple_landing")


class qpsSimpleLandingController(qpsLandingControllerInterface):
    """Detect marker N consecutive frames, then land directly.

    No position correction is applied — the drone lands wherever it hovers
    when the consecutive lock is acquired. Suitable for initial cage tests
    where the drone is pre-positioned above the marker.
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
        """Detect marker consecutively then command land.

        Loops until N consecutive detections of target_marker_id, timeout,
        camera failure, or abort. On lock, calls flight_manager.land().
        """
        self._active = True
        self._abort_requested = False
        consecutive = 0
        frames_searched = 0
        start_time = time.monotonic()

        try:
            while True:
                # ── Abort check ───────────────────────────────────
                if self._abort_requested:
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

                # ── Detect ────────────────────────────────────────
                detections = await self._detector.detect(frame)
                target = self._find_target(detections, target_marker_id)

                if target is not None:
                    consecutive += 1
                    logger.info(
                        "Marker %d detected (conf=%.2f) — consecutive=%d/%d",
                        target_marker_id, target.confidence,
                        consecutive, self._config.lock_frame_count,
                    )
                    if consecutive >= self._config.lock_frame_count:
                        logger.info("Lock acquired — commanding land")
                        await self._flight.land()
                        return qpsLandingResult(
                            outcome=qpsLandingOutcome.MARKER_FOUND,
                            search_duration_s=time.monotonic() - start_time,
                            frames_searched=frames_searched,
                            target_marker_id=target_marker_id,
                        )
                else:
                    if consecutive > 0:
                        logger.debug(
                            "Marker %d lost — resetting consecutive counter",
                            target_marker_id,
                        )
                    consecutive = 0

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
