#!/usr/bin/env python3
"""
QuackOps — Phase D: qpsHoverSearchController Pipeline Test

Validates the production qpsHoverSearchController against real camera +
real marker detector with a stubbed telemetry monitor (no FC required).

Architecture:
    qpsPiCameraManager (real, cam0)
        │
        ▼
    qpsMarkerDetector (real, DICT_4X4_50)
        │
        ▼
    qpsHoverSearchController (real)
        │
        ├──► qpsTelemetryMonitor (stub — returns fake GPS)
        └──► returns qpsLandingResult

Three scenarios tested:
    1. MARKER_FOUND  — marker placed in view, expect detection success
    2. ABORTED       — abort() called mid-search, expect graceful exit
    3. SEARCH_TIMEOUT — marker removed, search times out

Usage on the Pi from project root:
    cd ~/SeniorD/QuackOpsPi
    source venv/bin/activate
    python3 cameratests/test_phase_d_landing_pipeline.py

Pre-flight:
  - Print ArUco DICT_4X4_50 marker, ID 0
  - Have it nearby; the test will prompt you to place/remove it
  - Pi over SSH is fine — cv2.imshow is monkey-patched to no-op
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Make project importable when run from cameratests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Monkey-patch cv2.imshow / cv2.waitKey BEFORE importing the controller, so the
# preview window code in qpsHoverSearchController is silenced during the test.
# This lets the test run over headless SSH without an X server.
import cv2
cv2.imshow = lambda *a, **kw: None
cv2.waitKey = lambda *a, **kw: -1
cv2.destroyAllWindows = lambda *a, **kw: None

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_pi_camera_manager import qpsPiCameraManager
from quackops_pi.vision.qps_marker_detector import qpsMarkerDetector
from quackops_pi.mission.qps_hover_search_controller import qpsHoverSearchController
from quackops_pi.models.qps_landing_result import qpsLandingOutcome
from quackops_pi.models.qps_gps_position import qpsGPSPosition


# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qps.phase_d")


# ── Stub telemetry monitor ────────────────────────────────────────────

class _StubTelemetryMonitor:
    """Minimal qpsTelemetryMonitor stand-in for headless testing.

    qpsHoverSearchController only calls .get_gps_position() — and only when
    marker is found OR search aborts/times out (to record fallback_gps).
    Returns a fixed Stevens-area GPS so the result object is well-formed.
    """

    def __init__(self) -> None:
        self._fake_gps = qpsGPSPosition(
            latitude_deg=40.7437,         # Approx Stevens campus
            longitude_deg=-74.0272,
            altitude_m=10.0,              # Pretend we're hovering at 10m
            heading_deg=0.0,
            speed_m_s=0.0,
            timestamp=time.time(),
        )

    def get_gps_position(self) -> qpsGPSPosition:
        # Return a fresh copy with current timestamp on each call
        return qpsGPSPosition(
            latitude_deg=self._fake_gps.latitude_deg,
            longitude_deg=self._fake_gps.longitude_deg,
            altitude_m=self._fake_gps.altitude_m,
            heading_deg=self._fake_gps.heading_deg,
            speed_m_s=self._fake_gps.speed_m_s,
            timestamp=time.time(),
        )

    def get_drone_state(self):
        # Not called by qpsHoverSearchController, but kept for interface parity
        return None


# ── Helpers ───────────────────────────────────────────────────────────

def _prompt(msg: str) -> None:
    """Print a clear instruction and wait for the user to press Enter."""
    print()
    print("=" * 60)
    print(f"  {msg}")
    print("=" * 60)
    input("  Press Enter to continue... ")
    print()


def _result_summary(label: str, result) -> None:
    """Pretty-print a qpsLandingResult."""
    log.info(f"--- {label} ---")
    log.info(f"  outcome:          {result.outcome.name}")
    log.info(f"  duration:         {result.search_duration_s:.2f}s")
    log.info(f"  frames_searched:  {result.frames_searched}")
    log.info(f"  target_marker_id: {result.target_marker_id}")
    log.info(f"  marker_gps:       {result.marker_gps}")
    log.info(f"  fallback_gps:     {result.fallback_gps}")


# ── Test scenarios ────────────────────────────────────────────────────

async def scenario_1_marker_found(config, camera, detector, telemetry):
    """Marker is in view → expect MARKER_FOUND outcome."""
    log.info("=" * 60)
    log.info("Scenario 1: MARKER_FOUND")
    log.info("=" * 60)

    _prompt("Place the marker (DICT_4X4_50, ID 0) in cam0's view (bottom-facing).")

    controller = qpsHoverSearchController(camera, detector, telemetry, config)
    result = await controller.execute_marker_search(
        target_marker_id=config.target_marker_id
    )
    _result_summary("Result", result)

    if result.outcome == qpsLandingOutcome.MARKER_FOUND:
        log.info("✓ PASS: detection succeeded")
        assert result.marker_gps is not None, "marker_gps should be set on MARKER_FOUND"
        assert result.frames_searched > 0
        assert result.search_duration_s < config.search_timeout_s
        return True
    else:
        log.error(f"✗ FAIL: expected MARKER_FOUND, got {result.outcome.name}")
        return False


async def scenario_2_abort(config, camera, detector, telemetry):
    """Search aborted while running → expect ABORTED outcome."""
    log.info("=" * 60)
    log.info("Scenario 2: ABORTED")
    log.info("=" * 60)

    _prompt(
        "REMOVE the marker from cam0's view. The test will start a search, "
        "then abort it after 2 seconds."
    )

    controller = qpsHoverSearchController(camera, detector, telemetry, config)

    async def abort_after(delay_s: float):
        await asyncio.sleep(delay_s)
        log.info(f"  Calling abort() after {delay_s:.1f}s...")
        controller.abort()

    abort_task = asyncio.create_task(abort_after(2.0))
    result = await controller.execute_marker_search(
        target_marker_id=config.target_marker_id
    )
    await abort_task

    _result_summary("Result", result)

    if result.outcome == qpsLandingOutcome.ABORTED:
        log.info("✓ PASS: abort honored")
        assert result.fallback_gps is not None, "fallback_gps should be set on ABORTED"
        assert 1.5 < result.search_duration_s < 4.0, (
            f"abort timing off: {result.search_duration_s:.1f}s "
            "(expected ~2s)"
        )
        return True
    else:
        log.error(f"✗ FAIL: expected ABORTED, got {result.outcome.name}")
        return False


async def scenario_3_timeout(config, camera, detector, telemetry):
    """Marker not visible → expect SEARCH_TIMEOUT outcome.

    Override config.search_timeout_s temporarily to ~5s so the test doesn't
    wait the default 180s.
    """
    log.info("=" * 60)
    log.info("Scenario 3: SEARCH_TIMEOUT")
    log.info("=" * 60)

    _prompt(
        "Confirm the marker is NOT in cam0's view. Search will time out "
        "after 5 seconds."
    )

    # Temporarily override the timeout
    original_timeout = config.search_timeout_s
    config.search_timeout_s = 5.0

    try:
        controller = qpsHoverSearchController(camera, detector, telemetry, config)
        result = await controller.execute_marker_search(
            target_marker_id=config.target_marker_id
        )
    finally:
        config.search_timeout_s = original_timeout

    _result_summary("Result", result)

    if result.outcome == qpsLandingOutcome.SEARCH_TIMEOUT:
        log.info("✓ PASS: search timed out as expected")
        assert result.fallback_gps is not None, "fallback_gps should be set on TIMEOUT"
        assert 4.5 < result.search_duration_s < 6.5, (
            f"timeout duration off: {result.search_duration_s:.1f}s "
            "(expected ~5s)"
        )
        return True
    else:
        log.error(f"✗ FAIL: expected SEARCH_TIMEOUT, got {result.outcome.name}")
        return False


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 60)
    log.info("QuackOps — Phase D: Hover Search Controller Pipeline Test")
    log.info("=" * 60)

    config = qpsConfig()
    log.info(
        f"Config: {config.camera_resolution[0]}x{config.camera_resolution[1]} "
        f"@ {config.camera_fps}fps  dict={config.aruco_dictionary}  "
        f"target_id={config.target_marker_id}  "
        f"timeout={config.search_timeout_s}s"
    )

    # ── Build dependencies once, share across scenarios ──
    # The controller calls camera.start() and camera.stop() in each
    # execute_marker_search(), so we don't need to start the camera here.
    # But we DO build it once so we don't reinitialize libcamera repeatedly.
    camera = qpsPiCameraManager(config)
    detector = qpsMarkerDetector(config)
    telemetry = _StubTelemetryMonitor()

    log.info("Dependencies constructed:")
    log.info(f"  camera:    {type(camera).__name__}")
    log.info(f"  detector:  {type(detector).__name__}")
    log.info(f"  telemetry: {type(telemetry).__name__} (stub)")

    # Run scenarios
    results = []
    try:
        results.append(("MARKER_FOUND", await scenario_1_marker_found(
            config, camera, detector, telemetry)))
        results.append(("ABORTED", await scenario_2_abort(
            config, camera, detector, telemetry)))
        results.append(("SEARCH_TIMEOUT", await scenario_3_timeout(
            config, camera, detector, telemetry)))
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return

    # ── Summary ──
    log.info("")
    log.info("=" * 60)
    log.info("Phase D Summary")
    log.info("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        log.info(f"  {status}  Scenario: {name}")
    log.info("")

    all_pass = all(p for _, p in results)
    if all_pass:
        log.info("🎉 All scenarios PASSED — landing pipeline logic validated")
    else:
        log.error("✗ Some scenarios failed — see logs above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())