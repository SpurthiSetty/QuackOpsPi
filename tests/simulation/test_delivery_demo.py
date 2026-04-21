"""
QuackOps Delivery Demo — SITL + WebSocket + Webcam
===================================================
Wires all production classes together against ArduCopter SITL.

Prerequisites:
    1. Mission Planner SITL running (Multirotor, home at Stevens campus)
       Home: 40.7453, -74.0256
       pymavlink port: 5763  (Mission Planner holds 5760)
    2. Node.js backend running at ws://localhost:3001
       Send a startDelivery command from the web UI to kick off the mission.
    3. Laptop webcam available (device index 0).
    4. Printed ArUco marker (DICT_4X4_50, ID 0) ready to hold up to the
       webcam once the drone arrives at the destination and starts searching.

Usage:
    cd <repo root>
    python tests/simulation/test_delivery_demo.py

The script:
    1. Connects flight manager to SITL (tcp:localhost:5763)
    2. Connects backend client to Node.js WebSocket
    3. Waits for startDelivery from the web UI
    4. Arms → takeoffs → flies to destination
    5. Hovers and searches webcam for ArUco marker
    6. Lands → RTL → disarms → reports MISSION_COMPLETE

Press Ctrl+C at any time to trigger a clean shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

# ── Make sure the repo root is on sys.path when running as a script ───
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager import qpsFlightManager
from quackops_pi.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor
from quackops_pi.comms.qps_backend_client import qpsBackendClient
from quackops_pi.vision.qps_cv_camera_manager import qpsCVCameraManager
from quackops_pi.vision.qps_marker_detector import qpsMarkerDetector
from quackops_pi.mission.qps_hover_search_controller import qpsHoverSearchController
from quackops_pi.mission.qps_mission_controller_impl import qpsMissionControllerImpl

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qps.demo")


# ── Config overrides for SITL demo ────────────────────────────────────

def _build_demo_config() -> qpsConfig:
    """Return a qpsConfig tuned for the SITL delivery demo.

    Override individual fields here rather than editing simulation.json,
    since simulation.json uses the old MAVSDK key names.
    """
    config = qpsConfig()

    # ── Flight controller ──────────────────────────────────────────
    # pymavlink TCP to ArduCopter SITL (Mission Planner holds 5760;
    # we use the auxiliary output port 5763)
    config.connection_string = "tcp:localhost:5763"

    # ── Backend WebSocket ──────────────────────────────────────────
    # Node.js backend running locally; adjust if on another host
    config.backend_ws_url = "ws://10.155.37.228:3001"

    # ── Flight parameters ──────────────────────────────────────────
    config.orbit_altitude_m = 10.0      # cruise + hover altitude (m)
    config.target_marker_id = 0         # ArUco DICT_4X4_50 marker ID 0

    # ── Search parameters ──────────────────────────────────────────
    # Give the tester 90s to hold up the marker after the drone arrives
    config.search_timeout_s = 90.0

    # ── Camera ─────────────────────────────────────────────────────
    config.camera_resolution = (640, 480)
    config.camera_fps = 30
    config.aruco_dictionary = "DICT_4X4_50"

    # ── Telemetry streaming rate ───────────────────────────────────
    config.telemetry_polling_rate_hz = 2.0

    # ── Backend reconnect ──────────────────────────────────────────
    config.reconnection_interval_s = 3.0

    return config


# ── Main demo coroutine ───────────────────────────────────────────────

async def run_demo() -> None:
    """Wire all production components and run the full delivery mission."""

    config = _build_demo_config()

    logger.info("QuackOps Delivery Demo")
    logger.info("  SITL:    %s", config.connection_string)
    logger.info("  Backend: %s", config.backend_ws_url)
    logger.info("  Marker:  DICT_4X4_50 ID=%d", config.target_marker_id)
    logger.info("  Search timeout: %.0fs", config.search_timeout_s)

    # ── 1. Construct components (dependency injection) ─────────────
    flight_manager = qpsFlightManager(config)
    telemetry = qpsTelemetryMonitor(flight_manager, config)
    backend = qpsBackendClient(config)
    camera = qpsCVCameraManager(config, camera_id=0)
    detector = qpsMarkerDetector(config)
    landing_controller = qpsHoverSearchController(camera, detector, telemetry, config)
    mission_controller = qpsMissionControllerImpl(
        flight_manager, telemetry, landing_controller, backend, config
    )

    # ── 2. Connect flight manager ──────────────────────────────────
    logger.info("Connecting to SITL...")
    await flight_manager.connect()
    logger.info("SITL connected")

    # ── 3. Start telemetry monitor ─────────────────────────────────
    await telemetry.start()

    # ── 4. Disable pre-arm checks for SITL ────────────────────────
    # ARMING_CHECK=0 bypasses the GPS/baro checks that fail in pure SITL.
    # Do NOT call this on real hardware.
    logger.info("Setting SITL params (ARMING_CHECK=0)...")
    await flight_manager.set_sitl_params()

    # ── 5. Connect backend ─────────────────────────────────────────
    logger.info("Connecting to backend...")
    await backend.connect()

    # ── 6. Run mission ─────────────────────────────────────────────
    logger.info("Mission controller starting — send startDelivery from the web UI")
    await mission_controller.run()


# ── Entry point ───────────────────────────────────────────────────────

async def main() -> None:
    """Top-level coroutine with Ctrl+C handling and clean shutdown."""

    # Create a dummy config/flight_manager ref so shutdown can find them
    # even if run_demo() hasn't fully initialised yet.
    loop = asyncio.get_running_loop()

    # Register SIGINT handler (Ctrl+C) to cancel the main task gracefully
    main_task = asyncio.current_task()

    def _handle_sigint(*_args) -> None:
        logger.warning("Ctrl+C received — shutting down")
        if main_task is not None:
            main_task.cancel()

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
    else:
        # Windows: signal.signal works but loop.add_signal_handler doesn't
        signal.signal(signal.SIGINT, _handle_sigint)

    try:
        await run_demo()
    except asyncio.CancelledError:
        logger.info("Demo cancelled by user")
    except Exception as exc:
        logger.exception("Demo ended with unhandled exception: %s", exc)
    finally:
        logger.info("Demo finished")


if __name__ == "__main__":
    asyncio.run(main())
