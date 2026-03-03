"""QuackOps Pi — Autonomous Drone Delivery Companion Computer Software.

Entry point that wires all dependencies, starts subsystems, and runs
the mission state machine.
"""

import argparse
import asyncio
import logging
import sys

import mavsdk

from src.config.qps_config import qpsConfig
from src.flight.qps_flight_manager import qpsFlightManager
from src.vision.qps_pi_camera_manager import qpsPiCameraManager
from src.vision.qps_cv_camera_manager import qpsCVCameraManager
from src.vision.qps_marker_detector import qpsMarkerDetector
from src.communication.qps_backend_client import qpsBackendClient
from src.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor
from src.landing.qps_landing_controller import qpsLandingController
from src.mission.qps_mission_controller import qpsMissionController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("quackops-pi")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with config_path.
    """
    parser = argparse.ArgumentParser(
        description="QuackOps Pi — Autonomous Drone Delivery System",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.json",
        dest="config_path",
        help="Path to the JSON configuration file (default: config/default.json)",
    )
    return parser.parse_args()


async def main() -> None:
    """Wire dependencies and run the mission controller."""

    args = parse_args()

    # --- Load configuration ---
    logger.info("Loading configuration from %s", args.config_path)
    config = qpsConfig.load(args.config_path)

    # --- Create MAVSDK System ---
    drone = mavsdk.System()

    # --- Instantiate flight manager (always real — connects to SITL or hardware) ---
    flight_manager = qpsFlightManager(config)

    # --- Instantiate camera manager based on config ---
    if config.use_picamera2:
        logger.info("Using PiCamera2 camera manager (production)")
        camera_manager = qpsPiCameraManager(config)
    else:
        logger.info("Using OpenCV camera manager (simulation/desktop)")
        camera_manager = qpsCVCameraManager(config)

    # --- Instantiate marker detector (always real) ---
    marker_detector = qpsMarkerDetector(config)

    # --- Instantiate backend client (always real) ---
    backend_client = qpsBackendClient(config)

    # --- Instantiate telemetry monitor ---
    telemetry_monitor = qpsTelemetryMonitor(drone, config)

    # --- Instantiate landing controller ---
    landing_controller = qpsLandingController(
        config, flight_manager, camera_manager, marker_detector
    )

    # --- Instantiate mission controller ---
    mission_controller = qpsMissionController(
        config, flight_manager, landing_controller, backend_client, telemetry_monitor
    )

    # --- Register telemetry battery callbacks ---
    telemetry_monitor.set_battery_warning_callback(
        mission_controller.handle_battery_warning
    )
    telemetry_monitor.set_battery_critical_callback(
        mission_controller.handle_battery_critical
    )

    # --- Register backend command callback ---
    backend_client.on_command(mission_controller.handle_dispatch)

    # --- Start subsystems ---
    logger.info("Starting telemetry monitor...")
    await telemetry_monitor.start()

    logger.info("Connecting to backend at %s ...", config.backend_ws_url)
    await backend_client.connect()

    logger.info(
        "Connecting to flight controller at %s ...",
        config.mavsdk_connection_string,
    )
    await flight_manager.connect()

    # --- Run mission state machine ---
    logger.info("Starting mission controller...")
    await mission_controller.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt — shutting down...")
        # NOTE: shutdown() is async; in a real implementation the running
        # mission_controller.run() task would be cancelled and shutdown()
        # awaited in the finally block of main().
        sys.exit(0)
