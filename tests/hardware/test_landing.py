#!/usr/bin/env python3
"""
QuackOps Landing Controller Test — Indoor Cage (ALT_HOLD, no GPS)

Runs the full sequence:
    1. Start camera + stream server (watch at http://ssetty.local:8080)
    2. Connect flight manager (pymavlink RC override)
    3. Arm → Takeoff (ALT_HOLD)
    4. Run landing controller (simple or visual servo)
    5. Report result

Usage:
    python3 tests/hardware/test_landing.py --strategy simple
    python3 tests/hardware/test_landing.py --strategy servo
    python3 tests/hardware/test_landing.py --strategy simple --marker-id 0 --hover-time 3
    python3 tests/hardware/test_landing.py --strategy servo --gain 0.001 --tolerance 30

Ctrl+C at any time → emergency stop (throttle cut + force disarm)
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_rc_flight_manager import qpsRCFlightManager
from quackops_pi.vision.qps_pi_camera_manager import qpsPiCameraManager
from quackops_pi.vision.qps_marker_detector import qpsMarkerDetector
from quackops_pi.vision.qps_stream_server import qpsStreamServer
from quackops_pi.mission.qps_simple_landing_controller import qpsSimpleLandingController
from quackops_pi.mission.qps_visual_servo_landing_controller import qpsVisualServoLandingController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_landing")


async def run(args: argparse.Namespace) -> None:
    config = qpsConfig()

    # Apply CLI overrides
    config.lock_frame_count = args.lock_frames
    config.search_timeout_s = args.timeout
    if args.gain is not None:
        config.proportional_gain = args.gain
    if args.tolerance is not None:
        config.center_tolerance_px = args.tolerance

    # ── 1. Camera + Stream ────────────────────────────────────────────
    log.info("[1/5] Starting camera and stream server...")
    camera = qpsPiCameraManager(config)
    camera.start()
    time.sleep(1.0)  # camera warm-up

    detector = qpsMarkerDetector(config)
    stream = qpsStreamServer(camera, detector, config)
    stream.start(port=args.port)
    log.info("Stream at http://ssetty.local:%d", args.port)

    # ── 2. Flight Manager ─────────────────────────────────────────────
    log.info("[2/5] Connecting flight manager...")
    flight = qpsRCFlightManager(config)
    await flight.connect()

    # ── 3. Select landing controller ──────────────────────────────────
    if args.strategy == "simple":
        log.info("[3/5] Using SIMPLE landing controller")
        controller = qpsSimpleLandingController(camera, detector, flight, config)
    else:
        log.info("[3/5] Using VISUAL SERVO landing controller")
        controller = qpsVisualServoLandingController(camera, detector, flight, config)

    try:
        # ── 4. Arm + Takeoff ──────────────────────────────────────────
        log.info("[4/5] Arming and taking off to %.1fm...", args.altitude)
        log.info(">>> STAND CLEAR OF THE DRONE <<<")
        await asyncio.sleep(3.0)

        await flight.arm()
        await flight.takeoff(args.altitude)

        if args.hover_time > 0:
            log.info("Stabilizing hover for %.1fs...", args.hover_time)
            await asyncio.sleep(args.hover_time)

        # ── 5. Execute landing ────────────────────────────────────────
        log.info(
            "[5/5] Executing %s landing (marker ID=%d)...",
            args.strategy, args.marker_id,
        )
        result = await controller.execute_landing(args.marker_id)

        log.info("=" * 50)
        log.info("LANDING RESULT:")
        log.info("  Outcome        : %s", result.outcome.name)
        log.info("  Success        : %s", result.success)
        log.info("  Duration       : %.1fs", result.search_duration_s)
        log.info("  Frames searched: %d", result.frames_searched)
        log.info("=" * 50)

    except KeyboardInterrupt:
        log.warning("CTRL+C — EMERGENCY STOP")
        await flight.force_disarm()
        raise

    except Exception as exc:
        log.error("Flight error: %s", exc, exc_info=True)
        await flight.force_disarm()
        raise

    finally:
        stream.stop()
        camera.stop()
        await flight.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuackOps landing controller test (indoor cage)"
    )
    parser.add_argument(
        "--strategy", choices=["simple", "servo"], default="simple",
        help="Landing strategy (default: simple)",
    )
    parser.add_argument(
        "--marker-id", type=int, default=0,
        help="Target ArUco marker ID (default: 0)",
    )
    parser.add_argument(
        "--altitude", type=float, default=1.5,
        help="Takeoff altitude in meters (default: 1.5)",
    )
    parser.add_argument(
        "--hover-time", type=float, default=3.0,
        help="Seconds to hover before starting landing controller (default: 3.0)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Landing controller timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--lock-frames", type=int, default=5,
        help="Consecutive detections to confirm lock (default: 5)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Stream server port (default: 8080)",
    )
    # Visual servo specific
    parser.add_argument(
        "--gain", type=float, default=None,
        help="Proportional gain for visual servo (default: from config)",
    )
    parser.add_argument(
        "--tolerance", type=int, default=None,
        help="Center tolerance in pixels for visual servo (default: from config)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        log.info("Exiting")
        sys.exit(0)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
