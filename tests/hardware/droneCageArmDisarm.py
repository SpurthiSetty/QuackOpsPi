#!/usr/bin/env python3
"""
QuackOps Cage Test — Autonomous Arm → Takeoff → Hover → Land → Disarm
=====================================================================
Target:     ArduCopter 4.6.3 on Pixhawk 2.4.8 via TELEM1
Env:        Indoor cQuackOpsPi/Hardware_tests/dronecageTestFly.pyage, no GPS — requires EKF configured for no-GPS ops
Pi:         Raspberry Pi 5 (serial /dev/ttyAMA0 @ 57600)

REQUIRED PARAMETERS (set in Mission Planner BEFORE running):
    ARMING_CHECK    = 0
    GPS_TYPE        = 0
    FENCE_ENABLE    = 0
    EK3_SRC1_POSXY  = 0   (None)
    EK3_SRC1_VELXY  = 0   (None)
    EK3_SRC1_POSZ   = 1   (Baro)
    EK3_SRC1_VELZ   = 0   (None)
    EK3_SRC1_YAW    = 1   (Compass)

SAFETY:
    - Have RC transmitter ON with motor kill switch (RC6_OPTION=31)
    - Safety cable attached
    - Clear the cage area before running
    - Script has a hard timeout — drone will land after MAX_FLIGHT_TIME_S

Usage:
    python3 cage_test_hover.py
    python3 cage_test_hover.py --altitude 1.0 --hover-time 5
"""

import asyncio
import argparse
import logging
import time
import sys

from mavsdk import System
from mavsdk.action import ActionError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONNECTION_STRING = "serial:///dev/ttyAMA0:57600"

# Defaults (override via CLI args)
DEFAULT_TAKEOFF_ALT_M = 1.5      # meters — conservative for cage test
DEFAULT_HOVER_TIME_S = 8.0        # seconds to hover before landing
MAX_FLIGHT_TIME_S = 30.0          # hard safety timeout — land no matter what
HEALTH_CHECK_TIMEOUT_S = 15.0     # how long to wait for system health
POST_ARM_DELAY_S = 2.0            # settle time after arming
POST_TAKEOFF_WAIT_S = 5.0         # extra time to let takeoff stabilize
POST_LAND_TIMEOUT_S = 20.0        # max wait for landing to complete

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cage_test")

# ---------------------------------------------------------------------------
# Telemetry Monitor Task
# ---------------------------------------------------------------------------

async def monitor_telemetry(drone: System, stop_event: asyncio.Event):
    """Print telemetry in background so you can see what the drone is doing."""
    try:
        async for position in drone.telemetry.position():
            if stop_event.is_set():
                break
            log.info(
                f"ALT: {position.relative_altitude_m:.2f}m  "
                f"LAT: {position.latitude_deg:.6f}  "
                f"LON: {position.longitude_deg:.6f}"
            )
    except Exception as e:
        log.warning(f"Telemetry stream ended: {e}")


async def monitor_flight_mode(drone: System, stop_event: asyncio.Event):
    """Log flight mode changes."""
    try:
        async for mode in drone.telemetry.flight_mode():
            if stop_event.is_set():
                break
            log.info(f"FLIGHT MODE: {mode}")
    except Exception as e:
        log.warning(f"Flight mode stream ended: {e}")

# ---------------------------------------------------------------------------
# Health Check (adapted for no-GPS indoor)
# ---------------------------------------------------------------------------

async def wait_for_drone_ready(drone: System):
    """
    Wait for basic system health. With GPS_TYPE=0 and EKF no-position config,
    we can NOT expect is_global_position_ok or is_home_position_ok.
    We check: accelerometer, gyroscope, magnetometer, and that the system
    is connected / responds.
    """
    log.info("Waiting for drone connection and basic health...")
    start = time.monotonic()

    # Wait for connection first
    async for state in drone.core.connection_state():
        if state.is_connected:
            log.info("Connected to flight controller")
            break
        if time.monotonic() - start > HEALTH_CHECK_TIMEOUT_S:
            raise TimeoutError("Timed out waiting for connection")

    # Give a moment for telemetry to start flowing
    await asyncio.sleep(2.0)

    # Check what health data we can get — don't require GPS-related flags
    start = time.monotonic()
    async for health in drone.telemetry.health():
        checks = {
            "accelerometer": health.is_accelerometer_calibration_ok,
            "gyroscope": health.is_gyrometer_calibration_ok,
            "magnetometer": health.is_magnetometer_calibration_ok,
        }

        failed = [name for name, ok in checks.items() if not ok]
        passed = [name for name, ok in checks.items() if ok]

        if not failed:
            log.info(f"Health OK: {', '.join(passed)}")
            return

        elapsed = time.monotonic() - start
        if elapsed > HEALTH_CHECK_TIMEOUT_S:
            log.error(f"Health check timeout. FAILED: {', '.join(failed)}")
            raise TimeoutError(f"Sensors not ready: {', '.join(failed)}")

        log.info(
            f"Waiting... OK: {passed} | NOT READY: {failed} "
            f"({elapsed:.0f}/{HEALTH_CHECK_TIMEOUT_S:.0f}s)"
        )
        await asyncio.sleep(1.0)

# ---------------------------------------------------------------------------
# Wait for Landing (altitude-based, since landed_state is unreliable)
# ---------------------------------------------------------------------------

async def wait_for_landing(drone: System, timeout_s: float) -> bool:
    """
    Monitor relative altitude to detect landing.

    We use altitude rather than landed_state because ArduPilot does not
    reliably send EXTENDED_SYS_STATE needed for MAVSDK landed_state.
    """
    LANDED_ALT_THRESHOLD_M = 0.25
    LANDED_STABLE_COUNT = 5  # consecutive readings below threshold
    stable_count = 0
    start = time.monotonic()

    log.info(
        f"Monitoring altitude for landing "
        f"(threshold: {LANDED_ALT_THRESHOLD_M}m, "
        f"timeout: {timeout_s}s)..."
    )

    async for position in drone.telemetry.position():
        alt = position.relative_altitude_m
        elapsed = time.monotonic() - start

        if alt < LANDED_ALT_THRESHOLD_M:
            stable_count += 1
            if stable_count >= LANDED_STABLE_COUNT:
                log.info(f"Landing detected — altitude {alt:.2f}m (stable)")
                return True
        else:
            stable_count = 0

        if elapsed > timeout_s:
            log.warning(f"Landing timeout after {timeout_s}s (alt: {alt:.2f}m)")
            return False

    return False

# ---------------------------------------------------------------------------
# Main Test Sequence
# ---------------------------------------------------------------------------

async def run_test(takeoff_alt: float, hover_time: float):
    drone = System()
    stop_telemetry = asyncio.Event()

    log.info("=" * 60)
    log.info("QuackOps Cage Test — STARTING")
    log.info(f"  Takeoff altitude : {takeoff_alt}m")
    log.info(f"  Hover time       : {hover_time}s")
    log.info(f"  Hard timeout     : {MAX_FLIGHT_TIME_S}s")
    log.info(f"  Connection       : {CONNECTION_STRING}")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Connect
    # ------------------------------------------------------------------
    log.info("[1/6] Connecting to flight controller...")
    await drone.connect(system_address=CONNECTION_STRING)

    # ------------------------------------------------------------------
    # Step 2 — Health check
    # ------------------------------------------------------------------
    log.info("[2/6] Running health checks (GPS checks SKIPPED — indoor mode)...")
    await wait_for_drone_ready(drone)

    # Start background telemetry
    telem_task = asyncio.create_task(monitor_telemetry(drone, stop_telemetry))
    mode_task = asyncio.create_task(monitor_flight_mode(drone, stop_telemetry))

    try:
        # --------------------------------------------------------------
        # Step 3 — Arm
        # --------------------------------------------------------------
        log.info("[3/6] Arming motors...")
        log.info(">>> STAND CLEAR OF THE DRONE <<<")
        await asyncio.sleep(3.0)  # 3-second warning before arming

        try:
            await drone.action.arm()
            log.info("ARMED successfully")
        except ActionError as e:
            log.error(f"Arming FAILED: {e}")
            log.error(
                "Check that ARMING_CHECK=0, GPS_TYPE=0, and EK3_SRC params "
                "are set correctly."
            )
            return

        await asyncio.sleep(POST_ARM_DELAY_S)

        # Start safety timeout
        flight_start = time.monotonic()

        # --------------------------------------------------------------
        # Step 4 — Takeoff
        # --------------------------------------------------------------
        log.info(f"[4/6] Taking off to {takeoff_alt}m...")

        try:
            await drone.action.set_takeoff_altitude(takeoff_alt)
            await drone.action.takeoff()
            log.info("Takeoff command sent")
        except ActionError as e:
            log.error(f"Takeoff FAILED: {e}")
            log.info("Attempting to disarm...")
            try:
                await drone.action.disarm()
            except ActionError:
                log.error("Disarm also failed — USE KILL SWITCH")
            return

        # Wait for drone to reach altitude
        log.info(f"Waiting {POST_TAKEOFF_WAIT_S}s for takeoff to stabilize...")
        await asyncio.sleep(POST_TAKEOFF_WAIT_S)

        # --------------------------------------------------------------
        # Step 5 — Hover
        # --------------------------------------------------------------
        remaining_hover = hover_time
        log.info(f"[5/6] Hovering for {remaining_hover}s...")

        while remaining_hover > 0:
            # Safety timeout check
            flight_elapsed = time.monotonic() - flight_start
            if flight_elapsed > MAX_FLIGHT_TIME_S:
                log.warning(
                    f"SAFETY TIMEOUT ({MAX_FLIGHT_TIME_S}s) — "
                    "forcing landing NOW"
                )
                break

            wait_chunk = min(remaining_hover, 1.0)
            await asyncio.sleep(wait_chunk)
            remaining_hover -= wait_chunk
            log.info(f"  Hovering... {remaining_hover:.0f}s remaining")

        # --------------------------------------------------------------
        # Step 6 — Land
        # --------------------------------------------------------------
        log.info("[6/6] Commanding land...")

        try:
            await drone.action.land()
            log.info("Land command sent")
        except ActionError as e:
            log.error(f"Land FAILED: {e}")
            log.error("Attempting RTL...")
            try:
                await drone.action.return_to_launch()
            except ActionError:
                log.error("RTL also failed — USE KILL SWITCH")
            return

        # Wait for landing (altitude-based detection)
        landed = await wait_for_landing(drone, POST_LAND_TIMEOUT_S)

        if landed:
            log.info("Landing confirmed. Disarming...")
            await asyncio.sleep(2.0)  # settle before disarm
            try:
                await drone.action.disarm()
                log.info("DISARMED successfully")
            except ActionError as e:
                log.warning(
                    f"Disarm failed ({e}) — motors may auto-disarm. "
                    "This is normal if ArduPilot disarmed on its own."
                )
        else:
            log.warning(
                "Landing not confirmed by altitude — "
                "drone may still be airborne. CHECK VISUALLY."
            )

    finally:
        # Cleanup
        stop_telemetry.set()
        await asyncio.sleep(0.5)
        telem_task.cancel()
        mode_task.cancel()

    log.info("=" * 60)
    log.info("QuackOps Cage Test — COMPLETE")
    log.info("=" * 60)

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QuackOps cage test: arm → takeoff → hover → land"
    )
    parser.add_argument(
        "--altitude", type=float, default=DEFAULT_TAKEOFF_ALT_M,
        help=f"Takeoff altitude in meters (default: {DEFAULT_TAKEOFF_ALT_M})"
    )
    parser.add_argument(
        "--hover-time", type=float, default=DEFAULT_HOVER_TIME_S,
        help=f"Hover duration in seconds (default: {DEFAULT_HOVER_TIME_S})"
    )
    args = parser.parse_args()

    # Sanity bounds for cage testing
    if args.altitude > 3.5:
        log.warning(
            f"Altitude {args.altitude}m seems high for cage testing. "
            "Capping at 3.5m."
        )
        args.altitude = 3.5

    if args.hover_time > 30:
        log.warning(
            f"Hover time {args.hover_time}s is long. Capping at 30s."
        )
        args.hover_time = 30.0

    try:
        asyncio.run(run_test(args.altitude, args.hover_time))
    except KeyboardInterrupt:
        log.info("Test aborted by user (Ctrl+C)")
    except TimeoutError as e:
        log.error(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()