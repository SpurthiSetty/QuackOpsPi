"""
test_mavsdk_takeoff_land.py

Bench test: connect → health check → arm → takeoff → hover → land → disarm
Props OFF — verifies ArduCopter accepts the full command sequence over UART.

REQUIRED PARAMETERS (set in Mission Planner BEFORE running):
    ARMING_CHECK    = 0
    GPS_TYPE        = 0
    FENCE_ENABLE    = 0
    EK3_SRC1_POSXY  = 0   (None)
    EK3_SRC1_VELXY  = 0   (None)
    EK3_SRC1_POSZ   = 1   (Baro)
    EK3_SRC1_VELZ   = 0   (None)
    EK3_SRC1_YAW    = 1   (Compass)

Usage:
    source ~/SeniorD/QuackOpsPi/venv/bin/activate
    python test_mavsdk_takeoff_land.py
    python test_mavsdk_takeoff_land.py --altitude 1.0 --hover-time 5
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import time
import sys

from mavsdk import System
from mavsdk.action import ActionError

# ── Configuration ─────────────────────────────────────────────────────

CONNECTION_STRING = "serial:///dev/ttyAMA0:57600"

DEFAULT_TAKEOFF_ALT_M = 1.5
DEFAULT_HOVER_TIME_S = 8.0
MAX_FLIGHT_TIME_S = 30.0
HEALTH_CHECK_TIMEOUT_S = 15.0
POST_ARM_DELAY_S = 2.0
POST_TAKEOFF_WAIT_S = 5.0
POST_LAND_TIMEOUT_S = 20.0
LANDED_ALT_THRESHOLD_M = 0.25
LANDED_STABLE_COUNT = 5

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qps.bench_test")


# ── Background Telemetry ──────────────────────────────────────────────

async def monitor_telemetry(drone: System, stop: asyncio.Event) -> None:
    """Print position telemetry in background."""
    try:
        async for pos in drone.telemetry.position():
            if stop.is_set():
                break
            log.info(
                "  ALT: %.2fm  LAT: %.6f  LON: %.6f",
                pos.relative_altitude_m,
                pos.latitude_deg,
                pos.longitude_deg,
            )
    except Exception as e:
        log.warning("Telemetry stream ended: %s", e)


async def monitor_flight_mode(drone: System, stop: asyncio.Event) -> None:
    """Log flight mode changes."""
    try:
        async for mode in drone.telemetry.flight_mode():
            if stop.is_set():
                break
            log.info("  FLIGHT MODE → %s", mode)
    except Exception as e:
        log.warning("Flight mode stream ended: %s", e)


# ── Health Check (no-GPS indoor) ─────────────────────────────────────

async def wait_for_ready(drone: System) -> None:
    """Wait for connection and basic sensor health (GPS checks skipped)."""

    # Wait for heartbeat
    log.info("Waiting for connection...")
    start = time.monotonic()
    async for state in drone.core.connection_state():
        if state.is_connected:
            log.info("Connected to flight controller")
            break
        if time.monotonic() - start > HEALTH_CHECK_TIMEOUT_S:
            raise TimeoutError("Connection timeout")

    await asyncio.sleep(2.0)

    # Check sensor health — do NOT require GPS flags
    log.info("Checking sensor health (GPS checks SKIPPED — indoor mode)...")
    start = time.monotonic()
    async for health in drone.telemetry.health():
        checks = {
            "accelerometer": health.is_accelerometer_calibration_ok,
            "gyroscope": health.is_gyrometer_calibration_ok,
            "magnetometer": health.is_magnetometer_calibration_ok,
        }
        failed = [n for n, ok in checks.items() if not ok]

        if not failed:
            log.info("Health OK: %s", ", ".join(checks.keys()))
            return

        elapsed = time.monotonic() - start
        if elapsed > HEALTH_CHECK_TIMEOUT_S:
            raise TimeoutError(f"Sensors not ready: {', '.join(failed)}")

        log.info(
            "  Waiting... NOT READY: %s (%.0f/%.0fs)",
            failed, elapsed, HEALTH_CHECK_TIMEOUT_S,
        )
        await asyncio.sleep(1.0)


# ── Landing Detection (altitude-based) ───────────────────────────────

async def wait_for_landing(drone: System) -> bool:
    """Monitor altitude to detect landing.

    ArduPilot doesn't reliably send EXTENDED_SYS_STATE, so we can't
    use MAVSDK's landed_state. Instead, detect consecutive low-altitude
    readings.

    Note: with props off, altitude won't change, so this will trigger
    almost immediately (baro reads near zero).
    """
    stable_count = 0
    start = time.monotonic()

    log.info(
        "Monitoring altitude for landing (threshold: %.2fm, timeout: %.0fs)...",
        LANDED_ALT_THRESHOLD_M, POST_LAND_TIMEOUT_S,
    )

    async for pos in drone.telemetry.position():
        alt = pos.relative_altitude_m
        elapsed = time.monotonic() - start

        if abs(alt) < LANDED_ALT_THRESHOLD_M:
            stable_count += 1
            if stable_count >= LANDED_STABLE_COUNT:
                log.info("Landing detected — altitude %.2fm (stable)", alt)
                return True
        else:
            stable_count = 0

        if elapsed > POST_LAND_TIMEOUT_S:
            log.warning("Landing timeout after %.0fs (alt: %.2fm)", elapsed, alt)
            return False

    return False


# ── Main Test Sequence ────────────────────────────────────────────────

async def run_test(takeoff_alt: float, hover_time: float) -> None:
    drone = System()
    stop_telemetry = asyncio.Event()

    log.info("=" * 60)
    log.info("QuackOps Bench Test — Arm → Takeoff → Hover → Land")
    log.info("  Takeoff altitude : %.1fm", takeoff_alt)
    log.info("  Hover time       : %.1fs", hover_time)
    log.info("  Hard timeout     : %.0fs", MAX_FLIGHT_TIME_S)
    log.info("  Connection       : %s", CONNECTION_STRING)
    log.info("=" * 60)

    # ── Step 1: Connect ───────────────────────────────────────────────
    log.info("[1/6] Connecting...")
    await drone.connect(system_address=CONNECTION_STRING)

    # ── Step 2: Health check ──────────────────────────────────────────
    log.info("[2/6] Health check (no-GPS indoor mode)...")
    await wait_for_ready(drone)

    # Read initial telemetry snapshot
    try:
        async for battery in drone.telemetry.battery():
            log.info(
                "  Battery: %.1fV (%.0f%%)",
                battery.voltage_v,
                battery.remaining_percent * 100,
            )
            break
    except Exception as e:
        log.warning("  Battery read failed: %s", e)

    try:
        async for mode in drone.telemetry.flight_mode():
            log.info("  Current mode: %s", mode)
            break
    except Exception as e:
        log.warning("  Mode read failed: %s", e)

    # Start background telemetry
    telem_task = asyncio.create_task(monitor_telemetry(drone, stop_telemetry))
    mode_task = asyncio.create_task(monitor_flight_mode(drone, stop_telemetry))

    try:
        # ── Step 3: Arm ───────────────────────────────────────────────
        log.info("[3/6] Arming in 3 seconds...")
        log.info(">>> STAND CLEAR OF THE DRONE <<<")
        await asyncio.sleep(3.0)

        try:
            await drone.action.arm()
            log.info("ARMED successfully")
        except ActionError as e:
            log.error("ARM FAILED: %s", e)
            log.error(
                "Check: ARMING_CHECK=0, GPS_TYPE=0, EK3_SRC params set correctly"
            )
            return

        await asyncio.sleep(POST_ARM_DELAY_S)
        flight_start = time.monotonic()

        # ── Step 4: Takeoff ───────────────────────────────────────────
        log.info("[4/6] Taking off to %.1fm...", takeoff_alt)

        try:
            await drone.action.set_takeoff_altitude(takeoff_alt)
            await drone.action.takeoff()
            log.info("Takeoff command ACCEPTED")
        except ActionError as e:
            log.error("TAKEOFF FAILED: %s", e)
            log.info("Attempting disarm...")
            try:
                await drone.action.disarm()
                log.info("Disarmed after failed takeoff")
            except ActionError:
                log.error("Disarm also failed — kill motors manually if needed")
            return

        log.info("Waiting %.0fs for takeoff to stabilize...", POST_TAKEOFF_WAIT_S)
        await asyncio.sleep(POST_TAKEOFF_WAIT_S)

        # ── Step 5: Hover ─────────────────────────────────────────────
        log.info("[5/6] Hovering for %.0fs...", hover_time)
        remaining = hover_time

        while remaining > 0:
            # Safety timeout
            if time.monotonic() - flight_start > MAX_FLIGHT_TIME_S:
                log.warning(
                    "SAFETY TIMEOUT (%.0fs) — forcing land NOW",
                    MAX_FLIGHT_TIME_S,
                )
                break

            wait_chunk = min(remaining, 2.0)
            await asyncio.sleep(wait_chunk)
            remaining -= wait_chunk
            if remaining > 0:
                log.info("  Hovering... %.0fs remaining", remaining)

        # ── Step 6: Land ──────────────────────────────────────────────
        log.info("[6/6] Commanding land...")

        try:
            await drone.action.land()
            log.info("Land command ACCEPTED")
        except ActionError as e:
            log.error("LAND FAILED: %s", e)
            log.error("Attempting RTL...")
            try:
                await drone.action.return_to_launch()
            except ActionError:
                log.error("RTL also failed — KILL MOTORS MANUALLY")
            return

        # Wait for landing
        landed = await wait_for_landing(drone)

        if landed:
            log.info("Landing confirmed. Disarming...")
            await asyncio.sleep(2.0)
            try:
                await drone.action.disarm()
                log.info("DISARMED successfully")
            except ActionError as e:
                log.warning(
                    "Disarm failed (%s) — may have auto-disarmed, this is normal",
                    e,
                )
        else:
            log.warning(
                "Landing not confirmed by altitude — check drone visually"
            )

    finally:
        stop_telemetry.set()
        await asyncio.sleep(0.5)
        telem_task.cancel()
        mode_task.cancel()

    log.info("=" * 60)
    log.info("QuackOps Bench Test — COMPLETE")
    log.info("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuackOps bench test: arm → takeoff → hover → land → disarm"
    )
    parser.add_argument(
        "--altitude", type=float, default=DEFAULT_TAKEOFF_ALT_M,
        help=f"Takeoff altitude in meters (default: {DEFAULT_TAKEOFF_ALT_M})",
    )
    parser.add_argument(
        "--hover-time", type=float, default=DEFAULT_HOVER_TIME_S,
        help=f"Hover duration in seconds (default: {DEFAULT_HOVER_TIME_S})",
    )
    args = parser.parse_args()

    # Safety caps for bench/cage testing
    if args.altitude > 3.5:
        log.warning("Altitude %.1fm capped to 3.5m for safety", args.altitude)
        args.altitude = 3.5
    if args.hover_time > 30:
        log.warning("Hover time %.0fs capped to 30s for safety", args.hover_time)
        args.hover_time = 30.0

    print()
    print("=" * 60)
    print("  QuackOps MAVSDK Bench Test")
    print("  Arm → Takeoff → Hover → Land → Disarm")
    print()
    print("  PROPS MUST BE OFF FOR BENCH TESTING")
    print("=" * 60)
    print()

    confirm = input("Are props OFF? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Aborting — remove props first!")
        sys.exit(0)

    try:
        asyncio.run(run_test(args.altitude, args.hover_time))
    except KeyboardInterrupt:
        log.info("Test aborted by user (Ctrl+C)")
    except TimeoutError as e:
        log.error("Test failed: %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()