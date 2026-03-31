"""
test_mavsdk_connection.py

Quick bench test: connect to Pixhawk over UART, read telemetry, arm/disarm.
Props MUST be off before running.

Usage (on Pi):
    source ~/SeniorD/QuackOpsPi/venv/bin/activate
    python test_mavsdk_connection.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qps.hw_test")

# ── Configuration ─────────────────────────────────────────────────────

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 57600
MAVSDK_CONNECTION = f"serial://{SERIAL_PORT}:{BAUD_RATE}"

# How long to wait for various stages
CONNECT_TIMEOUT_S = 15.0
TELEMETRY_READ_S = 5.0
ARM_HOLD_S = 3.0


async def main() -> None:
    from mavsdk import System

    drone = System()

    # ── Step 1: Connect ───────────────────────────────────────────────
    logger.info("Connecting to Pixhawk via %s", MAVSDK_CONNECTION)
    await drone.connect(system_address=MAVSDK_CONNECTION)

    logger.info("Waiting for heartbeat...")
    try:
        async for state in drone.core.connection_state():
            if state.is_connected:
                logger.info("CONNECTED — heartbeat received")
                break
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for connection")
        sys.exit(1)

    # ── Step 2: Read telemetry snapshot ───────────────────────────────
    logger.info("Reading telemetry for %.0fs...", TELEMETRY_READ_S)

    # Battery
    try:
        async for battery in drone.telemetry.battery():
            logger.info(
                "  Battery: %.1fV  (%.0f%%)",
                battery.voltage_v,
                battery.remaining_percent * 100,
            )
            break
    except Exception as e:
        logger.warning("  Battery read failed: %s", e)

    # GPS info
    try:
        async for gps_info in drone.telemetry.gps_info():
            logger.info(
                "  GPS: fix_type=%s, satellites=%d",
                gps_info.fix_type,
                gps_info.num_satellites,
            )
            break
    except Exception as e:
        logger.warning("  GPS info read failed: %s", e)

    # Position
    try:
        async for position in drone.telemetry.position():
            logger.info(
                "  Position: lat=%.7f, lon=%.7f, alt=%.1fm",
                position.latitude_deg,
                position.longitude_deg,
                position.relative_altitude_m,
            )
            break
    except Exception as e:
        logger.warning("  Position read failed: %s", e)

    # Flight mode
    try:
        async for mode in drone.telemetry.flight_mode():
            logger.info("  Flight mode: %s", mode)
            break
    except Exception as e:
        logger.warning("  Flight mode read failed: %s", e)

    # Armed state
    try:
        async for armed in drone.telemetry.armed():
            logger.info("  Armed: %s", armed)
            break
    except Exception as e:
        logger.warning("  Armed state read failed: %s", e)

    # ── Step 3: Arm test ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 50)
    logger.info("PROPS MUST BE OFF — attempting arm in 3 seconds...")
    logger.info("=" * 50)
    await asyncio.sleep(3.0)

    try:
        await drone.action.arm()
        logger.info("ARM command accepted")
    except Exception as e:
        logger.error("ARM failed: %s", e)
        logger.info(
            "This is normal if arming checks fail (no GPS fix, etc.). "
            "You can set ARMING_CHECK=0 in Mission Planner to bypass."
        )
        logger.info("Test complete (arm failed — see above)")
        return

    # Verify armed state
    try:
        async for armed in drone.telemetry.armed():
            if armed:
                logger.info("CONFIRMED ARMED — motors should be live")
            else:
                logger.info("Arm command accepted but drone reports not armed")
            break
    except Exception as e:
        logger.warning("Could not verify armed state: %s", e)

    # Hold for a moment so you can hear/see motors
    logger.info("Holding armed state for %.0fs...", ARM_HOLD_S)
    await asyncio.sleep(ARM_HOLD_S)

    # ── Step 4: Disarm ────────────────────────────────────────────────
    logger.info("Disarming...")
    try:
        await drone.action.disarm()
        logger.info("DISARMED successfully")
    except Exception as e:
        logger.warning("Disarm failed: %s (may have auto-disarmed)", e)

    logger.info("")
    logger.info("Test complete!")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  QuackOps MAVSDK Hardware Connection Test")
    print("  PROPS MUST BE REMOVED / OFF")
    print("=" * 50 + "\n")

    confirm = input("Are props OFF? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Aborting — remove props first!")
        sys.exit(0)

    asyncio.run(main())