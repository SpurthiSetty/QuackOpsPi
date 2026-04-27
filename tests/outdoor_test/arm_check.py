#!/usr/bin/env python3
"""
QuackOps — Outdoor Arm/Disarm Check
=====================================
Connects, verifies GPS fix, arms, holds for HOLD_TIME_S, disarms.
No takeoff. Props on — stand clear.

Usage:
    python3 arm_check.py                  # default 5s hold
    python3 arm_check.py --hold 10        # 10s hold
    python3 arm_check.py --tcp            # connect via MAVProxy TCP
"""

import argparse
import logging
import sys
import time

from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERIAL_STRING   = "/dev/ttyAMA0"
TCP_STRING      = "tcp:localhost:5763"
BAUD_RATE       = 57600

DEFAULT_HOLD_S  = 5
MIN_SATELLITES  = 6
REQUIRED_FIX    = 3   # 3D fix

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arm_check")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def connect(use_tcp: bool):
    addr = TCP_STRING if use_tcp else SERIAL_STRING
    log.info(f"Connecting to {addr}...")
    conn = mavutil.mavlink_connection(addr, baud=BAUD_RATE)
    conn.wait_heartbeat(timeout=15)
    log.info(f"Connected — system {conn.target_system}, component {conn.target_component}")
    return conn


def wait_for_gps(conn) -> bool:
    log.info(f"Waiting for GPS fix ({MIN_SATELLITES}+ sats, 3D fix)...")
    start = time.monotonic()
    while time.monotonic() - start < 90.0:
        msg = conn.recv_match(type="GPS_RAW_INT", blocking=True, timeout=2.0)
        if msg:
            fix  = msg.fix_type
            sats = msg.satellites_visible
            log.info(f"  GPS | fix_type={fix}, satellites={sats}")
            if fix >= REQUIRED_FIX and sats >= MIN_SATELLITES:
                log.info(f"✓ GPS ready — {sats} satellites")
                return True
    log.error("GPS fix timeout — do not arm")
    return False


def arm(conn) -> bool:
    log.info("Arming...")
    conn.arducopter_arm()
    start = time.monotonic()
    while time.monotonic() - start < 10.0:
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg and msg.get_srcComponent() != 0:
            if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                log.info("✓ ARMED")
                return True
    log.error("Arm failed — check Mission Planner messages for reason")
    return False


def disarm(conn) -> None:
    log.info("Disarming...")
    conn.arducopter_disarm()
    time.sleep(1.0)
    msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=3.0)
    if msg and not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
        log.info("✓ DISARMED")
    else:
        log.warning("Disarm unconfirmed — check visually")


def force_disarm(conn) -> None:
    log.warning("FORCE DISARM")
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(hold_s: int, use_tcp: bool) -> None:
    log.info("=" * 50)
    log.info("QuackOps — Outdoor Arm/Disarm Check")
    log.info(f"  Hold time : {hold_s}s")
    log.info(f"  Transport : {'TCP (MAVProxy)' if use_tcp else 'Serial'}")
    log.info("=" * 50)

    conn = connect(use_tcp)

    if not wait_for_gps(conn):
        return

    log.info(">>> STAND CLEAR — arming in 3 seconds <<<")
    time.sleep(3.0)

    try:
        if not arm(conn):
            return

        log.info(f"Motors armed — holding for {hold_s}s (Ctrl+C to abort)...")
        for i in range(hold_s):
            log.info(f"  {i + 1}/{hold_s}s")
            time.sleep(1.0)

        disarm(conn)
        log.info("✓ Arm/disarm check passed")

    except KeyboardInterrupt:
        log.warning("CTRL+C — force disarming")
        force_disarm(conn)
        sys.exit(0)

    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        force_disarm(conn)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Outdoor arm/disarm check")
    parser.add_argument("--hold", type=int, default=DEFAULT_HOLD_S,
                        help=f"Seconds to hold armed (default: {DEFAULT_HOLD_S})")
    parser.add_argument("--tcp", action="store_true",
                        help="Connect via MAVProxy TCP instead of serial")
    args = parser.parse_args()
    run(hold_s=args.hold, use_tcp=args.tcp)


if __name__ == "__main__":
    main()