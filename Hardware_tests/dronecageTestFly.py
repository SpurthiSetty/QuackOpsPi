#!/usr/bin/env python3
"""
QuackOps Cage Test — RC Override Flight (ArduPilot)
====================================================
Target:     ArduCopter 4.6.3 on Pixhawk 2.4.8 via TELEM1
Env:        Indoor cage, no GPS, STABILIZE mode
Pi:         Raspberry Pi 5 (serial /dev/ttyAMA0 @ 57600)

WHY THIS SCRIPT EXISTS:
    MAVSDK manual_control uses the MANUAL_CONTROL MAVLink message,
    which ArduCopter ignores for motor output. ArduPilot requires
    RC_OVERRIDE messages to simulate stick input from a companion
    computer. This script uses pymavlink directly to send RC_OVERRIDE.

RC CHANNEL MAPPING (standard):
    CH1 = Roll       (1000-2000, center 1500)
    CH2 = Pitch      (1000-2000, center 1500)
    CH3 = Throttle   (1000-2000, 1000=min, 2000=max)
    CH4 = Yaw        (1000-2000, center 1500)

REQUIRED PARAMETERS (set in Mission Planner BEFORE running):
    ARMING_CHECK    = 0
    GPS_TYPE        = 0
    FENCE_ENABLE    = 0
    FS_THR_ENABLE   = 0        *** CRITICAL — disable RC failsafe ***
    FS_GCS_ENABLE   = 0        *** CRITICAL — disable GCS failsafe ***
    DISARM_DELAY    = 0        (if testing without props)
    EK3_SRC1_POSXY  = 0
    EK3_SRC1_VELXY  = 0
    EK3_SRC1_POSZ   = 1
    EK3_SRC1_VELZ   = 0
    EK3_SRC1_YAW    = 1

SAFETY:
    - Have RC transmitter ON with motor kill switch (RC6_OPTION=31)
    - Safety cable attached
    - Clear the cage area before running
    - Ctrl+C will immediately cut throttle and attempt disarm

Usage:
    python3 cage_test_rc_override.py
    python3 cage_test_rc_override.py --throttle-pct 50 --hover-time 5
    python3 cage_test_rc_override.py --throttle-pct 30 --hover-time 3  (conservative first test)
"""

import argparse
import logging
import time
import sys
import threading

from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONNECTION_STRING = "/dev/ttyAMA0"
BAUD_RATE = 57600

# RC PWM values
RC_MIN = 1000       # stick minimum
RC_MAX = 2000       # stick maximum
RC_CENTER = 1500    # stick center (neutral roll/pitch/yaw)
RC_THROTTLE_MIN = 1000  # throttle at bottom

# Defaults
DEFAULT_THROTTLE_PCT = 50    # percent of throttle (0-100)
DEFAULT_HOVER_TIME_S = 6.0
MAX_FLIGHT_TIME_S = 25.0

# Timing
RC_OVERRIDE_HZ = 20          # send rate — ArduPilot expects continuous input
RAMP_STEP_PCT = 5            # throttle ramp increment (percent)
RAMP_INTERVAL_S = 0.4        # time between ramp steps
DESCENT_TIME_S = 3.0         # time to descend before disarm
POST_ARM_DELAY_S = 2.0

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
# Helpers
# ---------------------------------------------------------------------------

def pct_to_pwm(throttle_pct: float) -> int:
    """Convert throttle percentage (0-100) to PWM value (1000-2000)."""
    return int(RC_THROTTLE_MIN + (throttle_pct / 100.0) * (RC_MAX - RC_MIN))


def send_rc_override(mav_conn, throttle_pwm: int):
    """
    Send RC_OVERRIDE for all 8 channels.
    CH1=Roll(center), CH2=Pitch(center), CH3=Throttle, CH4=Yaw(center)
    CH5-8 = 0 (no override)
    """
    mav_conn.mav.rc_channels_override_send(
        mav_conn.target_system,
        mav_conn.target_component,
        RC_CENTER,      # CH1 Roll
        RC_CENTER,      # CH2 Pitch
        throttle_pwm,   # CH3 Throttle
        RC_CENTER,      # CH4 Yaw
        0, 0, 0, 0      # CH5-8 no override
    )


class RCOverrideThread:
    """
    Background thread that continuously sends RC_OVERRIDE at a fixed rate.
    ArduPilot will trigger failsafe if RC input stops.
    """
    def __init__(self, mav_conn):
        self.mav_conn = mav_conn
        self.throttle_pwm = RC_THROTTLE_MIN
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        # Send a few idle commands to ensure motors stop
        for _ in range(10):
            send_rc_override(self.mav_conn, RC_THROTTLE_MIN)
            time.sleep(0.05)

    def set_throttle_pwm(self, pwm: int):
        self.throttle_pwm = max(RC_MIN, min(RC_MAX, pwm))

    def set_throttle_pct(self, pct: float):
        self.set_throttle_pwm(pct_to_pwm(pct))

    def _loop(self):
        interval = 1.0 / RC_OVERRIDE_HZ
        while self.running:
            try:
                send_rc_override(self.mav_conn, self.throttle_pwm)
            except Exception as e:
                log.warning(f"RC override send error: {e}")
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Connection & Arming
# ---------------------------------------------------------------------------

def connect_and_wait(port: str, baud: int) -> mavutil.mavlink_connection:
    """Connect to flight controller and wait for heartbeat."""
    log.info(f"Connecting to {port} @ {baud}...")
    mav_conn = mavutil.mavlink_connection(port, baud=baud)

    log.info("Waiting for heartbeat...")
    mav_conn.wait_heartbeat(timeout=15)
    log.info(
        f"Connected — system {mav_conn.target_system}, "
        f"component {mav_conn.target_component}"
    )
    return mav_conn


def arm(mav_conn) -> bool:
    """Send arm command and wait for confirmation."""
    log.info("Sending ARM command...")
    mav_conn.arducopter_arm()

    # Wait for arm confirmation
    start = time.monotonic()
    while time.monotonic() - start < 10:
        msg = mav_conn.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg:
            armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if armed:
                log.info("ARMED successfully")
                return True
    log.error("Arming failed — timed out waiting for armed confirmation")
    return False


def disarm(mav_conn):
    """Send disarm command."""
    log.info("Sending DISARM command...")
    mav_conn.arducopter_disarm()
    time.sleep(1.0)

    # Check if disarmed
    msg = mav_conn.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
    if msg:
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        if not armed:
            log.info("DISARMED successfully")
            return
    log.warning("Disarm may not have succeeded — check visually")


def force_disarm(mav_conn):
    """Force disarm (MAV_CMD_COMPONENT_ARM_DISARM with force flag)."""
    log.info("FORCE DISARM...")
    mav_conn.mav.command_long_send(
        mav_conn.target_system,
        mav_conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,     # confirmation
        0,     # 0=disarm
        21196, # force flag
        0, 0, 0, 0, 0
    )
    time.sleep(1.0)

# ---------------------------------------------------------------------------
# Main Test Sequence
# ---------------------------------------------------------------------------

def run_test(throttle_pct: float, hover_time: float):
    log.info("=" * 60)
    log.info("QuackOps Cage Test — RC OVERRIDE FLIGHT")
    log.info(f"  Throttle       : {throttle_pct}% ({pct_to_pwm(throttle_pct)} PWM)")
    log.info(f"  Hover time     : {hover_time}s")
    log.info(f"  Hard timeout   : {MAX_FLIGHT_TIME_S}s")
    log.info(f"  Override rate  : {RC_OVERRIDE_HZ} Hz")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Connect
    # ------------------------------------------------------------------
    log.info("[1/6] Connecting...")
    mav_conn = connect_and_wait(CONNECTION_STRING, BAUD_RATE)

    # ------------------------------------------------------------------
    # Step 2 — Start RC override thread at idle
    # ------------------------------------------------------------------
    log.info("[2/6] Starting RC override stream (idle throttle)...")
    rc_thread = RCOverrideThread(mav_conn)
    rc_thread.set_throttle_pct(0)
    rc_thread.start()

    # Let a few override messages flow before arming
    time.sleep(1.0)

    try:
        # --------------------------------------------------------------
        # Step 3 — Arm
        # --------------------------------------------------------------
        log.info("[3/6] Arming motors...")
        log.info(">>> STAND CLEAR OF THE DRONE <<<")
        time.sleep(3.0)

        if not arm(mav_conn):
            log.error(
                "Check ARMING_CHECK=0, GPS_TYPE=0, FS_THR_ENABLE=0, "
                "and EK3_SRC params."
            )
            rc_thread.stop()
            return

        time.sleep(POST_ARM_DELAY_S)
        flight_start = time.monotonic()

        # --------------------------------------------------------------
        # Step 4 — Ramp throttle up
        # --------------------------------------------------------------
        log.info(f"[4/6] Ramping throttle to {throttle_pct}%...")

        current_pct = 0.0
        while current_pct < throttle_pct:
            if time.monotonic() - flight_start > MAX_FLIGHT_TIME_S:
                log.warning("SAFETY TIMEOUT during ramp")
                break

            current_pct = min(current_pct + RAMP_STEP_PCT, throttle_pct)
            rc_thread.set_throttle_pct(current_pct)
            log.info(
                f"  Throttle: {current_pct:.0f}% "
                f"({pct_to_pwm(current_pct)} PWM)"
            )
            time.sleep(RAMP_INTERVAL_S)

        log.info(f"Throttle at target: {throttle_pct}%")

        # --------------------------------------------------------------
        # Step 5 — Hover
        # --------------------------------------------------------------
        log.info(f"[5/6] Holding throttle for {hover_time}s...")

        hover_start = time.monotonic()
        while time.monotonic() - hover_start < hover_time:
            if time.monotonic() - flight_start > MAX_FLIGHT_TIME_S:
                log.warning("SAFETY TIMEOUT during hover")
                break

            elapsed = time.monotonic() - hover_start
            log.info(
                f"  Holding | {elapsed:.0f}/{hover_time:.0f}s | "
                f"Throttle: {throttle_pct}%"
            )
            time.sleep(1.0)

        # --------------------------------------------------------------
        # Step 6 — Descend and disarm
        # --------------------------------------------------------------
        log.info("[6/6] Cutting throttle and disarming...")

        # Ramp down
        rc_thread.set_throttle_pct(throttle_pct * 0.5)
        log.info(f"Throttle reduced to {throttle_pct * 0.5:.0f}%")
        time.sleep(DESCENT_TIME_S / 2)

        # Idle
        rc_thread.set_throttle_pct(0)
        log.info("Throttle at idle")
        time.sleep(DESCENT_TIME_S / 2)

        # Stop override and disarm
        rc_thread.stop()
        time.sleep(0.5)
        disarm(mav_conn)

    except KeyboardInterrupt:
        log.info("CTRL+C — EMERGENCY STOP")
        rc_thread.set_throttle_pct(0)
        rc_thread.stop()
        time.sleep(0.3)
        force_disarm(mav_conn)
        sys.exit(0)

    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        rc_thread.set_throttle_pct(0)
        rc_thread.stop()
        time.sleep(0.3)
        force_disarm(mav_conn)
        sys.exit(1)

    log.info("=" * 60)
    log.info("QuackOps Cage Test — COMPLETE")
    log.info("=" * 60)

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QuackOps cage test: RC override flight (ArduPilot)"
    )
    parser.add_argument(
        "--throttle-pct", type=float, default=DEFAULT_THROTTLE_PCT,
        help=f"Throttle percentage 0-100 (default: {DEFAULT_THROTTLE_PCT})"
    )
    parser.add_argument(
        "--hover-time", type=float, default=DEFAULT_HOVER_TIME_S,
        help=f"Hover duration in seconds (default: {DEFAULT_HOVER_TIME_S})"
    )
    args = parser.parse_args()

    # Safety clamp
    if args.throttle_pct > 80:
        log.warning(
            f"Throttle {args.throttle_pct}% capped at 80% for cage safety."
        )
        args.throttle_pct = 80

    if args.throttle_pct < 0:
        args.throttle_pct = 0

    if args.hover_time > 20:
        log.warning("Hover time capped at 20s.")
        args.hover_time = 20.0

    try:
        run_test(args.throttle_pct, args.hover_time)
    except Exception as e:
        log.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()