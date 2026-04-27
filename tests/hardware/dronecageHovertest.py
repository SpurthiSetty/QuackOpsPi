#!/usr/bin/env python3
"""
QuackOps Cage Test — ALT_HOLD Hover Test
=========================================
Target:     ArduCopter 4.6.3 on Pixhawk 2.4.8 via TELEM1
Env:        Indoor cage, no GPS, ALT_HOLD mode
Pi:         Raspberry Pi 5 (serial /dev/ttyAMA0 @ 57600)

ALT_HOLD THROTTLE SEMANTICS:
    1500 PWM = hold current altitude (hover)
    >1500    = climb  (1650 = moderate climb)
    <1500    = descend (1350 = moderate descent)
    1000     = max descent / idle

SEQUENCE:
    1. Connect + start RC override thread at idle
    2. Switch to ALT_HOLD mode
    3. Arm
    4. Ramp throttle above 1500 to climb
    5. Set throttle to 1500 to hold altitude
    6. Hold for hover_time seconds
    7. Set throttle below 1500 to descend
    8. Set throttle to idle (1000)
    9. Disarm

REQUIRED PARAMETERS (set in Mission Planner BEFORE running):
    ARMING_CHECK    = 0
    GPS_TYPE        = 0
    FENCE_ENABLE    = 0
    FS_THR_ENABLE   = 0
    FS_GCS_ENABLE   = 0
    DISARM_DELAY    = 0
    EK3_SRC1_POSXY  = 0
    EK3_SRC1_VELXY  = 0
    EK3_SRC1_POSZ   = 1
    EK3_SRC1_VELZ   = 0
    EK3_SRC1_YAW    = 1

SAFETY:
    - RC transmitter ON with motor kill switch
    - Safety cable attached
    - Clear the cage area
    - Ctrl+C immediately cuts throttle and force disarms

Usage:
    python3 cage_test_alt_hold.py
    python3 cage_test_alt_hold.py --climb-pwm 1600 --hover-time 3  (conservative)
    python3 cage_test_alt_hold.py --climb-pwm 1700 --hover-time 5  (higher/longer)
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
RC_MIN = 1000
RC_MAX = 2000
RC_CENTER = 1500

# ALT_HOLD mode
ALT_HOLD_MODE = 2

# Defaults
DEFAULT_CLIMB_PWM = 1620       # moderate climb rate
DEFAULT_HOVER_TIME_S = 4.0
DEFAULT_CLIMB_TIME_S = 3.0     # seconds to climb before holding
DEFAULT_DESCEND_PWM = 1350     # moderate descent rate
DEFAULT_DESCEND_TIME_S = 4.0   # seconds to descend before idle
MAX_FLIGHT_TIME_S = 25.0       # hard safety timeout

# Timing
RC_OVERRIDE_HZ = 20
RAMP_STEP_PWM = 10             # PWM increment per ramp step
RAMP_INTERVAL_S = 0.1          # time between ramp steps
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
# RC Override Thread
# ---------------------------------------------------------------------------

class RCOverrideThread:
    """Background thread sending RC_OVERRIDE at a fixed rate."""

    def __init__(self, mav_conn):
        self.mav_conn = mav_conn
        self.lock = threading.Lock()
        self.throttle = RC_CENTER
        self.roll = RC_CENTER
        self.pitch = RC_CENTER
        self.yaw = RC_CENTER
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        # Send idle commands to ensure motors stop
        for _ in range(10):
            self.mav_conn.mav.rc_channels_override_send(
                self.mav_conn.target_system,
                self.mav_conn.target_component,
                RC_CENTER, RC_CENTER, RC_MIN, RC_CENTER,
                0, 0, 0, 0,
            )
            time.sleep(0.05)

    def set_throttle(self, pwm: int):
        with self.lock:
            self.throttle = max(RC_MIN, min(RC_MAX, pwm))

    def set_all(self, throttle=None, roll=None, pitch=None, yaw=None):
        with self.lock:
            if throttle is not None:
                self.throttle = max(RC_MIN, min(RC_MAX, throttle))
            if roll is not None:
                self.roll = max(RC_MIN, min(RC_MAX, roll))
            if pitch is not None:
                self.pitch = max(RC_MIN, min(RC_MAX, pitch))
            if yaw is not None:
                self.yaw = max(RC_MIN, min(RC_MAX, yaw))

    def _loop(self):
        interval = 1.0 / RC_OVERRIDE_HZ
        while self.running:
            with self.lock:
                t, r, p, y = self.throttle, self.roll, self.pitch, self.yaw
            try:
                self.mav_conn.mav.rc_channels_override_send(
                    self.mav_conn.target_system,
                    self.mav_conn.target_component,
                    r, p, t, y,
                    0, 0, 0, 0,
                )
            except Exception as e:
                log.warning(f"RC override send error: {e}")
            time.sleep(interval)

# ---------------------------------------------------------------------------
# Connection & Mode
# ---------------------------------------------------------------------------

def connect(port: str, baud: int):
    log.info(f"Connecting to {port} @ {baud}...")
    mav_conn = mavutil.mavlink_connection(port, baud=baud)
    log.info("Waiting for heartbeat...")
    mav_conn.wait_heartbeat(timeout=15)
    log.info(
        f"Connected — system {mav_conn.target_system}, "
        f"component {mav_conn.target_component}"
    )
    return mav_conn


def set_mode_alt_hold(mav_conn) -> bool:
    """Switch to ALT_HOLD mode and verify via heartbeat."""
    log.info("Switching to ALT_HOLD mode...")
    mav_conn.mav.command_long_send(
        mav_conn.target_system,
        mav_conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        ALT_HOLD_MODE,
        0, 0, 0, 0, 0,
    )

    # Verify mode switch via heartbeat
    start = time.monotonic()
    while time.monotonic() - start < 10:
        msg = mav_conn.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.get_srcComponent() != 0:
            if msg.custom_mode == ALT_HOLD_MODE:
                log.info("ALT_HOLD mode confirmed")
                return True
    log.error("Failed to switch to ALT_HOLD mode")
    return False

# ---------------------------------------------------------------------------
# Arm / Disarm
# ---------------------------------------------------------------------------

def arm(mav_conn) -> bool:
    log.info("Sending ARM command...")
    mav_conn.arducopter_arm()
    start = time.monotonic()
    while time.monotonic() - start < 10:
        msg = mav_conn.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            log.info("ARMED successfully")
            return True
    log.error("Arming failed — check parameters")
    return False


def disarm(mav_conn):
    log.info("Sending DISARM command...")
    mav_conn.arducopter_disarm()
    time.sleep(1.0)
    msg = mav_conn.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
    if msg and not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
        log.info("DISARMED successfully")
    else:
        log.warning("Disarm may not have succeeded — check visually")


def force_disarm(mav_conn):
    log.warning("FORCE DISARM")
    mav_conn.mav.command_long_send(
        mav_conn.target_system,
        mav_conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0,
    )
    time.sleep(1.0)

# ---------------------------------------------------------------------------
# Main Test Sequence
# ---------------------------------------------------------------------------

def run_test(climb_pwm: int, hover_time: float, climb_time: float,
             descend_pwm: int, descend_time: float):
    log.info("=" * 60)
    log.info("QuackOps Cage Test — ALT_HOLD HOVER")
    log.info(f"  Climb PWM      : {climb_pwm}")
    log.info(f"  Climb time     : {climb_time}s")
    log.info(f"  Hover time     : {hover_time}s")
    log.info(f"  Descend PWM    : {descend_pwm}")
    log.info(f"  Descend time   : {descend_time}s")
    log.info(f"  Hard timeout   : {MAX_FLIGHT_TIME_S}s")
    log.info("=" * 60)

    # ── 1. Connect ────────────────────────────────────────────────
    log.info("[1/7] Connecting...")
    mav_conn = connect(CONNECTION_STRING, BAUD_RATE)

    # ── 2. Start RC override at center (ALT_HOLD neutral) ────────
    log.info("[2/7] Starting RC override stream (center/neutral)...")
    rc = RCOverrideThread(mav_conn)
    rc.set_all(throttle=RC_CENTER, roll=RC_CENTER, pitch=RC_CENTER, yaw=RC_CENTER)
    rc.start()
    time.sleep(1.0)

    try:
        # ── 3. Switch to ALT_HOLD ────────────────────────────────
        log.info("[3/7] Setting ALT_HOLD mode...")
        if not set_mode_alt_hold(mav_conn):
            rc.stop()
            return

        # ── 4. Arm ───────────────────────────────────────────────
        log.info("[4/7] Arming motors...")
        log.info(">>> STAND CLEAR OF THE DRONE <<<")
        time.sleep(3.0)

        if not arm(mav_conn):
            rc.stop()
            return

        time.sleep(POST_ARM_DELAY_S)
        flight_start = time.monotonic()

        # ── 5. Climb — ramp throttle above 1500 ──────────────────
        log.info(f"[5/7] Climbing — ramping throttle to {climb_pwm} PWM...")

        current_pwm = RC_CENTER
        while current_pwm < climb_pwm:
            if time.monotonic() - flight_start > MAX_FLIGHT_TIME_S:
                log.warning("SAFETY TIMEOUT during climb")
                break
            current_pwm = min(current_pwm + RAMP_STEP_PWM, climb_pwm)
            rc.set_throttle(current_pwm)
            log.info(f"  Throttle: {current_pwm} PWM")
            time.sleep(RAMP_INTERVAL_S)

        log.info(f"Climbing at {climb_pwm} PWM for {climb_time}s...")
        climb_start = time.monotonic()
        while time.monotonic() - climb_start < climb_time:
            if time.monotonic() - flight_start > MAX_FLIGHT_TIME_S:
                log.warning("SAFETY TIMEOUT during climb hold")
                break
            elapsed = time.monotonic() - climb_start
            log.info(f"  Climbing | {elapsed:.1f}/{climb_time:.1f}s")
            time.sleep(0.5)

        # ── 6. Hover — throttle at 1500 (hold altitude) ──────────
        log.info(f"[6/7] Hovering — throttle at {RC_CENTER} PWM for {hover_time}s...")
        rc.set_throttle(RC_CENTER)

        hover_start = time.monotonic()
        while time.monotonic() - hover_start < hover_time:
            if time.monotonic() - flight_start > MAX_FLIGHT_TIME_S:
                log.warning("SAFETY TIMEOUT during hover")
                break
            elapsed = time.monotonic() - hover_start
            log.info(f"  Holding altitude | {elapsed:.1f}/{hover_time:.1f}s")
            time.sleep(1.0)

        # ── 7. Descend and disarm ─────────────────────────────────
        log.info(f"[7/7] Descending — throttle to {descend_pwm} PWM...")
        rc.set_throttle(descend_pwm)

        descend_start = time.monotonic()
        while time.monotonic() - descend_start < descend_time:
            if time.monotonic() - flight_start > MAX_FLIGHT_TIME_S:
                log.warning("SAFETY TIMEOUT during descent")
                break
            elapsed = time.monotonic() - descend_start
            log.info(f"  Descending | {elapsed:.1f}/{descend_time:.1f}s")
            time.sleep(0.5)

        # Idle throttle
        log.info("Throttle to idle (1000 PWM)")
        rc.set_throttle(RC_MIN)
        time.sleep(1.0)

        # Stop override and disarm
        rc.stop()
        time.sleep(0.5)
        disarm(mav_conn)

    except KeyboardInterrupt:
        log.info("CTRL+C — EMERGENCY STOP")
        rc.set_throttle(RC_MIN)
        rc.stop()
        time.sleep(0.3)
        force_disarm(mav_conn)
        sys.exit(0)

    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        rc.set_throttle(RC_MIN)
        rc.stop()
        time.sleep(0.3)
        force_disarm(mav_conn)
        sys.exit(1)

    total = time.monotonic() - flight_start
    log.info("=" * 60)
    log.info(f"QuackOps Cage Test — COMPLETE ({total:.1f}s total flight)")
    log.info("=" * 60)

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QuackOps cage test: ALT_HOLD hover (no GPS)"
    )
    parser.add_argument(
        "--climb-pwm", type=int, default=DEFAULT_CLIMB_PWM,
        help=f"Throttle PWM for climbing (default: {DEFAULT_CLIMB_PWM})"
    )
    parser.add_argument(
        "--climb-time", type=float, default=DEFAULT_CLIMB_TIME_S,
        help=f"Seconds to climb before holding (default: {DEFAULT_CLIMB_TIME_S})"
    )
    parser.add_argument(
        "--hover-time", type=float, default=DEFAULT_HOVER_TIME_S,
        help=f"Seconds to hold altitude (default: {DEFAULT_HOVER_TIME_S})"
    )
    parser.add_argument(
        "--descend-pwm", type=int, default=DEFAULT_DESCEND_PWM,
        help=f"Throttle PWM for descending (default: {DEFAULT_DESCEND_PWM})"
    )
    parser.add_argument(
        "--descend-time", type=float, default=DEFAULT_DESCEND_TIME_S,
        help=f"Seconds to descend before idle (default: {DEFAULT_DESCEND_TIME_S})"
    )
    args = parser.parse_args()

    # Safety clamps
    if args.climb_pwm > 1750:
        log.warning(f"Climb PWM {args.climb_pwm} capped at 1750 for cage safety")
        args.climb_pwm = 1750
    if args.climb_pwm < 1520:
        log.warning(f"Climb PWM {args.climb_pwm} raised to 1520 (below center won't climb)")
        args.climb_pwm = 1520
    if args.hover_time > 20:
        log.warning("Hover time capped at 20s")
        args.hover_time = 20.0

    try:
        run_test(
            climb_pwm=args.climb_pwm,
            hover_time=args.hover_time,
            climb_time=args.climb_time,
            descend_pwm=args.descend_pwm,
            descend_time=args.descend_time,
        )
    except Exception as e:
        log.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()