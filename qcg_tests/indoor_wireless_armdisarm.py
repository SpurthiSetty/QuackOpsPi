#!/usr/bin/env python3
"""
QuackOps — Phase 3: Wireless Arm/Disarm Test (pymavlink over MAVProxy UDP)

Architecture:
    Pixhawk ── UART ── Pi ── MAVProxy ── UDP 14551 ── this script
                                      └── UDP 14550 ── QGC (laptop)

Connects to the Pi's MAVProxy UDP endpoint, sets STABILIZE mode, sends ARM,
holds for a few seconds, then sends DISARM. Surfaces every STATUSTEXT message
so pre-arm rejection reasons are visible in the log.

Pre-flight checklist:
  - PROPS OFF
  - LiPo connected to Pixhawk
  - USB disconnected from laptop (test the wireless path cleanly)
  - MAVProxy running on Pi with --out=udpout:127.0.0.1:14551
  - QGC observing in parallel via UDP 14550
  - ARMING_CHECK=0 set on FC (bench-only)
  - FRAME_CLASS=1, FRAME_TYPE=1 set on FC

Usage (on the Pi, in the venv, in a SECOND SSH session — leave MAVProxy
running in the first):
    cd ~/SeniorD/QuackOpsPi
    source venv/bin/activate
    python3 tests/hardware/indoor_tests/wireless_arm_disarm.py
"""

import logging
import time

from pymavlink import mavutil

# ── Config ────────────────────────────────────────────────────────────
CONNECTION_STRING = "udpin:127.0.0.1:14551"   # MAVProxy pushes here
TARGET_SYSTEM = 1                              # ArduPilot autopilot sysid
TARGET_COMPONENT = 1                           # Autopilot component (NOT 0)
HOLD_ARMED_SECONDS = 5
ACK_TIMEOUT = 5.0
MODE_NAME = "STABILIZE"

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qps.wireless_arm")


# ── Helpers ───────────────────────────────────────────────────────────

def drain_messages(conn, duration_s, msg_types=None):
    """Drain incoming messages for `duration_s`, logging STATUSTEXT.

    Filtering by msg_types lets us call recv_match efficiently when we
    only care about specific messages.
    """
    end = time.time() + duration_s
    while time.time() < end:
        msg = conn.recv_match(type=msg_types, blocking=False)
        if msg is None:
            time.sleep(0.05)
            continue
        if msg.get_type() == "STATUSTEXT":
            text = msg.text.strip() if hasattr(msg, "text") else str(msg)
            if text:
                log.info(f"  [FC] {text}")


def wait_for_ack(conn, command, timeout):
    """Wait for COMMAND_ACK matching `command`. Logs STATUSTEXT meanwhile."""
    end = time.time() + timeout
    while time.time() < end:
        msg = conn.recv_match(
            type=["COMMAND_ACK", "STATUSTEXT"],
            blocking=False,
        )
        if msg is None:
            time.sleep(0.05)
            continue
        if msg.get_type() == "STATUSTEXT":
            text = msg.text.strip() if hasattr(msg, "text") else str(msg)
            if text:
                log.info(f"  [FC] {text}")
        elif msg.get_type() == "COMMAND_ACK" and msg.command == command:
            return msg
    return None


def get_armed_state(conn, timeout=2.0):
    """Read HEARTBEAT and return True if the SAFETY_ARMED bit is set."""
    end = time.time() + timeout
    while time.time() < end:
        msg = conn.recv_match(type="HEARTBEAT", blocking=False)
        if (
            msg is not None
            and msg.get_srcSystem() == TARGET_SYSTEM
            and msg.type != mavutil.mavlink.MAV_TYPE_GCS
        ):
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        time.sleep(0.05)
    return None


def set_mode(conn, mode_name):
    """Set flight mode by name and verify via heartbeat custom_mode.

    Uses MAV_CMD_DO_SET_MODE (more reliable on ArduCopter than SET_MODE).
    Filters out GCS heartbeats (compid=0) per past lessons.
    """
    mode_map = conn.mode_mapping()
    if mode_name not in mode_map:
        log.error(f"Unknown mode '{mode_name}'. Available: {list(mode_map.keys())}")
        return False
    target_mode = mode_map[mode_name]

    log.info(f"Switching to {mode_name} (custom_mode={target_mode})...")

    conn.mav.command_long_send(
        TARGET_SYSTEM,
        TARGET_COMPONENT,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,                                              # confirmation
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        target_mode,
        0, 0, 0, 0, 0,
    )

    end = time.time() + 5.0
    while time.time() < end:
        msg = conn.recv_match(type="HEARTBEAT", blocking=False)
        if msg is not None:
            # Skip GCS / MAVProxy heartbeats; only trust the autopilot's
            if (
                msg.get_srcSystem() == TARGET_SYSTEM
                and msg.type != mavutil.mavlink.MAV_TYPE_GCS
                and msg.custom_mode == target_mode
            ):
                log.info(f"  ✓ {mode_name} confirmed")
                return True
        time.sleep(0.05)

    log.error(f"  ✗ Mode change to {mode_name} not confirmed in 5s")
    return False


def send_arm_disarm(conn, arm: bool) -> bool:
    """Send MAV_CMD_COMPONENT_ARM_DISARM and return whether it was ACCEPTED.

    Hardcodes target_component=1 — MAVProxy passthrough reports compid=0 in
    heartbeats but the autopilot itself listens on compid=1.
    """
    action = "ARM" if arm else "DISARM"
    param1 = 1.0 if arm else 0.0

    log.info(f"Sending {action}...")
    conn.mav.command_long_send(
        TARGET_SYSTEM,
        TARGET_COMPONENT,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,                  # confirmation
        param1,             # 1 = arm, 0 = disarm
        0, 0, 0, 0, 0, 0,   # unused
    )

    ack = wait_for_ack(
        conn,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        timeout=ACK_TIMEOUT,
    )

    if ack is None:
        log.error(f"  ✗ {action} timed out — no COMMAND_ACK")
        return False

    result_names = {
        mavutil.mavlink.MAV_RESULT_ACCEPTED: "ACCEPTED",
        mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED: "TEMPORARILY_REJECTED",
        mavutil.mavlink.MAV_RESULT_DENIED: "DENIED",
        mavutil.mavlink.MAV_RESULT_UNSUPPORTED: "UNSUPPORTED",
        mavutil.mavlink.MAV_RESULT_FAILED: "FAILED",
        mavutil.mavlink.MAV_RESULT_IN_PROGRESS: "IN_PROGRESS",
    }
    name = result_names.get(ack.result, f"UNKNOWN({ack.result})")

    if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
        log.info(f"  ✓ {action} ACCEPTED")
        return True
    log.error(f"  ✗ {action} rejected: {name}")
    return False


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("QuackOps — Phase 3: Wireless Arm/Disarm Test")
    log.info(f"  Connection : {CONNECTION_STRING}")
    log.info(f"  Target     : sys={TARGET_SYSTEM} comp={TARGET_COMPONENT}")
    log.info(f"  Mode       : {MODE_NAME}")
    log.info(f"  Hold       : {HOLD_ARMED_SECONDS}s")
    log.info("=" * 60)

    log.info("[1/6] Connecting to MAVProxy via UDP...")
    conn = mavutil.mavlink_connection(CONNECTION_STRING)
    log.info("  Waiting for first heartbeat (10s timeout)...")
    hb = conn.wait_heartbeat(timeout=10)
    if hb is None:
        log.error("  ✗ No heartbeat in 10s. Is MAVProxy running with "
                  "--out=udpout:127.0.0.1:14551 ?")
        return
    log.info(f"  ✓ Heartbeat received — sysid={conn.target_system} "
             f"compid={conn.target_component}")
    log.info(f"  Using hardcoded target sys={TARGET_SYSTEM} comp={TARGET_COMPONENT}")

    log.info("[2/6] Draining boot messages (2s)...")
    drain_messages(conn, duration_s=2.0, msg_types=["STATUSTEXT"])

    log.info("[3/6] Setting flight mode...")
    if not set_mode(conn, MODE_NAME):
        log.error("Aborting — mode change failed")
        conn.close()
        return

    log.info("[4/6] Pre-arm: listening for any pre-arm errors (3s)...")
    drain_messages(conn, duration_s=3.0, msg_types=["STATUSTEXT"])

    armed_before = get_armed_state(conn)
    log.info(f"  Armed state before: {armed_before}")

    log.info(">>> STAND CLEAR — arming in 3 seconds <<<")
    time.sleep(3)

    log.info("[5/6] ARM")
    if not send_arm_disarm(conn, arm=True):
        log.error("Arm failed. Common fixes:")
        log.error("  - ARMING_CHECK should be 0 for indoor bench")
        log.error("  - FRAME_CLASS=1 and FRAME_TYPE=1 must be set")
        log.error("  - Battery voltage above BATT_LOW_VOLT threshold")
        log.error("  - Look at [FC] STATUSTEXT messages above for the reason")
        conn.close()
        return

    armed_after = get_armed_state(conn)
    log.info(f"  Armed state after ARM: {armed_after}")

    log.info(f"Holding armed for {HOLD_ARMED_SECONDS}s "
             f"(motors at MOT_SPIN_ARM idle)...")
    time.sleep(HOLD_ARMED_SECONDS)

    log.info("[6/6] DISARM")
    send_arm_disarm(conn, arm=False)

    armed_final = get_armed_state(conn)
    log.info(f"  Armed state final: {armed_final}")

    drain_messages(conn, duration_s=1.0, msg_types=["STATUSTEXT"])

    conn.close()
    log.info("=" * 60)
    log.info("Test complete")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
    except Exception:
        log.exception("Unexpected error")