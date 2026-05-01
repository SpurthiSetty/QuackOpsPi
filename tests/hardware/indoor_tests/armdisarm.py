#!/usr/bin/env python3
"""
QuackOps — Bench Arm/Disarm Test (pymavlink-based)

Connects to the FC over TELEM1, sets STABILIZE mode, sends ARM, waits for
result, holds armed, sends DISARM. Surfaces all STATUSTEXT messages so
you can see exactly why anything fails.

Pre-flight:
  - PROPS OFF
  - LiPo connected
  - ARMING_CHECK=0 set in QGC if testing indoors without GPS
  - No other process holding /dev/ttyAMA0 (kill MAVProxy, etc.)

Usage:
  python3 armdisarm.py
"""

import time
import logging
from pymavlink import mavutil

# --- Config ---
CONNECTION_STRING = "/dev/ttyAMA0"
BAUD = 57600
HOLD_ARMED_SECONDS = 5
ARM_ACK_TIMEOUT = 5.0
MODE_NAME = "STABILIZE"  # safest indoor mode

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qps.arm_disarm")


def drain_statustext(conn, duration=0.5, prefix="FC"):
    """Print all STATUSTEXT messages received within `duration` seconds."""
    end = time.time() + duration
    while time.time() < end:
        msg = conn.recv_match(type="STATUSTEXT", blocking=False)
        if msg is None:
            time.sleep(0.05)
            continue
        text = msg.text.strip() if hasattr(msg, "text") else str(msg)
        log.info(f"  [{prefix}] {text}")


def wait_for_ack(conn, command, timeout):
    """Wait for COMMAND_ACK matching `command`, also surfacing STATUSTEXT."""
    end = time.time() + timeout
    while time.time() < end:
        msg = conn.recv_match(type=["COMMAND_ACK", "STATUSTEXT"], blocking=False)
        if msg is None:
            time.sleep(0.05)
            continue
        if msg.get_type() == "STATUSTEXT":
            text = msg.text.strip() if hasattr(msg, "text") else str(msg)
            log.info(f"  [FC] {text}")
        elif msg.get_type() == "COMMAND_ACK" and msg.command == command:
            return msg
    return None


def get_armed_state(conn, timeout=2.0):
    """Read HEARTBEAT and return whether MAV_MODE_FLAG_SAFETY_ARMED bit is set."""
    end = time.time() + timeout
    while time.time() < end:
        msg = conn.recv_match(type="HEARTBEAT", blocking=False)
        if (msg is not None
                and msg.get_srcSystem() == conn.target_system
                and msg.type != mavutil.mavlink.MAV_TYPE_GCS):
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            return armed
        time.sleep(0.05)
    return None


def set_mode(conn, mode_name):
    """Set flight mode by name and verify via heartbeat custom_mode."""
    mode_map = conn.mode_mapping()
    if mode_name not in mode_map:
        log.error(f"Unknown mode '{mode_name}'. Available: {list(mode_map.keys())}")
        return False
    target_mode = mode_map[mode_name]

    log.info(f"Switching to {mode_name} mode (custom_mode={target_mode})...")
    
    # Method 1: SET_MODE message
    conn.mav.set_mode_send(
        conn.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        target_mode,
    )
    
    # Method 2: Also send MAV_CMD_DO_SET_MODE as a backup (more reliable on ArduCopter)
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        target_mode,
        0, 0, 0, 0, 0,
    )

    # Diagnostic: log every heartbeat we receive
    end = time.time() + 5.0
    heartbeats_seen = 0
    while time.time() < end:
        msg = conn.recv_match(type="HEARTBEAT", blocking=False)
        if msg is not None:
            heartbeats_seen += 1
            log.info(f"  HB: sys={msg.get_srcSystem()} comp={msg.get_srcComponent()} "
                     f"type={msg.type} custom_mode={msg.custom_mode} "
                     f"base_mode={msg.base_mode}")
            if (msg.get_srcSystem() == conn.target_system
                    and msg.type != mavutil.mavlink.MAV_TYPE_GCS
                    and msg.custom_mode == target_mode):
                log.info(f"✓ {mode_name} confirmed")
                return True
        time.sleep(0.05)
    log.error(f"✗ Mode change to {mode_name} not confirmed (saw {heartbeats_seen} heartbeats)")
    return False
"""
Add this function to:
  ~/SeniorD/QuackOpsPi/tests/hardware/indoor_tests/armdisarm.py

Best location: right after set_mode() and before main(),
i.e. around line 122 in your file.
"""

def send_arm_disarm(conn, arm: bool) -> bool:
    """Send MAV_CMD_COMPONENT_ARM_DISARM and verify via COMMAND_ACK.

    Returns True if the FC accepted the arm/disarm command, False otherwise.
    Surfaces all STATUSTEXT messages received during the wait for ACK so
    pre-arm rejection reasons are visible in the log.
    """
    action = "ARM" if arm else "DISARM"
    param1 = 1.0 if arm else 0.0

    log.info(f"Sending {action} command...")
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,           # confirmation
        param1,      # 1 = arm, 0 = disarm
        0, 0, 0, 0, 0, 0,  # unused params
    )

    ack = wait_for_ack(
        conn,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        timeout=ARM_ACK_TIMEOUT,
    )

    if ack is None:
        log.error(f"  ✗ {action} timed out — no COMMAND_ACK received")
        return False

    result_names = {
        mavutil.mavlink.MAV_RESULT_ACCEPTED: "ACCEPTED",
        mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED: "TEMPORARILY_REJECTED",
        mavutil.mavlink.MAV_RESULT_DENIED: "DENIED",
        mavutil.mavlink.MAV_RESULT_UNSUPPORTED: "UNSUPPORTED",
        mavutil.mavlink.MAV_RESULT_FAILED: "FAILED",
        mavutil.mavlink.MAV_RESULT_IN_PROGRESS: "IN_PROGRESS",
    }
    result_name = result_names.get(ack.result, f"UNKNOWN({ack.result})")

    if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
        log.info(f"  ✓ {action} ACCEPTED")
        return True
    else:
        log.error(f"  ✗ {action} rejected: {result_name}")
        return False

def main():
    log.info("=" * 60)
    log.info("QuackOps — Bench Arm/Disarm Test (pymavlink)")
    log.info(f"  Connection : {CONNECTION_STRING} @ {BAUD}")
    log.info(f"  Target mode: {MODE_NAME}")
    log.info(f"  Hold time  : {HOLD_ARMED_SECONDS}s")
    log.info("=" * 60)

    log.info("[1/6] Connecting...")
    conn = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD)
    conn.wait_heartbeat()
    log.info(f"✓ Connected — sysid={conn.target_system}, "
             f"compid={conn.target_component}")

    log.info("[2/6] Draining boot messages (2s)...")
    drain_statustext(conn, duration=2.0)

    log.info("[3/6] Setting flight mode...")
    if not set_mode(conn, MODE_NAME):
        log.error("Aborting — mode change failed")
        conn.close()
        return

    log.info("[4/6] Pre-arm: listening for any pre-arm errors (3s)...")
    drain_statustext(conn, duration=3.0)

    armed_before = get_armed_state(conn)
    log.info(f"  Armed state before: {armed_before}")

    log.info(">>> STAND CLEAR — arming in 3 seconds <<<")
    time.sleep(3)

    log.info("[5/6] ARM")
    if not send_arm_disarm(conn, arm=True):
        log.error("Arm failed. See [FC] messages above for reason.")
        log.info("Common fixes:")
        log.info("  - Set ARMING_CHECK=0 in QGC for indoor bench testing")
        log.info("  - Set FS_GCS_ENABLE=0 if 'GCS failsafe on' appears")
        log.info("  - Set FENCE_ENABLE=0 if 'Fence requires position' appears")
        conn.close()
        return

    armed_after = get_armed_state(conn)
    log.info(f"  Armed state after ARM: {armed_after}")

    log.info(f"Holding armed for {HOLD_ARMED_SECONDS}s...")
    time.sleep(HOLD_ARMED_SECONDS)

    log.info("[6/6] DISARM")
    send_arm_disarm(conn, arm=False)

    armed_final = get_armed_state(conn)
    log.info(f"  Armed state final: {armed_final}")

    drain_statustext(conn, duration=1.0)

    conn.close()
    log.info("=" * 60)
    log.info("Test complete")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
    except Exception as e:
        log.exception(f"Unexpected error: {e}")