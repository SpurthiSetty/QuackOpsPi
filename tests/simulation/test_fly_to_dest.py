"""
QuackOps SITL — Fly to Destination (pymavlink)
================================================
Single pymavlink connection to ArduCopter SITL. No port conflicts.

    Connect → Upload Mission → Arm → AUTO mode → Fly → RTL → Land

Keep Mission Planner connected (it holds port 5760).
pymavlink connects on port 5763.

Setting the Home Location:
    Mission Planner → SIMULATION tab → drag "Home Location" to
    Stevens campus (40.7453, -74.0256) → click Multirotor.

Usage:
    python test_fly_to_destination.py
"""

import logging
import math
import time

from pymavlink import mavutil, mavwp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quackops.fly_to")

# ── Configuration ─────────────────────────────────────────────────────

CONNECTION = "tcp:localhost:5763"

# ArduCopter mode IDs
MODE_STABILIZE = 0
MODE_AUTO = 3
MODE_GUIDED = 4
MODE_RTL = 6
MODE_LAND = 9

# Flight parameters
TAKEOFF_ALT_M = 10.0
CRUISE_SPEED_M_S = 5.0

# ── Destination ───────────────────────────────────────────────────────
# Stevens campus home: 40.7453, -74.0256

DESTINATION_LAT = 40.7440
DESTINATION_LON = -74.0230
DESTINATION_ALT_M = 10.0


def fly_to_destination(
    connection: str,
    dest_lat: float,
    dest_lon: float,
    dest_alt_m: float = 10.0,
    cruise_speed: float = 5.0,
) -> None:
    """Connect, upload mission, arm, fly to destination, RTL, land."""

    # ── 1. Connect ────────────────────────────────────────────────────
    logger.info("Connecting to %s ...", connection)
    mav = mavutil.mavlink_connection(connection)
    mav.wait_heartbeat()
    logger.info(
        "Connected! (sysid=%d, compid=%d)",
        mav.target_system, mav.target_component,
    )

    # ── 2. Request data streams ───────────────────────────────────────
    logger.info("Requesting data streams...")
    mav.mav.request_data_stream_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        4,  # 4 Hz
        1,  # start
    )

    # ── 3. Wait for GPS lock ──────────────────────────────────────────
    logger.info("Waiting for GPS lock...")
    home_lat, home_lon = _wait_for_gps(mav)
    logger.info("Home: lat=%.6f  lon=%.6f", home_lat, home_lon)

    distance_m = _haversine_m(home_lat, home_lon, dest_lat, dest_lon)
    logger.info(
        "Destination: lat=%.6f  lon=%.6f  (%.0fm away)",
        dest_lat, dest_lon, distance_m,
    )

    # ── 4. Wait for EKF to stabilize ─────────────────────────────────
    logger.info("Waiting for EKF to stabilize...")
    _wait_for_ekf(mav)
    logger.info("EKF ready!")

    # ── 5. Upload mission ─────────────────────────────────────────────
    logger.info("Uploading mission...")
    _upload_mission(mav, dest_lat, dest_lon, dest_alt_m, cruise_speed)
    logger.info("Mission uploaded")

    # ── 6. Disable pre-arm checks (SITL only) ─────────────────────────
    logger.info("Disabling pre-arm checks for SITL...")
    mav.mav.param_set_send(
        mav.target_system,
        mav.target_component,
        b'ARMING_CHECK',
        0,
        mavutil.mavlink.MAV_PARAM_TYPE_INT32,
    )
    ack = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=5)
    if ack:
        logger.info("  ARMING_CHECK set to %d", int(ack.param_value))
    time.sleep(2)

    # ── 7. Switch to GUIDED, arm, takeoff ─────────────────────────────
    logger.info("Switching to GUIDED mode...")
    _set_mode_verified(mav, MODE_GUIDED)

    logger.info("Arming...")
    mav.arducopter_arm()
    mav.motors_armed_wait()
    logger.info("Armed! (confirmed from heartbeat)")

    # Command takeoff in GUIDED mode
    logger.info("Commanding takeoff to %.0fm...", dest_alt_m)
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,        # confirmation
        0, 0, 0, 0, 0, 0,
        dest_alt_m,  # param7 = altitude
    )

    # Wait until airborne
    logger.info("Climbing...")
    while True:
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        if msg:
            alt = msg.relative_alt / 1000.0
            if alt >= dest_alt_m * 0.90:
                logger.info("  Reached %.1fm!", alt)
                break
            logger.info("  Alt: %.1fm", alt)
        time.sleep(0.5)

    # ── 8. Switch to AUTO (starts waypoint mission) ───────────────────
    logger.info("Switching to AUTO mode...")
    _set_mode_verified(mav, MODE_AUTO)
    # ── 9. Monitor flight ─────────────────────────────────────────────
    logger.info("Flying to destination...")
    _monitor_mission(mav, dest_lat, dest_lon, dest_alt_m)

    # ── 10. Loiter at destination ─────────────────────────────────────
    logger.info("At destination — loitering for 5 seconds...")
    time.sleep(5)

    # ── 11. RTL ───────────────────────────────────────────────────────
    logger.info("Returning to launch...")
    _set_mode_verified(mav, MODE_RTL)

    # Wait for landing
    logger.info("Waiting for landing...")
    _wait_for_landing(mav)
    logger.info("Landed!")

    # ── 12. Disarm ────────────────────────────────────────────────────
    logger.info("Disarming...")
    mav.arducopter_disarm()
    mav.motors_disarmed_wait()
    logger.info("Disarmed!")

    logger.info("═══ FLIGHT COMPLETE ═══")
    logger.info("Distance flown: ~%.0fm round trip", distance_m * 2)


# ── Mode switching ────────────────────────────────────────────────────

def _set_mode_verified(mav, mode_id: int, timeout: float = 10.0) -> bool:
    """Switch ArduCopter mode and verify via heartbeat.

    Uses MAV_CMD_DO_SET_MODE (produces ACKs) and confirms the mode
    actually changed by reading heartbeat custom_mode field.

    Args:
        mav:      pymavlink connection.
        mode_id:  ArduCopter mode number (3=AUTO, 6=RTL, etc.)
        timeout:  Seconds to wait for confirmation.

    Returns:
        True if mode confirmed, False on timeout.
    """
    mode_names = {0: "STABILIZE", 3: "AUTO", 4: "GUIDED", 6: "RTL", 9: "LAND"}
    mode_name = mode_names.get(mode_id, str(mode_id))

    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,  # confirmation
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
        0, 0, 0, 0, 0,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if msg and msg.custom_mode == mode_id:
            logger.info("  %s mode confirmed!", mode_name)
            return True
        elif msg:
            current = mode_names.get(msg.custom_mode, str(msg.custom_mode))
            logger.info("  Current mode: %s (waiting for %s)", current, mode_name)

    logger.error("  Failed to switch to %s mode (timeout)", mode_name)
    return False


# ── EKF readiness ─────────────────────────────────────────────────────

def _wait_for_ekf(mav, timeout: float = 30.0) -> None:
    """Wait until EKF has a valid position estimate.

    Checks SYS_STATUS for EKF health, or falls back to a
    time-based wait if the message isn't available.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = mav.recv_match(type="SYS_STATUS", blocking=True, timeout=2)
        if msg:
            # Check AHRS and GPS health bits in onboard_control_sensors_health
            ahrs_healthy = msg.onboard_control_sensors_health & mavutil.mavlink.MAV_SYS_STATUS_AHRS
            gps_healthy = msg.onboard_control_sensors_health & mavutil.mavlink.MAV_SYS_STATUS_SENSOR_GPS
            if ahrs_healthy and gps_healthy:
                return

    # Fallback: if we never got a clean SYS_STATUS, wait a fixed time
    logger.info("  EKF check inconclusive — waiting 10s as fallback")
    time.sleep(10)


# ── Mission upload ────────────────────────────────────────────────────

def _upload_mission(
    mav,
    dest_lat: float,
    dest_lon: float,
    dest_alt_m: float,
    speed: float,
) -> None:
    """Upload a waypoint mission: HOME → TAKEOFF → DESTINATION.

    Uses MISSION_ITEM_INT (the native MAVLink mission protocol).
    """
    wp_loader = mavwp.MAVWPLoader()

    # WP0 — Home (required by ArduPilot as first item)
    wp_loader.add(
        mavutil.mavlink.MAVLink_mission_item_int_message(
            mav.target_system,
            mav.target_component,
            0,  # seq
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1,  # current, autocontinue
            0, 0, 0, 0,  # params 1-4
            0, 0, 0,  # lat, lon, alt (auto-filled by ArduPilot)
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
    )

    # WP1 — Takeoff
    wp_loader.add(
        mavutil.mavlink.MAVLink_mission_item_int_message(
            mav.target_system,
            mav.target_component,
            1,  # seq
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 1,
            0, 0, 0, 0,  # params
            0, 0,         # lat, lon (ignored for copter takeoff)
            int(dest_alt_m),
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
    )

    # WP2 — Destination
    wp_loader.add(
        mavutil.mavlink.MAVLink_mission_item_int_message(
            mav.target_system,
            mav.target_component,
            2,  # seq
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1,
            0,     # hold time
            2.0,   # acceptance radius (meters)
            0, 0,
            int(dest_lat * 1e7),
            int(dest_lon * 1e7),
            dest_alt_m,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
    )

    # Send mission count
    mav.waypoint_count_send(wp_loader.count())

    # Send each waypoint when requested
    for i in range(wp_loader.count()):
        msg = mav.recv_match(type="MISSION_REQUEST", blocking=True, timeout=10)
        if msg is None:
            raise TimeoutError(f"Timeout waiting for MISSION_REQUEST for WP{i}")
        wp = wp_loader.wp(msg.seq)
        mav.mav.send(wp)
        logger.info("  Sent WP%d: %s", msg.seq, _describe_wp(msg.seq, wp))

    # Wait for mission ACK
    ack = mav.recv_match(type="MISSION_ACK", blocking=True, timeout=10)
    if ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
        logger.info("  Mission accepted by flight controller")
    else:
        raise RuntimeError(f"Mission upload failed: {ack}")


def _describe_wp(seq: int, wp) -> str:
    """Human-readable description of a waypoint."""
    if seq == 0:
        return "HOME"
    if wp.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
        return f"TAKEOFF to {wp.z:.0f}m"
    if wp.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
        lat = wp.x / 1e7 if abs(wp.x) > 1000 else wp.x
        lon = wp.y / 1e7 if abs(wp.y) > 1000 else wp.y
        return f"WAYPOINT lat={lat:.6f} lon={lon:.6f} alt={wp.z:.0f}m"
    return f"CMD={wp.command}"


def upload_multi_stop_mission(
    mav,
    waypoints: list[tuple[float, float, float]],
    takeoff_alt_m: float = 10.0,
) -> None:
    """Upload a multi-waypoint mission from (lat, lon, alt) tuples.

    Example:
        waypoints = [
            (40.7453, -74.0256, 15.0),  # Stevens Babbio
            (40.7440, -74.0230, 15.0),  # River Street
            (40.7365, -74.0273, 15.0),  # Sinatra Park
        ]
        upload_multi_stop_mission(mav, waypoints)
    """
    wp_loader = mavwp.MAVWPLoader()
    seq = 0

    # WP0 — Home
    wp_loader.add(
        mavutil.mavlink.MAVLink_mission_item_int_message(
            mav.target_system, mav.target_component,
            seq, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1, 0, 0, 0, 0, 0, 0, 0,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
    )
    seq += 1

    # WP1 — Takeoff
    wp_loader.add(
        mavutil.mavlink.MAVLink_mission_item_int_message(
            mav.target_system, mav.target_component,
            seq, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 1, 0, 0, 0, 0, 0, 0, int(takeoff_alt_m),
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
    )
    seq += 1

    # Navigation waypoints
    for lat, lon, alt in waypoints:
        wp_loader.add(
            mavutil.mavlink.MAVLink_mission_item_int_message(
                mav.target_system, mav.target_component,
                seq, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 1,
                0, 2.0, 0, 0,
                int(lat * 1e7), int(lon * 1e7), alt,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        )
        seq += 1

    # Send
    mav.waypoint_count_send(wp_loader.count())
    for i in range(wp_loader.count()):
        msg = mav.recv_match(type="MISSION_REQUEST", blocking=True, timeout=10)
        if msg is None:
            raise TimeoutError(f"Timeout waiting for MISSION_REQUEST for WP{i}")
        mav.mav.send(wp_loader.wp(msg.seq))

    ack = mav.recv_match(type="MISSION_ACK", blocking=True, timeout=10)
    if not (ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED):
        raise RuntimeError(f"Mission upload failed: {ack}")


# ── Flight monitoring ─────────────────────────────────────────────────

def _monitor_mission(
    mav,
    dest_lat: float,
    dest_lon: float,
    dest_alt: float,
) -> None:
    """Monitor flight progress until destination is reached."""
    last_log = 0.0
    while True:
        msg = mav.recv_match(
            type=["GLOBAL_POSITION_INT", "MISSION_CURRENT", "MISSION_ITEM_REACHED"],
            blocking=True,
            timeout=2.0,
        )
        if msg is None:
            continue

        now = time.time()

        if msg.get_type() == "GLOBAL_POSITION_INT":
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.relative_alt / 1000.0
            dist = _haversine_m(lat, lon, dest_lat, dest_lon)

            if now - last_log >= 2.0:
                logger.info(
                    "  GPS: lat=%.6f  lon=%.6f  alt=%.1fm  dist=%.0fm",
                    lat, lon, alt, dist,
                )
                last_log = now

            # Close enough to destination
            if dist < 5.0 and alt > (dest_alt * 0.7):
                logger.info("  Within 5m of destination!")
                return

        elif msg.get_type() == "MISSION_ITEM_REACHED":
            logger.info("  Reached mission item %d", msg.seq)
            # WP2 = destination
            if msg.seq >= 2:
                return

        elif msg.get_type() == "MISSION_CURRENT":
            # Only log mission current changes, not every message
            if now - last_log >= 2.0:
                logger.info("  Current mission item: %d", msg.seq)


# ── Helpers ───────────────────────────────────────────────────────────

def _wait_for_gps(mav) -> tuple[float, float]:
    """Wait until we have a valid GPS position."""
    while True:
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg and msg.lat != 0:
            return msg.lat / 1e7, msg.lon / 1e7


def _wait_for_landing(mav) -> None:
    """Wait until the drone has landed (altitude near zero, low speed)."""
    while True:
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg:
            alt = msg.relative_alt / 1000.0
            vz = abs(msg.vz / 100.0)
            if alt < 1.0 and vz < 0.3:
                return
        time.sleep(0.5)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    R = 6_371_000.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    fly_to_destination(
        connection=CONNECTION,
        dest_lat=DESTINATION_LAT,
        dest_lon=DESTINATION_LON,
        dest_alt_m=DESTINATION_ALT_M,
        cruise_speed=CRUISE_SPEED_M_S,
    )