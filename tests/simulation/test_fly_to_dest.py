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

import asyncio
import json
import logging
import math
import queue
import threading
import time

import websockets
from pymavlink import mavutil, mavwp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quackops.fly_to")


# ── WebSocket bridge (sync → async) ───────────────────────────────────

class _WSBridge:
    """Manages the WebSocket connection to the backend.

    Responsibilities:
    - Connects to the backend on start() and stays connected.
    - Receives incoming commands; signals wait_for_start_delivery() when
      a startDelivery command arrives.
    - Streams outgoing GPS position updates via send_position().

    Runs a single asyncio event loop on a daemon thread so the blocking
    pymavlink poll loop can call send_position() without awaiting.
    """

    WS_URI = "ws://10.155.36.45:3001/?role=pi"

    def __init__(self) -> None:
        self._out_q: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._start_delivery_event = threading.Event()
        self._start_delivery_data: dict | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-bridge")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def wait_for_start_delivery(self, timeout: float | None = None) -> dict | None:
        """Block the calling thread until a startDelivery command is received.

        Returns a dict with keys: order_id, lat, lng.
        Returns None if timeout expires before the command arrives.
        """
        if self._start_delivery_event.wait(timeout=timeout):
            return self._start_delivery_data
        return None

    def send_position(self, lat: float, lon: float, alt: float) -> None:
        """Thread-safe — call from the pymavlink polling loop."""
        self._out_q.put_nowait({"latitude_deg": lat, "longitude_deg": lon, "altitude_m": alt})

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_task())

    async def _ws_task(self) -> None:
        try:
            async with websockets.connect(self.WS_URI) as ws:
                logger.info("WebSocket connected to %s", self.WS_URI)
                recv = asyncio.create_task(self._recv_loop(ws))
                send = asyncio.create_task(self._send_loop(ws))
                await asyncio.gather(recv, send)
        except Exception as exc:
            logger.warning("WebSocket bridge error: %s", exc)

    async def _recv_loop(self, ws) -> None:
        """Handle incoming messages from the backend."""
        while not self._stop_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                msg = json.loads(raw)
                if msg.get("type") == "startDelivery":
                    self._start_delivery_data = {
                        "order_id": msg["orderId"],
                        "lat": float(msg["target"]["lat"]),
                        "lng": float(msg["target"]["lng"]),
                    }
                    logger.info(
                        "startDelivery received: orderId=%s  lat=%.6f  lng=%.6f",
                        self._start_delivery_data["order_id"],
                        self._start_delivery_data["lat"],
                        self._start_delivery_data["lng"],
                    )
                    self._start_delivery_event.set()
                else:
                    logger.info("WebSocket message: %s", msg)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.warning("WebSocket recv error: %s", exc)
                break

    async def _send_loop(self, ws) -> None:
        """Drain the outbound queue and send position updates."""
        while not self._stop_event.is_set():
            try:
                data = self._out_q.get_nowait()
                await ws.send(json.dumps({"position": data}))
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as exc:
                logger.warning("WebSocket send error: %s", exc)
                break


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
    ws_bridge: _WSBridge | None = None,
) -> None:
    """Connect, upload mission, arm, fly to destination, RTL, land.

    ws_bridge: pass an already-connected _WSBridge to stream GPS during flight.
    """

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
    arming_check_ok = _set_param_verified(mav, "ARMING_CHECK", 0)
    if not arming_check_ok:
        logger.error("  Failed to set ARMING_CHECK=0 — arming may fail")
    time.sleep(2)

    # ── 7. Switch to GUIDED, arm, takeoff ─────────────────────────────
    logger.info("Switching to GUIDED mode...")
    _set_mode_verified(mav, MODE_GUIDED)

    logger.info("Arming...")
    mav.arducopter_arm()

    # Drain STATUSTEXT + COMMAND_ACK to capture pre-arm failure reasons
    arm_accepted = False
    drain_deadline = time.time() + 3.0
    while time.time() < drain_deadline:
        msg = mav.recv_match(
            type=["COMMAND_ACK", "STATUSTEXT"],
            blocking=True,
            timeout=1,
        )
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT":
            text = msg.text.rstrip("\x00")
            if text:
                logger.info("  [FC] %s", text)
        elif msg.get_type() == "COMMAND_ACK":
            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info("  Arm command accepted")
                arm_accepted = True
                break
            else:
                logger.error("  Arm REJECTED (result=%d)", msg.result)
                # Keep draining to catch the PreArm reason from STATUSTEXT
    if not arm_accepted:
        logger.warning("  Arm may not have been accepted — checking heartbeat anyway")

    # Wait for armed state with a timeout (replaces infinite motors_armed_wait)
    logger.info("Waiting for motors armed...")
    arm_start = time.time()
    armed = False
    while time.time() - arm_start < 15.0:
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb and hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            armed = True
            break
    if armed:
        logger.info("Armed! (confirmed from heartbeat)")
    else:
        logger.error("  ARM TIMEOUT — motors never armed after 15s")
        # Final attempt to capture any STATUSTEXT pre-arm messages
        for _ in range(10):
            st = mav.recv_match(type="STATUSTEXT", blocking=True, timeout=0.5)
            if st:
                logger.error("  [FC] %s", st.text.rstrip("\x00"))
            else:
                break
        return

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

    # ── 9. Monitor flight (stream GPS over WebSocket) ─────────────────
    logger.info("Flying to destination...")
    _owns_bridge = ws_bridge is None
    if _owns_bridge:
        ws_bridge = _WSBridge()
        ws_bridge.start()
    try:
        _monitor_mission(mav, dest_lat, dest_lon, dest_alt_m, ws_bridge)
    finally:
        if _owns_bridge:
            ws_bridge.stop()

    # ── 10. Loiter at destination ─────────────────────────────────────
    logger.info("At destination — loitering for 5 seconds...")
    time.sleep(5)

    # ── 11. RTL ───────────────────────────────────────────────────────
    logger.info("Returning to launch...")
    _set_mode_verified(mav, MODE_RTL)

    # Wait for landing
    logger.info("Waiting for landing...")
    _wait_for_landing(mav, home_lat=home_lat, home_lon=home_lon)
    logger.info("Landed!")

    # ── 12. Disarm ────────────────────────────────────────────────────
    logger.info("Disarming...")
    mav.arducopter_disarm()
    # Wait for disarmed state with timeout (ArduCopter often auto-disarms after landing)
    disarm_start = time.time()
    disarmed = False
    while time.time() - disarm_start < 10.0:
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb and hb.get_srcComponent() != 0:
            if not (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                disarmed = True
                break
    if disarmed:
        logger.info("Disarmed!")
    else:
        logger.warning("Disarm not confirmed within 10s (may have auto-disarmed already)")

    logger.info("═══ FLIGHT COMPLETE ═══")
    logger.info("Distance flown: ~%.0fm round trip", distance_m * 2)


# ── Param setting with verification ───────────────────────────────────

def _set_param_verified(
    mav,
    param_id: str,
    value: float,
    max_attempts: int = 5,
) -> bool:
    """Set a parameter and verify via explicit read-back.

    Strategy:
        1. Send PARAM_SET with REAL32 type (ArduPilot stores all params
           as floats internally — INT32 can be silently ignored).
        2. Explicitly request the param back with PARAM_REQUEST_READ.
        3. Check the returned value matches.

    This avoids relying on the unsolicited PARAM_VALUE response from
    PARAM_SET, which can get lost in the data-stream flood.

    Args:
        mav:           pymavlink connection.
        param_id:      Parameter name (e.g. "ARMING_CHECK").
        value:         Desired value.
        max_attempts:  Number of set+verify cycles.

    Returns:
        True if the parameter was confirmed set, False otherwise.
    """
    param_id_bytes = param_id.encode("utf-8")

    for attempt in range(1, max_attempts + 1):
        # ── Send the PARAM_SET (use REAL32, not INT32) ────────────
        mav.mav.param_set_send(
            mav.target_system,
            mav.target_component,
            param_id_bytes,
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        time.sleep(0.5)  # give firmware time to process

        # ── Drain any buffered PARAM_VALUE messages ───────────────
        while True:
            stale = mav.recv_match(type="PARAM_VALUE", blocking=False)
            if stale is None:
                break

        # ── Explicitly request the param back ─────────────────────
        mav.mav.param_request_read_send(
            mav.target_system,
            mav.target_component,
            param_id_bytes,
            -1,  # use param_id string, not index
        )

        # ── Wait for the response ─────────────────────────────────
        deadline = time.time() + 3.0
        while time.time() < deadline:
            ack = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
            if ack is None:
                break
            received_id = ack.param_id.rstrip("\x00")
            if received_id == param_id:
                actual = int(ack.param_value)
                if actual == int(value):
                    logger.info(
                        "  %s confirmed = %d (attempt %d)",
                        param_id, int(value), attempt,
                    )
                    return True
                else:
                    logger.warning(
                        "  %s read back %d instead of %d, retrying...",
                        param_id, actual, int(value),
                    )
                    break
            # else: wrong param_id, keep draining

        if attempt < max_attempts:
            logger.warning(
                "  %s not confirmed (attempt %d/%d)",
                param_id, attempt, max_attempts,
            )

    logger.error(
        "  Failed to set %s=%d after %d attempts",
        param_id, int(value), max_attempts,
    )
    return False


# ── Mode switching ────────────────────────────────────────────────────

def _set_mode_verified(mav, mode_id: int, timeout: float = 10.0) -> bool:
    """Switch ArduCopter mode and verify via heartbeat.

    Uses MAV_CMD_DO_SET_MODE and confirms the mode actually changed by
    reading heartbeat custom_mode field.  Filters out GCS heartbeats
    from Mission Planner (compid=0) which report custom_mode=0
    (STABILIZE) and cause false flickering.

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
        if msg is None:
            continue
        # Ignore GCS heartbeats from Mission Planner (compid=0).
        # These always report custom_mode=0 (STABILIZE) and pollute
        # the verification loop.  We only care about the autopilot.
        if msg.get_srcComponent() == 0:
            continue
        if msg.custom_mode == mode_id:
            logger.info("  %s mode confirmed!", mode_name)
            return True
        else:
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
    ws_bridge: _WSBridge | None = None,
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

            if ws_bridge is not None:
                ws_bridge.send_position(lat, lon, alt)

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


def _wait_for_landing(mav, home_lat: float = 0.0, home_lon: float = 0.0, timeout: float = 120.0) -> None:
    """Wait until the drone has landed (altitude near zero, low speed).

    Logs progress every 3 seconds so the terminal isn't silent during
    the long RTL flight back.

    Args:
        mav:       pymavlink connection.
        home_lat:  Home latitude for distance-to-home logging (optional).
        home_lon:  Home longitude for distance-to-home logging (optional).
        timeout:   Maximum seconds to wait before giving up.
    """
    last_log = 0.0
    start = time.time()
    while time.time() - start < timeout:
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg:
            alt = msg.relative_alt / 1000.0
            vz = abs(msg.vz / 100.0)
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7

            # Log progress every 3 seconds
            now = time.time()
            if now - last_log >= 3.0:
                if home_lat != 0.0 and home_lon != 0.0:
                    dist = _haversine_m(lat, lon, home_lat, home_lon)
                    logger.info(
                        "  RTL: alt=%.1fm  vz=%.1fm/s  dist_to_home=%.0fm",
                        alt, msg.vz / 100.0, dist,
                    )
                else:
                    logger.info("  RTL: alt=%.1fm  vz=%.1fm/s", alt, msg.vz / 100.0)
                last_log = now

            if alt < 1.0 and vz < 0.3:
                return
        time.sleep(0.5)

    logger.error("  Landing timeout after %.0fs", timeout)


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
    # ── 0. Connect to backend, wait for startDelivery ─────────────────
    bridge = _WSBridge()
    bridge.start()

    logger.info("Connected to backend — waiting for startDelivery command...")
    cmd = bridge.wait_for_start_delivery()
    if cmd is None:
        logger.error("No startDelivery received — exiting")
        bridge.stop()
        raise SystemExit(1)

    logger.info(
        "Starting delivery: orderId=%s  destination=%.6f, %.6f",
        cmd["order_id"], cmd["lat"], cmd["lng"],
    )

    try:
        fly_to_destination(
            connection=CONNECTION,
            dest_lat=cmd["lat"],
            dest_lon=cmd["lng"],
            dest_alt_m=DESTINATION_ALT_M,
            cruise_speed=CRUISE_SPEED_M_S,
            ws_bridge=bridge,
        )
    finally:
        bridge.stop()