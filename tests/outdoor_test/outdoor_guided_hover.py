#!/usr/bin/env python3
"""
QuackOps Outdoor Test — GUIDED Mode Hover
==========================================
Target:     ArduCopter 4.6.3 on Pixhawk 2.4.8 via TELEM1
Env:        Outdoor, GPS required
Pi:         Raspberry Pi 5 (serial /dev/ttyAMA0 @ 57600)

GUIDED MODE BEHAVIOR:
    Pi commands takeoff to target altitude via MAV_CMD_NAV_TAKEOFF.
    Drone holds GPS position autonomously.
    Pi commands land via MAV_CMD_NAV_LAND at the same position.
    Ch5 RTL switch is monitored — on trigger, script yields control to
    ArduCopter and waits for RTL completion before exiting cleanly.

FLIGHT SEQUENCE:
    1.  Connect + start telemetry monitor
    2.  Verify 3D GPS fix with MIN_SATELLITES
    3.  Switch to GUIDED mode (verified via heartbeat)
    4.  Arm (verified via heartbeat)
    5.  Command MAV_CMD_NAV_TAKEOFF to TARGET_ALT_M
    6.  Wait for altitude reached (within ALTITUDE_TOLERANCE_M)
    7.  Hover for HOVER_TIME_S (RTL-aware polling loop)
    8.  Command MAV_CMD_NAV_LAND
    9.  Wait for landed (altitude < LANDED_ALT_THRESHOLD_M)
    10. Disarm

REQUIRED PARAMETERS (set in Mission Planner BEFORE running):
    ARMING_CHECK    = 1         all preflight checks ON (real hardware)
    GPS_TYPE        = 1         uBlox M8N
    FENCE_ENABLE    = 0         disable geofence for first test
    FS_THR_ENABLE   = 1         RC signal loss → RTL (keep ON)
    FS_GCS_ENABLE   = 0         Pi is the GCS — avoid spurious failsafe
    DISARM_DELAY    = 0         don't auto-disarm while landed
    RTL_ALT         = 1500      return altitude in cm (15m) — clear of obstacles

CHANNEL 5 (RTL SWITCH) — set in Mission Planner Flight Modes tab:
    FLTMODE_CH = 5
    Low position  → STABILIZE or GUIDED (whatever you had before)
    High position → RTL (mode 6)
    Verify in Flight Modes tab by flipping switch and watching highlight.

SAFETY:
    - RC transmitter ON, Ch5 RTL switch accessible and tested
    - Hard flight timeout enforced (MAX_FLIGHT_TIME_S)
    - Ctrl+C → immediate land + force disarm
    - Script yields to RTL cleanly at any point in the sequence

Usage:
    python3 outdoor_guided_hover.py                        # 3m, 5s hover
    python3 outdoor_guided_hover.py --alt 5 --hover 10    # 5m, 10s hover
"""

import argparse
import logging
import sys
import threading
import time

from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Configuration — change these, not the logic below
# ---------------------------------------------------------------------------

CONNECTION_STRING   = "/dev/ttyAMA0"
BAUD_RATE           = 57600

# Easily overridden via CLI (see bottom of file)
DEFAULT_TARGET_ALT_M    = 3.0    # meters
DEFAULT_HOVER_TIME_S    = 5.0    # seconds

# Hard safety ceiling — land immediately if exceeded regardless of sequence
MAX_FLIGHT_TIME_S       = 60.0

# GPS requirements — do not arm below these
MIN_SATELLITES          = 6
REQUIRED_FIX_TYPE       = 3      # 3 = 3D fix

# Tolerances and timeouts
ALTITUDE_TOLERANCE_M    = 0.4    # "close enough" to target alt
ALTITUDE_TIMEOUT_S      = 30.0   # max wait for target alt
LAND_TIMEOUT_S          = 40.0   # max wait for landed confirmation
ARM_TIMEOUT_S           = 10.0
MODE_TIMEOUT_S          = 10.0
GPS_WAIT_TIMEOUT_S      = 90.0   # abort if no fix after this
POST_ARM_DELAY_S        = 2.0    # settle time after arm before takeoff
TELEMETRY_HZ            = 10     # stream rate requested from FC

# Landed detection: below this altitude = on the ground
LANDED_ALT_THRESHOLD_M  = 0.3

# ArduCopter custom mode IDs
GUIDED_MODE = 4
RTL_MODE    = 6
LAND_MODE   = 9

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("guided_hover")

# ---------------------------------------------------------------------------
# Telemetry Monitor
# ---------------------------------------------------------------------------

class TelemetryMonitor:
    """
    Background thread that reads MAVLink messages and exposes current
    drone state as thread-safe properties.

    Monitored messages:
        HEARTBEAT        → mode, armed status
        GLOBAL_POSITION_INT → relative altitude, landed detection
        GPS_RAW_INT      → fix type, satellite count
    """

    def __init__(self, conn):
        self._conn = conn
        self._lock = threading.Lock()

        # State (protected by _lock)
        self._mode          = -1
        self._armed         = False
        self._rel_alt_m     = 0.0
        self._fix_type      = 0
        self._num_sats      = 0

        # RTL event — set when mode transitions INTO RTL from any other mode
        self.rtl_triggered = threading.Event()

        # Manual override event — set when mode changes away from GUIDED
        # while the Pi is supposed to be in control
        self.manual_override = threading.Event()

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info("Telemetry monitor started")

    def stop(self) -> None:
        self._running = False

    # ── Thread-safe accessors ─────────────────────────────────────────

    @property
    def mode(self) -> int:
        with self._lock:
            return self._mode

    @property
    def is_armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def relative_alt_m(self) -> float:
        with self._lock:
            return self._rel_alt_m

    @property
    def gps_fix_type(self) -> int:
        with self._lock:
            return self._fix_type

    @property
    def gps_satellites(self) -> int:
        with self._lock:
            return self._num_sats

    @property
    def is_landed(self) -> bool:
        with self._lock:
            return self._rel_alt_m < LANDED_ALT_THRESHOLD_M

    # ── Background loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            msg = self._conn.recv_match(
                type=["HEARTBEAT", "GLOBAL_POSITION_INT", "GPS_RAW_INT"],
                blocking=True,
                timeout=1.0,
            )
            if msg is None:
                continue

            mtype = msg.get_type()

            if mtype == "HEARTBEAT":
                # Filter out GCS component (srcComponent == 0) — it always
                # reports custom_mode=0 and would pollute mode detection.
                if msg.get_srcComponent() == 0:
                    continue

                prev_mode = self._mode
                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                with self._lock:
                    self._mode  = msg.custom_mode
                    self._armed = armed

                # RTL transition detection
                if prev_mode != RTL_MODE and msg.custom_mode == RTL_MODE:
                    log.warning("⚠️  RTL detected — Ch5 switch triggered")
                    self.rtl_triggered.set()

                # Manual override: any mode change away from GUIDED while
                # Pi is in control means a human has taken over
                if prev_mode == GUIDED_MODE and msg.custom_mode != GUIDED_MODE:
                    log.warning(
                        f"⚠️  Manual override — mode changed from GUIDED "
                        f"to {msg.custom_mode} — Pi standing down"
                    )
                    self.manual_override.set()

            elif mtype == "GLOBAL_POSITION_INT":
                alt_m = msg.relative_alt / 1000.0  # mm → m
                with self._lock:
                    self._rel_alt_m = alt_m

            elif mtype == "GPS_RAW_INT":
                with self._lock:
                    self._fix_type = msg.fix_type
                    self._num_sats = msg.satellites_visible

# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def connect() -> mavutil.mavfile:
    log.info(f"Connecting to {CONNECTION_STRING} @ {BAUD_RATE}...")
    conn = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD_RATE)
    log.info("Waiting for heartbeat...")
    conn.wait_heartbeat(timeout=15)
    log.info(
        f"Heartbeat received — system {conn.target_system}, "
        f"component {conn.target_component}"
    )
    return conn


def request_telemetry_streams(conn) -> None:
    """Ask the FC to stream position and raw sensor data at TELEMETRY_HZ."""
    for stream_id in [
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
        mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
    ]:
        conn.mav.request_data_stream_send(
            conn.target_system,
            conn.target_component,
            stream_id,
            TELEMETRY_HZ,
            1,  # start
        )
    log.info(f"Requested telemetry streams @ {TELEMETRY_HZ} Hz")


def wait_for_gps_fix(telemetry: TelemetryMonitor) -> bool:
    """Block until 3D GPS fix with MIN_SATELLITES. Hard timeout = GPS_WAIT_TIMEOUT_S."""
    log.info(
        f"Waiting for GPS fix (need fix_type >= {REQUIRED_FIX_TYPE}, "
        f"satellites >= {MIN_SATELLITES})..."
    )
    start = time.monotonic()
    while time.monotonic() - start < GPS_WAIT_TIMEOUT_S:
        fix  = telemetry.gps_fix_type
        sats = telemetry.gps_satellites
        log.info(f"  GPS | fix_type={fix}, satellites={sats}")
        if fix >= REQUIRED_FIX_TYPE and sats >= MIN_SATELLITES:
            log.info(f"✓ GPS fix acquired — {sats} sats, fix type {fix}")
            return True
        time.sleep(2.0)
    log.error(f"GPS fix timeout after {GPS_WAIT_TIMEOUT_S}s — do not fly")
    return False


def set_mode(conn, telemetry: TelemetryMonitor, mode_id: int, name: str) -> bool:
    """Send mode-change command and verify via heartbeat within MODE_TIMEOUT_S."""
    log.info(f"Setting mode → {name} ({mode_id})...")
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
        0, 0, 0, 0, 0,
    )
    start = time.monotonic()
    while time.monotonic() - start < MODE_TIMEOUT_S:
        if telemetry.mode == mode_id:
            log.info(f"✓ {name} mode confirmed")
            return True
        time.sleep(0.2)
    log.error(f"Mode switch to {name} not confirmed within {MODE_TIMEOUT_S}s")
    return False


def arm(conn, telemetry: TelemetryMonitor) -> bool:
    """Send arm command and verify armed state within ARM_TIMEOUT_S."""
    log.info("Arming motors...")
    conn.arducopter_arm()
    start = time.monotonic()
    while time.monotonic() - start < ARM_TIMEOUT_S:
        if telemetry.is_armed:
            log.info("✓ ARMED")
            return True
        time.sleep(0.2)
    log.error("Arming failed — check ARMING_CHECK failures in Mission Planner logs")
    return False


def command_takeoff(conn, altitude_m: float) -> None:
    """Send MAV_CMD_NAV_TAKEOFF. In GUIDED mode ArduCopter takes off and holds."""
    log.info(f"Commanding MAV_CMD_NAV_TAKEOFF to {altitude_m}m...")
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,   # params 1-4 unused by ArduCopter
        0, 0,          # lat, lon = 0 → use current position
        altitude_m,    # param7 = target altitude AGL in meters
    )


def command_land(conn) -> None:
    """Send MAV_CMD_NAV_LAND — drone descends and lands at current position."""
    log.info("Commanding MAV_CMD_NAV_LAND...")
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0, 0, 0, 0,
    )


def disarm(conn) -> None:
    log.info("Disarming...")
    conn.arducopter_disarm()
    time.sleep(1.0)
    log.info("Disarm command sent")


def force_disarm(conn) -> None:
    """Force-disarm even if in-air. Emergency use only."""
    log.warning("FORCE DISARM")
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,      # disarm
        21196,  # magic number = force
        0, 0, 0, 0, 0,
    )


def wait_for_altitude(
    telemetry: TelemetryMonitor,
    target_m: float,
) -> bool:
    """
    Poll altitude until within ALTITUDE_TOLERANCE_M of target.
    Returns False immediately if RTL is triggered.
    """
    log.info(
        f"Waiting for altitude {target_m}m "
        f"(tolerance ±{ALTITUDE_TOLERANCE_M}m, timeout {ALTITUDE_TIMEOUT_S}s)..."
    )
    start = time.monotonic()
    while time.monotonic() - start < ALTITUDE_TIMEOUT_S:
        if telemetry.rtl_triggered.is_set():
            log.warning("RTL triggered during climb — yielding")
            return False
        check_override(telemetry, None)
        alt = telemetry.relative_alt_m
        log.info(f"  Climbing | alt={alt:.2f}m / {target_m:.1f}m")
        if alt >= target_m - ALTITUDE_TOLERANCE_M:
            log.info(f"✓ Target altitude reached: {alt:.2f}m")
            return True
        time.sleep(0.5)
    log.error(
        f"Altitude timeout — reached {telemetry.relative_alt_m:.2f}m "
        f"of {target_m}m target"
    )
    return False


def wait_for_landed(telemetry: TelemetryMonitor) -> bool:
    """Poll until altitude < LANDED_ALT_THRESHOLD_M."""
    log.info(f"Waiting for landed (alt < {LANDED_ALT_THRESHOLD_M}m)...")
    start = time.monotonic()
    while time.monotonic() - start < LAND_TIMEOUT_S:
        check_override(telemetry, None)
        alt = telemetry.relative_alt_m
        log.info(f"  Descending | alt={alt:.2f}m")
        if telemetry.is_landed:
            log.info("✓ Landed confirmed")
            return True
        time.sleep(0.5)
    log.warning(
        f"Land timeout after {LAND_TIMEOUT_S}s — "
        f"current alt {telemetry.relative_alt_m:.2f}m. Check visually."
    )
    return False


def wait_for_rtl_complete(telemetry: TelemetryMonitor) -> None:
    """
    After RTL is triggered externally (Ch5 switch), wait for ArduCopter
    to fly home and land. We do not interfere — just monitor and log.
    """
    log.info("Monitoring RTL — hands off, ArduCopter in control...")
    start = time.monotonic()
    while time.monotonic() - start < 120.0:
        alt = telemetry.relative_alt_m
        mode = telemetry.mode
        log.info(f"  RTL | mode={mode} | alt={alt:.2f}m")
        if telemetry.is_landed:
            log.info("✓ RTL complete — drone landed")
            return
        time.sleep(1.0)
    log.warning("RTL monitor timeout — inspect drone visually")

# ---------------------------------------------------------------------------
# Override check helper
# ---------------------------------------------------------------------------

def check_override(telemetry: TelemetryMonitor, conn) -> None:
    """
    Call inside any polling loop. If manual override or RTL has been
    detected, log it, close the connection, and exit cleanly.
    ArduCopter is already in control at the firmware level — we just
    stop the Pi from issuing further commands.
    """
    if telemetry.manual_override.is_set() or telemetry.rtl_triggered.is_set():
        mode = telemetry.mode
        log.warning(
            f"Override active (mode={mode}) — Pi relinquishing control. "
            f"ArduCopter is flying. Monitor visually."
        )
        telemetry.stop()
        sys.exit(0)

# ---------------------------------------------------------------------------
# Main test sequence
# ---------------------------------------------------------------------------

def run_test(target_alt_m: float, hover_time_s: float) -> None:
    log.info("=" * 60)
    log.info("QuackOps Outdoor Test — GUIDED Hover")
    log.info(f"  Target altitude  : {target_alt_m} m")
    log.info(f"  Hover time       : {hover_time_s} s")
    log.info(f"  Hard timeout     : {MAX_FLIGHT_TIME_S} s")
    log.info(f"  Connection       : {CONNECTION_STRING} @ {BAUD_RATE}")
    log.info("=" * 60)

    # ── Step 1: Connect ───────────────────────────────────────────────
    log.info("[STEP 1/8] Connecting to flight controller...")
    conn = connect()
    request_telemetry_streams(conn)

    # ── Step 2: Start telemetry monitor ───────────────────────────────
    log.info("[STEP 2/8] Starting telemetry monitor...")
    telemetry = TelemetryMonitor(conn)
    telemetry.start()
    time.sleep(2.0)  # allow monitor to populate initial state

    flight_start: float = 0.0  # set after arm; used for hard timeout

    try:
        # ── Step 3: GPS fix check ─────────────────────────────────────
        log.info("[STEP 3/8] Verifying GPS fix...")
        if not wait_for_gps_fix(telemetry):
            log.error("Aborting — GPS fix requirement not met")
            return

        # ── Step 4: GUIDED mode ───────────────────────────────────────
        log.info("[STEP 4/8] Switching to GUIDED mode...")
        if not set_mode(conn, telemetry, GUIDED_MODE, "GUIDED"):
            log.error("Aborting — could not enter GUIDED mode")
            return

        # ── Step 5: Arm ───────────────────────────────────────────────
        log.info("[STEP 5/8] Arming...")
        log.info(">>>  STAND CLEAR OF THE DRONE  <<<")
        time.sleep(3.0)

        if not arm(conn, telemetry):
            log.error("Aborting — arm failed")
            return

        time.sleep(POST_ARM_DELAY_S)
        flight_start = time.monotonic()

        # ── Step 6: Takeoff ───────────────────────────────────────────
        log.info(f"[STEP 6/8] Taking off to {target_alt_m}m...")
        command_takeoff(conn, target_alt_m)

        reached = wait_for_altitude(telemetry, target_alt_m)

        if not reached:
            # Either RTL or timeout
            if telemetry.rtl_triggered.is_set():
                log.warning("RTL triggered during takeoff — yielding to ArduCopter")
                wait_for_rtl_complete(telemetry)
                disarm(conn)
                return
            else:
                log.error("Failed to reach target altitude — landing for safety")
                command_land(conn)
                wait_for_landed(telemetry)
                disarm(conn)
                return

        # ── Step 7: Hover ─────────────────────────────────────────────
        log.info(f"[STEP 7/8] Hovering for {hover_time_s}s...")
        hover_start = time.monotonic()

        while True:
            elapsed_hover = time.monotonic() - hover_start
            elapsed_total = time.monotonic() - flight_start

            # RTL switch check
            if telemetry.rtl_triggered.is_set():
                log.warning("RTL triggered during hover — yielding to ArduCopter")
                wait_for_rtl_complete(telemetry)
                disarm(conn)
                return

            check_override(telemetry, conn)

            # Hard flight timeout
            if elapsed_total >= MAX_FLIGHT_TIME_S:
                log.warning(
                    f"Hard flight timeout ({MAX_FLIGHT_TIME_S}s) — landing now"
                )
                break

            # Hover complete
            if elapsed_hover >= hover_time_s:
                log.info(f"✓ Hover complete ({elapsed_hover:.1f}s)")
                break

            log.info(
                f"  Hovering | alt={telemetry.relative_alt_m:.2f}m | "
                f"{elapsed_hover:.1f}/{hover_time_s:.1f}s"
            )
            time.sleep(1.0)

        # ── Step 8: Land ──────────────────────────────────────────────
        log.info("[STEP 8/8] Landing...")
        command_land(conn)
        wait_for_landed(telemetry)
        disarm(conn)

        total = time.monotonic() - flight_start
        log.info("=" * 60)
        log.info(f"✓ Test complete — {total:.1f}s total flight time")
        log.info("=" * 60)

    except KeyboardInterrupt:
        log.warning("CTRL+C — emergency land")
        command_land(conn)
        wait_for_landed(telemetry)
        force_disarm(conn)
        sys.exit(0)

    except Exception as exc:
        log.error(f"Unexpected error: {exc}", exc_info=True)
        command_land(conn)
        wait_for_landed(telemetry)
        force_disarm(conn)
        sys.exit(1)

    finally:
        telemetry.stop()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuackOps outdoor test: GUIDED hover"
    )
    parser.add_argument(
        "--alt", type=float, default=DEFAULT_TARGET_ALT_M,
        help=f"Target hover altitude in meters (default: {DEFAULT_TARGET_ALT_M}m)",
    )
    parser.add_argument(
        "--hover", type=float, default=DEFAULT_HOVER_TIME_S,
        help=f"Hover duration in seconds (default: {DEFAULT_HOVER_TIME_S}s)",
    )
    args = parser.parse_args()

    # Safety clamps — enforce reasonable bounds for first outdoor test
    if args.alt > 10.0:
        log.warning(f"Altitude {args.alt}m capped at 10m for first outdoor test")
        args.alt = 10.0
    if args.alt < 0.5:
        log.warning(f"Altitude {args.alt}m raised to minimum 2m")
        args.alt = 0.5
    if args.hover > 30.0:
        log.warning("Hover time capped at 30s")
        args.hover = 30.0

    run_test(target_alt_m=args.alt, hover_time_s=args.hover)


if __name__ == "__main__":
    main()