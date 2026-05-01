"""
outdoor_tests/test_common.py

Shared helpers for QuackOps outdoor flight test scripts.

All helpers operate only on data exposed by the existing qps* classes via their
public APIs — nothing in here modifies qps internals.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

_EARTH_RADIUS_M: float = 6_371_000.0


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two WGS-84 positions."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def ned_to_latlon(
    home_lat: float, home_lon: float, north_m: float, east_m: float
) -> Tuple[float, float]:
    """Flat-earth NED offset from home → (lat_deg, lon_deg).

    Valid for offsets up to a few hundred metres.
    """
    lat = home_lat + math.degrees(north_m / _EARTH_RADIUS_M)
    lon = home_lon + math.degrees(
        east_m / (_EARTH_RADIUS_M * math.cos(math.radians(home_lat)))
    )
    return lat, lon


# ══════════════════════════════════════════════════════════════════════════════
# TestDiagnostics
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagnostics:
    """Registers on the flight manager callback bus and captures EKF + vibration.

    Usage::

        diag = TestDiagnostics(fm)
        diag.set_home(lat, lon)
        snap = diag.snapshot(tm)
    """

    def __init__(self, fm: Any) -> None:
        # EKF_STATUS_REPORT fields
        self._vel_var: float = 0.0
        self._pos_horiz_var: float = 0.0
        self._pos_vert_var: float = 0.0
        self._compass_var: float = 0.0
        # VIBRATION fields
        self._vibe_x: float = 0.0
        self._vibe_y: float = 0.0
        self._vibe_z: float = 0.0
        # Home for drift computation
        self._home_lat: Optional[float] = None
        self._home_lon: Optional[float] = None

        fm.register_message_callback(self._on_message)

    def set_home(self, lat: float, lon: float) -> None:
        self._home_lat = lat
        self._home_lon = lon

    def _on_message(self, msg: Any) -> None:
        t = msg.get_type()
        if t == "EKF_STATUS_REPORT":
            self._vel_var = float(msg.velocity_variance)
            self._pos_horiz_var = float(msg.pos_horiz_variance)
            self._pos_vert_var = float(msg.pos_vert_variance)
            self._compass_var = float(msg.compass_variance)
        elif t == "VIBRATION":
            self._vibe_x = float(msg.vibration_x)
            self._vibe_y = float(msg.vibration_y)
            self._vibe_z = float(msg.vibration_z)

    def snapshot(self, tm: Any) -> Dict[str, Any]:
        """Return a flat dict suitable for CSV writing + human logging."""
        pos = tm.get_gps_position()
        state = tm.get_drone_state()

        lat = pos.latitude_deg if pos else None
        lon = pos.longitude_deg if pos else None
        alt_m = pos.altitude_m if pos else None
        heading = pos.heading_deg if pos else None
        speed = pos.speed_m_s if pos else None

        drift_m: Optional[float] = None
        if pos and self._home_lat is not None:
            drift_m = haversine_m(self._home_lat, self._home_lon, lat, lon)

        return {
            "timestamp": time.time(),
            "lat": lat,
            "lon": lon,
            "alt_m": alt_m,
            "drift_m": drift_m,
            "heading_deg": heading,
            "speed_m_s": speed,
            "mode": state.flight_mode if state else "UNKNOWN",
            "armed": state.is_armed if state else False,
            "batt_pct": state.battery_percent if state else 0.0,
            "batt_voltage": state.battery_voltage if state else 0.0,
            "sats": state.gps_num_satellites if state else 0,
            "fix_type": state.gps_fix_type if state else 0,
            # HDOP captured separately by wait_gps_ready helper; default 99.9
            "hdop": 99.9,
            "vel_var": self._vel_var,
            "pos_horiz_var": self._pos_horiz_var,
            "pos_vert_var": self._pos_vert_var,
            "compass_var": self._compass_var,
            "vibe_x": self._vibe_x,
            "vibe_y": self._vibe_y,
            "vibe_z": self._vibe_z,
        }


# ══════════════════════════════════════════════════════════════════════════════
# FailsafeWatcher
# ══════════════════════════════════════════════════════════════════════════════

class FailsafeWatcher:
    """Listens on the flight manager callback bus for STATUSTEXT failsafe strings.

    Battery failsafe wiring: uses qpsTelemetryMonitor.on_battery_critical /
    on_battery_warning so the same threshold logic isn't duplicated.

    Usage::

        fs = FailsafeWatcher(fm, tm)
        asyncio.create_task(fs.watch_mode_transitions())
        fs.set_expected_mode("GUIDED")
        ...
        if fs.any_triggered():
            ...
    """

    # Strings that indicate specific failsafes (lower-cased STATUSTEXT)
    _RC_STRINGS = ("failsafe: radio", "failsafe: rc", "radio failsafe")
    _GCS_STRINGS = ("gcs failsafe", "gcs timeout")
    _FENCE_STRINGS = ("fence breach", "fence: ")

    def __init__(self, fm: Any, tm: Any) -> None:
        self.rc_failsafe: asyncio.Event = asyncio.Event()
        self.gcs_failsafe: asyncio.Event = asyncio.Event()
        self.battery_failsafe: asyncio.Event = asyncio.Event()
        self.battery_warning: asyncio.Event = asyncio.Event()
        self.fence_breach: asyncio.Event = asyncio.Event()
        self.unexpected_mode_change: asyncio.Event = asyncio.Event()
        self.prearm_messages: list[str] = []

        self._tm = tm
        self._expected_mode: Optional[str] = None
        self._log = logging.getLogger("qps.failsafe_watcher")

        fm.register_message_callback(self._on_message)

        # Wire battery events via qpsTelemetryMonitor callbacks so threshold
        # logic is not duplicated.
        tm.on_battery_critical(lambda pct: self._on_battery_critical(pct))
        tm.on_battery_warning(lambda pct: self._on_battery_warning(pct))

    def set_expected_mode(self, mode_name: str) -> None:
        """Call before an intentional mode change so the watcher doesn't flag it."""
        self._expected_mode = mode_name

    def any_triggered(self) -> bool:
        return any([
            self.rc_failsafe.is_set(),
            self.gcs_failsafe.is_set(),
            self.battery_failsafe.is_set(),
            self.fence_breach.is_set(),
            self.unexpected_mode_change.is_set(),
        ])

    def triggered_names(self) -> list[str]:
        names = []
        if self.rc_failsafe.is_set():
            names.append("rc_failsafe")
        if self.gcs_failsafe.is_set():
            names.append("gcs_failsafe")
        if self.battery_failsafe.is_set():
            names.append("battery_failsafe")
        if self.fence_breach.is_set():
            names.append("fence_breach")
        if self.unexpected_mode_change.is_set():
            names.append("unexpected_mode_change")
        return names

    def _on_message(self, msg: Any) -> None:
        if msg.get_type() != "STATUSTEXT":
            return
        text = msg.text.rstrip("\x00").lower()
        if any(s in text for s in self._RC_STRINGS):
            self._log.warning("RC failsafe detected in STATUSTEXT: %s", msg.text.rstrip())
            self.rc_failsafe.set()
        if any(s in text for s in self._GCS_STRINGS):
            self._log.warning("GCS failsafe detected in STATUSTEXT: %s", msg.text.rstrip())
            self.gcs_failsafe.set()
        if any(s in text for s in self._FENCE_STRINGS):
            self._log.warning("Fence breach detected in STATUSTEXT: %s", msg.text.rstrip())
            self.fence_breach.set()
        if "prearm" in text:
            self.prearm_messages.append(msg.text.rstrip("\x00"))

    def _on_battery_critical(self, pct: float) -> None:
        self._log.warning("Battery CRITICAL: %.0f%% — setting battery_failsafe event", pct)
        self.battery_failsafe.set()

    def _on_battery_warning(self, pct: float) -> None:
        self._log.warning("Battery WARNING: %.0f%%", pct)
        self.battery_warning.set()

    async def watch_mode_transitions(self) -> None:
        """Background task: flag unexpected GUIDED → abort/pilot-takeover transitions."""
        prev_mode: Optional[str] = None
        while True:
            await asyncio.sleep(0.5)
            state = self._tm.get_drone_state()
            if state is None:
                continue
            mode = state.flight_mode
            if (
                prev_mode == "GUIDED"
                and mode in ("RTL", "LAND", "SMART_RTL", "STABILIZE", "ALT_HOLD", "LOITER")
                and self._expected_mode not in ("RTL", "LAND", "SMART_RTL")
            ):
                self._log.warning(
                    "Unexpected mode transition: GUIDED → %s (expected=%s) "
                    "— possible pilot takeover or FC failsafe",
                    mode, self._expected_mode,
                )
                self.unexpected_mode_change.set()
            prev_mode = mode


# ══════════════════════════════════════════════════════════════════════════════
# Flight helpers
# ══════════════════════════════════════════════════════════════════════════════

async def hover_with_logging(
    tm: Any,
    diag: TestDiagnostics,
    fs: FailsafeWatcher,
    duration_s: float,
    csv_writer: csv.DictWriter,
    log: logging.Logger,
    poll_hz: float = 2.0,
) -> str:
    """Log telemetry at poll_hz for duration_s.  Returns 'completed' or failsafe name(s).

    Exits early if any failsafe fires.
    """
    interval = 1.0 / poll_hz
    end_time = time.time() + duration_s
    while time.time() < end_time:
        snap = diag.snapshot(tm)
        csv_writer.writerow(snap)
        # Flush after every row so partial logs survive crashes
        csv_writer._file.flush()
        log.info(
            "t=%.0fs  alt=%.1fm  drift=%.1fm  mode=%s  armed=%s  batt=%.0f%%"
            "  sats=%d  hdop=%.1f  vel_var=%.3f  pos_h=%.3f  vx=%.1f  vy=%.1f  vz=%.1f",
            duration_s - (end_time - time.time()),
            snap["alt_m"] or 0.0,
            snap["drift_m"] or 0.0,
            snap["mode"],
            snap["armed"],
            snap["batt_pct"],
            snap["sats"],
            snap["hdop"],
            snap["vel_var"],
            snap["pos_horiz_var"],
            snap["vibe_x"],
            snap["vibe_y"],
            snap["vibe_z"],
        )
        if fs.any_triggered():
            names = ", ".join(fs.triggered_names())
            log.warning("Failsafe(s) triggered during hover: %s — exiting early", names)
            return names
        await asyncio.sleep(interval)
    return "completed"


async def wait_for_arrival(
    tm: Any,
    target_lat: float,
    target_lon: float,
    tolerance_m: float,
    timeout_s: float,
    fs: FailsafeWatcher,
    log: logging.Logger,
) -> None:
    """Poll GPS until within tolerance_m of target, or raise on timeout/failsafe."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fs.any_triggered():
            raise RuntimeError(
                f"Failsafe during wait_for_arrival: {', '.join(fs.triggered_names())}"
            )
        pos = tm.get_gps_position()
        if pos is not None:
            dist = haversine_m(pos.latitude_deg, pos.longitude_deg, target_lat, target_lon)
            log.debug("  Arrival check: %.1fm from target (tol=%.1fm)", dist, tolerance_m)
            if dist <= tolerance_m:
                log.info("Arrived at target (%.1fm)", dist)
                return
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"wait_for_arrival: did not reach target within {timeout_s}s"
    )


async def wait_gps_ready(
    fm: Any,
    tm: Any,
    min_sats: int = 12,
    max_hdop: float = 1.5,
    timeout_s: float = 120,
    log: Optional[logging.Logger] = None,
) -> None:
    """Block until GPS reports sufficient quality for flight.

    qpsDroneState doesn't carry HDOP, so we register a tiny local callback
    on fm to capture GPS_RAW_INT.eph (cm * 0.01 = HDOP).
    """
    if log is None:
        log = logging.getLogger("qps.wait_gps_ready")

    _hdop: list[float] = [99.9]

    def _hdop_cb(msg: Any) -> None:
        if msg.get_type() == "GPS_RAW_INT":
            _hdop[0] = msg.eph / 100.0 if msg.eph != 65535 else 99.9

    fm.register_message_callback(_hdop_cb)

    deadline = time.time() + timeout_s
    last_log = 0.0
    while time.time() < deadline:
        state = tm.get_drone_state()
        hdop = _hdop[0]
        sats = state.gps_num_satellites if state else 0
        fix = state.gps_fix_type if state else 0
        now = time.time()
        if now - last_log >= 1.0:
            log.info(
                "GPS: sats=%d  fix=%d  HDOP=%.1f  (need %d sats, HDOP<%.1f)",
                sats, fix, hdop, min_sats, max_hdop,
            )
            last_log = now
        if fix >= 3 and sats >= min_sats and hdop <= max_hdop:
            log.info("GPS ready: sats=%d  fix=%d  HDOP=%.1f", sats, fix, hdop)
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"GPS not ready after {timeout_s}s (sats={sats}, fix={fix}, HDOP={hdop:.1f})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Log-dir / file helpers
# ══════════════════════════════════════════════════════════════════════════════

_TELEMETRY_FIELDS = [
    "timestamp", "lat", "lon", "alt_m", "drift_m", "heading_deg", "speed_m_s",
    "mode", "armed", "batt_pct", "batt_voltage", "sats", "fix_type", "hdop",
    "vel_var", "pos_horiz_var", "pos_vert_var", "compass_var",
    "vibe_x", "vibe_y", "vibe_z",
]


def make_log_dir(base_dir: str | Path, test_name: str) -> Path:
    """Create {base_dir}/{YYYY-MM-DD_HH-MM-SS}_{test_name}/ and set up file logging."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(base_dir) / f"{ts}_{test_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "flight.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fh)

    return log_dir


# def open_telemetry_csv(log_dir: Path) -> csv.DictWriter:
#     """Open telemetry.csv in log_dir, write header, return DictWriter."""
#     f = open(log_dir / "telemetry.csv", "w", newline="")
#     writer = csv.DictWriter(f, fieldnames=_TELEMETRY_FIELDS, extrasaction="ignore")
#     writer.writeheader()
#     # Attach the file object so callers can flush it
#     writer.writer.stream = f  # type: ignore[attr-defined]
#     return writer

def open_telemetry_csv(log_dir):
    """Opens telemetry.csv in the log directory. Returns (writer, file_handle)."""
    f = open(log_dir / "telemetry.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=[
        "timestamp", "lat", "lon", "alt_m", "drift_m",
        "heading_deg", "speed_m_s", "mode", "armed",
        "batt_pct", "batt_voltage", "sats", "fix_type", "hdop",
        "vel_var", "pos_horiz_var", "pos_vert_var", "compass_var",
        "vibe_x", "vibe_y", "vibe_z",
    ])
    writer.writeheader()
    writer._file = f
    f.flush()
    return writer, f


class FailsafeLog:
    """Simple append-only log for failsafe events."""

    def __init__(self, log_dir: Path) -> None:
        self._f = open(log_dir / "failsafes.log", "w")
        self._f.write("unix_timestamp,event,details\n")

    def log(self, event_name: str, details: str = "") -> None:
        line = f"{time.time():.3f},{event_name},{details}\n"
        self._f.write(line)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def open_failsafe_log(log_dir: Path) -> FailsafeLog:
    return FailsafeLog(log_dir)
