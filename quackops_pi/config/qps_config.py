"""
qps_config.py

Aggregates all tunable system parameters.
Every qps class receives qpsConfig at construction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger("qps.config")


@dataclass
class qpsConfig:
    """Aggregated configuration for the QuackOps Pi module."""

    # ── Connection ────────────────────────────────────────────────────
    connection_string: str = "serial:///dev/ttyAMA0:57600"
    backend_ws_url: str = "ws://192.168.1.2:3001"

    # ── Telemetry ─────────────────────────────────────────────────────
    telemetry_polling_rate_hz: float = 2.0
    battery_warning_percent: float = 30.0
    battery_critical_percent: float = 15.0

    # ── Orbit search ──────────────────────────────────────────────────
    orbit_radius_m: float = 15.0
    orbit_num_points: int = 8
    orbit_altitude_m: float = 10.0
    search_timeout_s: float = 180.0
    target_marker_id: int = 0

    # ── Landing / goto ────────────────────────────────────────────────
    goto_arrival_tolerance_m: float = 2.0

    # ── Camera ────────────────────────────────────────────────────────
    camera_resolution: tuple[int, int] = (640, 480)
    camera_fps: int = 30

    # ── ArUco detection ───────────────────────────────────────────────
    aruco_dictionary: str = "DICT_4X4_50"

    # ── Pickup ────────────────────────────────────────────────────────
    pickup_timeout_s: float = 300.0

    # ── Stream server ─────────────────────────────────────────────────
    stream_port: int = 8080
    stream_fps: int = 15

    # ── Backend comms ─────────────────────────────────────────────────
    heartbeat_interval_s: float = 5.0
    reconnection_interval_s: float = 3.0

    # ── Landing controller — shared ───────────────────────────────────
    lock_frame_count: int = 5              # consecutive detections to confirm lock
    landing_strategy: str = "simple"       # "simple" or "servo"

    # ── Visual servo landing controller ───────────────────────────────
    proportional_gain: float = 0.001       # pixel offset → m/s velocity mapping
    center_tolerance_px: int = 30          # pixels from center = "centered"
    max_correction_velocity: float = 0.3   # m/s cap for cage safety

    # ── RC Flight Manager (indoor/no-GPS) ─────────────────────────────
    flight_manager_type: str = "gps"           # "gps" or "rc"
    rc_climb_throttle_pwm: int = 1650          # throttle PWM for climbing in ALT_HOLD
    rc_descend_throttle_pwm: int = 1350        # throttle PWM for descending in ALT_HOLD
    rc_climb_rate_m_per_s: float = 0.5         # estimated climb rate for time-based altitude
    max_rc_velocity_m_s: float = 0.5           # max velocity for NED-to-RC mapping
    max_rc_offset_pwm: int = 200               # max PWM offset from center for velocity commands
    rc_landing_duration_s: float = 5.0         # time to descend before disarm

    # ── I/O ───────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path) -> "qpsConfig":
        """Load config from JSON, falling back to defaults for missing keys."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found — using all defaults", path)
            return cls()

        with open(path) as f:
            data = json.load(f)

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        unknown = set(data.keys()) - valid_keys
        if unknown:
            logger.warning("Unknown config keys ignored: %s", unknown)

        return cls(**filtered)

    def to_file(self, path: str | Path) -> None:
        """Save current config to JSON."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        logger.info("Config saved to %s", path)