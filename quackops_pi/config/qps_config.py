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
    search_timeout_s: float = 60.0
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

    # ── Backend comms ─────────────────────────────────────────────────
    heartbeat_interval_s: float = 5.0
    reconnection_interval_s: float = 3.0

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