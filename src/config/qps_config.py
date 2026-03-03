from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class qpsConfig:
    """Central configuration for the QuackOps Pi application.

    Holds all tuneable parameters for flight control, vision processing,
    communication, and safety thresholds. Can be loaded from and saved to
    JSON files.
    """

    # --- Flight parameters ---
    cruise_altitude_m: float = 15.0
    takeoff_altitude_m: float = 10.0
    max_velocity_m_s: float = 5.0
    offboard_setpoint_rate_hz: float = 20.0
    mavsdk_connection_string: str = "udp://:14540"
    simulation_mode: bool = True

    # --- Landing / search parameters ---
    search_timeout_s: float = 30.0
    retry_altitude_adjustment_m: float = 2.0
    landing_altitude_threshold_m: float = 0.5
    position_tolerance_m: float = 0.2
    descent_rate_m_s: float = 0.3
    marker_lock_frames: int = 5
    max_correction_velocity_m_s: float = 1.0
    proportional_gain: float = 0.5

    # --- Battery thresholds ---
    battery_warning_percent: float = 30.0
    battery_critical_percent: float = 15.0

    # --- Telemetry ---
    telemetry_polling_rate_hz: float = 10.0
    gps_streaming_interval_ms: float = 500.0

    # --- Backend communication ---
    backend_ws_url: str = "ws://localhost:3001"
    backend_http_url: str = "http://localhost:3001"
    reconnection_interval_s: float = 10.0
    max_reconnection_attempts: int = -1
    heartbeat_interval_s: float = 5.0
    message_queue_max_size: int = 1000

    # --- Camera ---
    bottom_camera_id: int = 0
    camera_resolution: tuple[int, int] = (640, 480)
    camera_fps: int = 30
    use_picamera2: bool = False

    # --- ArUco / Vision ---
    aruco_dictionary: str = "DICT_6X6_250"
    marker_size_m: float = 0.15
    min_detection_confidence: float = 0.6
    camera_matrix: Optional[list[list[float]]] = None
    distortion_coefficients: Optional[list[float]] = None

    # --- Mission ---
    pickup_timeout_s: float = 300.0
    preflight_min_gps_fix: int = 3
    preflight_min_satellites: int = 8
    base_marker_id: int = 0

    # -----------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------

    @classmethod
    def load(cls, filepath: str) -> "qpsConfig":
        """Load a qpsConfig from a JSON file.

        Missing keys fall back to their dataclass defaults.

        Args:
            filepath: Path to the JSON configuration file.

        Returns:
            qpsConfig: A new configuration instance populated from the file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data: dict = json.load(f)

        # camera_resolution is stored as a list in JSON; convert to tuple
        if "camera_resolution" in data and isinstance(data["camera_resolution"], list):
            data["camera_resolution"] = tuple(data["camera_resolution"])

        return cls(**data)

    def save(self, filepath: str) -> None:
        """Save the current configuration to a JSON file.

        Args:
            filepath: Destination path for the JSON file.
        """
        data = asdict(self)
        # Convert tuple to list for JSON compatibility
        if isinstance(data.get("camera_resolution"), tuple):
            data["camera_resolution"] = list(data["camera_resolution"])

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
