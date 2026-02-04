"""
Configuration management for the QuackOps Pi CV Module.

This module provides centralized configuration handling with support for
environment variables, config files, and runtime overrides.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for camera modules."""
    
    front_camera_id: int = 0
    bottom_camera_id: int = 1
    resolution_width: int = 640
    resolution_height: int = 480
    framerate: int = 30
    use_picamera: bool = True  # Use PiCamera2 vs OpenCV VideoCapture


@dataclass
class VisionConfig:
    """Configuration for computer vision processing."""
    
    aruco_dictionary: str = "DICT_6X6_250"
    marker_size_cm: float = 10.0
    detection_confidence_threshold: float = 0.8
    max_detection_distance_cm: float = 200.0
    min_detection_distance_cm: float = 20.0
    pose_estimation_enabled: bool = True


@dataclass
class FlightConfig:
    """Configuration for flight controller communication."""
    
    mavsdk_system_address: str = "udp://:14540"
    pixhawk_serial_port: str = "/dev/ttyUSB0"
    pixhawk_baud_rate: int = 57600
    connection_timeout_sec: float = 30.0
    command_timeout_sec: float = 10.0
    landing_descent_rate_mps: float = 0.3
    precision_landing_enabled: bool = True
    return_to_launch_altitude_m: float = 10.0


@dataclass
class CommunicationConfig:
    """Configuration for backend communication."""
    
    backend_base_url: str = "http://localhost:8000"
    websocket_url: str = "ws://localhost:8000/ws/drone"
    api_timeout_sec: float = 10.0
    reconnect_interval_sec: float = 5.0
    heartbeat_interval_sec: float = 1.0
    telemetry_update_interval_sec: float = 0.5


@dataclass
class LandingConfig:
    """Configuration for landing coordination."""
    
    approach_altitude_m: float = 5.0
    final_descent_altitude_m: float = 1.0
    marker_lock_threshold_pixels: int = 50
    max_correction_velocity_mps: float = 0.5
    landing_timeout_sec: float = 60.0
    abort_if_marker_lost_sec: float = 5.0


@dataclass 
class Config:
    """
    Main configuration container for the QuackOps Pi CV Module.
    
    Aggregates all sub-configurations and provides methods for loading
    from files or environment variables.
    
    Usage:
        # Load default configuration
        config = Config()
        
        # Load from file
        config = Config.from_file("/path/to/config.json")
        
        # Load with environment overrides
        config = Config.from_environment()
    """
    
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    landing: LandingConfig = field(default_factory=LandingConfig)
    
    # General settings
    log_level: str = "INFO"
    debug_mode: bool = False
    simulation_mode: bool = False
    
    @classmethod
    def from_file(cls, filepath: str) -> "Config":
        """Load configuration from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Config file not found: {filepath}, using defaults")
            return cls()
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls._from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise ValueError(f"Invalid JSON in config file: {filepath}")
    
    @classmethod
    def from_environment(cls) -> "Config":
        """Load configuration with environment variable overrides."""
        config = cls()
        
        # Map environment variables to config fields
        env_mappings = {
            "QPS_LOG_LEVEL": ("log_level", str),
            "QPS_DEBUG_MODE": ("debug_mode", lambda x: x.lower() == "true"),
            "QPS_SIMULATION_MODE": ("simulation_mode", lambda x: x.lower() == "true"),
            "QPS_BACKEND_URL": ("communication.backend_base_url", str),
            "QPS_WEBSOCKET_URL": ("communication.websocket_url", str),
            "QPS_MAVSDK_ADDRESS": ("flight.mavsdk_system_address", str),
            "QPS_PIXHAWK_PORT": ("flight.pixhawk_serial_port", str),
        }
        
        for env_var, (attr_path, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                config._set_nested_attr(attr_path, converter(value))
                logger.debug(f"Config override from env: {env_var}={value}")
        
        return config
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()
        
        # Update sub-configs if present
        if "camera" in data:
            config.camera = CameraConfig(**data["camera"])
        if "vision" in data:
            config.vision = VisionConfig(**data["vision"])
        if "flight" in data:
            config.flight = FlightConfig(**data["flight"])
        if "communication" in data:
            config.communication = CommunicationConfig(**data["communication"])
        if "landing" in data:
            config.landing = LandingConfig(**data["landing"])
        
        # Update top-level settings
        for key in ["log_level", "debug_mode", "simulation_mode"]:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def _set_nested_attr(self, attr_path: str, value: Any) -> None:
        """Set a nested attribute using dot notation."""
        parts = attr_path.split(".")
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        from dataclasses import asdict
        return {
            "camera": asdict(self.camera),
            "vision": asdict(self.vision),
            "flight": asdict(self.flight),
            "communication": asdict(self.communication),
            "landing": asdict(self.landing),
            "log_level": self.log_level,
            "debug_mode": self.debug_mode,
            "simulation_mode": self.simulation_mode,
        }
    
    def save(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
