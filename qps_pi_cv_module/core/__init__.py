"""
Core module containing base classes, interfaces, and configuration.
"""

from .config import (
    Config,
    CameraConfig,
    VisionConfig,
    FlightConfig,
    CommunicationConfig,
    LandingConfig,
)
from .interfaces import (
    DroneState,
    LandingPhase,
    MarkerDetection,
    DronePosition,
    DroneVelocity,
    DroneTelemetry,
    IMarkerDetector,
    IFlightController,
    ICameraManager,
    IBackendClient,
    ILandingCoordinator,
)
from .landing_coordinator import LandingCoordinator
from .pi_cv_module import PiCvModule, ModuleStatus

__all__ = [
    # Main module
    "PiCvModule",
    "ModuleStatus",
    # Config
    "Config",
    "CameraConfig",
    "VisionConfig",
    "FlightConfig",
    "CommunicationConfig",
    "LandingConfig",
    # Data classes
    "DroneState",
    "LandingPhase",
    "MarkerDetection",
    "DronePosition",
    "DroneVelocity",
    "DroneTelemetry",
    # Interfaces
    "IMarkerDetector",
    "IFlightController",
    "ICameraManager",
    "IBackendClient",
    "ILandingCoordinator",
    # Implementation
    "LandingCoordinator",
]
