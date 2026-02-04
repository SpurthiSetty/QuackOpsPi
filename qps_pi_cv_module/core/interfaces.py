"""
Abstract base classes and interfaces for the QuackOps Pi CV Module.

These define the contracts that concrete implementations must follow,
enabling dependency injection and easier testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Any
from enum import Enum, auto
import numpy as np


class DroneState(Enum):
    """Enumeration of possible drone states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    IDLE = auto()
    ARMING = auto()
    TAKING_OFF = auto()
    IN_FLIGHT = auto()
    NAVIGATING = auto()
    APPROACHING_LANDING = auto()
    PRECISION_LANDING = auto()
    LANDING = auto()
    LANDED = auto()
    RETURNING_TO_BASE = auto()
    ERROR = auto()


class LandingPhase(Enum):
    """Phases of the precision landing sequence."""
    NOT_STARTED = auto()
    APPROACHING = auto()
    SEARCHING_MARKER = auto()
    MARKER_ACQUIRED = auto()
    DESCENDING = auto()
    FINAL_APPROACH = auto()
    TOUCHDOWN = auto()
    COMPLETED = auto()
    ABORTED = auto()


@dataclass
class MarkerDetection:
    """Data class representing a detected ArUco marker."""
    marker_id: int
    corners: np.ndarray  # 4x2 array of corner coordinates
    center_x: float
    center_y: float
    area: float
    confidence: float
    timestamp: float
    
    # Pose estimation (if available)
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    distance_cm: Optional[float] = None
    
    @property
    def is_pose_available(self) -> bool:
        """Check if pose estimation data is available."""
        return self.rotation_vector is not None and self.translation_vector is not None


@dataclass
class DronePosition:
    """Data class representing drone position."""
    latitude: float
    longitude: float
    altitude_m: float
    relative_altitude_m: float
    heading_deg: float
    timestamp: float


@dataclass
class DroneVelocity:
    """Data class representing drone velocity."""
    velocity_north_mps: float
    velocity_east_mps: float
    velocity_down_mps: float
    groundspeed_mps: float
    timestamp: float


@dataclass
class DroneTelemetry:
    """Comprehensive drone telemetry data."""
    position: DronePosition
    velocity: DroneVelocity
    battery_percent: float
    is_armed: bool
    is_in_air: bool
    flight_mode: str
    gps_fix_type: int
    satellite_count: int
    state: DroneState


class IMarkerDetector(ABC):
    """Interface for marker detection implementations."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[MarkerDetection]:
        """
        Detect ArUco markers in the given frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detected markers
        """
        pass
    
    @abstractmethod
    def estimate_pose(
        self, 
        detection: MarkerDetection,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> MarkerDetection:
        """
        Estimate the 3D pose of a detected marker.
        
        Args:
            detection: The marker detection to estimate pose for
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            
        Returns:
            Updated detection with pose information
        """
        pass


class IFlightController(ABC):
    """Interface for flight controller implementations."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the flight controller."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the flight controller."""
        pass
    
    @abstractmethod
    async def arm(self) -> bool:
        """Arm the drone."""
        pass
    
    @abstractmethod
    async def disarm(self) -> bool:
        """Disarm the drone."""
        pass
    
    @abstractmethod
    async def takeoff(self, altitude_m: float) -> bool:
        """Command the drone to take off to specified altitude."""
        pass
    
    @abstractmethod
    async def land(self) -> bool:
        """Command the drone to land."""
        pass
    
    @abstractmethod
    async def goto(
        self, 
        latitude: float, 
        longitude: float, 
        altitude_m: float,
        heading_deg: Optional[float] = None
    ) -> bool:
        """Command the drone to go to a specific location."""
        pass
    
    @abstractmethod
    async def set_velocity(
        self,
        velocity_north_mps: float,
        velocity_east_mps: float,
        velocity_down_mps: float,
        yaw_deg: Optional[float] = None
    ) -> bool:
        """Set the drone's velocity vector."""
        pass
    
    @abstractmethod
    async def get_telemetry(self) -> DroneTelemetry:
        """Get current telemetry data."""
        pass
    
    @abstractmethod
    async def return_to_launch(self) -> bool:
        """Command the drone to return to launch point."""
        pass


class ICameraManager(ABC):
    """Interface for camera management implementations."""
    
    @abstractmethod
    def start(self) -> bool:
        """Start the camera capture."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the camera capture."""
        pass
    
    @abstractmethod
    def get_frame(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """Get the latest frame from the specified camera."""
        pass
    
    @abstractmethod
    def get_camera_matrix(self, camera_id: int = 0) -> np.ndarray:
        """Get the camera intrinsic matrix for pose estimation."""
        pass
    
    @abstractmethod
    def get_distortion_coeffs(self, camera_id: int = 0) -> np.ndarray:
        """Get the camera distortion coefficients."""
        pass


class IBackendClient(ABC):
    """Interface for backend communication implementations."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the backend server."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the backend server."""
        pass
    
    @abstractmethod
    async def send_telemetry(self, telemetry: DroneTelemetry) -> bool:
        """Send telemetry data to the backend."""
        pass
    
    @abstractmethod
    async def send_landing_status(
        self, 
        phase: LandingPhase, 
        marker_detected: bool,
        details: Optional[dict] = None
    ) -> bool:
        """Send landing status update to the backend."""
        pass
    
    @abstractmethod
    def register_command_callback(
        self, 
        callback: Callable[[str, dict], Any]
    ) -> None:
        """Register callback for commands received from backend."""
        pass


class ILandingCoordinator(ABC):
    """Interface for landing coordination implementations."""
    
    @abstractmethod
    async def start_landing_sequence(
        self,
        target_marker_id: int
    ) -> bool:
        """
        Start the precision landing sequence.
        
        Args:
            target_marker_id: The ArUco marker ID to land on
            
        Returns:
            True if landing completed successfully
        """
        pass
    
    @abstractmethod
    async def abort_landing(self) -> bool:
        """Abort the current landing sequence."""
        pass
    
    @abstractmethod
    def get_current_phase(self) -> LandingPhase:
        """Get the current landing phase."""
        pass
    
    @abstractmethod
    def is_landing_in_progress(self) -> bool:
        """Check if a landing sequence is currently in progress."""
        pass
