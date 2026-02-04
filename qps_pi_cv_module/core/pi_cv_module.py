"""
Main QuackOps Pi CV Module orchestrator.

This is the primary entry point for the Raspberry Pi companion computer,
coordinating all subsystems for autonomous drone delivery operations.
"""

import asyncio
import logging
import signal
import sys
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from .config import Config
from .interfaces import (
    DroneState,
    LandingPhase,
    DroneTelemetry,
    MarkerDetection,
)
from .landing_coordinator import LandingCoordinator
from ..vision.marker_detector import MarkerDetector
from ..vision.camera_manager import CameraManager
from ..flight.flight_controller import FlightController
from ..communication.backend_client import BackendClient, HttpBackendClient

logger = logging.getLogger(__name__)


@dataclass
class ModuleStatus:
    """Status of the Pi CV Module."""
    is_running: bool = False
    flight_controller_connected: bool = False
    backend_connected: bool = False
    cameras_active: bool = False
    current_state: DroneState = DroneState.DISCONNECTED
    active_mission: Optional[str] = None
    error_message: Optional[str] = None


class PiCvModule:
    """
    Main orchestrator for the QuackOps Raspberry Pi CV Module.
    
    This class initializes and coordinates all subsystems:
    - Flight Controller (MAVSDK)
    - Camera Manager
    - Marker Detector
    - Landing Coordinator
    - Backend Communication
    
    It provides high-level methods for drone operations and handles
    commands received from the web backend.
    
    Attributes:
        config: Module configuration
        flight_controller: Flight control interface
        camera_manager: Camera management interface
        marker_detector: Marker detection interface
        landing_coordinator: Precision landing coordinator
        backend_client: WebSocket backend client
        http_client: HTTP backend client
        
    Example:
        >>> config = Config.from_file("config.json")
        >>> module = PiCvModule(config)
        >>> await module.start()
        >>> # Module now running and responding to backend commands
        >>> await module.stop()
    """
    
    def __init__(self, config: Config):
        """
        Initialize the Pi CV Module.
        
        Args:
            config: Module configuration settings
        """
        self.config = config
        self._status = ModuleStatus()
        
        # Configure logging
        self._setup_logging()
        
        # Initialize subsystems
        self.flight_controller = FlightController(config.flight)
        self.camera_manager = CameraManager(config.camera)
        self.marker_detector = MarkerDetector(config.vision)
        self.backend_client = BackendClient(config.communication)
        self.http_client = HttpBackendClient(config.communication)
        
        # Landing coordinator (initialized after other subsystems)
        self.landing_coordinator = LandingCoordinator(
            config.landing,
            self.flight_controller,
            self.marker_detector,
            self.camera_manager,
            self.backend_client
        )
        
        # Background tasks
        self._telemetry_task: Optional[asyncio.Task] = None
        self._command_handler_task: Optional[asyncio.Task] = None
        
        # Shutdown event
        self._shutdown_event = asyncio.Event()
        
        # Register command callback
        self.backend_client.register_command_callback(self._handle_command)
        
        # Register signal handlers
        self._setup_signal_handlers()
        
        logger.info("PiCvModule initialized")
    
    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if self.config.debug_mode:
            logging.getLogger("qps_pi_cv_module").setLevel(logging.DEBUG)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self) -> bool:
        """
        Start the Pi CV Module.
        
        Initializes all subsystems and begins operation.
        
        Returns:
            True if started successfully
        """
        logger.info("Starting PiCvModule...")
        
        try:
            # Start cameras
            if not self.camera_manager.start():
                logger.error("Failed to start cameras")
                self._status.error_message = "Camera initialization failed"
                return False
            
            self._status.cameras_active = True
            logger.info("Cameras started")
            
            # Connect to flight controller
            if not self.config.simulation_mode:
                if not await self.flight_controller.connect():
                    logger.error("Failed to connect to flight controller")
                    self._status.error_message = "Flight controller connection failed"
                    return False
                
                self._status.flight_controller_connected = True
                logger.info("Flight controller connected")
            
            # Connect to backend
            if await self.backend_client.connect():
                self._status.backend_connected = True
                logger.info("Backend connected")
            else:
                logger.warning("Backend connection failed, will retry")
            
            # Start background tasks
            self._telemetry_task = asyncio.create_task(self._telemetry_loop())
            
            self._status.is_running = True
            self._status.current_state = DroneState.IDLE
            
            logger.info("PiCvModule started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start PiCvModule: {e}")
            self._status.error_message = str(e)
            return False
    
    async def stop(self) -> None:
        """Stop the Pi CV Module and cleanup resources."""
        logger.info("Stopping PiCvModule...")
        
        self._status.is_running = False
        
        # Cancel background tasks
        if self._telemetry_task and not self._telemetry_task.done():
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect from backend
        await self.backend_client.disconnect()
        self._status.backend_connected = False
        
        # Disconnect from flight controller
        await self.flight_controller.disconnect()
        self._status.flight_controller_connected = False
        
        # Stop cameras
        self.camera_manager.stop()
        self._status.cameras_active = False
        
        logger.info("PiCvModule stopped")
    
    async def run(self) -> None:
        """
        Run the module until shutdown is requested.
        
        This is a blocking call that runs the main event loop.
        """
        if not await self.start():
            logger.error("Failed to start module, exiting")
            return
        
        logger.info("Module running, waiting for shutdown...")
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
        
        # Cleanup
        await self.stop()
    
    async def execute_delivery(
        self,
        order_id: str,
        destination_lat: float,
        destination_lon: float,
        target_marker_id: int
    ) -> bool:
        """
        Execute a full delivery mission.
        
        Args:
            order_id: Order identifier
            destination_lat: Delivery latitude
            destination_lon: Delivery longitude
            target_marker_id: Landing pad marker ID
            
        Returns:
            True if delivery completed successfully
        """
        logger.info(f"Starting delivery for order {order_id}")
        self._status.active_mission = order_id
        
        try:
            # Update order status
            await self.http_client.update_order_status(
                order_id, "in_transit"
            )
            
            # Arm and takeoff
            if not await self.flight_controller.arm():
                raise Exception("Failed to arm drone")
            
            if not await self.flight_controller.takeoff(
                self.config.flight.return_to_launch_altitude_m
            ):
                raise Exception("Failed to takeoff")
            
            # Navigate to destination
            if not await self.flight_controller.goto(
                destination_lat,
                destination_lon,
                self.config.landing.approach_altitude_m
            ):
                raise Exception("Navigation failed")
            
            # Wait for arrival (simplified - should use proper position checking)
            await asyncio.sleep(5)
            
            # Update status
            await self.http_client.update_order_status(
                order_id, "arriving"
            )
            
            # Execute precision landing
            if not await self.landing_coordinator.start_landing_sequence(
                target_marker_id
            ):
                raise Exception("Landing sequence failed")
            
            # Delivery complete!
            await self.http_client.update_order_status(
                order_id, "delivered",
                {"landing_accuracy_cm": self.landing_coordinator._state.final_accuracy_cm}
            )
            
            logger.info(f"Delivery {order_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Delivery failed: {e}")
            await self.http_client.update_order_status(
                order_id, "failed",
                {"error": str(e)}
            )
            return False
        
        finally:
            self._status.active_mission = None
    
    async def return_to_base(self) -> bool:
        """
        Return drone to launch point.
        
        Returns:
            True if RTL command successful
        """
        logger.info("Returning to base")
        
        # Takeoff first if not in air
        telemetry = await self.flight_controller.get_telemetry()
        if not telemetry.is_in_air:
            if not await self.flight_controller.takeoff(
                self.config.flight.return_to_launch_altitude_m
            ):
                return False
        
        return await self.flight_controller.return_to_launch()
    
    async def emergency_land(self) -> bool:
        """
        Execute emergency landing at current position.
        
        Returns:
            True if landing initiated
        """
        logger.warning("Emergency landing initiated")
        
        # Abort any ongoing landing sequence
        if self.landing_coordinator.is_landing_in_progress():
            await self.landing_coordinator.abort_landing()
        
        return await self.flight_controller.land()
    
    async def _handle_command(self, command: str, data: Dict[str, Any]) -> None:
        """
        Handle commands received from the backend.
        
        Args:
            command: Command type
            data: Command data/parameters
        """
        logger.info(f"Handling command: {command}")
        
        try:
            if command == "takeoff":
                altitude = data.get("altitude", 5.0)
                await self.flight_controller.arm()
                await self.flight_controller.takeoff(altitude)
            
            elif command == "land":
                await self.flight_controller.land()
            
            elif command == "goto":
                await self.flight_controller.goto(
                    data["latitude"],
                    data["longitude"],
                    data.get("altitude", 10.0)
                )
            
            elif command == "rtl":
                await self.return_to_base()
            
            elif command == "emergency_land":
                await self.emergency_land()
            
            elif command == "start_delivery":
                await self.execute_delivery(
                    data["order_id"],
                    data["destination_lat"],
                    data["destination_lon"],
                    data["target_marker_id"]
                )
            
            elif command == "abort_landing":
                await self.landing_coordinator.abort_landing()
            
            elif command == "hold":
                await self.flight_controller.hold()
            
            else:
                logger.warning(f"Unknown command: {command}")
        
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            await self.backend_client.send_event("command_error", {
                "command": command,
                "error": str(e)
            })
    
    async def _telemetry_loop(self) -> None:
        """Background task to stream telemetry to backend."""
        while self._status.is_running:
            try:
                # Get telemetry
                telemetry = await self.flight_controller.get_telemetry()
                
                # Send to backend
                await self.backend_client.send_telemetry(telemetry)
                
                # Update status
                self._status.current_state = telemetry.state
                
                await asyncio.sleep(
                    self.config.communication.telemetry_update_interval_sec
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry loop error: {e}")
                await asyncio.sleep(1)
    
    def get_status(self) -> ModuleStatus:
        """Get current module status."""
        return self._status
    
    def get_marker_detection_frame(self) -> Optional[tuple]:
        """
        Get a frame with marker detections drawn.
        
        Returns:
            Tuple of (frame, detections) or None if unavailable
        """
        frame = self.camera_manager.get_bottom_frame()
        if frame is None:
            return None
        
        camera_id = self.camera_manager.config.bottom_camera_id
        camera_matrix = self.camera_manager.get_camera_matrix(camera_id)
        dist_coeffs = self.camera_manager.get_distortion_coeffs(camera_id)
        
        detections = self.marker_detector.detect_with_pose(
            frame, camera_matrix, dist_coeffs
        )
        
        annotated = self.marker_detector.draw_detections(
            frame, detections, True, camera_matrix, dist_coeffs
        )
        
        return annotated, detections


async def main():
    """Main entry point for running the Pi CV Module."""
    # Load configuration
    config = Config.from_environment()
    
    # Create and run module
    module = PiCvModule(config)
    await module.run()


if __name__ == "__main__":
    asyncio.run(main())
