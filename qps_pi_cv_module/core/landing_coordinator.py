"""
Landing Coordinator for precision vision-based landing.

This module coordinates the landing sequence using computer vision
and flight control to achieve precision landing on ArUco markers.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass

from ..core.interfaces import (
    ILandingCoordinator,
    IFlightController,
    IMarkerDetector,
    ICameraManager,
    IBackendClient,
    LandingPhase,
    MarkerDetection,
)
from ..core.config import LandingConfig

logger = logging.getLogger(__name__)


@dataclass
class LandingState:
    """Internal state for landing coordination."""
    target_marker_id: int = -1
    phase: LandingPhase = LandingPhase.NOT_STARTED
    start_time: float = 0.0
    last_detection_time: float = 0.0
    last_detection: Optional[MarkerDetection] = None
    abort_reason: Optional[str] = None
    final_accuracy_cm: Optional[float] = None


class LandingCoordinator(ILandingCoordinator):
    """
    Coordinates precision landing using computer vision.
    
    This class orchestrates the entire landing sequence:
    1. Approach phase - descend to approach altitude
    2. Search phase - look for target marker
    3. Acquire phase - lock onto marker
    4. Descent phase - controlled descent with visual tracking
    5. Final approach - slow descent to touchdown
    6. Touchdown - complete landing
    
    The coordinator uses feedback from the marker detector to
    adjust the drone's position during descent, ensuring accurate
    landing on the designated marker.
    
    Attributes:
        config: Landing configuration settings
        flight_controller: Flight controller interface
        marker_detector: Marker detection interface
        camera_manager: Camera management interface
        backend_client: Backend communication interface
        
    Example:
        >>> coordinator = LandingCoordinator(
        ...     config, flight_controller, marker_detector, 
        ...     camera_manager, backend_client
        ... )
        >>> success = await coordinator.start_landing_sequence(target_marker_id=42)
    """
    
    def __init__(
        self,
        config: LandingConfig,
        flight_controller: IFlightController,
        marker_detector: IMarkerDetector,
        camera_manager: ICameraManager,
        backend_client: Optional[IBackendClient] = None
    ):
        """
        Initialize the landing coordinator.
        
        Args:
            config: Landing configuration settings
            flight_controller: Flight controller interface
            marker_detector: Marker detection interface
            camera_manager: Camera management interface
            backend_client: Optional backend communication interface
        """
        self.config = config
        self._flight_controller = flight_controller
        self._marker_detector = marker_detector
        self._camera_manager = camera_manager
        self._backend_client = backend_client
        
        # Internal state
        self._state = LandingState()
        self._is_landing = False
        self._abort_requested = False
        
        # Vision processing task
        self._vision_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._phase_callbacks: list = []
        
        # PID controllers for position correction
        self._x_pid = SimplePID(kp=0.3, ki=0.01, kd=0.1)
        self._y_pid = SimplePID(kp=0.3, ki=0.01, kd=0.1)
        
        logger.info("LandingCoordinator initialized")
    
    async def start_landing_sequence(self, target_marker_id: int) -> bool:
        """
        Start the precision landing sequence.
        
        Args:
            target_marker_id: The ArUco marker ID to land on
            
        Returns:
            True if landing completed successfully
        """
        if self._is_landing:
            logger.warning("Landing sequence already in progress")
            return False
        
        logger.info(f"Starting landing sequence for marker {target_marker_id}")
        
        # Initialize state
        self._state = LandingState(
            target_marker_id=target_marker_id,
            phase=LandingPhase.APPROACHING,
            start_time=time.time()
        )
        self._is_landing = True
        self._abort_requested = False
        
        # Reset PID controllers
        self._x_pid.reset()
        self._y_pid.reset()
        
        try:
            # Phase 1: Approach altitude
            await self._notify_phase_change(LandingPhase.APPROACHING)
            
            if not await self._approach_phase():
                return await self._handle_abort("Approach phase failed")
            
            # Phase 2: Search for marker
            await self._notify_phase_change(LandingPhase.SEARCHING_MARKER)
            
            if not await self._search_phase():
                return await self._handle_abort("Marker not found")
            
            # Phase 3: Acquire marker lock
            await self._notify_phase_change(LandingPhase.MARKER_ACQUIRED)
            
            # Start offboard control for precision landing
            if not await self._flight_controller.start_offboard():
                return await self._handle_abort("Failed to start offboard mode")
            
            try:
                # Phase 4: Controlled descent
                await self._notify_phase_change(LandingPhase.DESCENDING)
                
                if not await self._descent_phase():
                    return await self._handle_abort("Descent phase failed")
                
                # Phase 5: Final approach
                await self._notify_phase_change(LandingPhase.FINAL_APPROACH)
                
                if not await self._final_approach_phase():
                    return await self._handle_abort("Final approach failed")
                
            finally:
                await self._flight_controller.stop_offboard()
            
            # Phase 6: Touchdown
            await self._notify_phase_change(LandingPhase.TOUCHDOWN)
            
            if not await self._touchdown_phase():
                return await self._handle_abort("Touchdown failed")
            
            # Success!
            await self._notify_phase_change(LandingPhase.COMPLETED)
            
            elapsed = time.time() - self._state.start_time
            logger.info(
                f"Landing completed successfully in {elapsed:.1f}s, "
                f"accuracy: {self._state.final_accuracy_cm:.1f}cm"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Landing sequence error: {e}")
            return await self._handle_abort(str(e))
        
        finally:
            self._is_landing = False
    
    async def abort_landing(self) -> bool:
        """
        Abort the current landing sequence.
        
        Returns:
            True if abort was successful
        """
        if not self._is_landing:
            return True
        
        logger.warning("Landing abort requested")
        self._abort_requested = True
        
        # Stop any velocity commands
        await self._flight_controller.set_velocity(0, 0, 0)
        
        # Hold position
        await self._flight_controller.hold()
        
        await self._notify_phase_change(LandingPhase.ABORTED)
        
        return True
    
    def get_current_phase(self) -> LandingPhase:
        """Get the current landing phase."""
        return self._state.phase
    
    def is_landing_in_progress(self) -> bool:
        """Check if a landing sequence is currently in progress."""
        return self._is_landing
    
    def register_phase_callback(
        self, 
        callback: Callable[[LandingPhase, Optional[dict]], Any]
    ) -> None:
        """Register callback for phase changes."""
        self._phase_callbacks.append(callback)
    
    async def _approach_phase(self) -> bool:
        """Execute approach phase - descend to approach altitude."""
        logger.info(f"Approaching at {self.config.approach_altitude_m}m")
        
        # Get current telemetry
        telemetry = await self._flight_controller.get_telemetry()
        current_alt = telemetry.position.relative_altitude_m
        
        # Descend to approach altitude
        while current_alt > self.config.approach_altitude_m + 0.5:
            if self._abort_requested:
                return False
            
            # Descend slowly
            await self._flight_controller.set_velocity(
                0, 0, self.config.landing_descent_rate_mps
            )
            
            await asyncio.sleep(0.1)
            telemetry = await self._flight_controller.get_telemetry()
            current_alt = telemetry.position.relative_altitude_m
        
        # Stop descent
        await self._flight_controller.set_velocity(0, 0, 0)
        
        return True
    
    async def _search_phase(self) -> bool:
        """Execute search phase - look for target marker."""
        logger.info(f"Searching for marker {self._state.target_marker_id}")
        
        timeout = self.config.landing_timeout_sec
        start = time.time()
        
        while time.time() - start < timeout:
            if self._abort_requested:
                return False
            
            # Get frame from bottom camera
            frame = self._camera_manager.get_bottom_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            
            # Detect markers
            detections = self._marker_detector.detect(frame)
            
            # Look for target marker
            target = self._marker_detector.find_target_marker(
                detections, 
                self._state.target_marker_id
            )
            
            if target is not None:
                self._state.last_detection = target
                self._state.last_detection_time = time.time()
                logger.info(f"Marker {self._state.target_marker_id} found!")
                return True
            
            await asyncio.sleep(0.1)
        
        logger.warning("Marker search timeout")
        return False
    
    async def _descent_phase(self) -> bool:
        """Execute descent phase with visual tracking."""
        logger.info("Beginning controlled descent")
        
        telemetry = await self._flight_controller.get_telemetry()
        current_alt = telemetry.position.relative_altitude_m
        
        # Get camera info for pose estimation
        camera_matrix = self._camera_manager.get_camera_matrix(
            self._camera_manager.config.bottom_camera_id
        )
        dist_coeffs = self._camera_manager.get_distortion_coeffs(
            self._camera_manager.config.bottom_camera_id
        )
        
        # Get frame dimensions for center calculation
        frame = self._camera_manager.get_bottom_frame()
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width / 2
            frame_center_y = frame_height / 2
        else:
            frame_center_x = self._camera_manager.config.resolution_width / 2
            frame_center_y = self._camera_manager.config.resolution_height / 2
        
        marker_lost_time = 0.0
        
        while current_alt > self.config.final_descent_altitude_m:
            if self._abort_requested:
                return False
            
            # Get frame and detect marker
            frame = self._camera_manager.get_bottom_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            
            detections = self._marker_detector.detect_with_pose(
                frame, camera_matrix, dist_coeffs
            )
            
            target = self._marker_detector.find_target_marker(
                detections,
                self._state.target_marker_id
            )
            
            if target is not None:
                # Update state
                self._state.last_detection = target
                self._state.last_detection_time = time.time()
                marker_lost_time = 0.0
                
                # Calculate position error from frame center
                error_x = target.center_x - frame_center_x
                error_y = target.center_y - frame_center_y
                
                # Apply PID control to calculate velocity corrections
                # Invert because marker right of center means drone needs to go right
                velocity_east = self._x_pid.update(error_x) * -0.01
                velocity_north = self._y_pid.update(error_y) * 0.01
                
                # Clamp velocities
                max_vel = self.config.max_correction_velocity_mps
                velocity_east = max(-max_vel, min(max_vel, velocity_east))
                velocity_north = max(-max_vel, min(max_vel, velocity_north))
                
                # Apply descent and correction
                await self._flight_controller.set_velocity(
                    velocity_north,
                    velocity_east,
                    self.config.landing_descent_rate_mps
                )
                
            else:
                # Marker lost
                if marker_lost_time == 0.0:
                    marker_lost_time = time.time()
                
                # Check if lost too long
                if time.time() - marker_lost_time > self.config.abort_if_marker_lost_sec:
                    logger.warning("Marker lost for too long, aborting")
                    return False
                
                # Continue descending slowly without correction
                await self._flight_controller.set_velocity(
                    0, 0, self.config.landing_descent_rate_mps * 0.5
                )
            
            await asyncio.sleep(0.05)
            
            telemetry = await self._flight_controller.get_telemetry()
            current_alt = telemetry.position.relative_altitude_m
        
        return True
    
    async def _final_approach_phase(self) -> bool:
        """Execute final approach - slow descent to touchdown."""
        logger.info("Final approach")
        
        # Slower descent rate for final approach
        final_descent_rate = self.config.landing_descent_rate_mps * 0.5
        
        # Continue tracking and descending
        telemetry = await self._flight_controller.get_telemetry()
        current_alt = telemetry.position.relative_altitude_m
        
        while current_alt > 0.3:  # Until very close to ground
            if self._abort_requested:
                return False
            
            # Get frame and detect
            frame = self._camera_manager.get_bottom_frame()
            if frame is not None:
                detections = self._marker_detector.detect(frame)
                target = self._marker_detector.find_target_marker(
                    detections,
                    self._state.target_marker_id
                )
                
                if target is not None:
                    self._state.last_detection = target
                    self._state.last_detection_time = time.time()
            
            # Descend slowly
            await self._flight_controller.set_velocity(0, 0, final_descent_rate)
            
            await asyncio.sleep(0.05)
            telemetry = await self._flight_controller.get_telemetry()
            current_alt = telemetry.position.relative_altitude_m
        
        # Calculate final accuracy if we have a recent detection
        if self._state.last_detection and self._state.last_detection.distance_cm:
            self._state.final_accuracy_cm = self._state.last_detection.distance_cm
        else:
            self._state.final_accuracy_cm = -1.0
        
        return True
    
    async def _touchdown_phase(self) -> bool:
        """Execute touchdown - command final landing."""
        logger.info("Touchdown")
        
        # Stop offboard control and command land
        await self._flight_controller.stop_offboard()
        
        # Use the standard land command for final touchdown
        return await self._flight_controller.land()
    
    async def _handle_abort(self, reason: str) -> bool:
        """Handle landing abort."""
        self._state.abort_reason = reason
        self._state.phase = LandingPhase.ABORTED
        
        logger.error(f"Landing aborted: {reason}")
        
        # Notify backend
        if self._backend_client:
            await self._backend_client.send_landing_status(
                LandingPhase.ABORTED,
                marker_detected=False,
                details={"reason": reason}
            )
        
        # Notify callbacks
        for callback in self._phase_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(LandingPhase.ABORTED, {"reason": reason})
                else:
                    callback(LandingPhase.ABORTED, {"reason": reason})
            except Exception as e:
                logger.error(f"Phase callback error: {e}")
        
        self._is_landing = False
        return False
    
    async def _notify_phase_change(self, phase: LandingPhase) -> None:
        """Notify all listeners of phase change."""
        self._state.phase = phase
        
        details = {
            "marker_id": self._state.target_marker_id,
            "elapsed_time": time.time() - self._state.start_time,
        }
        
        if self._state.last_detection:
            details["last_detection_age"] = time.time() - self._state.last_detection_time
        
        # Notify backend
        if self._backend_client:
            await self._backend_client.send_landing_status(
                phase,
                marker_detected=self._state.last_detection is not None,
                details=details
            )
        
        # Notify callbacks
        for callback in self._phase_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(phase, details)
                else:
                    callback(phase, details)
            except Exception as e:
                logger.error(f"Phase callback error: {e}")


class SimplePID:
    """Simple PID controller for position correction."""
    
    def __init__(self, kp: float, ki: float, kd: float):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()
    
    def update(self, error: float) -> float:
        """
        Update PID controller with current error.
        
        Args:
            error: Current error value
            
        Returns:
            Control output
        """
        current_time = time.time()
        dt = current_time - self._last_time
        
        if dt <= 0:
            dt = 0.01
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self._integral += error * dt
        self._integral = max(-100, min(100, self._integral))  # Clamp
        i_term = self.ki * self._integral
        
        # Derivative term
        derivative = (error - self._last_error) / dt
        d_term = self.kd * derivative
        
        # Update state
        self._last_error = error
        self._last_time = current_time
        
        return p_term + i_term + d_term
    
    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()
