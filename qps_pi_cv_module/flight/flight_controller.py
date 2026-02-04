"""
Flight Controller implementation using MAVSDK.

This module provides high-level flight control commands that wrap
the MAVSDK library for communication with PX4 autopilot systems.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass

from ..core.interfaces import (
    IFlightController,
    DroneState,
    DronePosition,
    DroneVelocity,
    DroneTelemetry,
)
from ..core.config import FlightConfig

logger = logging.getLogger(__name__)


class FlightController(IFlightController):
    """
    Flight controller interface using MAVSDK for PX4 communication.
    
    This class provides an async interface to control the drone through
    MAVSDK, wrapping common flight operations like takeoff, land,
    goto, and velocity control.
    
    Attributes:
        config: Flight configuration settings
        drone: MAVSDK System instance
        state: Current drone state
        
    Example:
        >>> controller = FlightController(config)
        >>> await controller.connect()
        >>> await controller.arm()
        >>> await controller.takeoff(5.0)
        >>> await controller.land()
    """
    
    def __init__(self, config: FlightConfig):
        """
        Initialize the flight controller.
        
        Args:
            config: Flight configuration settings
        """
        self.config = config
        self._drone = None
        self._state = DroneState.DISCONNECTED
        self._is_connected = False
        
        # Telemetry cache
        self._latest_position: Optional[DronePosition] = None
        self._latest_velocity: Optional[DroneVelocity] = None
        self._battery_percent: float = 0.0
        self._is_armed: bool = False
        self._is_in_air: bool = False
        self._flight_mode: str = "UNKNOWN"
        self._gps_fix_type: int = 0
        self._satellite_count: int = 0
        
        # Telemetry update tasks
        self._telemetry_tasks: list = []
        
        # Callbacks
        self._state_change_callbacks: list = []
        
        logger.info(
            f"FlightController initialized (address={config.mavsdk_system_address})"
        )
    
    async def connect(self) -> bool:
        """
        Establish connection to the flight controller.
        
        Returns:
            True if connection established successfully
        """
        if self._is_connected:
            logger.warning("Already connected to flight controller")
            return True
        
        try:
            from mavsdk import System
            
            self._set_state(DroneState.CONNECTING)
            
            self._drone = System()
            await self._drone.connect(
                system_address=self.config.mavsdk_system_address
            )
            
            # Wait for connection with timeout
            logger.info("Waiting for drone connection...")
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    logger.info("Connected to drone")
                    break
            
            # Start telemetry tasks
            await self._start_telemetry_listeners()
            
            self._is_connected = True
            self._set_state(DroneState.IDLE)
            
            return True
            
        except ImportError:
            logger.error("MAVSDK not installed. Install with: pip install mavsdk")
            self._set_state(DroneState.ERROR)
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._set_state(DroneState.ERROR)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the flight controller."""
        # Cancel telemetry tasks
        for task in self._telemetry_tasks:
            task.cancel()
        self._telemetry_tasks.clear()
        
        self._is_connected = False
        self._drone = None
        self._set_state(DroneState.DISCONNECTED)
        logger.info("Disconnected from flight controller")
    
    async def arm(self) -> bool:
        """
        Arm the drone.
        
        Returns:
            True if armed successfully
        """
        if not self._is_connected:
            logger.error("Cannot arm: not connected")
            return False
        
        try:
            self._set_state(DroneState.ARMING)
            await self._drone.action.arm()
            logger.info("Drone armed")
            self._is_armed = True
            return True
        except Exception as e:
            logger.error(f"Failed to arm: {e}")
            self._set_state(DroneState.ERROR)
            return False
    
    async def disarm(self) -> bool:
        """
        Disarm the drone.
        
        Returns:
            True if disarmed successfully
        """
        if not self._is_connected:
            return False
        
        try:
            await self._drone.action.disarm()
            logger.info("Drone disarmed")
            self._is_armed = False
            self._set_state(DroneState.IDLE)
            return True
        except Exception as e:
            logger.error(f"Failed to disarm: {e}")
            return False
    
    async def takeoff(self, altitude_m: float) -> bool:
        """
        Command the drone to take off.
        
        Args:
            altitude_m: Target altitude in meters
            
        Returns:
            True if takeoff initiated successfully
        """
        if not self._is_connected:
            logger.error("Cannot takeoff: not connected")
            return False
        
        try:
            self._set_state(DroneState.TAKING_OFF)
            
            # Set takeoff altitude
            await self._drone.action.set_takeoff_altitude(altitude_m)
            
            # Initiate takeoff
            await self._drone.action.takeoff()
            logger.info(f"Takeoff initiated (target altitude: {altitude_m}m)")
            
            # Wait for drone to reach altitude
            await self._wait_for_altitude(altitude_m, tolerance=0.5)
            
            self._set_state(DroneState.IN_FLIGHT)
            return True
            
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            self._set_state(DroneState.ERROR)
            return False
    
    async def land(self) -> bool:
        """
        Command the drone to land.
        
        Returns:
            True if landing initiated successfully
        """
        if not self._is_connected:
            return False
        
        try:
            self._set_state(DroneState.LANDING)
            await self._drone.action.land()
            logger.info("Landing initiated")
            
            # Wait for landing to complete
            await self._wait_for_landed()
            
            self._set_state(DroneState.LANDED)
            return True
            
        except Exception as e:
            logger.error(f"Landing failed: {e}")
            self._set_state(DroneState.ERROR)
            return False
    
    async def goto(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        heading_deg: Optional[float] = None
    ) -> bool:
        """
        Command the drone to go to a specific location.
        
        Args:
            latitude: Target latitude in degrees
            longitude: Target longitude in degrees
            altitude_m: Target altitude in meters (MSL)
            heading_deg: Optional heading in degrees
            
        Returns:
            True if goto command accepted
        """
        if not self._is_connected:
            return False
        
        try:
            self._set_state(DroneState.NAVIGATING)
            
            yaw_deg = heading_deg if heading_deg is not None else float('nan')
            
            await self._drone.action.goto_location(
                latitude,
                longitude,
                altitude_m,
                yaw_deg
            )
            
            logger.info(
                f"Navigating to ({latitude:.6f}, {longitude:.6f}) "
                f"at {altitude_m}m"
            )
            return True
            
        except Exception as e:
            logger.error(f"Goto command failed: {e}")
            return False
    
    async def set_velocity(
        self,
        velocity_north_mps: float,
        velocity_east_mps: float,
        velocity_down_mps: float,
        yaw_deg: Optional[float] = None
    ) -> bool:
        """
        Set the drone's velocity vector (NED frame).
        
        Args:
            velocity_north_mps: Velocity north in m/s
            velocity_east_mps: Velocity east in m/s  
            velocity_down_mps: Velocity down in m/s
            yaw_deg: Optional yaw angle in degrees
            
        Returns:
            True if velocity set successfully
        """
        if not self._is_connected:
            return False
        
        try:
            # Use offboard mode for velocity control
            from mavsdk.offboard import VelocityNedYaw
            
            if yaw_deg is None:
                # Maintain current heading
                yaw_deg = self._latest_position.heading_deg if self._latest_position else 0.0
            
            await self._drone.offboard.set_velocity_ned(
                VelocityNedYaw(
                    velocity_north_mps,
                    velocity_east_mps,
                    velocity_down_mps,
                    yaw_deg
                )
            )
            return True
            
        except Exception as e:
            logger.error(f"Set velocity failed: {e}")
            return False
    
    async def start_offboard(self) -> bool:
        """Start offboard control mode."""
        if not self._is_connected:
            return False
        
        try:
            from mavsdk.offboard import VelocityNedYaw
            
            # Set initial setpoint (required before starting offboard)
            await self._drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            )
            
            await self._drone.offboard.start()
            logger.info("Offboard mode started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start offboard: {e}")
            return False
    
    async def stop_offboard(self) -> bool:
        """Stop offboard control mode."""
        if not self._is_connected:
            return False
        
        try:
            await self._drone.offboard.stop()
            logger.info("Offboard mode stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop offboard: {e}")
            return False
    
    async def get_telemetry(self) -> DroneTelemetry:
        """
        Get current telemetry data.
        
        Returns:
            DroneTelemetry with latest available data
        """
        position = self._latest_position or DronePosition(
            latitude=0.0,
            longitude=0.0,
            altitude_m=0.0,
            relative_altitude_m=0.0,
            heading_deg=0.0,
            timestamp=time.time()
        )
        
        velocity = self._latest_velocity or DroneVelocity(
            velocity_north_mps=0.0,
            velocity_east_mps=0.0,
            velocity_down_mps=0.0,
            groundspeed_mps=0.0,
            timestamp=time.time()
        )
        
        return DroneTelemetry(
            position=position,
            velocity=velocity,
            battery_percent=self._battery_percent,
            is_armed=self._is_armed,
            is_in_air=self._is_in_air,
            flight_mode=self._flight_mode,
            gps_fix_type=self._gps_fix_type,
            satellite_count=self._satellite_count,
            state=self._state
        )
    
    async def return_to_launch(self) -> bool:
        """
        Command the drone to return to launch point.
        
        Returns:
            True if RTL command accepted
        """
        if not self._is_connected:
            return False
        
        try:
            self._set_state(DroneState.RETURNING_TO_BASE)
            await self._drone.action.return_to_launch()
            logger.info("Return to launch initiated")
            return True
        except Exception as e:
            logger.error(f"RTL failed: {e}")
            return False
    
    async def hold(self) -> bool:
        """Command the drone to hold current position."""
        if not self._is_connected:
            return False
        
        try:
            await self._drone.action.hold()
            logger.info("Hold command sent")
            return True
        except Exception as e:
            logger.error(f"Hold failed: {e}")
            return False
    
    async def _start_telemetry_listeners(self) -> None:
        """Start background tasks to listen for telemetry updates."""
        # Position listener
        self._telemetry_tasks.append(
            asyncio.create_task(self._position_listener())
        )
        
        # Velocity listener
        self._telemetry_tasks.append(
            asyncio.create_task(self._velocity_listener())
        )
        
        # Battery listener
        self._telemetry_tasks.append(
            asyncio.create_task(self._battery_listener())
        )
        
        # Flight mode listener
        self._telemetry_tasks.append(
            asyncio.create_task(self._flight_mode_listener())
        )
        
        # GPS info listener
        self._telemetry_tasks.append(
            asyncio.create_task(self._gps_listener())
        )
        
        # Armed/In-air listener
        self._telemetry_tasks.append(
            asyncio.create_task(self._armed_listener())
        )
    
    async def _position_listener(self) -> None:
        """Listen for position updates."""
        try:
            async for position in self._drone.telemetry.position():
                self._latest_position = DronePosition(
                    latitude=position.latitude_deg,
                    longitude=position.longitude_deg,
                    altitude_m=position.absolute_altitude_m,
                    relative_altitude_m=position.relative_altitude_m,
                    heading_deg=0.0,  # Updated by heading listener
                    timestamp=time.time()
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Position listener error: {e}")
    
    async def _velocity_listener(self) -> None:
        """Listen for velocity updates."""
        try:
            async for velocity in self._drone.telemetry.velocity_ned():
                self._latest_velocity = DroneVelocity(
                    velocity_north_mps=velocity.north_m_s,
                    velocity_east_mps=velocity.east_m_s,
                    velocity_down_mps=velocity.down_m_s,
                    groundspeed_mps=(
                        velocity.north_m_s**2 + velocity.east_m_s**2
                    ) ** 0.5,
                    timestamp=time.time()
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Velocity listener error: {e}")
    
    async def _battery_listener(self) -> None:
        """Listen for battery updates."""
        try:
            async for battery in self._drone.telemetry.battery():
                self._battery_percent = battery.remaining_percent * 100
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Battery listener error: {e}")
    
    async def _flight_mode_listener(self) -> None:
        """Listen for flight mode changes."""
        try:
            async for mode in self._drone.telemetry.flight_mode():
                self._flight_mode = str(mode)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Flight mode listener error: {e}")
    
    async def _gps_listener(self) -> None:
        """Listen for GPS info updates."""
        try:
            async for gps in self._drone.telemetry.gps_info():
                self._gps_fix_type = gps.fix_type.value
                self._satellite_count = gps.num_satellites
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"GPS listener error: {e}")
    
    async def _armed_listener(self) -> None:
        """Listen for armed state and in-air updates."""
        try:
            async for armed in self._drone.telemetry.armed():
                self._is_armed = armed
            async for in_air in self._drone.telemetry.in_air():
                self._is_in_air = in_air
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Armed listener error: {e}")
    
    async def _wait_for_altitude(
        self, 
        target_altitude: float, 
        tolerance: float = 1.0,
        timeout: float = 60.0
    ) -> bool:
        """Wait for drone to reach target altitude."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._latest_position:
                current_alt = self._latest_position.relative_altitude_m
                if abs(current_alt - target_altitude) <= tolerance:
                    return True
            await asyncio.sleep(0.5)
        
        logger.warning(f"Timeout waiting for altitude {target_altitude}m")
        return False
    
    async def _wait_for_landed(self, timeout: float = 60.0) -> bool:
        """Wait for drone to land."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self._is_in_air:
                return True
            await asyncio.sleep(0.5)
        
        logger.warning("Timeout waiting for landing")
        return False
    
    def _set_state(self, state: DroneState) -> None:
        """Set state and notify callbacks."""
        old_state = self._state
        self._state = state
        
        if old_state != state:
            logger.debug(f"State changed: {old_state.name} -> {state.name}")
            for callback in self._state_change_callbacks:
                try:
                    callback(old_state, state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")
    
    def register_state_callback(
        self, 
        callback: Callable[[DroneState, DroneState], None]
    ) -> None:
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    @property
    def state(self) -> DroneState:
        """Get current drone state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to flight controller."""
        return self._is_connected
    
    @property
    def is_armed(self) -> bool:
        """Check if drone is armed."""
        return self._is_armed
    
    @property
    def is_in_air(self) -> bool:
        """Check if drone is airborne."""
        return self._is_in_air
