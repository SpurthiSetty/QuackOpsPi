from abc import ABC, abstractmethod

from src.models.qps_gps_position import qpsGPSPosition


class qpsIFlightManager(ABC):
    """Abstract interface for all flight-control operations.

    Wraps the autopilot SDK (e.g. MAVSDK) so that production and mock
    implementations can be swapped via dependency injection.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Establish a connection to the flight controller.

        Returns:
            bool: True if the connection was established successfully.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the flight controller."""
        ...

    @abstractmethod
    async def arm(self) -> bool:
        """Arm the drone motors.

        Returns:
            bool: True if arming succeeded.
        """
        ...

    @abstractmethod
    async def disarm(self) -> bool:
        """Disarm the drone motors.

        Returns:
            bool: True if disarming succeeded.
        """
        ...

    @abstractmethod
    async def takeoff(self, altitude_m: float) -> bool:
        """Command the drone to take off to a specified altitude.

        Args:
            altitude_m: Target altitude in metres above the takeoff point.

        Returns:
            bool: True if the takeoff command was accepted.
        """
        ...

    @abstractmethod
    async def land(self) -> bool:
        """Command the drone to land at its current position.

        Returns:
            bool: True if the land command was accepted.
        """
        ...

    @abstractmethod
    async def return_to_launch(self) -> bool:
        """Command the drone to return to its launch position and land.

        Returns:
            bool: True if the RTL command was accepted.
        """
        ...

    @abstractmethod
    async def upload_mission(self, waypoints: list[qpsGPSPosition]) -> bool:
        """Upload a waypoint mission to the flight controller.

        Args:
            waypoints: Ordered list of GPS waypoints for the mission.

        Returns:
            bool: True if the mission was uploaded successfully.
        """
        ...

    @abstractmethod
    async def start_mission(self) -> bool:
        """Start the uploaded waypoint mission.

        Returns:
            bool: True if the mission was started successfully.
        """
        ...

    @abstractmethod
    async def is_mission_complete(self) -> bool:
        """Check whether the current waypoint mission has finished.

        Returns:
            bool: True if the mission has reached its final waypoint.
        """
        ...

    @abstractmethod
    async def start_offboard(self) -> bool:
        """Enter offboard control mode for direct velocity commands.

        Returns:
            bool: True if offboard mode was engaged successfully.
        """
        ...

    @abstractmethod
    async def stop_offboard(self) -> bool:
        """Exit offboard control mode.

        Returns:
            bool: True if offboard mode was disengaged successfully.
        """
        ...

    @abstractmethod
    async def send_velocity_ned(
        self, north_m_s: float, east_m_s: float, down_m_s: float
    ) -> bool:
        """Send a velocity setpoint in the NED (North-East-Down) frame.

        Args:
            north_m_s: Velocity component towards north in m/s.
            east_m_s: Velocity component towards east in m/s.
            down_m_s: Velocity component downward in m/s (positive = descend).

        Returns:
            bool: True if the setpoint was sent successfully.
        """
        ...

    @abstractmethod
    async def send_hover_setpoint(self) -> None:
        """Send a zero-velocity setpoint to hold the current position."""
        ...
