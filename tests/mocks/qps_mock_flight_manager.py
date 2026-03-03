from typing import Optional

from src.config.qps_config import qpsConfig
from src.interfaces.qps_i_flight_manager import qpsIFlightManager
from src.models.qps_gps_position import qpsGPSPosition


class qpsMockFlightManager(qpsIFlightManager):
    """Mock flight manager for unit and integration testing.

    Returns configurable results for every operation and logs all
    commands for later assertion.
    """

    def __init__(self, config: qpsConfig) -> None:
        """Initialise the mock flight manager.

        Args:
            config: Application configuration (stored but largely unused).
        """
        self.config: qpsConfig = config
        self.arm_should_succeed: bool = True
        self.takeoff_should_succeed: bool = True
        self.mission_complete: bool = False
        self.command_log: list[str] = []
        self.simulated_position: Optional[qpsGPSPosition] = None

    # ------------------------------------------------------------------
    # qpsIFlightManager implementation
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Simulate a successful connection to the flight controller.

        Returns:
            bool: Always True.
        """
        self.command_log.append("connect")
        return True

    async def disconnect(self) -> None:
        """Simulate disconnection from the flight controller."""
        self.command_log.append("disconnect")

    async def arm(self) -> bool:
        """Simulate arming with a configurable result.

        Returns:
            bool: Value of self.arm_should_succeed.
        """
        self.command_log.append("arm")
        return self.arm_should_succeed

    async def disarm(self) -> bool:
        """Simulate disarming.

        Returns:
            bool: Always True.
        """
        self.command_log.append("disarm")
        return True

    async def takeoff(self, altitude_m: float) -> bool:
        """Simulate takeoff with a configurable result.

        Args:
            altitude_m: Target altitude (logged).

        Returns:
            bool: Value of self.takeoff_should_succeed.
        """
        self.command_log.append(f"takeoff:{altitude_m}")
        return self.takeoff_should_succeed

    async def land(self) -> bool:
        """Simulate a landing command.

        Returns:
            bool: Always True.
        """
        self.command_log.append("land")
        return True

    async def return_to_launch(self) -> bool:
        """Simulate a return-to-launch command.

        Returns:
            bool: Always True.
        """
        self.command_log.append("return_to_launch")
        return True

    async def upload_mission(self, waypoints: list[qpsGPSPosition]) -> bool:
        """Simulate uploading a waypoint mission.

        Args:
            waypoints: The waypoints (logged by count).

        Returns:
            bool: Always True.
        """
        self.command_log.append(f"upload_mission:{len(waypoints)}")
        return True

    async def start_mission(self) -> bool:
        """Simulate starting a mission.

        Returns:
            bool: Always True.
        """
        self.command_log.append("start_mission")
        return True

    async def is_mission_complete(self) -> bool:
        """Return the pre-configured mission-complete flag.

        Returns:
            bool: Value of self.mission_complete.
        """
        return self.mission_complete

    async def start_offboard(self) -> bool:
        """Simulate entering offboard mode.

        Returns:
            bool: Always True.
        """
        self.command_log.append("start_offboard")
        return True

    async def stop_offboard(self) -> bool:
        """Simulate exiting offboard mode.

        Returns:
            bool: Always True.
        """
        self.command_log.append("stop_offboard")
        return True

    async def send_velocity_ned(
        self, north_m_s: float, east_m_s: float, down_m_s: float
    ) -> bool:
        """Log a velocity setpoint.

        Args:
            north_m_s: North velocity.
            east_m_s: East velocity.
            down_m_s: Down velocity.

        Returns:
            bool: Always True.
        """
        self.command_log.append(f"velocity_ned:{north_m_s},{east_m_s},{down_m_s}")
        return True

    async def send_hover_setpoint(self) -> None:
        """Log a hover setpoint command."""
        self.command_log.append("hover_setpoint")

    # ------------------------------------------------------------------
    # Test-helper methods
    # ------------------------------------------------------------------

    def set_arm_result(self, success: bool) -> None:
        """Configure whether arm() should succeed.

        Args:
            success: Desired arm result.
        """
        self.arm_should_succeed = success

    def set_mission_complete(self, complete: bool) -> None:
        """Configure the return value of is_mission_complete().

        Args:
            complete: Desired mission-complete flag.
        """
        self.mission_complete = complete

    def get_command_log(self) -> list[str]:
        """Return the ordered list of commands received.

        Returns:
            list[str]: Command log entries.
        """
        return list(self.command_log)
