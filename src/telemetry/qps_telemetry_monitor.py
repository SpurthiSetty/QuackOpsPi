from typing import Optional, Callable

import mavsdk

from src.config.qps_config import qpsConfig
from src.models.qps_drone_state import qpsDroneState
from src.models.qps_gps_position import qpsGPSPosition


class qpsTelemetryMonitor:
    """Monitors real-time telemetry data from the drone via MAVSDK.

    Subscribes to MAVSDK telemetry streams for position, battery, flight
    status, and GPS info. Fires callbacks when battery thresholds are
    breached.
    """

    def __init__(self, drone: mavsdk.System, config: qpsConfig) -> None:
        """Initialise the telemetry monitor.

        Args:
            drone: The MAVSDK System instance to subscribe to.
            config: Application configuration with polling rates and
                battery thresholds.
        """
        self.drone: mavsdk.System = drone
        self.config: qpsConfig = config

        self.drone_state: Optional[qpsDroneState] = None
        self.gps_position: Optional[qpsGPSPosition] = None
        self.running: bool = False

        self._on_battery_warning: Optional[Callable] = None
        self._on_battery_critical: Optional[Callable] = None

    async def start(self) -> None:
        """Start all telemetry monitoring tasks.

        Launches async tasks for position, battery, flight status,
        and GPS info monitoring.
        """
        # TODO: Set self.running = True. Create asyncio tasks for
        #  _monitor_position, _monitor_battery, _monitor_flight_status,
        #  _monitor_gps_info.
        pass

    def stop(self) -> None:
        """Stop all telemetry monitoring tasks."""
        # TODO: Set self.running = False. Cancel all monitoring tasks.
        pass

    def get_drone_state(self) -> qpsDroneState:
        """Return the latest aggregated drone state.

        Returns:
            qpsDroneState: The most recent telemetry snapshot.
        """
        # TODO: Return self.drone_state.
        pass

    def get_gps_position(self) -> qpsGPSPosition:
        """Return the latest GPS position.

        Returns:
            qpsGPSPosition: The most recent GPS position reading.
        """
        # TODO: Return self.gps_position.
        pass

    def set_battery_warning_callback(self, callback: Callable) -> None:
        """Register a callback for low-battery warnings.

        The callback is invoked when battery drops below
        config.battery_warning_percent.

        Args:
            callback: A callable with no arguments.
        """
        # TODO: Store callback in self._on_battery_warning.
        pass

    def set_battery_critical_callback(self, callback: Callable) -> None:
        """Register a callback for critical-battery alerts.

        The callback is invoked when battery drops below
        config.battery_critical_percent.

        Args:
            callback: A callable with no arguments.
        """
        # TODO: Store callback in self._on_battery_critical.
        pass

    async def _monitor_position(self) -> None:
        """Subscribe to MAVSDK position telemetry and update gps_position.

        Runs continuously while self.running is True.
        """
        # TODO: async for position in self.drone.telemetry.position():
        #  Update self.gps_position with new readings.
        pass

    async def _monitor_battery(self) -> None:
        """Subscribe to MAVSDK battery telemetry and fire callbacks.

        Checks remaining percent against warning and critical thresholds.
        Runs continuously while self.running is True.
        """
        # TODO: async for battery in self.drone.telemetry.battery():
        #  Update battery_percent in drone_state. Fire callbacks if thresholds
        #  are breached.
        pass

    async def _monitor_flight_status(self) -> None:
        """Subscribe to MAVSDK flight mode and armed state telemetry.

        Runs continuously while self.running is True.
        """
        # TODO: Subscribe to in_air and armed streams. Update drone_state
        #  fields accordingly.
        pass

    async def _monitor_gps_info(self) -> None:
        """Subscribe to MAVSDK GPS info telemetry.

        Updates fix type and satellite count in drone state.
        Runs continuously while self.running is True.
        """
        # TODO: async for gps_info in self.drone.telemetry.gps_info():
        #  Update gps_fix_type and satellite_count in drone_state.
        pass
