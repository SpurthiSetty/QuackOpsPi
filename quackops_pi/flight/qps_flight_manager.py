from __future__ import annotations

import asyncio
import logging
from typing import List

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.mission import MissionError, MissionItem, MissionPlan
from mavsdk.offboard import OffboardError, VelocityNedYaw

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager_interface import qpsFlightManagerInterface
from quackops_pi.models.qps_gps_position import qpsGPSPosition

logger = logging.getLogger("qps.flight")

_CONNECT_TIMEOUT_S: float = 30.0
_TAKEOFF_TOLERANCE_M: float = 1.5
_TAKEOFF_TIMEOUT_S: float = 30.0


class qpsFlightManager(qpsFlightManagerInterface):
    """Production flight manager — communicates with Pixhawk over serial via MAVSDK.

    Connection string from config defaults to ``serial:///dev/ttyAMA0:57600``
    (TELEM1 at MAVLink2 / 57600 baud).  Pass a UDP string for SITL.

    ArduPilot-specific notes:
    - ``pause_mission()``  → ``action.hold()``  (switches to GUIDED hold)
    - ``goto_location()``  → ``action.goto_location()`` (GUIDED mode goto)
    - Landed-state telemetry is unreliable on ArduPilot/MAVSDK; takeoff
      completion is detected by polling relative altitude instead.
    """

    def __init__(self, config: qpsConfig) -> None:
        self._config = config
        self._drone = System()
        self._connected = False

    # ── Connection ────────────────────────────────────────────────────

    async def connect(self) -> None:
        logger.info("Connecting to Pixhawk: %s", self._config.connection_string)
        await self._drone.connect(system_address=self._config.connection_string)

        try:
            async with asyncio.timeout(_CONNECT_TIMEOUT_S):
                async for state in self._drone.core.connection_state():
                    if state.is_connected:
                        self._connected = True
                        logger.info("Flight controller connected")
                        return
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"Timed out connecting to {self._config.connection_string} "
                f"after {_CONNECT_TIMEOUT_S:.0f} s"
            )

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Flight manager disconnected")

    # ── Arming ────────────────────────────────────────────────────────

    async def arm(self) -> None:
        logger.info("Arming drone")
        try:
            await self._drone.action.arm()
        except ActionError as exc:
            logger.error("Arm failed: %s", exc)
            raise
        logger.info("Drone armed")

    async def disarm(self) -> None:
        logger.info("Disarming drone")
        try:
            await self._drone.action.disarm()
        except ActionError as exc:
            logger.error("Disarm failed: %s", exc)
            raise
        logger.info("Drone disarmed")

    # ── Basic flight ──────────────────────────────────────────────────

    async def takeoff(self, altitude_m: float) -> None:
        logger.info("Taking off to %.1f m", altitude_m)
        try:
            await self._drone.action.set_takeoff_altitude(altitude_m)
            await self._drone.action.takeoff()
        except ActionError as exc:
            logger.error("Takeoff command failed: %s", exc)
            raise

        # ArduPilot landed_state / in_air telemetry is unreliable — poll altitude instead.
        try:
            await asyncio.wait_for(
                self._wait_for_altitude(altitude_m - _TAKEOFF_TOLERANCE_M),
                timeout=_TAKEOFF_TIMEOUT_S,
            )
            logger.info("Takeoff complete")
        except asyncio.TimeoutError:
            logger.warning(
                "Target altitude not reached within %.0f s — continuing", _TAKEOFF_TIMEOUT_S
            )

    async def land(self) -> None:
        logger.info("Landing")
        try:
            await self._drone.action.land()
        except ActionError as exc:
            logger.error("Land command failed: %s", exc)
            raise
        logger.info("Land command accepted")

    async def return_to_launch(self) -> None:
        logger.info("Return to launch")
        try:
            await self._drone.action.return_to_launch()
        except ActionError as exc:
            logger.error("RTL command failed: %s", exc)
            raise
        logger.info("RTL command accepted")

    # ── Waypoint missions ─────────────────────────────────────────────

    async def upload_mission(self, waypoints: List[qpsGPSPosition]) -> None:
        logger.info("Uploading mission: %d waypoints", len(waypoints))
        items = [
            MissionItem(
                latitude_deg=wp.lat,
                longitude_deg=wp.lon,
                relative_altitude_m=wp.altitude,
                speed_m_s=float("nan"),
                is_fly_through=True,
                gimbal_pitch_deg=float("nan"),
                gimbal_yaw_deg=float("nan"),
                camera_action=MissionItem.CameraAction.NONE,
                loiter_time_s=float("nan"),
                camera_photo_interval_s=float("nan"),
                acceptance_radius_m=float("nan"),
                yaw_deg=float("nan"),
                camera_photo_distance_m=float("nan"),
                vehicle_action=MissionItem.VehicleAction.NONE,
            )
            for wp in waypoints
        ]
        try:
            await self._drone.mission.upload_mission(MissionPlan(items))
        except MissionError as exc:
            logger.error("Mission upload failed: %s", exc)
            raise
        logger.info("Mission uploaded")

    async def start_mission(self) -> None:
        logger.info("Starting mission")
        try:
            await self._drone.mission.start_mission()
        except MissionError as exc:
            logger.error("Start mission failed: %s", exc)
            raise
        logger.info("Mission started")

    async def pause_mission(self) -> None:
        """Pause running mission. On ArduPilot this switches to GUIDED hold."""
        logger.info("Pausing mission (GUIDED hold)")
        try:
            await self._drone.action.hold()
        except ActionError as exc:
            logger.error("Pause mission failed: %s", exc)
            raise
        logger.info("Mission paused")

    async def is_mission_finished(self) -> bool:
        async for progress in self._drone.mission.mission_progress():
            finished = progress.total > 0 and progress.current >= progress.total
            logger.debug("Mission progress: %d / %d", progress.current, progress.total)
            return finished
        return False

    # ── Direct navigation ─────────────────────────────────────────────

    async def goto_location(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        yaw_deg: float = float("nan"),
    ) -> None:
        """Fly to a GPS position. On ArduPilot this uses GUIDED mode goto."""
        logger.info(
            "goto_location lat=%.6f lon=%.6f alt=%.1f m yaw=%.1f°",
            latitude_deg, longitude_deg, altitude_m, yaw_deg,
        )
        try:
            await self._drone.action.goto_location(
                latitude_deg, longitude_deg, altitude_m, yaw_deg
            )
        except ActionError as exc:
            logger.error("goto_location failed: %s", exc)
            raise
        logger.info("goto_location command accepted")

    # ── Offboard control ──────────────────────────────────────────────

    async def start_offboard(self) -> None:
        """Enter offboard mode. Sends a zero-velocity setpoint first (required by PX4)."""
        logger.info("Starting offboard mode")
        # A valid setpoint must exist before offboard can be engaged.
        await self._drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        try:
            await self._drone.offboard.start()
        except OffboardError as exc:
            logger.error("start_offboard failed: %s", exc)
            raise
        logger.info("Offboard mode started")

    async def stop_offboard(self) -> None:
        logger.info("Stopping offboard mode")
        try:
            await self._drone.offboard.stop()
        except OffboardError as exc:
            logger.error("stop_offboard failed: %s", exc)
            raise
        logger.info("Offboard mode stopped")

    async def send_velocity_ned(
        self,
        north_m_s: float,
        east_m_s: float,
        down_m_s: float,
        yaw_deg_s: float = 0.0,
    ) -> None:
        await self._drone.offboard.set_velocity_ned(
            VelocityNedYaw(north_m_s, east_m_s, down_m_s, yaw_deg_s)
        )

    async def send_hover_setpoint(self) -> None:
        await self.send_velocity_ned(0.0, 0.0, 0.0, 0.0)

    # ── Helpers ───────────────────────────────────────────────────────

    async def _wait_for_altitude(self, target_m: float) -> None:
        """Stream position telemetry until relative altitude >= target_m."""
        async for pos in self._drone.telemetry.position():
            if pos.relative_altitude_m >= target_m:
                logger.debug("Altitude reached: %.1f m", pos.relative_altitude_m)
                return
