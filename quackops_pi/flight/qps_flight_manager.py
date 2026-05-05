"""
qps_flight_manager.py

GPS-based production flight manager.
Extends qpsFlightManagerBase with GPS-dependent flight operations.

Connection strings:
    Serial (Pi hardware) : "serial:///dev/ttyAMA0:57600"
    SITL (TCP)           : "tcp:localhost:5763"
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from pymavlink import mavutil, mavwp

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager_base import qpsFlightManagerBase
from quackops_pi.models.qps_gps_position import qpsGPSPosition

logger = logging.getLogger("qps.flight_manager")


class qpsFlightManager(qpsFlightManagerBase):
    """GPS-based production flight manager.

    Implements takeoff, landing, waypoint missions, and direct navigation
    using GUIDED and AUTO modes with GPS position feedback.
    All shared pymavlink infrastructure is inherited from qpsFlightManagerBase.
    """

    def __init__(self, config: qpsConfig) -> None:
        super().__init__(config)

    # ── Basic flight ──────────────────────────────────────────────────

    async def takeoff(self, altitude_m: float) -> None:
        """Command takeoff via MAV_CMD_NAV_TAKEOFF (must already be in GUIDED mode).

        Waits until GLOBAL_POSITION_INT reports relative_alt >= 90% of target.
        Logs a warning (does not raise) if the target is not reached in 30s.
        """
        logger.info("Taking off to %.1fm...", altitude_m)
        self._gps_event.clear()

        await self._send_and_await_ack(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            lambda: self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,              # confirmation
                0, 0, 0, 0,     # params 1-4 unused
                0, 0,           # params 5-6 unused (lat/lon ignored for copter)
                altitude_m,     # param7 = target altitude (m)
            ),
            timeout=5.0,
        )

        await self._wait_for_altitude(altitude_m * 0.90)

    async def land(self) -> None:
        """Switch to LAND mode and wait for touchdown (alt < 1m, vz < 0.3 m/s)."""
        logger.info("Landing...")
        self._land_event.clear()
        await self.set_mode(self.MODE_LAND)
        try:
            await asyncio.wait_for(self._land_event.wait(), timeout=120.0)
            logger.info("Landed!")
        except asyncio.TimeoutError:
            logger.warning("Land timeout — drone may still be descending")

    async def return_to_launch(self) -> None:
        """Switch to RTL mode. Does not block until landing."""
        logger.info("Return to launch...")
        await self.set_mode(self.MODE_RTL)
        logger.info("RTL mode active")

    # ── Waypoint missions ─────────────────────────────────────────────

    async def upload_mission(self, waypoints: List[qpsGPSPosition]) -> None:
        """Upload a waypoint mission via the MAVLink mission protocol.

        Automatically prepends:
            WP0 — HOME  (lat/lon/alt auto-filled by ArduPilot)
            WP1 — NAV_TAKEOFF  (altitude = first waypoint's altitude)

        Then appends the caller's waypoints as NAV_WAYPOINT items.
        Uses MISSION_ITEM_INT (not the deprecated MISSION_ITEM).

        Raises RuntimeError if the upload handshake times out or is rejected.
        """
        if not waypoints:
            raise ValueError("upload_mission requires at least one waypoint")

        logger.info("Uploading mission: %d nav waypoints", len(waypoints))
        takeoff_alt = int(waypoints[0].altitude_m)
        wp_loader = mavwp.MAVWPLoader()
        seq = 0

        # WP0 — Home (FC fills lat/lon/alt from current home position)
        wp_loader.add(mavutil.mavlink.MAVLink_mission_item_int_message(
            self._mav.target_system, self._mav.target_component,
            seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1,           # current=0, autocontinue=1
            0, 0, 0, 0,     # hold, accept_radius, pass_radius, yaw
            0, 0, 0,        # lat, lon, alt  (auto-filled)
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        ))
        seq += 1

        # WP1 — Takeoff
        wp_loader.add(mavutil.mavlink.MAVLink_mission_item_int_message(
            self._mav.target_system, self._mav.target_component,
            seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 1,
            0, 0, 0, 0,         # params (pitch, empty, empty, yaw)
            0, 0, takeoff_alt,  # lat/lon ignored for copter takeoff; alt required
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        ))
        seq += 1

        # WP2+ — Navigation waypoints from caller
        for wp in waypoints:
            wp_loader.add(mavutil.mavlink.MAVLink_mission_item_int_message(
                self._mav.target_system, self._mav.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 1,
                0,       # hold time (s)
                2.0,     # acceptance radius (m)
                0, 0,    # pass_radius, yaw
                int(wp.latitude_deg * 1e7),
                int(wp.longitude_deg * 1e7),
                wp.altitude_m,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            ))
            seq += 1

        self._mission_total_items = wp_loader.count()

        # Clear stale mission protocol state
        self._mission_ack_event.clear()
        while not self._mission_request_queue.empty():
            try:
                self._mission_request_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Kick off the upload handshake by sending the item count
        self._mav.waypoint_count_send(wp_loader.count())

        # Respond to each MISSION_REQUEST with the requested item
        for i in range(wp_loader.count()):
            try:
                req = await asyncio.wait_for(
                    self._mission_request_queue.get(), timeout=10.0
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Timeout waiting for MISSION_REQUEST for WP{i}"
                )
            self._mav.mav.send(wp_loader.wp(req.seq))
            logger.debug("  Sent WP%d", req.seq)

        # Wait for MISSION_ACK
        try:
            await asyncio.wait_for(self._mission_ack_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for MISSION_ACK")

        if self._mission_ack_result != mavutil.mavlink.MAV_MISSION_ACCEPTED:
            raise RuntimeError(
                f"Mission upload rejected: result={self._mission_ack_result}"
            )
        logger.info(
            "Mission accepted by flight controller (%d items)", wp_loader.count()
        )

    async def start_mission(self) -> None:
        """Switch to AUTO mode to begin the uploaded waypoint mission."""
        logger.info("Starting mission...")
        await self.set_mode(self.MODE_AUTO)
        logger.info("Mission running")

    async def pause_mission(self) -> None:
        """Pause the running mission by switching to GUIDED mode hold.

        On ArduPilot, switching to GUIDED while in AUTO pauses the mission
        and holds current position.
        """
        logger.info("Pausing mission (GUIDED hold)...")
        await self.set_mode(self.MODE_GUIDED)
        logger.info("Mission paused")

    async def is_mission_finished(self) -> bool:
        """Return True if the last uploaded mission item has been reached."""
        if self._mission_total_items <= 0:
            return False
        return self._mission_item_reached >= self._mission_total_items - 1

    # ── Direct navigation ─────────────────────────────────────────────

    async def goto_location(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        yaw_deg: float = float("nan"),
    ) -> None:
        """Fly to a GPS coordinate in GUIDED mode.

        Uses SET_POSITION_TARGET_GLOBAL_INT (not MAV_CMD_DO_REPOSITION, which
        ArduCopter 4.x rejects with MAV_RESULT_UNSUPPORTED in GUIDED mode).
        This message produces no COMMAND_ACK — arrival detection is the
        caller's responsibility via wait_for_arrival.
        """
        logger.info(
            "goto_location lat=%.6f lon=%.6f alt=%.1fm",
            latitude_deg, longitude_deg, altitude_m,
        )
        # SET_POSITION_TARGET_GLOBAL_INT has no ACK; fire and return.
        # type_mask 0xFF8 enables position (bits 0-2 clear) and ignores
        # velocity, acceleration, yaw, and yaw_rate (bits 3-11 set).
        self._mav.mav.set_position_target_global_int_send(
            0,                                                        # time_boot_ms (ignored)
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0xFF8,                                                    # type_mask
            int(latitude_deg * 1e7),                                  # lat_int (degE7)
            int(longitude_deg * 1e7),                                 # lon_int (degE7)
            float(altitude_m),                                        # alt (m above home)
            0.0, 0.0, 0.0,                                            # vx, vy, vz (ignored)
            0.0, 0.0, 0.0,                                            # afx, afy, afz (ignored)
            0.0, 0.0,                                                 # yaw, yaw_rate (ignored)
        )
        logger.info("goto_location sent")

    # ── Offboard (not supported on ArduCopter) ────────────────────────

    async def start_offboard(self) -> None:
        raise NotImplementedError(
            "Offboard mode not supported on ArduCopter; "
            "use GUIDED + goto_location or velocity override"
        )

    async def stop_offboard(self) -> None:
        raise NotImplementedError(
            "Offboard mode not supported on ArduCopter; "
            "use GUIDED + goto_location or velocity override"
        )

    async def send_velocity_ned(
        self,
        north_m_s: float,   # noqa: ARG002
        east_m_s: float,    # noqa: ARG002
        down_m_s: float,    # noqa: ARG002
        yaw_deg_s: float = 0.0,  # noqa: ARG002
    ) -> None:
        raise NotImplementedError(
            "Offboard mode not supported on ArduCopter; "
            "use GUIDED + goto_location or velocity override"
        )

    async def send_hover_setpoint(self) -> None:
        raise NotImplementedError(
            "Offboard mode not supported on ArduCopter; "
            "use GUIDED + goto_location or velocity override"
        )

    # ── Private helpers ───────────────────────────────────────────────

    async def _wait_for_altitude(self, target_m: float) -> None:
        """Poll GLOBAL_POSITION_INT until relative altitude reaches target_m.

        Logs a warning (does not raise) if target is not reached within 30s.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30.0

        while loop.time() < deadline:
            if self._last_gps_msg is not None:
                alt = self._last_gps_msg.relative_alt / 1000.0
                logger.info("  Climbing: %.1fm / %.1fm", alt, target_m)
                if alt >= target_m:
                    logger.info("Altitude reached: %.1fm", alt)
                    return
            self._gps_event.clear()
            try:
                await asyncio.wait_for(self._gps_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

        logger.warning(
            "Altitude %.1fm not reached within 30s — continuing", target_m
        )
