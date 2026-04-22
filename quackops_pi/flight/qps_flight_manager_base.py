"""
qps_flight_manager_base.py

Abstract base providing shared pymavlink infrastructure for all
concrete flight manager implementations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from pymavlink import mavutil

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager_interface import qpsFlightManagerInterface

logger = logging.getLogger("qps.flight_manager")


class qpsFlightManagerBase(qpsFlightManagerInterface):
    """Abstract base providing shared pymavlink infrastructure.

    Implements connection, reader loop, arming, and internal helpers.
    GPS-specific and RC-specific flight methods remain abstract — implemented
    by qpsFlightManager and qpsRCFlightManager respectively.
    """

    # ── ArduCopter flight mode IDs ────────────────────────────────────
    MODE_STABILIZE: int = 0
    MODE_ALT_HOLD: int = 2
    MODE_AUTO: int = 3
    MODE_GUIDED: int = 4
    MODE_RTL: int = 6
    MODE_LAND: int = 9

    _MODE_NAMES: dict[int, str] = {
        0: "STABILIZE",
        2: "ALT_HOLD",
        3: "AUTO",
        4: "GUIDED",
        6: "RTL",
        9: "LAND",
    }

    def __init__(self, config: qpsConfig) -> None:
        self._config = config
        self._mav: Optional[Any] = None          # pymavlink connection object
        self._reader_task: Optional[asyncio.Task] = None
        self._connected: bool = False

        # ── Telemetry callback list ───────────────────────────────────
        self._message_callbacks: list[Callable[[Any], None]] = []

        # ── COMMAND_ACK state ─────────────────────────────────────────
        self._ack_event: asyncio.Event = asyncio.Event()
        self._ack_result: Optional[int] = None
        self._expected_command_id: Optional[int] = None

        # ── Heartbeat state ───────────────────────────────────────────
        self._last_heartbeat: Optional[Any] = None
        self._heartbeat_event: asyncio.Event = asyncio.Event()

        # ── GPS / GLOBAL_POSITION_INT state ───────────────────────────
        self._last_gps_msg: Optional[Any] = None
        self._gps_event: asyncio.Event = asyncio.Event()

        # ── SYS_STATUS / EKF state ────────────────────────────────────
        self._last_sys_status: Optional[Any] = None
        self._sys_status_event: asyncio.Event = asyncio.Event()

        # ── Landing detection ─────────────────────────────────────────
        self._land_event: asyncio.Event = asyncio.Event()

        # ── Mission tracking ──────────────────────────────────────────
        self._current_mission_item: int = 0
        self._mission_item_reached: int = -1
        self._mission_total_items: int = 0
        self._mission_request_queue: asyncio.Queue = asyncio.Queue()
        self._mission_ack_event: asyncio.Event = asyncio.Event()
        self._mission_ack_result: Optional[int] = None

        # ── Param read-back state ─────────────────────────────────────
        self._param_id_waiting: Optional[str] = None
        self._param_value: Optional[float] = None
        self._param_event: asyncio.Event = asyncio.Event()

    # ── Connection ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open MAVLink connection, wait for heartbeat, request streams, start reader."""
        logger.info("Connecting to %s ...", self._config.connection_string)
        self._mav = await asyncio.to_thread(
            mavutil.mavlink_connection, self._config.connection_string
        )
        await asyncio.to_thread(self._mav.wait_heartbeat)
        logger.info(
            "Connected! (sysid=%d, compid=%d)",
            self._mav.target_system,
            self._mav.target_component,
        )

        # Request all telemetry streams at 4 Hz
        self._mav.mav.request_data_stream_send(
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            4,   # Hz
            1,   # start
        )

        self._connected = True
        self._reader_task = asyncio.create_task(
            self._reader_loop(), name="mav-reader"
        )
        logger.info("Reader loop started")

    async def disconnect(self) -> None:
        """Cancel the reader loop and close the MAVLink connection."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        self._connected = False
        logger.info("Flight manager disconnected")

    # ── Callback registration ─────────────────────────────────────────

    def register_message_callback(self, callback: Callable[[Any], None]) -> None:
        """Register a callback that receives every MAVLink message.

        qpsTelemetryMonitor calls this during setup. The callback runs
        within the reader loop's asyncio task.
        """
        self._message_callbacks.append(callback)

    # ── Reader loop ───────────────────────────────────────────────────

    async def _reader_loop(self) -> None:
        """Single asyncio.Task — the only consumer of MAVLink messages.

        For every message received:
        1. Dispatch to all registered telemetry callbacks.
        2. Signal the appropriate internal asyncio.Event so command methods
           can wake up (COMMAND_ACK, HEARTBEAT, GLOBAL_POSITION_INT, etc.).
        """
        logger.debug("Reader loop running")
        while True:
            try:
                msg = await asyncio.to_thread(
                    self._mav.recv_match, blocking=True, timeout=1.0
                )
            except asyncio.CancelledError:
                logger.debug("Reader loop cancelled")
                raise
            except Exception as exc:
                logger.warning("Reader recv error: %s", exc)
                continue

            if msg is None:
                continue

            msg_type = msg.get_type()

            # ── 1. Dispatch to telemetry callbacks ────────────────
            for cb in self._message_callbacks:
                try:
                    cb(msg)
                except Exception as exc:
                    logger.warning("Telemetry callback error: %s", exc)

            # ── 2. Internal event signalling ──────────────────────

            if msg_type == "STATUSTEXT":
                text = msg.text.rstrip("\x00")
                if text:
                    logger.info("[FC] %s", text)

            elif msg_type == "COMMAND_ACK":
                if (
                    self._expected_command_id is not None
                    and msg.command == self._expected_command_id
                ):
                    self._ack_result = msg.result
                    self._ack_event.set()

            elif msg_type == "HEARTBEAT":
                # Filter GCS heartbeats (Mission Planner sends compid=0 with
                # custom_mode=0/STABILIZE, which causes false mode-switch failures)
                if msg.get_srcComponent() != 0:
                    self._last_heartbeat = msg
                    self._heartbeat_event.set()

            elif msg_type == "GLOBAL_POSITION_INT":
                self._last_gps_msg = msg
                self._gps_event.set()
                # Landing detection: low altitude + low vertical speed
                alt_m = msg.relative_alt / 1000.0
                vz_m_s = abs(msg.vz / 100.0)
                if alt_m < 1.0 and vz_m_s < 0.3:
                    self._land_event.set()

            elif msg_type == "SYS_STATUS":
                self._last_sys_status = msg
                self._sys_status_event.set()

            elif msg_type == "MISSION_REQUEST":
                await self._mission_request_queue.put(msg)

            elif msg_type == "MISSION_ACK":
                self._mission_ack_result = msg.type
                self._mission_ack_event.set()

            elif msg_type == "MISSION_CURRENT":
                self._current_mission_item = msg.seq

            elif msg_type == "MISSION_ITEM_REACHED":
                self._mission_item_reached = msg.seq

            elif msg_type == "PARAM_VALUE":
                if self._param_id_waiting is not None:
                    received_id = msg.param_id.rstrip("\x00")
                    if received_id == self._param_id_waiting:
                        self._param_value = msg.param_value
                        self._param_event.set()

    # ── Arming ────────────────────────────────────────────────────────

    async def arm(self) -> None:
        """Arm the drone.

        Sends arducopter_arm(), awaits COMMAND_ACK for MAV_CMD_COMPONENT_ARM_DISARM,
        then confirms armed state via heartbeat base_mode flag.
        """
        logger.info("Arming...")
        await self._send_and_await_ack(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            self._mav.arducopter_arm,
            timeout=5.0,
        )
        await self._wait_heartbeat_condition(
            lambda hb: bool(
                hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            ),
            timeout=15.0,
        )
        logger.info("Armed! (confirmed via heartbeat)")

    async def disarm(self) -> None:
        """Disarm the drone.

        Sends arducopter_disarm(), awaits COMMAND_ACK, confirms via heartbeat.
        """
        logger.info("Disarming...")
        await self._send_and_await_ack(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            self._mav.arducopter_disarm,
            timeout=5.0,
        )
        await self._wait_heartbeat_condition(
            lambda hb: not bool(
                hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            ),
            timeout=15.0,
        )
        logger.info("Disarmed!")

    # ── Internal helpers ──────────────────────────────────────────────

    async def _send_and_await_ack(
        self,
        command_id: int,
        send_fn: Callable[[], None],
        timeout: float = 5.0,
    ) -> None:
        """Send a MAVLink command and await its COMMAND_ACK.

        Raises RuntimeError on timeout or if result != MAV_RESULT_ACCEPTED.
        """
        self._ack_event.clear()
        self._ack_result = None
        self._expected_command_id = command_id
        send_fn()
        try:
            await asyncio.wait_for(self._ack_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._expected_command_id = None
            raise RuntimeError(
                f"Timeout waiting for ACK for command {command_id} after {timeout}s"
            )
        self._expected_command_id = None
        if self._ack_result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(
                f"Command {command_id} rejected: result={self._ack_result}"
            )

    async def _wait_heartbeat_condition(
        self,
        predicate: Callable[[Any], bool],
        timeout: float = 15.0,
    ) -> None:
        """Wait until an autopilot heartbeat satisfies predicate.

        Pattern:
            1. Clear event
            2. Check predicate on _last_heartbeat (catches heartbeat that arrived
               between last check and the clear)
            3. Wait for event
            4. Check predicate
            5. Repeat until satisfied or timeout

        Raises RuntimeError on timeout.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            self._heartbeat_event.clear()
            if self._last_heartbeat is not None and predicate(self._last_heartbeat):
                return
            remaining = deadline - loop.time()
            try:
                await asyncio.wait_for(
                    self._heartbeat_event.wait(), timeout=min(remaining, 2.0)
                )
            except asyncio.TimeoutError:
                pass
            if self._last_heartbeat is not None and predicate(self._last_heartbeat):
                return
        raise RuntimeError(
            f"Heartbeat condition not met within {timeout}s"
        )

    async def set_mode(self, mode_id: int, timeout: float = 10.0) -> None:
        """Switch flight mode and verify via heartbeat custom_mode field.

        Uses MAV_CMD_DO_SET_MODE and confirms via heartbeat (not COMMAND_ACK)
        because ArduPilot mode switches don't always produce an ACK but always
        update custom_mode in subsequent heartbeats.
        """
        mode_name = self._MODE_NAMES.get(mode_id, str(mode_id))
        logger.info("Switching to %s mode...", mode_name)
        self._mav.mav.command_long_send(
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,   # confirmation
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
            0, 0, 0, 0, 0,
        )
        await self._wait_heartbeat_condition(
            lambda hb: hb.custom_mode == mode_id,
            timeout=timeout,
        )
        logger.info("%s mode confirmed", mode_name)

    async def _wait_for_gps(self) -> tuple[float, float]:
        """Wait until a valid (non-zero) GPS position is available."""
        while True:
            if self._last_gps_msg is not None and self._last_gps_msg.lat != 0:
                return (
                    self._last_gps_msg.lat / 1e7,
                    self._last_gps_msg.lon / 1e7,
                )
            self._gps_event.clear()
            try:
                await asyncio.wait_for(self._gps_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.debug("Waiting for GPS lock...")

    async def _wait_for_ekf(self, timeout: float = 30.0) -> None:
        """Wait until EKF reports AHRS and GPS health in SYS_STATUS.

        Falls back to a 10-second sleep if SYS_STATUS never shows both healthy.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self._last_sys_status is not None:
                sensors = self._last_sys_status.onboard_control_sensors_health
                ahrs_ok = sensors & mavutil.mavlink.MAV_SYS_STATUS_AHRS
                gps_ok = sensors & mavutil.mavlink.MAV_SYS_STATUS_SENSOR_GPS
                if ahrs_ok and gps_ok:
                    return
            self._sys_status_event.clear()
            remaining = deadline - loop.time()
            try:
                await asyncio.wait_for(
                    self._sys_status_event.wait(), timeout=min(remaining, 2.0)
                )
            except asyncio.TimeoutError:
                pass
        logger.info("EKF check inconclusive — waiting 10s as fallback")
        await asyncio.sleep(10)

    async def _set_param(
        self,
        param_id: str,
        value: float,
        max_attempts: int = 5,
    ) -> bool:
        """Set a parameter and verify via explicit PARAM_REQUEST_READ readback.

        Uses REAL32 type (ArduPilot stores all params as floats internally).
        Returns True if confirmed, False after max_attempts failures.
        """
        param_id_bytes = param_id.encode("utf-8")

        for attempt in range(1, max_attempts + 1):
            self._mav.mav.param_set_send(
                self._mav.target_system,
                self._mav.target_component,
                param_id_bytes,
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            await asyncio.sleep(0.5)

            self._param_id_waiting = param_id
            self._param_value = None
            self._param_event.clear()

            self._mav.mav.param_request_read_send(
                self._mav.target_system,
                self._mav.target_component,
                param_id_bytes,
                -1,   # use param_id string, not index
            )

            try:
                await asyncio.wait_for(self._param_event.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._param_id_waiting = None
                logger.warning(
                    "  %s param read timeout (attempt %d/%d)",
                    param_id, attempt, max_attempts,
                )
                continue

            self._param_id_waiting = None
            if (
                self._param_value is not None
                and int(self._param_value) == int(value)
            ):
                logger.info(
                    "  %s confirmed = %d (attempt %d)",
                    param_id, int(value), attempt,
                )
                return True

            logger.warning(
                "  %s read back %s instead of %d, retrying...",
                param_id, self._param_value, int(value),
            )

        logger.error(
            "  Failed to set %s=%d after %d attempts",
            param_id, int(value), max_attempts,
        )
        return False

    async def set_sitl_params(self) -> None:
        """Disable pre-arm safety checks for SITL testing (ARMING_CHECK=0).

        Not for use in production. Call before arm() when testing against SITL.
        """
        logger.info("Setting SITL params (ARMING_CHECK=0)...")
        ok = await self._set_param("ARMING_CHECK", 0)
        if not ok:
            logger.error("Failed to set ARMING_CHECK=0 — arming may fail")
        await asyncio.sleep(2)
