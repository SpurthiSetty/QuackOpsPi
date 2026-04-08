"""
qps_telemetry_monitor.py

Subscribes to live drone telemetry via qpsFlightManager's message callback system.

Pattern B: qpsFlightManager owns the MAVLink connection and reader loop.
This class registers _on_message with the flight manager at construction time
and updates internal state on every MAVLink message the reader loop dispatches.
No polling — all updates are push-driven.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from quackops_pi.config import qpsConfig
from quackops_pi.models import qpsGPSPosition, qpsDroneState

logger = logging.getLogger("qps.telemetry_monitor")

# ArduCopter custom_mode → human-readable name
_FLIGHT_MODE_NAMES: dict[int, str] = {
    0:  "STABILIZE",
    1:  "ACRO",
    2:  "ALT_HOLD",
    3:  "AUTO",
    4:  "GUIDED",
    5:  "LOITER",
    6:  "RTL",
    7:  "CIRCLE",
    9:  "LAND",
    11: "DRIFT",
    13: "SPORT",
    16: "POSHOLD",
    17: "BRAKE",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL",
}

# MAV_MODE_FLAG_SAFETY_ARMED bit (0x80 = 128)
_ARMED_FLAG: int = 0x80


class qpsTelemetryMonitor:
    """Provides current drone telemetry on demand.

    Registers _on_message with qpsFlightManager on construction so it receives
    every MAVLink message dispatched by the reader loop. State is updated in-place
    on _drone_state (a mutable dataclass).

    Battery threshold callbacks fire once per threshold crossing (not repeatedly).
    """

    def __init__(self, flight_manager: Any, config: qpsConfig) -> None:
        self._config = config
        self._gps_position: Optional[qpsGPSPosition] = None
        self._drone_state: qpsDroneState = qpsDroneState()
        self._battery_warning_callback: Optional[Callable] = None
        self._battery_critical_callback: Optional[Callable] = None
        self._running: bool = False

        # Track whether each threshold has already fired this session
        self._warning_fired: bool = False
        self._critical_fired: bool = False

        # Register immediately — callbacks arrive as soon as reader loop starts
        flight_manager.register_message_callback(self._on_message)

    # ── Public accessors ──────────────────────────────────────────────

    def get_gps_position(self) -> Optional[qpsGPSPosition]:
        """Return the latest GPS snapshot, or None if no data received yet."""
        return self._gps_position

    def get_drone_state(self) -> Optional[qpsDroneState]:
        """Return the current aggregated drone state."""
        return self._drone_state

    # ── Callback registration ─────────────────────────────────────────

    def on_battery_warning(self, callback: Callable) -> None:
        """Register a callback fired once when battery drops below warning threshold."""
        self._battery_warning_callback = callback

    def on_battery_critical(self, callback: Callable) -> None:
        """Register a callback fired once when battery drops below critical threshold."""
        self._battery_critical_callback = callback

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Mark monitor active. The reader loop is managed by qpsFlightManager."""
        self._running = True
        logger.info("Telemetry monitor started")

    async def stop(self) -> None:
        """Mark monitor inactive. Incoming messages are ignored while stopped."""
        self._running = False
        logger.info("Telemetry monitor stopped")

    # ── Message handler ───────────────────────────────────────────────

    def _on_message(self, msg: Any) -> None:
        """Dispatch a MAVLink message to the appropriate state-update handler.

        Called from the reader loop's asyncio task on every message received.
        Guards on _running so updates stop cleanly after stop() is called.
        """
        if not self._running:
            return

        msg_type = msg.get_type()

        if msg_type == "GLOBAL_POSITION_INT":
            self._handle_global_position_int(msg)
        elif msg_type == "HEARTBEAT":
            if msg.get_srcComponent() != 0:   # filter GCS (Mission Planner)
                self._handle_heartbeat(msg)
        elif msg_type == "BATTERY_STATUS":
            self._handle_battery_status(msg)
        elif msg_type == "SYS_STATUS":
            self._handle_sys_status(msg)
        elif msg_type == "GPS_RAW_INT":
            self._handle_gps_raw_int(msg)

    # ── Per-message-type handlers ─────────────────────────────────────

    def _handle_global_position_int(self, msg: Any) -> None:
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt_m = msg.relative_alt / 1000.0

        # hdg is in centidegrees (0-36000); 65535 = unknown
        heading_deg = msg.hdg / 100.0 if msg.hdg != 65535 else 0.0

        # vx/vy are cm/s in body-NE frame; compute groundspeed magnitude
        vx_m_s = msg.vx / 100.0
        vy_m_s = msg.vy / 100.0
        speed_m_s = (vx_m_s ** 2 + vy_m_s ** 2) ** 0.5

        self._gps_position = qpsGPSPosition(
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_m=alt_m,
            heading_deg=heading_deg,
            speed_m_s=speed_m_s,
            timestamp=time.time(),
        )
        self._drone_state.gps_position = self._gps_position
        # Infer in_air from altitude (> 0.5m above home)
        self._drone_state.in_air = alt_m > 0.5

    def _handle_heartbeat(self, msg: Any) -> None:
        self._drone_state.is_armed = bool(msg.base_mode & _ARMED_FLAG)
        mode_id = int(msg.custom_mode)
        self._drone_state.flight_mode = _FLIGHT_MODE_NAMES.get(mode_id, str(mode_id))

    def _handle_battery_status(self, msg: Any) -> None:
        # battery_remaining: 0-100 percent; -1 = unknown
        if msg.battery_remaining >= 0:
            pct = float(msg.battery_remaining)
            self._drone_state.battery_percent = pct
            self._check_battery_thresholds(pct)

        # voltages: list of cell voltages in mV; 65535 = not available
        if msg.voltages and msg.voltages[0] != 65535:
            self._drone_state.battery_voltage = msg.voltages[0] / 1000.0

    def _handle_sys_status(self, msg: Any) -> None:
        # SYS_STATUS also carries battery info — use it as a fallback if no
        # BATTERY_STATUS is present (older ArduPilot builds)
        if msg.battery_remaining >= 0:
            pct = float(msg.battery_remaining)
            self._drone_state.battery_percent = pct
            self._check_battery_thresholds(pct)

        # voltage_battery: in mV; 65535 = unknown
        if msg.voltage_battery != 65535:
            self._drone_state.battery_voltage = msg.voltage_battery / 1000.0

    def _handle_gps_raw_int(self, msg: Any) -> None:
        self._drone_state.gps_fix_type = msg.fix_type
        self._drone_state.gps_num_satellites = msg.satellites_visible

    # ── Battery threshold logic ───────────────────────────────────────

    def _check_battery_thresholds(self, pct: float) -> None:
        """Fire warning/critical callbacks once per threshold crossing."""
        if pct <= self._config.battery_critical_percent and not self._critical_fired:
            self._critical_fired = True
            logger.warning("BATTERY CRITICAL: %.0f%%", pct)
            if self._battery_critical_callback is not None:
                try:
                    self._battery_critical_callback(pct)
                except Exception as exc:
                    logger.warning("Battery critical callback error: %s", exc)
        elif pct <= self._config.battery_warning_percent and not self._warning_fired:
            self._warning_fired = True
            logger.warning("BATTERY WARNING: %.0f%%", pct)
            if self._battery_warning_callback is not None:
                try:
                    self._battery_warning_callback(pct)
                except Exception as exc:
                    logger.warning("Battery warning callback error: %s", exc)
