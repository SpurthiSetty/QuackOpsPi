"""
qps_telemetry_monitor.py

Subscribes to live drone telemetry and provides current state on demand.
Stub — needs full MAVSDK subscription implementation.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable

from quackops_pi.config import qpsConfig
from quackops_pi.models import qpsGPSPosition, qpsDroneState

logger = logging.getLogger("qps.telemetry_monitor")


class qpsTelemetryMonitor:
    """Provides current drone telemetry on demand."""

    def __init__(self, system, config: qpsConfig) -> None:
        self._system = system  # mavsdk.System
        self._config = config
        self._gps_position: Optional[qpsGPSPosition] = None
        self._drone_state: Optional[qpsDroneState] = None
        self._battery_warning_callback: Optional[Callable] = None
        self._battery_critical_callback: Optional[Callable] = None
        self._running: bool = False

    # ── Public accessors ──────────────────────────────────────────────

    def get_gps_position(self) -> Optional[qpsGPSPosition]:
        return self._gps_position

    def get_drone_state(self) -> Optional[qpsDroneState]:
        return self._drone_state

    # ── Callback registration ─────────────────────────────────────────

    def on_battery_warning(self, callback: Callable) -> None:
        self._battery_warning_callback = callback

    def on_battery_critical(self, callback: Callable) -> None:
        self._battery_critical_callback = callback

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        # TODO: Subscribe to MAVSDK telemetry streams
        self._running = True
        logger.info("Telemetry monitor started")

    async def stop(self) -> None:
        self._running = False
        logger.info("Telemetry monitor stopped")