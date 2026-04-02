"""
qps_flight_manager_interface.py

Defines the contract for commanding drone flight operations.
Implementations: qpsFlightManager (production), qpsMockFlightManager (testing)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from quackops_pi.models import qpsGPSPosition


class qpsFlightManagerInterface(ABC):
    """Contract for all drone flight commands."""

    # ── Connection ────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    # ── Arming ────────────────────────────────────────────────────────

    @abstractmethod
    async def arm(self) -> None:
        ...

    @abstractmethod
    async def disarm(self) -> None:
        ...

    # ── Basic flight ──────────────────────────────────────────────────

    @abstractmethod
    async def takeoff(self, altitude_m: float) -> None:
        ...

    @abstractmethod
    async def land(self) -> None:
        ...

    @abstractmethod
    async def return_to_launch(self) -> None:
        ...

    # ── Waypoint missions ─────────────────────────────────────────────

    @abstractmethod
    async def upload_mission(self, waypoints: List[qpsGPSPosition]) -> None:
        ...

    @abstractmethod
    async def start_mission(self) -> None:
        ...

    @abstractmethod
    async def pause_mission(self) -> None:
        """Pause the running mission. ArduPilot: switch to GUIDED mode hold."""
        ...

    @abstractmethod
    async def is_mission_finished(self) -> bool:
        ...

    # ── Direct navigation ─────────────────────────────────────────────

    @abstractmethod
    async def goto_location(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        yaw_deg: float = float("nan"),
    ) -> None:
        """Fly to a specific GPS coordinate and loiter."""
        ...

    # ── Offboard control ──────────────────────────────────────────────

    @abstractmethod
    async def start_offboard(self) -> None:
        ...

    @abstractmethod
    async def stop_offboard(self) -> None:
        ...

    @abstractmethod
    async def send_velocity_ned(
        self, north_m_s: float, east_m_s: float, down_m_s: float, yaw_deg_s: float = 0.0
    ) -> None:
        ...

    @abstractmethod
    async def send_hover_setpoint(self) -> None:
        ...