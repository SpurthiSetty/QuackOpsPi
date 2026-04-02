"""
qps_drone_state.py

Holds aggregated drone health.
Produced by: qpsTelemetryMonitor
Consumed by: qpsMissionController
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .qps_gps_position import qpsGPSPosition


@dataclass
class qpsDroneState:
    """Mutable aggregated drone state updated by telemetry monitor."""

    gps_position: Optional[qpsGPSPosition] = None
    battery_percent: float = 0.0
    battery_voltage: float = 0.0
    is_armed: bool = False
    in_air: bool = False
    flight_mode: str = "UNKNOWN"
    gps_fix_type: int = 0
    gps_num_satellites: int = 0