from dataclasses import dataclass

from src.models.qps_gps_position import qpsGPSPosition


@dataclass
class qpsDroneState:
    """Aggregated snapshot of the drone's current state from telemetry."""

    gps_position: qpsGPSPosition
    battery_percent: float
    is_armed: bool
    is_in_air: bool
    flight_mode: str
    gps_fix_type: int
    satellite_count: int
