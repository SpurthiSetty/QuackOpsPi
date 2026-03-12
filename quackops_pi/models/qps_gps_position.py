"""
qps_gps_position.py

Holds a single GPS coordinate snapshot.
Produced by: qpsTelemetryMonitor
Consumed by: qpsBackendClient, qpsMissionController
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class qpsGPSPosition:
    """Immutable GPS snapshot. Serializable to JSON for WebSocket streaming."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    heading_deg: float
    speed_m_s: float
    timestamp: float  # Unix epoch seconds

    def to_dict(self) -> dict:
        return {
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m,
            "heading_deg": self.heading_deg,
            "speed_m_s": self.speed_m_s,
            "timestamp": self.timestamp,
        }