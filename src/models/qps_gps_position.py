from dataclasses import dataclass, asdict


@dataclass
class qpsGPSPosition:
    """Represents a GPS position with velocity and heading information."""

    latitude: float
    longitude: float
    altitude_m: float
    relative_altitude_m: float
    heading_deg: float
    ground_speed_m_s: float
    timestamp: float

    def to_dict(self) -> dict:
        """Serialize the GPS position to a dictionary for JSON transmission.

        Returns:
            dict: Dictionary representation of the GPS position.
        """
        return asdict(self)
