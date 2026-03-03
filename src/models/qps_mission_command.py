from dataclasses import dataclass, field

from src.enums.qps_command_type import qpsCommandType


@dataclass
class qpsMissionCommand:
    """A command received from the backend instructing the drone to act."""

    command_type: qpsCommandType
    order_id: str
    destination_lat: float
    destination_lon: float
    delivery_marker_id: int
    payload: dict = field(default_factory=dict)
