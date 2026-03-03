from dataclasses import dataclass, field, asdict

from src.enums.qps_status_type import qpsStatusType


@dataclass
class qpsStatusMessage:
    """A status message to be sent to the backend."""

    status_type: qpsStatusType
    order_id: str
    data: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Serialize the status message to a dictionary for JSON transmission.

        Returns:
            dict: Dictionary representation of the status message with the
                  status_type converted to its string value.
        """
        d = asdict(self)
        d["status_type"] = self.status_type.value
        return d
