from enum import Enum


class qpsCommandType(Enum):
    """Commands that can be received from the backend."""

    DISPATCH = "DISPATCH"
    PICKUP_CONFIRMED = "PICKUP_CONFIRMED"
    ABORT = "ABORT"
