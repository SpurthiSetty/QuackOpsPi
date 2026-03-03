from enum import Enum


class qpsLandingResult(Enum):
    """Possible outcomes of a precision landing attempt."""

    SUCCESS = "SUCCESS"
    FALLBACK_LAND = "FALLBACK_LAND"
    ABORTED = "ABORTED"
