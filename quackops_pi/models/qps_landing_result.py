"""
qps_landing_result.py

Holds the outcome of a marker search / landing attempt.
Produced by: qpsLandingController
Consumed by: qpsMissionController
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .qps_gps_position import qpsGPSPosition


class qpsLandingOutcome(Enum):
    """Discrete outcomes of the marker search phase."""
    MARKER_FOUND = auto()
    SEARCH_TIMEOUT = auto()
    CAMERA_FAILURE = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class qpsLandingResult:
    """Immutable result of a single marker search attempt."""

    outcome: qpsLandingOutcome
    marker_gps: Optional[qpsGPSPosition] = None
    fallback_gps: Optional[qpsGPSPosition] = None
    search_duration_s: float = 0.0
    frames_searched: int = 0
    target_marker_id: int = -1

    @property
    def success(self) -> bool:
        return self.outcome == qpsLandingOutcome.MARKER_FOUND