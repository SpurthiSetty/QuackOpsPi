"""
qps_landing_controller_interface.py

Defines the contract for landing controllers.
Implementations own the full landing sequence: detect → (optionally correct) → land.
MissionController calls execute_landing() and handles the result without knowing the strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quackops_pi.models.qps_landing_result import qpsLandingResult


class qpsLandingControllerInterface(ABC):
    """Contract for all landing controller strategies."""

    @abstractmethod
    async def execute_landing(self, target_marker_id: int) -> qpsLandingResult:
        """Execute the full landing sequence.

        The controller owns the entire flow: detect marker, optionally correct
        position, command landing, and return the result.

        Args:
            target_marker_id: ArUco marker ID to search for.

        Returns:
            qpsLandingResult with outcome and diagnostics.
        """
        ...

    @abstractmethod
    def abort(self) -> None:
        """Signal the landing sequence to stop at the next iteration."""
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether a landing sequence is currently in progress."""
        ...
