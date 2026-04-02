"""
qps_camera_manager_interface.py

Defines the contract for acquiring camera frames on demand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class qpsCameraManagerInterface(ABC):
    """Contract for camera frame acquisition."""

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def get_frame(self) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    def is_running(self) -> bool:
        ...