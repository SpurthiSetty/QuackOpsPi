"""
qps_backend_client_interface.py

Defines the contract for bidirectional communication with the Node.js backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from quackops_pi.models import qpsGPSPosition


class qpsBackendClientInterface(ABC):
    """Contract for backend communication."""

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def send_status(self, status_type: str, data: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    async def stream_gps(self, position: qpsGPSPosition) -> None:
        ...

    @abstractmethod
    def on_command(self, callback: Callable) -> None:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...