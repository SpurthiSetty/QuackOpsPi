from abc import ABC, abstractmethod
from typing import Callable

from src.models.qps_status_message import qpsStatusMessage
from src.models.qps_gps_position import qpsGPSPosition


class qpsIBackendClient(ABC):
    """Abstract interface for communication with the remote Node.js backend."""

    @abstractmethod
    async def connect(self) -> bool:
        """Open a WebSocket connection to the backend server.

        Returns:
            bool: True if the connection was established successfully.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the WebSocket connection gracefully."""
        ...

    @abstractmethod
    async def send_status(self, message: qpsStatusMessage) -> bool:
        """Send a status message to the backend.

        Args:
            message: The status message to transmit.

        Returns:
            bool: True if the message was sent (or queued) successfully.
        """
        ...

    @abstractmethod
    async def stream_gps(self, position: qpsGPSPosition) -> None:
        """Stream a GPS position update to the backend.

        Args:
            position: The current GPS position to send.
        """
        ...

    @abstractmethod
    def on_command(self, callback: Callable) -> None:
        """Register a callback to be invoked when a command is received.

        Args:
            callback: A callable that accepts a qpsMissionCommand.
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check whether the WebSocket connection is currently active.

        Returns:
            bool: True if connected.
        """
        ...
