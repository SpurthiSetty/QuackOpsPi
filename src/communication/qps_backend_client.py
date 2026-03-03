from typing import Optional, Callable

import aiohttp

from src.config.qps_config import qpsConfig
from src.interfaces.qps_i_backend_client import qpsIBackendClient
from src.models.qps_status_message import qpsStatusMessage
from src.models.qps_gps_position import qpsGPSPosition


class qpsBackendClient(qpsIBackendClient):
    """Production WebSocket client for communication with the Node.js backend.

    Maintains a persistent WebSocket connection, sends status updates and
    GPS streams, and listens for incoming mission commands.  Includes
    automatic reconnection, heartbeat, and message queuing.
    """

    def __init__(self, config: qpsConfig) -> None:
        """Initialise the backend client.

        Args:
            config: Application configuration containing WebSocket URL,
                reconnection parameters, and queue settings.
        """
        self.config: qpsConfig = config
        self.ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self.connected: bool = False
        self.message_queue: list[qpsStatusMessage] = []
        self.command_callback: Optional[Callable] = None

    async def connect(self) -> bool:
        """Open a WebSocket connection to the backend server.

        Returns:
            bool: True if the connection was established successfully.
        """
        # TODO: Create aiohttp.ClientSession. Connect to
        #  self.config.backend_ws_url. Store ws_connection. Set connected = True.
        #  Launch _listen_for_commands, _send_heartbeat, _flush_queue tasks.
        pass

    async def disconnect(self) -> None:
        """Close the WebSocket connection gracefully."""
        # TODO: Close self.ws_connection. Set connected = False. Cancel
        #  background tasks.
        pass

    async def send_status(self, message: qpsStatusMessage) -> bool:
        """Send a status message to the backend.

        If not connected, queues the message for later delivery.

        Args:
            message: The status message to transmit.

        Returns:
            bool: True if the message was sent or queued successfully.
        """
        # TODO: If connected, serialize message.to_dict() and send via
        #  ws_connection.send_json(). Otherwise, append to message_queue
        #  (respecting config.message_queue_max_size).
        pass

    async def stream_gps(self, position: qpsGPSPosition) -> None:
        """Stream a GPS position update to the backend.

        Args:
            position: The current GPS position to send.
        """
        # TODO: Serialize position.to_dict() and send as a GPS update
        #  message via the WebSocket.
        pass

    def on_command(self, callback: Callable) -> None:
        """Register a callback for incoming commands.

        Args:
            callback: A callable that accepts a qpsMissionCommand.
        """
        # TODO: Store callback in self.command_callback.
        pass

    def is_connected(self) -> bool:
        """Check whether the WebSocket connection is active.

        Returns:
            bool: True if connected.
        """
        # TODO: Return self.connected.
        pass

    async def _listen_for_commands(self) -> None:
        """Listen for incoming messages on the WebSocket.

        Parses received JSON into qpsMissionCommand objects and invokes
        the registered command_callback.
        """
        # TODO: async for msg in self.ws_connection: parse JSON, build
        #  qpsMissionCommand, invoke self.command_callback if set.
        pass

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect to the backend on connection loss.

        Retries up to config.max_reconnection_attempts times with
        config.reconnection_interval_s between attempts.
        """
        # TODO: While not connected and attempts remain, sleep then
        #  attempt connect(). Log each attempt.
        pass

    async def _flush_queue(self) -> None:
        """Send any queued messages after reconnection."""
        # TODO: Iterate self.message_queue and send each message.
        #  Clear queue on success.
        pass

    async def _send_heartbeat(self) -> None:
        """Periodically send heartbeat messages to keep the connection alive.

        Runs every config.heartbeat_interval_s while connected.
        """
        # TODO: While self.connected, sleep for heartbeat_interval_s,
        #  send a heartbeat JSON message.
        pass
