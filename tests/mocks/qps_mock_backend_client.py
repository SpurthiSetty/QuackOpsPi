from typing import Optional, Callable

from src.config.qps_config import qpsConfig
from src.interfaces.qps_i_backend_client import qpsIBackendClient
from src.models.qps_status_message import qpsStatusMessage
from src.models.qps_gps_position import qpsGPSPosition
from src.models.qps_mission_command import qpsMissionCommand


class qpsMockBackendClient(qpsIBackendClient):
    """Mock backend client that logs all outgoing messages for assertions.

    Allows tests to inject commands and inspect sent status/GPS messages.
    """

    def __init__(self, config: qpsConfig) -> None:
        """Initialise the mock backend client.

        Args:
            config: Application configuration (stored but largely unused).
        """
        self.config: qpsConfig = config
        self.connected: bool = True
        self.sent_messages: list[qpsStatusMessage] = []
        self.gps_log: list[qpsGPSPosition] = []
        self.command_callback: Optional[Callable] = None
        self.pending_commands: list[qpsMissionCommand] = []

    # ------------------------------------------------------------------
    # qpsIBackendClient implementation
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Simulate a successful WebSocket connection.

        Returns:
            bool: Always True.
        """
        self.connected = True
        return True

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self.connected = False

    async def send_status(self, message: qpsStatusMessage) -> bool:
        """Record a status message for later inspection.

        Args:
            message: The status message to log.

        Returns:
            bool: Always True.
        """
        self.sent_messages.append(message)
        return True

    async def stream_gps(self, position: qpsGPSPosition) -> None:
        """Record a GPS position for later inspection.

        Args:
            position: The GPS position to log.
        """
        self.gps_log.append(position)

    def on_command(self, callback: Callable) -> None:
        """Register a command callback.

        Args:
            callback: A callable that accepts a qpsMissionCommand.
        """
        self.command_callback = callback

    def is_connected(self) -> bool:
        """Check mock connection state.

        Returns:
            bool: Current connected flag.
        """
        return self.connected

    # ------------------------------------------------------------------
    # Test-helper methods
    # ------------------------------------------------------------------

    def inject_command(self, command: qpsMissionCommand) -> None:
        """Simulate receiving a command from the backend.

        Invokes the registered callback immediately if one is set,
        otherwise queues the command.

        Args:
            command: The command to inject.
        """
        if self.command_callback is not None:
            self.command_callback(command)
        else:
            self.pending_commands.append(command)

    def get_sent_messages(self) -> list[qpsStatusMessage]:
        """Return all status messages that were sent.

        Returns:
            list[qpsStatusMessage]: Ordered list of sent messages.
        """
        return list(self.sent_messages)

    def get_gps_log(self) -> list[qpsGPSPosition]:
        """Return all GPS positions that were streamed.

        Returns:
            list[qpsGPSPosition]: Ordered list of GPS positions.
        """
        return list(self.gps_log)
