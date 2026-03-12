import pytest
import pytest_asyncio

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.models.qps_status_type import qpsStatusType
from quackops_pi.models.qps_status_message import qpsStatusMessage
from quackops_pi.models.qps_gps_position import qpsGPSPosition
from quackops_pi.comms.qps_mock_backend_client import qpsMockBackendClient


@pytest.fixture
def config() -> qpsConfig:
    """Create a default test configuration."""
    return qpsConfig(simulation_mode=True)


@pytest.fixture
def client(config: qpsConfig) -> qpsMockBackendClient:
    """Create a mock backend client for testing."""
    return qpsMockBackendClient(config)


class TestBackendClientConnection:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect(self, client: qpsMockBackendClient) -> None:
        """Verify connect sets connected state."""
        # TODO: Call connect, assert is_connected() is True.
        pass

    @pytest.mark.asyncio
    async def test_disconnect(self, client: qpsMockBackendClient) -> None:
        """Verify disconnect clears connected state."""
        # TODO: Call disconnect, assert is_connected() is False.
        pass


class TestBackendClientMessages:
    """Tests for message sending."""

    @pytest.mark.asyncio
    async def test_send_status_logs_message(
        self, client: qpsMockBackendClient
    ) -> None:
        """Verify sent status messages are logged."""
        # TODO: Send a status message, assert it appears in get_sent_messages().
        pass

    @pytest.mark.asyncio
    async def test_stream_gps_logs_position(
        self, client: qpsMockBackendClient
    ) -> None:
        """Verify streamed GPS positions are logged."""
        # TODO: Stream a position, assert it appears in get_gps_log().
        pass


class TestBackendClientCommands:
    """Tests for command injection."""

    def test_inject_command_invokes_callback(
        self, client: qpsMockBackendClient
    ) -> None:
        """Verify injected commands reach the registered callback."""
        # TODO: Register a callback, inject a command, assert callback invoked.
        pass
