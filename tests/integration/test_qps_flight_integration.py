import pytest
import pytest_asyncio

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_mock_flight_manager import qpsMockFlightManager


@pytest.fixture
def config() -> qpsConfig:
    """Create a default test configuration."""
    return qpsConfig(simulation_mode=True)


@pytest.fixture
def flight_manager(config: qpsConfig) -> qpsMockFlightManager:
    """Create a mock flight manager for integration testing."""
    return qpsMockFlightManager(config)


class TestFlightIntegrationConnect:
    """Integration tests for flight connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_arm_takeoff_land_disarm(
        self, flight_manager: qpsMockFlightManager
    ) -> None:
        """Verify the full connect-arm-takeoff-land-disarm sequence."""
        # TODO: Run the full flight sequence using mock. Assert command_log
        #  contains the expected entries in order.
        pass

    @pytest.mark.asyncio
    async def test_arm_failure_aborts_sequence(
        self, flight_manager: qpsMockFlightManager
    ) -> None:
        """Verify that an arm failure prevents takeoff."""
        # TODO: Set arm_should_succeed = False. Attempt sequence. Assert
        #  takeoff was never called.
        pass


class TestFlightIntegrationOffboard:
    """Integration tests for offboard control."""

    @pytest.mark.asyncio
    async def test_offboard_velocity_commands(
        self, flight_manager: qpsMockFlightManager
    ) -> None:
        """Verify offboard velocity commands are logged correctly."""
        # TODO: Start offboard, send velocity commands, stop offboard.
        #  Assert command_log.
        pass
