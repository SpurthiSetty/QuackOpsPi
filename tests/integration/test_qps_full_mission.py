import pytest
import pytest_asyncio

from src.config.qps_config import qpsConfig
from src.enums.qps_mission_state import qpsMissionState
from src.enums.qps_command_type import qpsCommandType
from src.models.qps_mission_command import qpsMissionCommand
from src.mission.qps_mission_controller import qpsMissionController
from src.landing.qps_landing_controller import qpsLandingController
from src.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor
from tests.mocks.qps_mock_flight_manager import qpsMockFlightManager
from tests.mocks.qps_mock_camera_manager import qpsMockCameraManager
from tests.mocks.qps_mock_marker_detector import qpsMockMarkerDetector
from tests.mocks.qps_mock_backend_client import qpsMockBackendClient


@pytest.fixture
def config() -> qpsConfig:
    """Create a default test configuration."""
    return qpsConfig(simulation_mode=True)


class TestFullMission:
    """End-to-end integration tests for a complete delivery mission."""

    @pytest.mark.asyncio
    async def test_full_delivery_and_return(self, config: qpsConfig) -> None:
        """Test the complete mission: dispatch → deliver → pickup → return.

        Verifies the state machine walks through every expected state
        from IDLE to MISSION_COMPLETE.
        """
        # TODO: Wire all mocks. Inject DISPATCH command. Step through states.
        #  Inject PICKUP_CONFIRMED. Step through return states. Assert
        #  final state is MISSION_COMPLETE.
        pass

    @pytest.mark.asyncio
    async def test_mission_abort_mid_delivery(self, config: qpsConfig) -> None:
        """Test aborting a mission during the delivery leg."""
        # TODO: Inject DISPATCH, advance to EN_ROUTE_TO_DELIVERY, inject
        #  ABORT. Verify appropriate state transition and RTL.
        pass

    @pytest.mark.asyncio
    async def test_mission_battery_critical_during_flight(
        self, config: qpsConfig
    ) -> None:
        """Test critical battery handling during flight."""
        # TODO: Advance mission to EN_ROUTE_TO_DELIVERY. Trigger
        #  handle_battery_critical. Assert EMERGENCY_RTL state.
        pass
