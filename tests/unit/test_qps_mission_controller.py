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


@pytest.fixture
def mock_flight_manager(config: qpsConfig) -> qpsMockFlightManager:
    """Create a mock flight manager."""
    return qpsMockFlightManager(config)


@pytest.fixture
def mock_camera_manager(config: qpsConfig) -> qpsMockCameraManager:
    """Create a mock camera manager."""
    return qpsMockCameraManager(config)


@pytest.fixture
def mock_marker_detector(config: qpsConfig) -> qpsMockMarkerDetector:
    """Create a mock marker detector."""
    return qpsMockMarkerDetector(config)


@pytest.fixture
def mock_backend_client(config: qpsConfig) -> qpsMockBackendClient:
    """Create a mock backend client."""
    return qpsMockBackendClient(config)


class TestMissionControllerInit:
    """Tests for qpsMissionController initialisation."""

    def test_initial_state_is_idle(
        self,
        config: qpsConfig,
        mock_flight_manager: qpsMockFlightManager,
        mock_camera_manager: qpsMockCameraManager,
        mock_marker_detector: qpsMockMarkerDetector,
        mock_backend_client: qpsMockBackendClient,
    ) -> None:
        """Verify the controller starts in IDLE state."""
        # TODO: Instantiate qpsMissionController and assert state == IDLE.
        pass

    def test_transition_to_changes_state(
        self,
        config: qpsConfig,
        mock_flight_manager: qpsMockFlightManager,
        mock_camera_manager: qpsMockCameraManager,
        mock_marker_detector: qpsMockMarkerDetector,
        mock_backend_client: qpsMockBackendClient,
    ) -> None:
        """Verify transition_to updates the state."""
        # TODO: Call transition_to(PRE_FLIGHT) and assert state changed.
        pass


class TestMissionControllerDispatch:
    """Tests for dispatch command handling."""

    def test_handle_dispatch_stores_order_info(
        self,
        config: qpsConfig,
        mock_flight_manager: qpsMockFlightManager,
        mock_camera_manager: qpsMockCameraManager,
        mock_marker_detector: qpsMockMarkerDetector,
        mock_backend_client: qpsMockBackendClient,
    ) -> None:
        """Verify handle_dispatch populates order fields."""
        # TODO: Call handle_dispatch with a DISPATCH command. Assert
        #  current_order_id, destination, current_delivery_marker_id are set.
        pass


class TestMissionControllerBattery:
    """Tests for battery warning and critical handling."""

    def test_handle_battery_warning(
        self,
        config: qpsConfig,
        mock_flight_manager: qpsMockFlightManager,
        mock_camera_manager: qpsMockCameraManager,
        mock_marker_detector: qpsMockMarkerDetector,
        mock_backend_client: qpsMockBackendClient,
    ) -> None:
        """Verify battery warning triggers appropriate state change."""
        # TODO: Set up controller in an in-flight state, call
        #  handle_battery_warning, assert correct transition.
        pass

    def test_handle_battery_critical(
        self,
        config: qpsConfig,
        mock_flight_manager: qpsMockFlightManager,
        mock_camera_manager: qpsMockCameraManager,
        mock_marker_detector: qpsMockMarkerDetector,
        mock_backend_client: qpsMockBackendClient,
    ) -> None:
        """Verify critical battery triggers emergency action."""
        # TODO: Set up controller, call handle_battery_critical, assert
        #  EMERGENCY_RTL or GROUNDED_AWAITING_RETRIEVAL.
        pass
