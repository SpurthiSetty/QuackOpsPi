import pytest
import numpy

from src.config.qps_config import qpsConfig
from src.enums.qps_landing_result import qpsLandingResult
from src.landing.qps_landing_controller import qpsLandingController
from tests.mocks.qps_mock_flight_manager import qpsMockFlightManager
from tests.mocks.qps_mock_camera_manager import qpsMockCameraManager
from tests.mocks.qps_mock_marker_detector import qpsMockMarkerDetector


@pytest.fixture
def config() -> qpsConfig:
    """Create a default test configuration."""
    return qpsConfig(simulation_mode=True)


@pytest.fixture
def landing_controller(
    config: qpsConfig,
) -> qpsLandingController:
    """Create a landing controller with mock dependencies."""
    flight = qpsMockFlightManager(config)
    camera = qpsMockCameraManager(config)
    detector = qpsMockMarkerDetector(config)
    return qpsLandingController(config, flight, camera, detector)


class TestLandingControllerInit:
    """Tests for qpsLandingController initialisation."""

    def test_initial_state(self, landing_controller: qpsLandingController) -> None:
        """Verify the controller starts in a non-landing state."""
        # TODO: Assert is_landing is False and current_target_marker_id is None.
        pass


class TestLandingControllerExecution:
    """Tests for the execute_landing sequence."""

    @pytest.mark.asyncio
    async def test_execute_landing_marker_found(
        self, landing_controller: qpsLandingController
    ) -> None:
        """Verify successful landing when marker is detected."""
        # TODO: Pre-load mock detections, call execute_landing, assert SUCCESS.
        pass

    @pytest.mark.asyncio
    async def test_execute_landing_marker_not_found(
        self, landing_controller: qpsLandingController
    ) -> None:
        """Verify fallback landing when marker is never detected."""
        # TODO: Set no detection, call execute_landing, assert FALLBACK_LAND.
        pass


class TestVelocityCorrection:
    """Tests for compute_velocity_correction."""

    def test_correction_towards_centre(
        self, landing_controller: qpsLandingController
    ) -> None:
        """Verify correction vector points towards frame centre."""
        # TODO: Create a detection offset from centre, call
        #  compute_velocity_correction, assert correct direction.
        pass
