import pytest
import pytest_asyncio

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.models.qps_landing_result import qpsLandingOutcome, qpsLandingResult
from quackops_pi.mission.qps_landing_controller import qpsLandingController
from quackops_pi.flight.qps_mock_flight_manager import qpsMockFlightManager
from quackops_pi.vision.qps_mock_camera_manager import qpsMockCameraManager
from quackops_pi.vision.qps_mock_marker_detector import qpsMockMarkerDetector


@pytest.fixture
def config() -> qpsConfig:
    """Create a default test configuration."""
    return qpsConfig(simulation_mode=True)


@pytest.fixture
def landing_controller(config: qpsConfig) -> qpsLandingController:
    """Create a landing controller wired to mock dependencies."""
    flight = qpsMockFlightManager(config)
    camera = qpsMockCameraManager(config)
    detector = qpsMockMarkerDetector(config)
    return qpsLandingController(config, flight, camera, detector)


class TestLandingIntegration:
    """Integration tests for the full precision-landing pipeline."""

    @pytest.mark.asyncio
    async def test_full_landing_with_marker(
        self, landing_controller: qpsLandingController
    ) -> None:
        """Test landing sequence when marker is consistently detected."""
        # TODO: Pre-load detections showing marker approaching centre,
        #  call execute_landing, assert SUCCESS.
        pass

    @pytest.mark.asyncio
    async def test_landing_with_intermittent_detection(
        self, landing_controller: qpsLandingController
    ) -> None:
        """Test landing when marker detection is intermittent."""
        # TODO: Script detections with gaps, call execute_landing,
        #  verify retry behaviour.
        pass

    @pytest.mark.asyncio
    async def test_landing_abort(
        self, landing_controller: qpsLandingController
    ) -> None:
        """Test that abort stops the landing sequence."""
        # TODO: Start execute_landing, call abort mid-sequence, assert ABORTED.
        pass
