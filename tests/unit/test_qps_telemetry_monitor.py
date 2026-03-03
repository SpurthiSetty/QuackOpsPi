import pytest

from src.config.qps_config import qpsConfig
from src.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor


@pytest.fixture
def config() -> qpsConfig:
    """Create a default test configuration."""
    return qpsConfig(simulation_mode=True)


class TestTelemetryMonitorInit:
    """Tests for qpsTelemetryMonitor initialisation."""

    def test_initial_state(self, config: qpsConfig) -> None:
        """Verify the monitor starts in a non-running state."""
        # TODO: Instantiate with a mock drone, assert running is False.
        pass


class TestTelemetryCallbacks:
    """Tests for battery callback registration."""

    def test_set_battery_warning_callback(self, config: qpsConfig) -> None:
        """Verify the warning callback is stored."""
        # TODO: Register a callback, verify it is stored.
        pass

    def test_set_battery_critical_callback(self, config: qpsConfig) -> None:
        """Verify the critical callback is stored."""
        # TODO: Register a callback, verify it is stored.
        pass


class TestTelemetryData:
    """Tests for telemetry data access."""

    def test_get_drone_state_initially_none(self, config: qpsConfig) -> None:
        """Verify drone state is None before monitoring starts."""
        # TODO: Assert get_drone_state() returns None initially.
        pass

    def test_get_gps_position_initially_none(self, config: qpsConfig) -> None:
        """Verify GPS position is None before monitoring starts."""
        # TODO: Assert get_gps_position() returns None initially.
        pass
