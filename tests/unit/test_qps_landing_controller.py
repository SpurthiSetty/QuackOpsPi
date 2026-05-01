"""
tests/unit/test_qps_landing_controller.py

Unit tests for qpsLandingController using mock camera, detector,
and a lightweight stub telemetry monitor — no hardware required.
"""
from __future__ import annotations

import asyncio
import math
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.mission.qps_landing_controller import qpsLandingController
from quackops_pi.models.qps_drone_state import qpsDroneState
from quackops_pi.models.qps_gps_position import qpsGPSPosition
from quackops_pi.models.qps_landing_result import qpsLandingOutcome
from quackops_pi.models.qps_marker_detection import qpsMarkerDetection
from quackops_pi.vision.qps_mock_camera_manager import qpsMockCameraManager
from quackops_pi.vision.qps_mock_marker_detector import qpsMockMarkerDetector


# ── Stub telemetry monitor ────────────────────────────────────────────────────

class _StubTelemetry:
    """Minimal stand-in for qpsTelemetryMonitor.

    Only exposes the two methods qpsLandingController actually calls.
    """

    def __init__(
        self,
        gps: Optional[qpsGPSPosition] = None,
        state: Optional[qpsDroneState] = None,
    ) -> None:
        self._gps = gps
        self._state = state

    def get_gps_position(self) -> Optional[qpsGPSPosition]:
        return self._gps

    def get_drone_state(self) -> Optional[qpsDroneState]:
        return self._state


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_config(**overrides) -> qpsConfig:
    """Return a qpsConfig with a short timeout by default."""
    cfg = qpsConfig()
    cfg.search_timeout_s = overrides.pop("search_timeout_s", 999.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_gps(lat: float = 40.0, lon: float = -74.0, heading: float = 0.0) -> qpsGPSPosition:
    return qpsGPSPosition(
        latitude_deg=lat,
        longitude_deg=lon,
        altitude_m=10.0,
        heading_deg=heading,
        speed_m_s=0.0,
        timestamp=0.0,
    )


def _make_state(gps: Optional[qpsGPSPosition] = None) -> qpsDroneState:
    state = qpsDroneState()
    state.gps_position = gps or _make_gps()
    return state


def _make_detection(
    marker_id: int = 0,
    tvec: tuple = (1.0, -2.0, 5.0),
    confidence: float = 0.9,
) -> qpsMarkerDetection:
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    return qpsMarkerDetection(
        marker_id=marker_id,
        corners=corners,
        center_px=(5.0, 5.0),
        confidence=confidence,
        tvec=tvec,
        rvec=(0.0, 0.0, 0.0),
        distance_m=tvec[2],
    )


def _make_controller(
    camera: Optional[qpsMockCameraManager] = None,
    detector: Optional[qpsMockMarkerDetector] = None,
    telemetry: Optional[_StubTelemetry] = None,
    config: Optional[qpsConfig] = None,
) -> qpsLandingController:
    cfg = config or _make_config()
    cam = camera or qpsMockCameraManager(cfg)
    det = detector or qpsMockMarkerDetector(cfg)
    tel = telemetry or _StubTelemetry(gps=_make_gps(), state=_make_state())
    return qpsLandingController(cam, det, tel, cfg)


# ── Initial state ─────────────────────────────────────────────────────────────

class TestInitialState:
    def test_not_searching_at_start(self) -> None:
        ctrl = _make_controller()
        assert ctrl.is_searching is False


# ── Marker found paths ────────────────────────────────────────────────────────

class TestMarkerFound:
    @pytest.mark.asyncio
    async def test_marker_found_on_first_frame(self) -> None:
        det = qpsMockMarkerDetector(_make_config())
        det.set_constant_detection(_make_detection(marker_id=0))

        ctrl = _make_controller(detector=det)
        result = await ctrl.execute_marker_search(target_marker_id=0)

        assert result.outcome == qpsLandingOutcome.MARKER_FOUND
        assert result.marker_gps is not None
        assert result.frames_searched == 1
        assert result.target_marker_id == 0

    @pytest.mark.asyncio
    async def test_marker_found_after_several_misses(self) -> None:
        det = qpsMockMarkerDetector(_make_config())
        det.set_scripted_detections([
            [],
            [],
            [_make_detection(marker_id=0)],
        ])

        ctrl = _make_controller(detector=det)
        result = await ctrl.execute_marker_search(target_marker_id=0)

        assert result.outcome == qpsLandingOutcome.MARKER_FOUND
        assert result.frames_searched == 3

    @pytest.mark.asyncio
    async def test_is_searching_flag_is_true_during_search(self) -> None:
        """The flag must be True while execute_marker_search is running."""
        cfg = _make_config()
        det = qpsMockMarkerDetector(cfg)
        det.set_no_detection()
        ctrl = _make_controller(detector=det, config=cfg)

        task = asyncio.create_task(ctrl.execute_marker_search(target_marker_id=0))
        await asyncio.sleep(0)  # let the task start and reach its first yield
        assert ctrl.is_searching is True
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_is_searching_false_after_completion(self) -> None:
        det = qpsMockMarkerDetector(_make_config())
        det.set_constant_detection(_make_detection(marker_id=0))

        ctrl = _make_controller(detector=det)
        await ctrl.execute_marker_search(target_marker_id=0)

        assert ctrl.is_searching is False


# ── Timeout / no marker paths ─────────────────────────────────────────────────

class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_search_timeout(self) -> None:
        cfg = _make_config(search_timeout_s=0.0)
        det = qpsMockMarkerDetector(cfg)
        det.set_no_detection()
        gps = _make_gps(lat=40.1, lon=-74.1)
        tel = _StubTelemetry(gps=gps, state=_make_state(gps))

        ctrl = _make_controller(detector=det, telemetry=tel, config=cfg)
        result = await ctrl.execute_marker_search(target_marker_id=0)

        assert result.outcome == qpsLandingOutcome.SEARCH_TIMEOUT
        assert result.marker_gps is None
        assert result.fallback_gps is not None
        assert result.fallback_gps.latitude_deg == pytest.approx(40.1)

    @pytest.mark.asyncio
    async def test_wrong_marker_id_causes_timeout(self) -> None:
        cfg = _make_config(search_timeout_s=0.0)
        det = qpsMockMarkerDetector(cfg)
        det.set_constant_detection(_make_detection(marker_id=99))  # searching for 0

        ctrl = _make_controller(detector=det, config=cfg)
        result = await ctrl.execute_marker_search(target_marker_id=0)

        assert result.outcome == qpsLandingOutcome.SEARCH_TIMEOUT


# ── Abort path ────────────────────────────────────────────────────────────────

class TestAbort:
    @pytest.mark.asyncio
    async def test_abort_returns_aborted_outcome(self) -> None:
        cfg = _make_config(search_timeout_s=999.0)
        det = qpsMockMarkerDetector(cfg)
        det.set_no_detection()
        gps = _make_gps(lat=40.5, lon=-74.5)
        tel = _StubTelemetry(gps=gps, state=_make_state(gps))

        ctrl = _make_controller(detector=det, telemetry=tel, config=cfg)
        task = asyncio.create_task(ctrl.execute_marker_search(target_marker_id=0))
        await asyncio.sleep(0)  # let search_loop reach its first yield point
        ctrl.abort()
        result = await task

        assert result.outcome == qpsLandingOutcome.ABORTED
        assert result.fallback_gps is not None
        assert result.fallback_gps.latitude_deg == pytest.approx(40.5)

    @pytest.mark.asyncio
    async def test_abort_before_search_has_no_effect(self) -> None:
        """Calling abort() before execute_marker_search does nothing."""
        det = qpsMockMarkerDetector(_make_config())
        det.set_constant_detection(_make_detection(marker_id=0))
        ctrl = _make_controller(detector=det)

        ctrl.abort()  # before search starts — should be ignored
        result = await ctrl.execute_marker_search(target_marker_id=0)

        assert result.outcome == qpsLandingOutcome.MARKER_FOUND


# ── Camera failure paths ──────────────────────────────────────────────────────

class TestCameraFailure:
    @pytest.mark.asyncio
    async def test_camera_fails_to_start(self) -> None:
        cfg = _make_config()
        cam = qpsMockCameraManager(cfg)
        cam.fail_on_start = True

        ctrl = _make_controller(camera=cam, config=cfg)
        result = await ctrl.execute_marker_search(target_marker_id=0)

        assert result.outcome == qpsLandingOutcome.CAMERA_FAILURE
        assert result.frames_searched == 0
        assert result.search_duration_s == pytest.approx(0.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_camera_returns_none_frame(self) -> None:
        cfg = _make_config()
        cam = qpsMockCameraManager(cfg)
        cam.return_none = True

        ctrl = _make_controller(camera=cam, config=cfg)
        result = await ctrl.execute_marker_search(target_marker_id=0)

        assert result.outcome == qpsLandingOutcome.CAMERA_FAILURE


# ── Marker GPS computation ────────────────────────────────────────────────────

_R = 6_371_000.0  # matches qpsLandingController._EARTH_RADIUS_M


class TestComputeMarkerGps:
    """Tests for _compute_marker_gps — the core pose-to-GPS math."""

    def _ctrl(self, heading: float) -> qpsLandingController:
        gps = _make_gps(lat=40.0, lon=-74.0, heading=heading)
        state = _make_state(gps)
        tel = _StubTelemetry(gps=gps, state=state)
        return _make_controller(telemetry=tel)

    def test_north_heading_offsets(self) -> None:
        """heading=0°: camera_x→east, camera_-y→north with no rotation."""
        ctrl = self._ctrl(heading=0.0)
        detection = _make_detection(tvec=(1.0, -2.0, 5.0))  # body_east=1, body_north=2
        state = _make_state(_make_gps(lat=40.0, lon=-74.0, heading=0.0))

        result = ctrl._compute_marker_gps(detection, state)

        assert result is not None
        expected_delta_lat = 2.0 / _R * (180.0 / math.pi)
        expected_delta_lon = 1.0 / (_R * math.cos(math.radians(40.0))) * (180.0 / math.pi)
        assert result.latitude_deg == pytest.approx(40.0 + expected_delta_lat, rel=1e-6)
        assert result.longitude_deg == pytest.approx(-74.0 + expected_delta_lon, rel=1e-6)

    def test_east_heading_rotates_offsets(self) -> None:
        """heading=90°: body_north maps to ned_east, body_east maps to ned_south."""
        ctrl = self._ctrl(heading=90.0)
        # tvec=(1,-2,5) → body_east=1, body_north=2
        # heading=90°: ned_north = 2*cos90 - 1*sin90 = -1
        #              ned_east  = 2*sin90 + 1*cos90 =  2
        detection = _make_detection(tvec=(1.0, -2.0, 5.0))
        state = _make_state(_make_gps(lat=40.0, lon=-74.0, heading=90.0))

        result = ctrl._compute_marker_gps(detection, state)

        assert result is not None
        expected_delta_lat = -1.0 / _R * (180.0 / math.pi)
        expected_delta_lon = 2.0 / (_R * math.cos(math.radians(40.0))) * (180.0 / math.pi)
        assert result.latitude_deg == pytest.approx(40.0 + expected_delta_lat, rel=1e-6)
        assert result.longitude_deg == pytest.approx(-74.0 + expected_delta_lon, rel=1e-6)

    def test_returns_none_when_drone_state_is_none(self) -> None:
        ctrl = _make_controller()
        detection = _make_detection()
        assert ctrl._compute_marker_gps(detection, None) is None

    def test_returns_none_when_gps_position_is_none(self) -> None:
        ctrl = _make_controller()
        state = qpsDroneState()  # gps_position defaults to None
        detection = _make_detection()
        assert ctrl._compute_marker_gps(detection, state) is None

    def test_returns_none_when_tvec_is_none(self) -> None:
        ctrl = _make_controller()
        corners = np.zeros((4, 2))
        detection = qpsMarkerDetection(
            marker_id=0, corners=corners, center_px=(0.0, 0.0),
            confidence=0.9, tvec=None, distance_m=None,
        )
        state = _make_state()
        assert ctrl._compute_marker_gps(detection, state) is None


# ── Multiple detections — picks highest confidence ────────────────────────────

class TestFindTarget:
    def test_picks_highest_confidence_match(self) -> None:
        low = _make_detection(marker_id=0, confidence=0.5)
        high = _make_detection(marker_id=0, confidence=0.95)
        wrong = _make_detection(marker_id=1, confidence=0.99)

        result = qpsLandingController._find_target([low, wrong, high], target_id=0)
        assert result is high

    def test_returns_none_when_no_match(self) -> None:
        detections = [_make_detection(marker_id=5)]
        assert qpsLandingController._find_target(detections, target_id=0) is None

    def test_returns_none_for_empty_list(self) -> None:
        assert qpsLandingController._find_target([], target_id=0) is None
