"""
qps_mission_controller_landing.py

Landing orchestration methods for qpsMissionController.
Orbit calculation + happy/fallback landing paths.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import List, TYPE_CHECKING

from quackops_pi.config import qpsConfig
from quackops_pi.models import qpsGPSPosition, qpsLandingOutcome, qpsLandingResult

if TYPE_CHECKING:
    from quackops_pi.comms import qpsBackendClientInterface
    from quackops_pi.flight import qpsFlightManagerInterface
    from quackops_pi.mission.qps_landing_controller import qpsLandingController
    from quackops_pi.telemetry import qpsTelemetryMonitor

logger = logging.getLogger("qps.mission_controller")

_EARTH_RADIUS_M = 6_371_000.0


class qpsMissionControllerLandingMixin:
    """Landing orchestration methods.

    Assumes the host class exposes:
        self._flight_manager: qpsFlightManagerInterface
        self._landing_controller: qpsLandingController
        self._telemetry: qpsTelemetryMonitor
        self._backend: qpsBackendClientInterface
        self._config: qpsConfig
    """

    # ── Top-level delivery landing sequence ───────────────────────────

    async def _execute_delivery_landing(
        self,
        delivery_gps: qpsGPSPosition,
        target_marker_id: int,
    ) -> None:
        """Full delivery landing sequence per the activity diagram."""

        # 1. Calculate orbit waypoints
        orbit_waypoints = self._calculate_orbit_waypoints(
            center=delivery_gps,
            radius_m=self._config.orbit_radius_m,
            num_points=self._config.orbit_num_points,
            altitude_m=self._config.orbit_altitude_m,
        )
        logger.info(
            "Orbit planned: %d waypoints, radius=%.1fm, alt=%.1fm",
            len(orbit_waypoints), self._config.orbit_radius_m, self._config.orbit_altitude_m,
        )

        # 2. Upload and start orbit mission
        await self._flight_manager.upload_mission(orbit_waypoints)
        await self._flight_manager.start_mission()
        logger.info("Orbit mission started")

        # 3. Run marker search concurrently with orbit
        search_result: qpsLandingResult = (
            await self._landing_controller.execute_marker_search(target_marker_id)
        )

        logger.info(
            "Search complete: outcome=%s, duration=%.1fs, frames=%d",
            search_result.outcome.name, search_result.search_duration_s,
            search_result.frames_searched,
        )

        # 4. Branch on search outcome
        if search_result.success:
            await self._handle_marker_found(search_result)
        else:
            await self._handle_marker_not_found(search_result, delivery_gps)

    # ── Happy path ────────────────────────────────────────────────────

    async def _handle_marker_found(self, result: qpsLandingResult) -> None:
        assert result.marker_gps is not None

        logger.info(
            "Marker found at lat=%.7f, lon=%.7f — initiating landing",
            result.marker_gps.latitude_deg, result.marker_gps.longitude_deg,
        )

        await self._flight_manager.pause_mission()
        await self._flight_manager.goto_location(
            latitude_deg=result.marker_gps.latitude_deg,
            longitude_deg=result.marker_gps.longitude_deg,
            altitude_m=self._config.orbit_altitude_m,
        )
        await self._wait_for_arrival(
            target_lat=result.marker_gps.latitude_deg,
            target_lon=result.marker_gps.longitude_deg,
            tolerance_m=self._config.goto_arrival_tolerance_m,
        )

        await self._flight_manager.land()
        await self._wait_for_landed_state()

        final_gps = self._telemetry.get_gps_position()
        await self._backend.send_status(
            status_type="DELIVERY_COMPLETE",
            data={
                "marker_id": result.target_marker_id,
                "marker_gps": result.marker_gps.to_dict(),
                "landed_gps": final_gps.to_dict() if final_gps else None,
                "search_duration_s": result.search_duration_s,
                "frames_searched": result.frames_searched,
            },
        )
        logger.info("Delivery complete — landed at marker position")

    # ── Fallback path ─────────────────────────────────────────────────

    async def _handle_marker_not_found(
        self,
        result: qpsLandingResult,
        delivery_gps: qpsGPSPosition,
    ) -> None:
        logger.warning(
            "Marker not found (outcome=%s) — fallback landing at delivery center",
            result.outcome.name,
        )

        await self._flight_manager.pause_mission()
        await self._flight_manager.goto_location(
            latitude_deg=delivery_gps.latitude_deg,
            longitude_deg=delivery_gps.longitude_deg,
            altitude_m=self._config.orbit_altitude_m,
        )
        await self._wait_for_arrival(
            target_lat=delivery_gps.latitude_deg,
            target_lon=delivery_gps.longitude_deg,
            tolerance_m=self._config.goto_arrival_tolerance_m,
        )

        await self._flight_manager.land()
        await self._wait_for_landed_state()

        final_gps = self._telemetry.get_gps_position()
        await self._backend.send_status(
            status_type="FALLBACK_LANDING",
            data={
                "landed_gps": final_gps.to_dict() if final_gps else None,
                "target_marker_id": result.target_marker_id,
                "search_duration_s": result.search_duration_s,
                "frames_searched": result.frames_searched,
                "reason": result.outcome.name,
            },
        )
        await self._backend.send_status(
            status_type="MARKER_NOT_FOUND",
            data={
                "target_marker_id": result.target_marker_id,
                "delivery_gps": delivery_gps.to_dict(),
                "message": (
                    f"Orbit search for marker {result.target_marker_id} "
                    f"timed out after {result.search_duration_s:.1f}s "
                    f"({result.frames_searched} frames). "
                    f"Landed at delivery center."
                ),
            },
        )
        logger.info("Fallback landing complete — backend notified")

    # ── Orbit waypoint calculation ────────────────────────────────────

    @staticmethod
    def _calculate_orbit_waypoints(
        center: qpsGPSPosition,
        radius_m: float,
        num_points: int,
        altitude_m: float,
    ) -> List[qpsGPSPosition]:
        """Generate N evenly-spaced GPS waypoints on a circle around center."""
        if num_points < 3:
            raise ValueError(f"Orbit requires at least 3 waypoints, got {num_points}")

        waypoints: List[qpsGPSPosition] = []
        center_lat_rad = math.radians(center.latitude_deg)

        for i in range(num_points):
            bearing_rad = 2.0 * math.pi * i / num_points
            north_m = radius_m * math.cos(bearing_rad)
            east_m = radius_m * math.sin(bearing_rad)

            delta_lat_deg = (north_m / _EARTH_RADIUS_M) * (180.0 / math.pi)
            delta_lon_deg = (
                east_m / (_EARTH_RADIUS_M * math.cos(center_lat_rad))
            ) * (180.0 / math.pi)

            waypoints.append(
                qpsGPSPosition(
                    latitude_deg=center.latitude_deg + delta_lat_deg,
                    longitude_deg=center.longitude_deg + delta_lon_deg,
                    altitude_m=altitude_m,
                    heading_deg=0.0,
                    speed_m_s=0.0,
                    timestamp=0.0,
                )
            )
        return waypoints

    # ── Wait helpers ──────────────────────────────────────────────────

    async def _wait_for_arrival(
        self, target_lat: float, target_lon: float, tolerance_m: float,
    ) -> None:
        poll_interval = 1.0 / max(self._config.telemetry_polling_rate_hz, 1.0)
        while True:
            gps = self._telemetry.get_gps_position()
            if gps is not None:
                dist = self._haversine_m(
                    gps.latitude_deg, gps.longitude_deg, target_lat, target_lon,
                )
                if dist <= tolerance_m:
                    return
            await asyncio.sleep(poll_interval)

    async def _wait_for_landed_state(self) -> None:
        poll_interval = 1.0 / max(self._config.telemetry_polling_rate_hz, 1.0)
        while True:
            drone_state = self._telemetry.get_drone_state()
            if drone_state is not None and not drone_state.in_air:
                return
            await asyncio.sleep(poll_interval)

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))