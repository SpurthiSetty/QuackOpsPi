"""
qps_mission_controller_impl.py

State machine orchestrator for the full QuackOps delivery mission.

Flow (happy path):
    IDLE → AWAITING_DISPATCH → PRE_FLIGHT → ARMING → TAKEOFF
    → EN_ROUTE_TO_DELIVERY → SEARCHING_FOR_MARKER → LANDING
    → DELIVERY_COMPLETE → EN_ROUTE_TO_BASE → MISSION_COMPLETE

Error paths:
    - Pre-flight check failure  → send PRE_FLIGHT_FAILED, back to IDLE
    - Arm failure               → send ARM_FAILED, back to IDLE
    - Any airborne exception    → emergency RTL
    - Battery critical callback → set emergency RTL flag (handled in poll loop)

GPS streaming:
    Runs as an asyncio Task from EN_ROUTE_TO_DELIVERY through MISSION_COMPLETE,
    pushing telemetry GPS to the backend at telemetry_polling_rate_hz.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from quackops_pi.flight.qps_flight_manager_interface import qpsFlightManagerInterface
from quackops_pi.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor
from quackops_pi.comms.qps_backend_client_interface import qpsBackendClientInterface
from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.models.qps_gps_position import qpsGPSPosition
from quackops_pi.models.qps_mission_command import qpsMissionCommand
from quackops_pi.models.qps_mission_state import qpsMissionState
from quackops_pi.models.qps_status_type import qpsStatusType

logger = logging.getLogger("qps.mission_controller")

# How long to poll is_mission_finished between checks (seconds)
_MISSION_POLL_INTERVAL_S: float = 1.0

# Timeout waiting for RTL to land back at home (seconds)
_RTL_LAND_TIMEOUT_S: float = 180.0


class qpsMissionControllerImpl:
    """Full-mission state machine for autonomous drone delivery.

    Dependencies are injected at construction — no class creates its own.
    The landing_controller argument accepts qpsHoverSearchController or the
    orbit-based qpsLandingController — both return qpsLandingResult.

    Usage:
        controller = qpsMissionControllerImpl(fm, telemetry, landing, backend, config)
        await controller.run()     # blocks until MISSION_COMPLETE or error
        await controller.shutdown()
    """

    def __init__(
        self,
        flight_manager: qpsFlightManagerInterface,
        telemetry: qpsTelemetryMonitor,
        landing_controller: Any,   # qpsHoverSearchController | qpsLandingController
        backend: qpsBackendClientInterface,
        config: qpsConfig,
    ) -> None:
        self._fm = flight_manager
        self._telemetry = telemetry
        self._landing = landing_controller
        self._backend = backend
        self._config = config

        self._state: qpsMissionState = qpsMissionState.IDLE

        # Populated when startDelivery arrives
        self._order_id: str = ""
        self._dest_lat: float = 0.0
        self._dest_lon: float = 0.0
        self._target_marker_id: int = config.target_marker_id

        # Dispatch synchronisation
        self._dispatch_event: asyncio.Event = asyncio.Event()
        self._pending_command: Optional[qpsMissionCommand] = None

        # Set by backend when returnToSource is received
        self._return_to_source_event: asyncio.Event = asyncio.Event()

        # GPS streaming task (started on EN_ROUTE, cancelled at MISSION_COMPLETE)
        self._gps_stream_task: Optional[asyncio.Task] = None

        # Set by battery-critical callback; checked in flight poll loops
        self._emergency_rtl: bool = False

    # ── Public API ────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main entry point — runs the full mission lifecycle.

        Blocks until MISSION_COMPLETE or an unrecoverable error.  On Ctrl+C
        the caller's try/finally should invoke shutdown().
        """
        logger.info("Mission controller starting")

        # Wire battery critical callback
        self._telemetry.on_battery_critical(self._on_battery_critical)

        # ── IDLE → AWAITING_DISPATCH ───────────────────────────────
        await self._set_state(qpsMissionState.AWAITING_DISPATCH)
        self._backend.on_command(self._on_command)
        self._backend.on_return_to_source(self._on_return_to_source)
        await self._backend.send_status("ready", {})

        logger.info("Waiting for startDelivery command from backend...")
        await self._dispatch_event.wait()
        cmd = self._pending_command
        if cmd is None:
            logger.error("Dispatch event fired but command is None")
            return

        self._order_id = cmd.order_id
        self._dest_lat = cmd.destination_lat
        self._dest_lon = cmd.destination_lon
        self._target_marker_id = cmd.delivery_marker_id
        logger.info(
            "Dispatch received: orderId=%s  dest=%.6f,%.6f  markerId=%d",
            self._order_id, self._dest_lat, self._dest_lon, self._target_marker_id,
        )

        # ── AWAITING_DISPATCH → PRE_FLIGHT ─────────────────────────
        await self._set_state(qpsMissionState.PRE_FLIGHT)
        if not await self._run_preflight_checks():
            await self._backend.send_status(
                qpsStatusType.PRE_FLIGHT_FAILED.value,
                {"order_id": self._order_id, "reason": "preflight checks failed"},
            )
            await self._set_state(qpsMissionState.IDLE)
            return

        # Upload mission during pre-flight (before arming, while FC is quiet)
        dest_wp = qpsGPSPosition(
            latitude_deg=self._dest_lat,
            longitude_deg=self._dest_lon,
            altitude_m=self._config.orbit_altitude_m,
            heading_deg=0.0,
            speed_m_s=0.0,
            timestamp=time.time(),
        )
        try:
            await self._fm.upload_mission([dest_wp])
            logger.info("Mission uploaded to flight controller")
        except Exception as exc:
            logger.error("Mission upload failed: %s", exc)
            await self._backend.send_status(
                qpsStatusType.PRE_FLIGHT_FAILED.value,
                {"order_id": self._order_id, "reason": f"mission upload failed: {exc}"},
            )
            await self._set_state(qpsMissionState.IDLE)
            return

        # ── PRE_FLIGHT → ARMING ────────────────────────────────────
        await self._set_state(qpsMissionState.ARMING)
        try:
            # Switch to GUIDED mode before arming so MAV_CMD_NAV_TAKEOFF is accepted.
            # pause_mission() is the only interface method that switches to GUIDED.
            logger.info("Switching to GUIDED mode for arm + takeoff")
            await self._fm.pause_mission()
            await self._fm.arm()
        except Exception as exc:
            logger.error("Arm failed: %s", exc)
            await self._backend.send_status(
                qpsStatusType.ARM_FAILED.value,
                {"order_id": self._order_id, "reason": str(exc)},
            )
            await self._set_state(qpsMissionState.IDLE)
            return

        # ── ARMING → TAKEOFF ───────────────────────────────────────
        await self._set_state(qpsMissionState.TAKEOFF)
        try:
            await self._fm.takeoff(self._config.orbit_altitude_m)
        except Exception as exc:
            logger.error("Takeoff failed: %s", exc)
            await self._emergency_rtl_procedure()
            return

        # ── TAKEOFF → EN_ROUTE_TO_DELIVERY ─────────────────────────
        await self._set_state(qpsMissionState.EN_ROUTE_TO_DELIVERY)
        try:
            await self._fm.start_mission()
        except Exception as exc:
            logger.error("start_mission failed: %s", exc)
            await self._emergency_rtl_procedure()
            return

        # Start GPS streaming as a background task
        self._gps_stream_task = asyncio.create_task(
            self._stream_gps_loop(), name="gps-stream"
        )

        # Monitor mission until destination waypoint reached
        logger.info("Flying to destination (%.6f, %.6f)...", self._dest_lat, self._dest_lon)
        try:
            await self._wait_mission_finished()
        except RuntimeError as exc:
            logger.error("Mission monitoring interrupted: %s", exc)
            await self._emergency_rtl_procedure()
            return

        # ── EN_ROUTE_TO_DELIVERY → SEARCHING_FOR_MARKER ────────────
        await self._set_state(qpsMissionState.SEARCHING_FOR_MARKER)
        try:
            # Pause AUTO mission — FC switches to GUIDED hold
            await self._fm.pause_mission()
        except Exception as exc:
            logger.warning("pause_mission failed: %s — continuing search anyway", exc)

        logger.info("Running marker search (ID=%d)...", self._target_marker_id)
        try:
            result = await self._landing.execute_marker_search(self._target_marker_id)
        except Exception as exc:
            logger.error("Marker search raised: %s — using fallback GPS", exc)
            result = None

        if result is not None and result.success:
            logger.info(
                "Marker found after %.1fs (%d frames)",
                result.search_duration_s, result.frames_searched,
            )
        else:
            outcome = result.outcome.name if result else "EXCEPTION"
            logger.warning(
                "Marker search ended with %s — landing at destination GPS",
                outcome,
            )
            await self._backend.send_status(
                qpsStatusType.MARKER_NOT_FOUND.value,
                {"order_id": self._order_id, "outcome": outcome},
            )
            await self._backend.send_status(
                qpsStatusType.FALLBACK_LAND.value,
                {"order_id": self._order_id},
            )

        # ── SEARCHING_FOR_MARKER → LANDING ─────────────────────────
        await self._set_state(qpsMissionState.LANDING)
        try:
            await self._fm.land()
        except Exception as exc:
            logger.error("land() failed: %s", exc)
            # Best effort — try RTL if land fails
            await self._emergency_rtl_procedure()
            return

        # ── LANDING → DELIVERY_COMPLETE ────────────────────────────
        await self._set_state(qpsMissionState.DELIVERY_COMPLETE)
        landed_gps = self._telemetry.get_gps_position()
        await self._backend.send_status(
            qpsStatusType.DELIVERY_COMPLETE.value,
            {
                "order_id": self._order_id,
                "latitude_deg": landed_gps.latitude_deg if landed_gps else self._dest_lat,
                "longitude_deg": landed_gps.longitude_deg if landed_gps else self._dest_lon,
            },
        )

        # ── DELIVERY_COMPLETE: wait for backend returnToSource ──────
        logger.info("Waiting for returnToSource from backend...")
        await self._return_to_source_event.wait()
        logger.info("returnToSource received — initiating RTL")

        # ── DELIVERY_COMPLETE → EN_ROUTE_TO_BASE ───────────────────
        await self._set_state(qpsMissionState.EN_ROUTE_TO_BASE)
        try:
            await self._fm.return_to_launch()
        except Exception as exc:
            logger.error("return_to_launch failed: %s", exc)
            # Already on the ground — just continue to disarm

        logger.info("RTL active — waiting for home landing...")
        await self._wait_for_home_landing()

        # ── EN_ROUTE_TO_BASE → MISSION_COMPLETE ────────────────────
        await self._set_state(qpsMissionState.MISSION_COMPLETE)

        # Stop GPS stream
        if self._gps_stream_task is not None and not self._gps_stream_task.done():
            self._gps_stream_task.cancel()
            try:
                await self._gps_stream_task
            except asyncio.CancelledError:
                pass
            self._gps_stream_task = None

        try:
            await self._fm.disarm()
        except Exception as exc:
            logger.warning("disarm failed (may have auto-disarmed): %s", exc)

        await self._backend.send_status(
            qpsStatusType.MISSION_COMPLETE.value,
            {"order_id": self._order_id},
        )
        logger.info("═══ MISSION COMPLETE ═══")

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel streams, disconnect backend and flight manager."""
        logger.info("Mission controller shutting down")
        if self._gps_stream_task is not None and not self._gps_stream_task.done():
            self._gps_stream_task.cancel()
            try:
                await self._gps_stream_task
            except asyncio.CancelledError:
                pass
        await self._backend.disconnect()
        await self._fm.disconnect()

    # ── State management ──────────────────────────────────────────────

    async def _set_state(self, new_state: qpsMissionState) -> None:
        """Transition to new_state, logging the change."""
        old_name = self._state.name
        self._state = new_state
        logger.info("State: %s → %s", old_name, new_state.name)

    # ── Phase helpers ─────────────────────────────────────────────────

    async def _run_preflight_checks(self) -> bool:
        """Validate drone health before committing to flight.

        Checks GPS fix quality and battery level.

        Returns:
            True if all checks pass, False otherwise.
        """
        state = self._telemetry.get_drone_state()
        if state is None:
            logger.error("Preflight: no telemetry data available")
            return False

        if state.gps_fix_type < 3:
            logger.error(
                "Preflight: GPS fix too weak (fix_type=%d, need >=3)",
                state.gps_fix_type,
            )
            return False

        if state.battery_percent < self._config.battery_warning_percent:
            logger.error(
                "Preflight: battery too low (%.0f%% < %.0f%% warning threshold)",
                state.battery_percent, self._config.battery_warning_percent,
            )
            return False

        logger.info(
            "Preflight OK: GPS fix_type=%d  battery=%.0f%%  sats=%d",
            state.gps_fix_type, state.battery_percent, state.gps_num_satellites,
        )
        return True

    async def _wait_mission_finished(self) -> None:
        """Poll is_mission_finished() until the destination waypoint is reached.

        Raises RuntimeError if battery-critical emergency RTL is triggered.
        """
        last_log = time.time()
        while True:
            if self._emergency_rtl:
                raise RuntimeError("Emergency RTL triggered by battery-critical callback")

            if await self._fm.is_mission_finished():
                logger.info("Mission finished — destination waypoint reached")
                return

            # Log progress at a reduced rate to avoid log spam
            if time.time() - last_log >= 5.0:
                gps = self._telemetry.get_gps_position()
                if gps:
                    logger.info(
                        "En route: lat=%.6f  lon=%.6f  alt=%.1fm",
                        gps.latitude_deg, gps.longitude_deg, gps.altitude_m,
                    )
                last_log = time.time()

            await asyncio.sleep(_MISSION_POLL_INTERVAL_S)

    async def _wait_for_home_landing(self) -> None:
        """Wait for the full RTL flight home: liftoff first, then touchdown.

        Phase 1 — wait for airborne (in_air or alt > 2m), timeout 30s.
            Needed because after delivery landing the drone is already on the
            ground (in_air == False).  Checking landed immediately would return
            at once and skip the entire RTL flight.

        Phase 2 — wait for landed (not in_air), timeout _RTL_LAND_TIMEOUT_S.
        """
        _AIRBORNE_TIMEOUT_S = 30.0
        _AIRBORNE_ALT_M = 2.0

        loop = asyncio.get_running_loop()
        last_log = time.time()

        # ── Phase 1: wait for liftoff ─────────────────────────────
        logger.info("RTL: waiting for liftoff...")
        airborne_deadline = loop.time() + _AIRBORNE_TIMEOUT_S
        while loop.time() < airborne_deadline:
            gps = self._telemetry.get_gps_position()
            state = self._telemetry.get_drone_state()
            airborne = (state and state.in_air) or (gps and gps.altitude_m > _AIRBORNE_ALT_M)
            if airborne:
                logger.info(
                    "RTL: airborne (alt=%.1fm)",
                    gps.altitude_m if gps else 0.0,
                )
                break
            await asyncio.sleep(1.0)
        else:
            logger.warning(
                "RTL liftoff not detected within %.0fs — skipping RTL wait",
                _AIRBORNE_TIMEOUT_S,
            )
            return

        # ── Phase 2: wait for home landing ────────────────────────
        logger.info("RTL: flying home, waiting for landing...")
        land_deadline = loop.time() + _RTL_LAND_TIMEOUT_S
        while loop.time() < land_deadline:
            state = self._telemetry.get_drone_state()
            gps = self._telemetry.get_gps_position()
            landed = state and not state.in_air
            if landed:
                logger.info("Home landing confirmed (not in_air)")
                return

            if time.time() - last_log >= 5.0:
                if gps:
                    logger.info(
                        "RTL: lat=%.6f  lon=%.6f  alt=%.1fm",
                        gps.latitude_deg, gps.longitude_deg, gps.altitude_m,
                    )
                last_log = time.time()

            await asyncio.sleep(1.0)

        logger.warning("RTL land timeout after %.0fs — continuing", _RTL_LAND_TIMEOUT_S)

    async def _emergency_rtl_procedure(self) -> None:
        """Best-effort emergency RTL.  Called on any mid-flight exception."""
        logger.error("EMERGENCY RTL initiated")
        await self._set_state(qpsMissionState.EMERGENCY_RTL)
        try:
            await self._fm.return_to_launch()
        except Exception as exc:
            logger.error("Emergency RTL command failed: %s", exc)
        # Stop GPS stream if running
        if self._gps_stream_task is not None and not self._gps_stream_task.done():
            self._gps_stream_task.cancel()

    # ── GPS streaming ─────────────────────────────────────────────────

    async def _stream_gps_loop(self) -> None:
        """Periodically read telemetry GPS and push it to the backend.

        Runs at config.telemetry_polling_rate_hz.  Cancelled by MissionController
        at MISSION_COMPLETE.
        """
        interval = 1.0 / self._config.telemetry_polling_rate_hz
        try:
            while True:
                position = self._telemetry.get_gps_position()
                if position is not None:
                    await self._backend.stream_gps(position)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("GPS stream task cancelled")
            raise

    # ── Callbacks ─────────────────────────────────────────────────────

    def _on_command(self, command: qpsMissionCommand) -> None:
        """Invoked synchronously by qpsBackendClient when a command arrives."""
        self._pending_command = command
        self._dispatch_event.set()

    def _on_return_to_source(self, order_id: str) -> None:
        """Invoked synchronously by qpsBackendClient when returnToSource arrives."""
        logger.info("returnToSource acknowledged for orderId=%s", order_id)
        self._return_to_source_event.set()

    def _on_battery_critical(self, pct: float) -> None:
        """Invoked synchronously by qpsTelemetryMonitor on critical battery.

        Sets emergency_rtl flag; the flight poll loops check this each iteration
        and trigger RTL from whatever airborne state they are in.
        """
        logger.error(
            "BATTERY CRITICAL (%.0f%%) — emergency RTL will trigger at next poll",
            pct,
        )
        self._emergency_rtl = True
