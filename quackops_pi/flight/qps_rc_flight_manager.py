"""
qps_rc_flight_manager.py

RC-override flight manager for indoor/no-GPS cage testing.
Uses ALT_HOLD mode with RC_OVERRIDE messages for motor control.

A background thread sends RC_OVERRIDE at 20 Hz to prevent ArduPilot's
RC failsafe. Public flight methods are async for interface compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import List, Optional

from pymavlink import mavutil

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager_base import qpsFlightManagerBase
from quackops_pi.models.qps_gps_position import qpsGPSPosition

logger = logging.getLogger("qps.rc_flight_manager")

RC_MIN: int = 1000
RC_MAX: int = 2000
RC_CENTER: int = 1500
RC_OVERRIDE_HZ: int = 20


class qpsRCFlightManager(qpsFlightManagerBase):
    """RC-override flight manager for indoor cage testing without GPS.

    Uses ALT_HOLD mode and RC_OVERRIDE messages for motor control.
    A background thread sends RC_OVERRIDE at 20 Hz to prevent failsafe.
    Altitude targeting is time-based — no GPS altitude feedback indoors.
    """

    def __init__(self, config: qpsConfig) -> None:
        super().__init__(config)
        self._rc_thread: Optional[threading.Thread] = None
        self._rc_running: bool = False
        self._rc_lock: threading.Lock = threading.Lock()
        self._throttle_pwm: int = RC_CENTER
        self._roll_pwm: int = RC_CENTER
        self._pitch_pwm: int = RC_CENTER
        self._yaw_pwm: int = RC_CENTER

    # ── Connection ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open MAVLink connection and start RC override thread."""
        await super().connect()
        self._rc_running = True
        self._rc_thread = threading.Thread(
            target=self._rc_override_loop, daemon=True, name="rc-override"
        )
        self._rc_thread.start()
        self._set_rc_channels(
            throttle=RC_CENTER, roll=RC_CENTER, pitch=RC_CENTER, yaw=RC_CENTER
        )
        logger.info("RC override thread started")

    async def disconnect(self) -> None:
        """Stop RC override thread, idle motors, then close MAVLink connection."""
        self._rc_running = False
        if self._rc_thread is not None:
            self._rc_thread.join(timeout=2.0)
            self._rc_thread = None

        # Send a few idle overrides to ensure motors stop before closing
        if self._mav is not None:
            for _ in range(5):
                try:
                    self._mav.mav.rc_channels_override_send(
                        self._mav.target_system,
                        self._mav.target_component,
                        RC_CENTER, RC_CENTER, RC_MIN, RC_CENTER,
                        0, 0, 0, 0,
                    )
                except Exception:
                    pass
                time.sleep(0.05)

        await super().disconnect()

    # ── RC override thread ────────────────────────────────────────────

    def _rc_override_loop(self) -> None:
        """Background thread: sends RC_OVERRIDE at RC_OVERRIDE_HZ."""
        interval = 1.0 / RC_OVERRIDE_HZ
        while self._rc_running:
            with self._rc_lock:
                throttle = self._throttle_pwm
                roll = self._roll_pwm
                pitch = self._pitch_pwm
                yaw = self._yaw_pwm
            try:
                self._mav.mav.rc_channels_override_send(
                    self._mav.target_system,
                    self._mav.target_component,
                    roll, pitch, throttle, yaw,
                    0, 0, 0, 0,  # CH5-8 no override
                )
            except Exception as exc:
                logger.warning("RC override send error: %s", exc)
            time.sleep(interval)

    def _set_rc_channels(
        self,
        throttle: Optional[int] = None,
        roll: Optional[int] = None,
        pitch: Optional[int] = None,
        yaw: Optional[int] = None,
    ) -> None:
        """Update RC channel PWM values thread-safely."""
        with self._rc_lock:
            if throttle is not None:
                self._throttle_pwm = max(RC_MIN, min(RC_MAX, throttle))
            if roll is not None:
                self._roll_pwm = max(RC_MIN, min(RC_MAX, roll))
            if pitch is not None:
                self._pitch_pwm = max(RC_MIN, min(RC_MAX, pitch))
            if yaw is not None:
                self._yaw_pwm = max(RC_MIN, min(RC_MAX, yaw))

    # ── Basic flight ──────────────────────────────────────────────────

    async def takeoff(self, altitude_m: float) -> None:
        """ALT_HOLD takeoff using time-based throttle ramping.

        WARNING: No GPS altitude feedback indoors. altitude_m is a best-effort
        estimate based on rc_climb_rate_m_per_s from config.
        """
        logger.warning(
            "RC takeoff: altitude is time-based estimate only — no GPS feedback"
        )
        await self._set_mode(self.MODE_ALT_HOLD)
        self._set_rc_channels(throttle=RC_CENTER)
        logger.info("ALT_HOLD engaged, ready for takeoff")

        # Ramp throttle from center to climb value in 10 PWM steps every 0.1s
        climb_pwm = self._config.rc_climb_throttle_pwm
        current = RC_CENTER
        while current < climb_pwm:
            current = min(current + 10, climb_pwm)
            self._set_rc_channels(throttle=current)
            await asyncio.sleep(0.1)

        # Hold climb throttle for time proportional to target altitude
        climb_duration = altitude_m / self._config.rc_climb_rate_m_per_s
        logger.info(
            "Climbing to ~%.1fm (%.1fs at %d PWM)...",
            altitude_m, climb_duration, climb_pwm,
        )
        await asyncio.sleep(climb_duration)

        # Return to hold-altitude throttle
        self._set_rc_channels(throttle=RC_CENTER)
        logger.info("Takeoff complete (time-based, no altitude feedback)")

    async def land(self) -> None:
        """ALT_HOLD descent and disarm.

        Descends at rc_descend_throttle_pwm for rc_landing_duration_s, then
        idles motors and disarms.
        """
        descend_pwm = self._config.rc_descend_throttle_pwm
        duration = self._config.rc_landing_duration_s
        logger.info("Landing: throttle %d for %.1fs...", descend_pwm, duration)
        self._set_rc_channels(throttle=descend_pwm)
        await asyncio.sleep(duration)

        self._set_rc_channels(throttle=RC_MIN)
        await asyncio.sleep(1.0)
        await self.disarm()
        logger.info("Landing complete")

    async def send_velocity_ned(
        self,
        north_m_s: float,
        east_m_s: float,
        down_m_s: float,    # noqa: ARG002 — horizontal corrections only
        yaw_deg_s: float = 0.0,  # noqa: ARG002 — yaw not used
    ) -> None:
        """Map NED velocity to RC channel offsets for visual servo corrections.

        Only north/south (pitch channel) and east/west (roll channel) are applied.
        down_m_s and yaw_deg_s are ignored.

        Pitch: nose-down = forward = north = lower PWM on pitch channel.
        Roll: right = east = higher PWM on roll channel.
        """
        max_vel = self._config.max_rc_velocity_m_s
        max_offset = self._config.max_rc_offset_pwm

        pitch_offset = int(-north_m_s / max_vel * max_offset)
        roll_offset = int(east_m_s / max_vel * max_offset)

        pitch_pwm = max(RC_MIN, min(RC_MAX, RC_CENTER + pitch_offset))
        roll_pwm = max(RC_MIN, min(RC_MAX, RC_CENTER + roll_offset))

        self._set_rc_channels(roll=roll_pwm, pitch=pitch_pwm)

    async def send_hover_setpoint(self) -> None:
        """Return all channels to center (hold position in ALT_HOLD)."""
        self._set_rc_channels(
            throttle=RC_CENTER, roll=RC_CENTER, pitch=RC_CENTER, yaw=RC_CENTER
        )

    async def force_disarm(self) -> None:
        """Force disarm regardless of flight state. EMERGENCY USE ONLY."""
        logger.warning("FORCE DISARM")
        self._set_rc_channels(
            throttle=RC_MIN, roll=RC_CENTER, pitch=RC_CENTER, yaw=RC_CENTER
        )
        await asyncio.sleep(0.3)
        self._mav.mav.command_long_send(
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 21196, 0, 0, 0, 0, 0,
        )
        await asyncio.sleep(1.0)
        logger.warning("Force disarm sent")

    # ── GPS methods — not available in RC/indoor mode ─────────────────

    async def return_to_launch(self) -> None:
        raise NotImplementedError(
            "Method requires GPS — not available in RC/indoor mode"
        )

    async def upload_mission(self, waypoints: List[qpsGPSPosition]) -> None:
        raise NotImplementedError(
            "Method requires GPS — not available in RC/indoor mode"
        )

    async def start_mission(self) -> None:
        raise NotImplementedError(
            "Method requires GPS — not available in RC/indoor mode"
        )

    async def pause_mission(self) -> None:
        raise NotImplementedError(
            "Method requires GPS — not available in RC/indoor mode"
        )

    async def is_mission_finished(self) -> bool:
        raise NotImplementedError(
            "Method requires GPS — not available in RC/indoor mode"
        )

    async def goto_location(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        yaw_deg: float = float("nan"),
    ) -> None:
        raise NotImplementedError(
            "Method requires GPS — not available in RC/indoor mode"
        )

    async def start_offboard(self) -> None:
        raise NotImplementedError(
            "Method requires GPS — not available in RC/indoor mode"
        )

    async def stop_offboard(self) -> None:
        raise NotImplementedError(
            "Method requires GPS — not available in RC/indoor mode"
        )
