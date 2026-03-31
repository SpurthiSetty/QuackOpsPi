"""
QuackOps SITL Flight Test
==========================
Proves the full flight flow against ArduCopter SITL:
    Connect → Arm → Takeoff → Fly Waypoints → Land

Run with Mission Planner's SIMULATION tab active (Multirotor).
MAVSDK connects on tcpout://localhost:5763.

The mission flies a small triangle ~50m from the home position
at 10m altitude, then returns and lands.
"""

import asyncio
import logging
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quackops.flight_test")

# ── Configuration ─────────────────────────────────────────────────────
CONNECTION_STRING = "tcpout://localhost:5763"
TAKEOFF_ALTITUDE_M = 10.0

# Triangle waypoints offset from home (meters → approx degree offsets)
# 1 degree lat ≈ 111,111m, 1 degree lon ≈ 111,111m * cos(lat)
# At lat=-35.36: cos(-35.36) ≈ 0.815
# 50m north ≈ 0.00045 deg lat, 50m east ≈ 0.00055 deg lon
WAYPOINT_OFFSETS = [
    (0.00045, 0.0),         # 50m north
    (0.00022, 0.00055),     # 25m north, 50m east
    (-0.00022, 0.00028),    # 25m south, 25m east
]


async def run():
    drone = System()

    # ── 1. Connect ────────────────────────────────────────────────────
    logger.info("Connecting to SITL on %s ...", CONNECTION_STRING)
    await drone.connect(system_address=CONNECTION_STRING)

    logger.info("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            logger.info("Connected!")
            break

    # ── 2. Request telemetry streams ──────────────────────────────────
    logger.info("Requesting telemetry rates...")
    await _set_telemetry_rates(drone)

    # ── 3. Wait for GPS fix and healthy state ─────────────────────────
    # ── 3. Wait for GPS lock ──────────────────────────────────────────
    logger.info("Waiting for GPS lock...")
    async for position in drone.telemetry.position():
        if position.latitude_deg != 0.0:
            home_lat = position.latitude_deg
            home_lon = position.longitude_deg
            logger.info(
                "Home: lat=%.6f  lon=%.6f  alt=%.1fm",
                home_lat, home_lon, position.relative_altitude_m,
            )
            break

    # Grab home position for reference
    async for position in drone.telemetry.position():
        home_lat = position.latitude_deg
        home_lon = position.longitude_deg
        logger.info(
            "Home: lat=%.6f  lon=%.6f  alt=%.1fm",
            home_lat, home_lon, position.relative_altitude_m,
        )
        break

    # ── 4. Build waypoint mission ─────────────────────────────────────
    mission_items = []
    # ── 4. Build waypoint mission ─────────────────────────────────────

    for i, (dlat, dlon) in enumerate(WAYPOINT_OFFSETS):
        wp = MissionItem(
            latitude_deg=home_lat + dlat,
            longitude_deg=home_lon + dlon,
            relative_altitude_m=TAKEOFF_ALTITUDE_M,
            speed_m_s=2.5,
            is_fly_through=True,
            gimbal_pitch_deg=float("nan"),
            gimbal_yaw_deg=float("nan"),
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=float("nan"),
            camera_photo_interval_s=float("nan"),
            acceptance_radius_m=2.0,
            yaw_deg=float("nan"),
            camera_photo_distance_m=float("nan"),
            vehicle_action=MissionItem.VehicleAction.NONE,
        )
        mission_items.append(wp)
        logger.info(
            "  WP%d: lat=%.6f  lon=%.6f  alt=%.0fm",
            i + 1, home_lat + dlat, home_lon + dlon, TAKEOFF_ALTITUDE_M,
        )

    mission_plan = MissionPlan(mission_items)

    # ── 5. Upload mission ─────────────────────────────────────────────
    logger.info("Uploading mission (%d waypoints)...", len(mission_items))
    await drone.mission.upload_mission(mission_plan)
    logger.info("Mission uploaded")

    # ── 6. Arm ────────────────────────────────────────────────────────
  # ── 6. Arm and takeoff via Action plugin ──────────────────────────
    logger.info("Setting takeoff altitude to %.0fm...", TAKEOFF_ALTITUDE_M)
    await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE_M)

    logger.info("Arming...")
    await drone.action.arm()
    logger.info("Armed!")

    logger.info("Taking off...")
    await drone.action.takeoff()

    # Wait until we're airborne and near target altitude
    logger.info("Waiting to reach altitude...")
    async for position in drone.telemetry.position():
        alt = position.relative_altitude_m
        if alt >= TAKEOFF_ALTITUDE_M * 0.90:
            logger.info("  Reached %.1fm — ready for mission", alt)
            break
        logger.info("  Climbing... %.1fm", alt)
        await asyncio.sleep(1)

    # ── 7. Start waypoint mission ─────────────────────────────────────
    logger.info("Starting mission...")
    await drone.mission.start_mission()
    logger.info("Mission started — watch the drone in Mission Planner!")

    # ── 8. Monitor mission progress ───────────────────────────────────
    logger.info("Monitoring mission progress...")
    position_task = asyncio.create_task(_log_position(drone))

    async for progress in drone.mission.mission_progress():
        logger.info(
            "Mission progress: %d / %d",
            progress.current, progress.total,
        )
        if progress.current == progress.total:
            logger.info("All waypoints reached!")
            break

    position_task.cancel()
    try:
        await position_task
    except asyncio.CancelledError:
        pass

    # ── 9. Return and land ────────────────────────────────────────────
    logger.info("Returning to launch...")
    await drone.action.return_to_launch()

    # Wait for drone to land
    logger.info("Waiting for drone to land...")
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            logger.info("Landed!")
            break

    # ── 10. Disarm ────────────────────────────────────────────────────
    await asyncio.sleep(2)  # Let motors spin down
    logger.info("Disarming...")
    try:
        await drone.action.disarm()
        logger.info("Disarmed!")
    except Exception as e:
        logger.info("Auto-disarmed by firmware (normal): %s", e)

    logger.info("═══ FLIGHT TEST COMPLETE ═══")
    logger.info(
        "Next step: integrate this into qpsFlightManager "
        "and qpsTelemetryMonitor production implementations."
    )


async def _set_telemetry_rates(drone: System) -> None:
    """Request telemetry stream rates from ArduPilot."""
    rate_setters = [
        ("position", drone.telemetry.set_rate_position, 2.0),
        ("battery", drone.telemetry.set_rate_battery, 1.0),
        ("in_air", drone.telemetry.set_rate_in_air, 1.0),
        ("altitude", drone.telemetry.set_rate_altitude, 1.0),
    ]
    for name, setter, rate in rate_setters:
        try:
            await setter(rate)
            logger.info("  ✓ %s @ %.0f Hz", name, rate)
        except Exception as e:
            logger.warning("  ⚠ %s rate request failed: %s", name, e)


async def _log_position(drone: System) -> None:
    """Background task: log GPS position every update."""
    try:
        async for position in drone.telemetry.position():
            logger.info(
                "  POS: lat=%.6f  lon=%.6f  alt=%.1fm",
                position.latitude_deg,
                position.longitude_deg,
                position.relative_altitude_m,
            )
    except asyncio.CancelledError:
        return


if __name__ == "__main__":
    asyncio.run(run())