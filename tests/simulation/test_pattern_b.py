"""
test_pattern_b.py

Smoke test — Pattern B reader loop + callback dispatch.

Connects to ArduCopter SITL on tcp:localhost:5763, prints GPS position and
drone state once per second for 10 seconds, then disconnects cleanly.

Verifies:
  - qpsFlightManager connects and starts the reader loop
  - qpsTelemetryMonitor receives messages via the callback
  - GPS position and drone state are populated by real telemetry

Usage (with SITL running on port 5763):
    python tests/simulation/test_pattern_b.py
"""

from __future__ import annotations

import asyncio
import logging

import sys
from pathlib import Path

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager import qpsFlightManager
from quackops_pi.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test.pattern_b")


async def main() -> None:
    config = qpsConfig(connection_string="tcp:localhost:5763")

    fm = qpsFlightManager(config)
    tm = qpsTelemetryMonitor(fm, config)

    tm.on_battery_warning(lambda pct: logger.warning("Battery warning: %.0f%%", pct))
    tm.on_battery_critical(lambda pct: logger.error("Battery critical: %.0f%%", pct))

    await fm.connect()
    await tm.start()

    logger.info("Monitoring telemetry for 10 seconds...")
    for i in range(10):
        await asyncio.sleep(1)

        gps = tm.get_gps_position()
        state = tm.get_drone_state()

        if gps:
            logger.info(
                "[%2ds] GPS  lat=%.6f  lon=%.6f  alt=%.1fm  hdg=%.0f deg  spd=%.1fm/s",
                i + 1,
                gps.latitude_deg,
                gps.longitude_deg,
                gps.altitude_m,
                gps.heading_deg,
                gps.speed_m_s,
            )
        else:
            logger.info("[%2ds] GPS: no data yet", i + 1)

        if state:
            logger.info(
                "       armed=%-5s  mode=%-12s  in_air=%-5s  "
                "batt=%.0f%%  fix=%d  sats=%d",
                state.is_armed,
                state.flight_mode,
                state.in_air,
                state.battery_percent,
                state.gps_fix_type,
                state.gps_num_satellites,
            )

    await tm.stop()
    await fm.disconnect()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
