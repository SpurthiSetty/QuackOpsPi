"""
outdoor_tests/test_lateral_oneway.py

One-way lateral movement: takeoff, fly N metres at a bearing, hover, land at destination.

This is a focused variant of test_lateral.py with NO return-to-home: the drone
lands wherever the pattern ends.

Usage:
    python3 outdoor_tests/test_lateral_oneway.py \
        --distance 3 --bearing 0 --alt 2 \
        [--hover-before 3] [--hover-at-each 5] \
        [--config outdoor_tests/config/outdoor_production.json] \
        [--log-dir outdoor_tests/logs/]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from pathlib import Path

# ── Repo-root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager import qpsFlightManager
from quackops_pi.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor

from test_common import (
    TestDiagnostics,
    FailsafeWatcher,
    FailsafeLog,
    hover_with_logging,
    wait_for_arrival,
    wait_gps_ready,
    ned_to_latlon,
    make_log_dir,
    open_telemetry_csv,
    open_failsafe_log,
)

# ── Safety caps ───────────────────────────────────────────────────────────────

ALT_MIN_M: float = 0.5
ALT_MAX_M: float = 10.0
HOVER_MIN_S: float = 1.0
HOVER_MAX_S: float = 30.0
DISTANCE_MAX_M: float = 5.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("qps.test_lateral_oneway")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="QuackOps one-way lateral flight test")
    parser.add_argument("--alt", type=float, default=2.0,
                        help=f"Flight altitude in metres [{ALT_MIN_M}, {ALT_MAX_M}]")
    parser.add_argument("--hover-before", type=float, default=3.0,
                        help="Seconds to hover at takeoff before translating")
    parser.add_argument("--hover-at-each", type=float, default=5.0,
                        help="Seconds to hover at destination before landing")
    parser.add_argument("--distance", type=float, default=3.0,
                        help=f"Distance to translate in metres (capped at {DISTANCE_MAX_M}m)")
    parser.add_argument("--bearing", type=float, default=0.0,
                        help="Bearing in degrees from north (0=N, 90=E, 180=S, 270=W)")
    parser.add_argument("--config", type=str,
                        default=str(here / "config" / "outdoor_production.json"))
    parser.add_argument("--log-dir", type=str, default=str(here / "logs"))
    args = parser.parse_args()

    # Safety caps
    args.alt = max(ALT_MIN_M, min(ALT_MAX_M, args.alt))
    args.hover_before = max(HOVER_MIN_S, min(HOVER_MAX_S, args.hover_before))
    args.hover_at_each = max(HOVER_MIN_S, min(HOVER_MAX_S, args.hover_at_each))
    args.distance = max(0.5, min(DISTANCE_MAX_M, args.distance))
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> int:
    args = parse_args()

    log_dir = make_log_dir(args.log_dir, "lateral_oneway")
    log.info("Log directory: %s", log_dir)
    log.info(
        "Parameters: alt=%.1fm  distance=%.1fm  bearing=%.0f°"
        "  hover_before=%.1fs  hover_at_each=%.1fs",
        args.alt, args.distance, args.bearing,
        args.hover_before, args.hover_at_each,
    )

    config = qpsConfig.from_file(args.config)
    log.info("Connection: %s", config.connection_string)

    # Compute single destination waypoint (NED offset from home)
    b_rad = math.radians(args.bearing)
    dest_n = math.cos(b_rad) * args.distance
    dest_e = math.sin(b_rad) * args.distance
    log.info("Destination (NED from home): N=%.2fm E=%.2fm", dest_n, dest_e)

    fm = qpsFlightManager(config)
    tm = qpsTelemetryMonitor(fm, config)
    diag = TestDiagnostics(fm)
    fs = FailsafeWatcher(fm, tm)

    csv_writer, csv_file = open_telemetry_csv(log_dir)
    fs_log: FailsafeLog = open_failsafe_log(log_dir)

    watcher_task: asyncio.Task | None = None
    in_flight = False
    home_lat: float = 0.0
    home_lon: float = 0.0

    try:
        log.info("Connecting to flight controller...")
        await fm.connect()
        await tm.start()

        watcher_task = asyncio.create_task(
            fs.watch_mode_transitions(), name="failsafe-watcher"
        )

        log.info("Waiting for GPS lock...")
        await wait_gps_ready(fm, tm, min_sats=10, max_hdop=1.4, log=log)

        home = tm.get_gps_position()
        if home is None:
            raise RuntimeError("No GPS position available after GPS ready check")
        home_lat = home.latitude_deg
        home_lon = home.longitude_deg
        diag.set_home(home_lat, home_lon)
        log.info("Home: lat=%.7f  lon=%.7f", home_lat, home_lon)

        await fm.set_mode(fm.MODE_GUIDED)
        fs.set_expected_mode("GUIDED")

        log.info(">>> STAND CLEAR — arming in 3 seconds <<<")
        await asyncio.sleep(3)

        log.info("Arming...")
        await fm.arm()
        in_flight = True

        log.info("Taking off to %.1fm...", args.alt)
        await fm.takeoff(args.alt)
        log.info("Takeoff complete")

        # ── Pre-translation hover ───────────────────────────────────────────
        log.info("Pre-translation hover for %.1fs...", args.hover_before)
        result = await hover_with_logging(
            tm, diag, fs, args.hover_before, csv_writer, log,
            poll_hz=config.telemetry_polling_rate_hz,
        )
        if result != "completed":
            fs_log.log(result, "during pre-translation hover")
            log.warning("Pre-translation hover cut short: %s — landing in place", result)
            await _land_in_place(fm, tm, fs, log)
            in_flight = False
            return 1

        # ── Translate to destination ────────────────────────────────────────
        tgt_lat, tgt_lon = ned_to_latlon(home_lat, home_lon, dest_n, dest_e)
        log.info(
            "Translating to destination: N=%.2fm E=%.2fm → lat=%.7f lon=%.7f",
            dest_n, dest_e, tgt_lat, tgt_lon,
        )

        await fm.goto_location(tgt_lat, tgt_lon, args.alt)
        try:
            await wait_for_arrival(
                tm, tgt_lat, tgt_lon,
                tolerance_m=config.goto_arrival_tolerance_m,
                timeout_s=30.0,
                fs=fs,
                log=log,
            )
        except TimeoutError:
            log.warning("Arrival timeout — proceeding to land at current position")
        except RuntimeError as exc:
            log.warning("Failsafe during transit: %s — landing in place", exc)
            fs_log.log("failsafe_in_transit", str(exc))
            await _land_in_place(fm, tm, fs, log)
            in_flight = False
            return 1

        # ── Hover at destination ────────────────────────────────────────────
        log.info("Hovering at destination for %.1fs before landing...", args.hover_at_each)
        result = await hover_with_logging(
            tm, diag, fs, args.hover_at_each, csv_writer, log,
            poll_hz=config.telemetry_polling_rate_hz,
        )
        if result != "completed":
            fs_log.log(result, "during destination hover")
            log.warning("Destination hover cut short: %s — landing in place", result)

        # ── Land at destination (no return) ────────────────────────────────
        log.info("Landing at destination (no return-to-home)")
        await _land_in_place(fm, tm, fs, log)
        in_flight = False

    except (KeyboardInterrupt, asyncio.CancelledError):
        log.warning("Interrupted — landing in place")
        if in_flight:
            await _land_in_place(fm, tm, fs, log)
        return 1

    except Exception:
        log.exception("Unexpected error — landing in place")
        if in_flight:
            await _land_in_place(fm, tm, fs, log)
        return 1

    finally:
        if watcher_task is not None:
            watcher_task.cancel()
            try:
                await watcher_task
            except asyncio.CancelledError:
                pass
        await tm.stop()
        await fm.disconnect()
        fs_log.close()
        csv_file.close()
        log.info("Done. Logs in: %s", log_dir)
        print(f"\nLogs saved to: {log_dir}")

    return 0


_ABORT_MODES = ("LAND", "RTL", "SMART_RTL", "STABILIZE", "ALT_HOLD", "LOITER")


async def _land_in_place(
    fm: qpsFlightManager,
    tm: qpsTelemetryMonitor,
    fs: FailsafeWatcher,
    log: logging.Logger,
) -> None:
    """Land at current position. Handles already-landing mode gracefully."""
    state = tm.get_drone_state()
    if state is not None and state.flight_mode in ("LAND", "RTL", "SMART_RTL"):
        log.info(
            "FC already in %s — skipping fm.land(), attempting disarm only",
            state.flight_mode,
        )
        try:
            await fm.disarm()
            log.info("Disarmed")
        except RuntimeError:
            log.info("Already disarmed")
        return

    log.info("Switching to LAND at current position...")
    try:
        fs.set_expected_mode("LAND")
        await fm.land()
    except Exception:
        log.exception("Land command failed — manual intervention required")
        return

    try:
        await fm.disarm()
        log.info("Landed and disarmed")
    except RuntimeError:
        log.info("Already disarmed after landing")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))