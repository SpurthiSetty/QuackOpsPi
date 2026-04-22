"""
outdoor_tests/test_lateral.py

Lateral movement flight test: hover, fly a geometric pattern, return to home, land.

Usage:
    python3 outdoor_tests/test_lateral.py \
        --pattern square --size 3 --alt 3 \
        [--hover-before 3] [--hover-at-each 5] \
        [--config outdoor_tests/config/outdoor_production.json] \
        [--log-dir outdoor_tests/logs/]

    # Line pattern with custom bearing (degrees from north):
    python3 outdoor_tests/test_lateral.py --pattern line --size 5 --bearing 90
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from pathlib import Path
from typing import List, Tuple

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
LEG_MAX_M: float = 5.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("qps.test_lateral")


# ── Pattern generator ─────────────────────────────────────────────────────────

def build_pattern(
    pattern: str, size: float, bearing_deg: float = 0.0
) -> List[Tuple[float, float]]:
    """Return list of (north_m, east_m) NED offsets from home.

    square  — CW square, returns to origin as final point
    line    — fly to one point along bearing, return to origin
    triangle — equilateral, one vertex north, two at ±120°
    """
    if pattern == "square":
        s = size
        return [(s, 0.0), (s, s), (0.0, s), (0.0, 0.0)]

    elif pattern == "line":
        b = math.radians(bearing_deg)
        n = math.cos(b) * size
        e = math.sin(b) * size
        return [(n, e), (0.0, 0.0)]

    elif pattern == "triangle":
        pts = []
        for i in range(3):
            angle = math.radians(90 + i * 120)  # first vertex points north
            pts.append((math.cos(angle) * size, math.sin(angle) * size))
        pts.append((0.0, 0.0))
        return pts

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="QuackOps lateral movement flight test")
    parser.add_argument("--alt", type=float, default=3.0,
                        help=f"Flight altitude in metres [{ALT_MIN_M}, {ALT_MAX_M}]")
    parser.add_argument("--hover-before", type=float, default=3.0,
                        help="Seconds to hover at takeoff before starting pattern")
    parser.add_argument("--hover-at-each", type=float, default=5.0,
                        help="Seconds to hover at each pattern waypoint")
    parser.add_argument("--pattern", choices=["square", "line", "triangle"],
                        default="square", help="Movement pattern")
    parser.add_argument("--size", type=float, default=3.0,
                        help=f"Pattern leg length in metres (capped at {LEG_MAX_M}m)")
    parser.add_argument("--bearing", type=float, default=0.0,
                        help="Bearing in degrees from north (line pattern only)")
    parser.add_argument("--config", type=str,
                        default=str(here / "config" / "outdoor_production.json"))
    parser.add_argument("--log-dir", type=str, default=str(here / "logs"))
    args = parser.parse_args()

    # Safety caps
    args.alt = max(ALT_MIN_M, min(ALT_MAX_M, args.alt))
    args.hover_before = max(HOVER_MIN_S, min(HOVER_MAX_S, args.hover_before))
    args.hover_at_each = max(HOVER_MIN_S, min(HOVER_MAX_S, args.hover_at_each))
    args.size = max(0.5, min(LEG_MAX_M, args.size))
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> int:
    args = parse_args()

    log_dir = make_log_dir(args.log_dir, f"lateral_{args.pattern}")
    log.info("Log directory: %s", log_dir)
    log.info(
        "Parameters: alt=%.1fm  pattern=%s  size=%.1fm  bearing=%.0f°"
        "  hover_before=%.1fs  hover_at_each=%.1fs",
        args.alt, args.pattern, args.size, args.bearing,
        args.hover_before, args.hover_at_each,
    )

    config = qpsConfig.from_file(args.config)
    log.info("Connection: %s", config.connection_string)

    pattern = build_pattern(args.pattern, args.size, args.bearing)
    log.info("Pattern waypoints (NED from home): %s", pattern)

    fm = qpsFlightManager(config)
    tm = qpsTelemetryMonitor(fm, config)
    diag = TestDiagnostics(fm)
    fs = FailsafeWatcher(fm, tm)

    csv_writer = open_telemetry_csv(log_dir)
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
        await wait_gps_ready(fm, tm, min_sats=12, max_hdop=1.5, log=log)

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

        # ── Pre-pattern hover ────────────────────────────────────────────────
        log.info("Pre-pattern hover for %.1fs...", args.hover_before)
        result = await hover_with_logging(
            tm, diag, fs, args.hover_before, csv_writer, log,
            poll_hz=config.telemetry_polling_rate_hz,
        )
        if result != "completed":
            fs_log.log(result, "during pre-pattern hover")
            log.warning("Pre-pattern hover cut short: %s — returning to home", result)
            await _return_home_and_land(fm, tm, fs, home_lat, home_lon, args.alt, config, log)
            in_flight = False
            return 1

        # ── Pattern loop ─────────────────────────────────────────────────────
        aborted = False
        for i, (north_m, east_m) in enumerate(pattern):
            tgt_lat, tgt_lon = ned_to_latlon(home_lat, home_lon, north_m, east_m)
            log.info(
                "Waypoint %d/%d: N=%.1fm E=%.1fm → lat=%.7f lon=%.7f",
                i + 1, len(pattern), north_m, east_m, tgt_lat, tgt_lon,
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
                log.warning("Arrival timeout at WP%d — continuing to next", i + 1)
            except RuntimeError as exc:
                log.warning("Failsafe during transit to WP%d: %s", i + 1, exc)
                fs_log.log("failsafe_in_transit", str(exc))
                aborted = True
                break

            result = await hover_with_logging(
                tm, diag, fs, args.hover_at_each, csv_writer, log,
                poll_hz=config.telemetry_polling_rate_hz,
            )
            if result != "completed":
                fs_log.log(result, f"during hover at WP{i+1}")
                log.warning("Hover at WP%d cut short: %s", i + 1, result)
                aborted = True
                break

        # ── Always return to home before landing ─────────────────────────────
        if aborted:
            log.info("Pattern aborted — returning to home for safety landing")
        else:
            log.info("Pattern complete — returning to home")

        await _return_home_and_land(fm, tm, fs, home_lat, home_lon, args.alt, config, log)
        in_flight = False

    except (KeyboardInterrupt, asyncio.CancelledError):
        log.warning("Interrupted — attempting emergency return and land")
        if in_flight and home_lat != 0.0:
            await _return_home_and_land(fm, tm, fs, home_lat, home_lon, args.alt, config, log)
        elif in_flight:
            await _emergency_land(fm, fs, log)
        return 1

    except Exception:
        log.exception("Unexpected error — emergency land")
        if in_flight:
            await _emergency_land(fm, fs, log)
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
        log.info("Done. Logs in: %s", log_dir)
        print(f"\nLogs saved to: {log_dir}")

    return 0


async def _return_home_and_land(
    fm: qpsFlightManager,
    tm: qpsTelemetryMonitor,
    fs: FailsafeWatcher,
    home_lat: float,
    home_lon: float,
    alt_m: float,
    config: qpsConfig,
    log: logging.Logger,
) -> None:
    log.info("Flying to home (%.7f, %.7f) at %.1fm...", home_lat, home_lon, alt_m)
    try:
        await fm.goto_location(home_lat, home_lon, alt_m)
        await wait_for_arrival(
            tm, home_lat, home_lon,
            tolerance_m=config.goto_arrival_tolerance_m,
            timeout_s=60.0,
            fs=fs,
            log=log,
        )
    except (TimeoutError, RuntimeError) as exc:
        log.warning("Return-to-home arrival check failed: %s — landing anyway", exc)
    log.info("Landing...")
    fs.set_expected_mode("LAND")
    await fm.land()
    await fm.disarm()
    log.info("Landed and disarmed")


async def _emergency_land(
    fm: qpsFlightManager, fs: FailsafeWatcher, log: logging.Logger
) -> None:
    try:
        fs.set_expected_mode("LAND")
        await fm.land()
        await fm.disarm()
        log.info("Emergency land/disarm complete")
    except Exception:
        log.exception("Emergency land also failed — manual intervention required")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
