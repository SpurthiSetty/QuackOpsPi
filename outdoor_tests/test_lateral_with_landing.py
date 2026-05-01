"""
outdoor_tests/test_lateral_with_landing.py

Lateral movement flight test with ArUco marker-triggered landing.

Identical to test_lateral.py through the pattern loop. After returning to home
position, replaces the final land with a qpsHoverSearchController marker
search before landing.

Usage:
    python3 outdoor_tests/test_lateral_with_landing.py \
        --pattern square --size 3 --alt 3 --search-timeout 15
    python3 outdoor_tests/test_lateral_with_landing.py \
        --pattern square --size 3 --alt 3 --no-marker-search
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from pathlib import Path
from typing import List, Tuple

# ── Repo-root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

# Monkey-patch cv2 preview so qpsHoverSearchController runs headless over SSH
cv2.imshow = lambda *a, **kw: None
cv2.waitKey = lambda *a, **kw: -1
cv2.destroyAllWindows = lambda *a, **kw: None

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager import qpsFlightManager
from quackops_pi.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor
from quackops_pi.mission.qps_hover_search_controller import qpsHoverSearchController
from quackops_pi.models.qps_landing_result import qpsLandingOutcome

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
    open_recording_sink,
    build_camera_stack,
)

# ── Safety caps ───────────────────────────────────────────────────────────────

ALT_MIN_M: float = 0.5
ALT_MAX_M: float = 10.0
HOVER_MIN_S: float = 1.0
HOVER_MAX_S: float = 30.0
LEG_MAX_M: float = 5.0
SEARCH_MIN_S: float = 5.0
SEARCH_MAX_S: float = 60.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("qps.test_lateral_with_landing")


# ── Pattern generator (unchanged from test_lateral.py) ───────────────────────

def build_pattern(
    pattern: str, size: float, bearing_deg: float = 0.0
) -> List[Tuple[float, float]]:
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
            angle = math.radians(90 + i * 120)
            pts.append((math.cos(angle) * size, math.sin(angle) * size))
        pts.append((0.0, 0.0))
        return pts
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="QuackOps lateral movement + marker-search landing test"
    )
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
    parser.add_argument(
        "--search-timeout", type=float, default=15.0,
        help=f"Max seconds to search for marker after returning home "
             f"[{SEARCH_MIN_S}, {SEARCH_MAX_S}] (default 15.0)",
    )
    parser.add_argument(
        "--marker-id", type=int, default=None,
        help="ArUco marker ID to search for (default: from config)",
    )
    parser.add_argument(
        "--no-marker-search", action="store_true", default=False,
        help="Skip marker search and hover for --search-timeout seconds "
             "at home before landing",
    )
    parser.add_argument("--config", type=str,
                        default=str(here / "config" / "outdoor_production.json"))
    parser.add_argument("--log-dir", type=str, default=str(here / "logs"))
    args = parser.parse_args()

    args.alt = max(ALT_MIN_M, min(ALT_MAX_M, args.alt))
    args.hover_before = max(HOVER_MIN_S, min(HOVER_MAX_S, args.hover_before))
    args.hover_at_each = max(HOVER_MIN_S, min(HOVER_MAX_S, args.hover_at_each))
    args.size = max(0.5, min(LEG_MAX_M, args.size))
    args.search_timeout = max(SEARCH_MIN_S, min(SEARCH_MAX_S, args.search_timeout))
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> int:
    args = parse_args()

    log_dir = make_log_dir(args.log_dir, f"lateral_with_landing_{args.pattern}")
    log.info("Log directory: %s", log_dir)
    log.info(
        "Parameters: alt=%.1fm  pattern=%s  size=%.1fm  bearing=%.0f°"
        "  hover_before=%.1fs  hover_at_each=%.1fs"
        "  search_timeout=%.1fs  no_search=%s",
        args.alt, args.pattern, args.size, args.bearing,
        args.hover_before, args.hover_at_each,
        args.search_timeout, args.no_marker_search,
    )

    config = qpsConfig.from_file(args.config)
    config.search_timeout_s = args.search_timeout
    if args.marker_id is not None:
        config.target_marker_id = args.marker_id
    log.info("Connection: %s", config.connection_string)

    pattern = build_pattern(args.pattern, args.size, args.bearing)
    log.info("Pattern waypoints (NED from home): %s", pattern)

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

    # Camera stack state — tracked for safe cleanup
    camera_started = False
    stream_started = False
    writer = None
    det_file = None
    ts_file = None
    camera = None
    stream = None
    search_controller = None

    try:
        # ── Camera + recording stack ─────────────────────────────────────────
        width, height = config.camera_resolution
        sink, writer, det_file, ts_file = open_recording_sink(
            log_dir, width, height, config.stream_fps
        )
        camera, detector, stream = await build_camera_stack(config)
        stream.register_frame_sink(sink)
        await camera.start()
        camera_started = True
        stream.start(port=config.stream_port)
        stream_started = True
        log.info("MJPEG stream: http://0.0.0.0:%d/", config.stream_port)
        log.info("Recording: %s", log_dir / "video.mp4")

        # ── Connect + telemetry ──────────────────────────────────────────────
        log.info("Connecting to flight controller...")
        await fm.connect()
        await tm.start()

        search_controller = qpsHoverSearchController(camera, detector, tm, config)

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

        # ── Pre-pattern hover ────────────────────────────────────────────────
        log.info("Pre-pattern hover for %.1fs...", args.hover_before)
        result = await hover_with_logging(
            tm, diag, fs, args.hover_before, csv_writer, log,
            poll_hz=config.telemetry_polling_rate_hz,
        )
        if result != "completed":
            fs_log.log(result, "during pre-pattern hover")
            log.warning("Pre-pattern hover cut short: %s — returning to home", result)
            await _fly_home(fm, tm, fs, home_lat, home_lon, args.alt, config, log)
            await _land_and_disarm(fm, fs, log)
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

        # ── Return home ──────────────────────────────────────────────────────
        if aborted:
            log.info("Pattern aborted — returning home, skipping marker search")
            await _fly_home(fm, tm, fs, home_lat, home_lon, args.alt, config, log)
            await _land_and_disarm(fm, fs, log)
        else:
            log.info("Pattern complete — returning home for marker search")
            await _fly_home(fm, tm, fs, home_lat, home_lon, args.alt, config, log)

            # ── Marker search (or timed hover fallback) ──────────────────────
            if args.no_marker_search:
                log.info(
                    "Hovering for %.1fs (no marker search)...", args.search_timeout
                )
                hover_result = await hover_with_logging(
                    tm, diag, fs, args.search_timeout, csv_writer, log,
                    poll_hz=config.telemetry_polling_rate_hz,
                )
                if hover_result != "completed":
                    fs_log.log(hover_result, "early exit during home hover")
                    log.warning("Home hover ended early: %s", hover_result)
            else:
                log.info(
                    "Searching for marker ID=%d (timeout=%.1fs)...",
                    config.target_marker_id, config.search_timeout_s,
                )
                # NOTE: qpsHoverSearchController calls camera.start() and
                # camera.stop() internally. Since the camera is already running
                # for the MJPEG stream, the controller's start() may attempt
                # to re-initialise an already-open camera. The controller's
                # stop() also closes the camera, so the stream goes dark during
                # descent. Accepted for testing purposes.
                landing_result = await search_controller.execute_marker_search(
                    config.target_marker_id
                )
                log.info(
                    "Search outcome: %s  duration=%.2fs  frames=%d",
                    landing_result.outcome.name,
                    landing_result.search_duration_s,
                    landing_result.frames_searched,
                )
                with open(log_dir / "search_result.json", "w") as f:
                    json.dump({
                        "outcome": landing_result.outcome.name,
                        "duration_s": landing_result.search_duration_s,
                        "frames_searched": landing_result.frames_searched,
                        "target_marker_id": landing_result.target_marker_id,
                        "marker_gps": (
                            {
                                "lat": landing_result.marker_gps.latitude_deg,
                                "lon": landing_result.marker_gps.longitude_deg,
                                "alt": landing_result.marker_gps.altitude_m,
                            } if landing_result.marker_gps else None
                        ),
                        "fallback_gps": (
                            {
                                "lat": landing_result.fallback_gps.latitude_deg,
                                "lon": landing_result.fallback_gps.longitude_deg,
                                "alt": landing_result.fallback_gps.altitude_m,
                            } if landing_result.fallback_gps else None
                        ),
                    }, f, indent=2)
                if landing_result.outcome != qpsLandingOutcome.MARKER_FOUND:
                    fs_log.log(
                        "search_no_marker_found",
                        f"outcome={landing_result.outcome.name}",
                    )

            await _land_and_disarm(fm, fs, log)

        in_flight = False

    except (KeyboardInterrupt, asyncio.CancelledError):
        log.warning("Interrupted — attempting emergency return and land")
        if in_flight and home_lat != 0.0:
            await _fly_home(fm, tm, fs, home_lat, home_lon, args.alt, config, log)
            await _land_and_disarm(fm, fs, log)
        elif in_flight:
            await _emergency_land(fm, fs, tm, log)
        return 1

    except Exception:
        log.exception("Unexpected error — emergency land")
        if in_flight:
            await _emergency_land(fm, fs, tm, log)
        return 1

    finally:
        if watcher_task is not None:
            watcher_task.cancel()
            try:
                await watcher_task
            except asyncio.CancelledError:
                pass
        try:
            if stream_started and stream is not None:
                stream.stop()
        except Exception:
            log.exception("Error stopping stream")
        try:
            if camera_started and camera is not None:
                await camera.stop()
        except Exception:
            log.exception("Error stopping camera")
        try:
            if writer is not None:
                writer.release()
        except Exception:
            log.exception("Error releasing video writer")
        try:
            if det_file is not None:
                det_file.close()
            if ts_file is not None:
                ts_file.close()
        except Exception:
            log.exception("Error closing recorder files")
        await tm.stop()
        await fm.disconnect()
        fs_log.close()
        csv_file.close()
        log.info("Done. Logs in: %s", log_dir)
        print(f"\nLogs saved to: {log_dir}")

    return 0


_ABORT_MODES = ("LAND", "RTL", "SMART_RTL", "STABILIZE", "ALT_HOLD", "LOITER")


async def _fly_home(
    fm: qpsFlightManager,
    tm: qpsTelemetryMonitor,
    fs: FailsafeWatcher,
    home_lat: float,
    home_lon: float,
    alt_m: float,
    config: qpsConfig,
    log: logging.Logger,
) -> None:
    """Navigate to home position and wait for arrival."""
    state = tm.get_drone_state()
    if state is not None and state.flight_mode in _ABORT_MODES:
        log.warning(
            "Current mode is %s — skipping RTH to avoid fighting pilot/FC takeover",
            state.flight_mode,
        )
        return

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
        log.warning("Return-to-home arrival check failed: %s — continuing anyway", exc)


async def _land_and_disarm(
    fm: qpsFlightManager,
    fs: FailsafeWatcher,
    log: logging.Logger,
) -> None:
    """Command LAND mode and disarm."""
    log.info("Landing...")
    fs.set_expected_mode("LAND")
    await fm.land()
    await fm.disarm()
    log.info("Landed and disarmed")


async def _emergency_land(
    fm: qpsFlightManager,
    fs: FailsafeWatcher,
    tm: qpsTelemetryMonitor,
    log: logging.Logger,
) -> None:
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
    try:
        fs.set_expected_mode("LAND")
        await fm.land()
    except Exception:
        log.exception("Emergency land failed — manual intervention required")
        return
    try:
        await fm.disarm()
        log.info("Emergency land/disarm complete")
    except RuntimeError:
        log.info("Already disarmed after landing")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
