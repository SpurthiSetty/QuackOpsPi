"""
outdoor_tests/test_hover_with_landing.py

Hover-in-place flight test with ArUco marker-triggered landing.

Identical to test_hover.py up through takeoff, then replaces the timed hover
with a qpsHoverSearchController marker search. On MARKER_FOUND or
SEARCH_TIMEOUT the drone proceeds to land at its current position.

Usage:
    python3 outdoor_tests/test_hover_with_landing.py --alt 1 --search-timeout 15
    python3 outdoor_tests/test_hover_with_landing.py --alt 1 --no-marker-search
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

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
    wait_gps_ready,
    make_log_dir,
    open_telemetry_csv,
    open_failsafe_log,
    open_recording_sink,
    build_camera_stack,
)

# ── Safety caps ───────────────────────────────────────────────────────────────

ALT_MIN_M: float = 0.5
ALT_MAX_M: float = 10.0
SEARCH_MIN_S: float = 5.0
SEARCH_MAX_S: float = 60.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("qps.test_hover_with_landing")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="QuackOps hover + marker-search landing test"
    )
    parser.add_argument(
        "--alt", type=float, default=3.0,
        help=f"Takeoff altitude in metres [{ALT_MIN_M}, {ALT_MAX_M}] (default 3.0)",
    )
    parser.add_argument(
        "--search-timeout", type=float, default=15.0,
        help=f"Max seconds to search for marker before landing anyway "
             f"[{SEARCH_MIN_S}, {SEARCH_MAX_S}] (default 15.0)",
    )
    parser.add_argument(
        "--marker-id", type=int, default=None,
        help="ArUco marker ID to search for (default: from config)",
    )
    parser.add_argument(
        "--no-marker-search", action="store_true", default=False,
        help="Skip marker search and hover for --search-timeout seconds "
             "before landing (fallback path matching old test_hover behavior)",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(here / "config" / "outdoor_production.json"),
        help="Path to qpsConfig JSON file",
    )
    parser.add_argument(
        "--log-dir", type=str, default=str(here / "logs"),
        help="Base directory for log output",
    )
    args = parser.parse_args()
    args.alt = max(ALT_MIN_M, min(ALT_MAX_M, args.alt))
    args.search_timeout = max(SEARCH_MIN_S, min(SEARCH_MAX_S, args.search_timeout))
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> int:
    args = parse_args()

    log_dir = make_log_dir(args.log_dir, "hover_with_landing")
    log.info("Log directory: %s", log_dir)
    log.info(
        "Parameters: alt=%.1fm  search_timeout=%.1fs  marker_id=%s  no_search=%s",
        args.alt, args.search_timeout,
        args.marker_id if args.marker_id is not None else "from config",
        args.no_marker_search,
    )

    config = qpsConfig.from_file(args.config)
    config.search_timeout_s = args.search_timeout
    if args.marker_id is not None:
        config.target_marker_id = args.marker_id
    log.info("Connection: %s", config.connection_string)
    log.info(
        "Search: marker_id=%d  timeout=%.1fs",
        config.target_marker_id, config.search_timeout_s,
    )

    fm = qpsFlightManager(config)
    tm = qpsTelemetryMonitor(fm, config)
    diag = TestDiagnostics(fm)
    fs = FailsafeWatcher(fm, tm)

    csv_writer, csv_file = open_telemetry_csv(log_dir)
    fs_log: FailsafeLog = open_failsafe_log(log_dir)

    watcher_task: asyncio.Task | None = None
    in_flight = False

    # Camera stack state — tracked for safe cleanup
    camera_started = False
    stream_started = False
    writer = None
    det_file = None
    ts_file = None
    camera = None
    stream = None

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

        search_controller = qpsHoverSearchController(camera, detector, tm, config)

        # ── Connect + telemetry ──────────────────────────────────────────────
        log.info("Connecting to flight controller...")
        await fm.connect()
        await tm.start()

        watcher_task = asyncio.create_task(
            fs.watch_mode_transitions(), name="failsafe-watcher"
        )

        # ── GPS preflight ────────────────────────────────────────────────────
        log.info("Waiting for GPS lock...")
        await wait_gps_ready(fm, tm, min_sats=10, max_hdop=1.4, log=log)

        home = tm.get_gps_position()
        if home is None:
            raise RuntimeError("No GPS position available after GPS ready check")
        diag.set_home(home.latitude_deg, home.longitude_deg)
        log.info(
            "Home: lat=%.7f  lon=%.7f  alt=%.1fm",
            home.latitude_deg, home.longitude_deg, home.altitude_m,
        )

        # ── Pre-flight battery check ─────────────────────────────────────────
        state = tm.get_drone_state()
        batt_v = state.battery_voltage if state else 0.0
        batt_pct = state.battery_percent if state else 0.0
        log.info("Pre-flight battery: %.0f%%  %.2fV", batt_pct, batt_v)
        if 0 < batt_v < 11.6:
            raise RuntimeError(
                f"Battery too low to fly: {batt_v:.2f}V (minimum 11.6V)"
            )

        # ── Mode → GUIDED ────────────────────────────────────────────────────
        await fm.set_mode(fm.MODE_GUIDED)
        fs.set_expected_mode("GUIDED")

        log.info(">>> STAND CLEAR — arming in 3 seconds <<<")
        await asyncio.sleep(3)
        if fs.prearm_messages:
            log.warning("Pre-arm warnings from FC (arm may be rejected):")
            for m in fs.prearm_messages:
                log.warning("  FC: %s", m)

        # ── Arm + takeoff ────────────────────────────────────────────────────
        log.info("Arming...")
        await fm.arm()
        in_flight = True

        log.info("Taking off to %.1fm...", args.alt)
        await fm.takeoff(args.alt)
        log.info("Takeoff complete")

        # ── Marker search (or timed hover fallback) ──────────────────────────
        if args.no_marker_search:
            log.info("Hovering for %.1fs (no marker search)...", args.search_timeout)
            hover_result = await hover_with_logging(
                tm, diag, fs, args.search_timeout, csv_writer, log,
                poll_hz=config.telemetry_polling_rate_hz,
            )
            if hover_result != "completed":
                fs_log.log(hover_result, "early exit during hover")
                log.warning("Hover ended early: %s", hover_result)
        else:
            log.info(
                "Searching for marker ID=%d (timeout=%.1fs)...",
                config.target_marker_id, config.search_timeout_s,
            )
            # NOTE: qpsHoverSearchController calls camera.start() and
            # camera.stop() internally at the start and end of
            # execute_marker_search. Since the camera is already running for
            # the MJPEG stream, the controller's start() may attempt to
            # re-initialise an already-open camera (a known limitation).
            # The controller's stop() at the end of the search also closes
            # the camera, so the stream goes dark during descent. This is
            # accepted for testing purposes — recording trails off naturally.
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

        # ── Land ─────────────────────────────────────────────────────────────
        log.info("Landing...")
        fs.set_expected_mode("LAND")
        await fm.land()
        log.info("Landed")

        state = tm.get_drone_state()
        if state is None or state.is_armed:
            await fm.disarm()
        in_flight = False
        log.info("Disarmed")

    except (KeyboardInterrupt, asyncio.CancelledError):
        log.warning("Interrupted — attempting emergency land/disarm")
        if in_flight:
            await _emergency_land(fm, fs, tm, log)
        return 1

    except Exception:
        log.exception("Unexpected error during flight — attempting emergency land/disarm")
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
