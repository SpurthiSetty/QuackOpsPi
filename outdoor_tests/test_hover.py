"""
outdoor_tests/test_hover.py

Hover-in-place flight test with telemetry logging and failsafe monitoring.

Usage:
    python3 outdoor_tests/test_hover.py [--alt 3] [--hover 10] \
        [--config outdoor_tests/config/outdoor_production.json] \
        [--log-dir outdoor_tests/logs/]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── Repo-root on path so 'quackops_pi' is importable ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.flight.qps_flight_manager import qpsFlightManager
from quackops_pi.telemetry.qps_telemetry_monitor import qpsTelemetryMonitor

from test_common import (
    TestDiagnostics,
    FailsafeWatcher,
    FailsafeLog,
    hover_with_logging,
    wait_gps_ready,
    make_log_dir,
    open_telemetry_csv,
    open_failsafe_log,
)

# ── Safety caps ───────────────────────────────────────────────────────────────

ALT_MIN_M: float = 0.5
ALT_MAX_M: float = 10.0
HOVER_MIN_S: float = 1.0
HOVER_MAX_S: float = 30.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("qps.test_hover")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="QuackOps hover flight test")
    parser.add_argument(
        "--alt", type=float, default=3.0,
        help=f"Takeoff altitude in metres [{ALT_MIN_M}, {ALT_MAX_M}] (default 3.0)",
    )
    parser.add_argument(
        "--hover", type=float, default=10.0,
        help=f"Hover duration in seconds [{HOVER_MIN_S}, {HOVER_MAX_S}] (default 10.0)",
    )
    parser.add_argument(
        "--config", type=str, default=str(here / "config" / "outdoor_production.json"),
        help="Path to qpsConfig JSON file",
    )
    parser.add_argument(
        "--log-dir", type=str, default=str(here / "logs"),
        help="Base directory for log output",
    )
    args = parser.parse_args()
    # Enforce safety caps
    args.alt = max(ALT_MIN_M, min(ALT_MAX_M, args.alt))
    args.hover = max(HOVER_MIN_S, min(HOVER_MAX_S, args.hover))
    return args


async def main() -> int:
    args = parse_args()

    # ── Log directory ────────────────────────────────────────────────────────
    log_dir = make_log_dir(args.log_dir, "hover")
    log.info("Log directory: %s", log_dir)
    log.info("Parameters: alt=%.1fm  hover=%.1fs", args.alt, args.hover)

    # ── Load config ──────────────────────────────────────────────────────────
    config = qpsConfig.from_file(args.config)
    log.info("Connection: %s", config.connection_string)

    # ── Instantiate qps stack ────────────────────────────────────────────────
    fm = qpsFlightManager(config)
    tm = qpsTelemetryMonitor(fm, config)
    diag = TestDiagnostics(fm)
    fs = FailsafeWatcher(fm, tm)

    #csv_writer = open_telemetry_csv(log_dir)
    csv_writer, csv_file = open_telemetry_csv(log_dir)
    fs_log: FailsafeLog = open_failsafe_log(log_dir)

    watcher_task: asyncio.Task | None = None
    in_flight = False

    try:
        # ── Connect ──────────────────────────────────────────────────────────
        log.info("Connecting to flight controller...")
        await fm.connect()
        await tm.start()

        watcher_task = asyncio.create_task(
            fs.watch_mode_transitions(), name="failsafe-watcher"
        )

        # ── GPS preflight ────────────────────────────────────────────────────
        log.info("Waiting for GPS lock...")
        await wait_gps_ready(fm, tm, min_sats=10, max_hdop=1.4, log=log)

        # ── Capture home position ────────────────────────────────────────────
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
        if 0 < batt_v < 1:
            raise RuntimeError(
                f"Battery too low to fly: {batt_v:.2f}V (stop testing below 11.6V, full=12.6V)"
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

        # ── Arm ──────────────────────────────────────────────────────────────
        log.info("Arming...")
        await fm.arm()
        in_flight = True

        # ── Takeoff ──────────────────────────────────────────────────────────
        log.info("Taking off to %.1fm...", args.alt)
        await fm.takeoff(args.alt)
        log.info("Takeoff complete")

        # ── Hover with telemetry logging ─────────────────────────────────────
        log.info("Hovering for %.1fs...", args.hover)
        result = await hover_with_logging(
            tm, diag, fs, args.hover, csv_writer, log,
            poll_hz=config.telemetry_polling_rate_hz,
        )

        if result != "completed":
            fs_log.log(result, "early exit during hover")
            log.warning("Hover ended early: %s", result)

        # ── Land ─────────────────────────────────────────────────────────────
        log.info("Landing...")
        fs.set_expected_mode("LAND")
        await fm.land()
        log.info("Landed")

        # ── Disarm ───────────────────────────────────────────────────────────
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
