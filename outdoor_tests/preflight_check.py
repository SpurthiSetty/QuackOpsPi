"""
outdoor_tests/preflight_check.py

Standalone preflight parameter checker.  Connects via MAVProxy UDP (does NOT
use the qps package — no async, no Python venv required beyond pymavlink).

Run AFTER MAVProxy is already up:
    python3 outdoor_tests/preflight_check.py [--connection udpin:127.0.0.1:14550]

Exit 0 = all pass.  Exit 1 = one or more failures.

Battery voltage thresholds are prompted interactively if the expected values
don't match a standard cell count, because the correct values depend on the
actual pack used on the day.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Optional

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not installed. Run: pip install pymavlink", file=sys.stderr)
    sys.exit(2)

# ── ANSI colours ─────────────────────────────────────────────────────────────

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

def _ok(s: str) -> str:  return f"{_GREEN}✓{_RESET} {s}"
def _fail(s: str) -> str: return f"{_RED}✗{_RESET} {s}"
def _warn(s: str) -> str: return f"{_YELLOW}!{_RESET} {s}"


# ── Parameter definitions ─────────────────────────────────────────────────────
# Each entry: (param_name, expected_value, note)
# expected_value=None means "prompt user"

_PARAMS: list[tuple[str, Optional[float], str]] = [
    ("ARMING_CHECK",    1.0,   "Pre-arm checks ON"),
    ("FS_THR_ENABLE",   3.0,   "RC failsafe → LAND"),
    ("FS_GCS_ENABLE",   5.0,   "GCS failsafe → LAND"),
    ("FS_GCS_TIMEOUT",  5.0,   "GCS timeout 5 s"),
    ("BATT_FS_LOW_ACT", 0.0,   "Low battery → WARN ONLY (pilot decides)"),
    ("BATT_FS_CRT_ACT", 1.0,   "Critical battery → LAND"),
    ("BATT_LOW_VOLT",   None,  "Low-voltage threshold (user-confirmed)"),
    ("BATT_CRT_VOLT",   None,  "Critical-voltage threshold (user-confirmed)"),
    ("FENCE_ENABLE",    0.0,   "Geofence OFF"),
    ("FENCE_ACTION",    0.0,   "Fence breach → report only (no failsafe)"),
    ("RTL_ALT",         1500.0,"RTL altitude 15 m (above obstacles)"),
    ("DISARM_DELAY",    10.0,  "Auto-disarm 10 s after landing"),
]


# ── MAVLink helpers ───────────────────────────────────────────────────────────

def _read_param(
    mav: Any, param_id: str, retries: int = 5, timeout_s: float = 3.0
) -> Optional[float]:
    """Request a parameter and wait for PARAM_VALUE.  Returns None on failure."""
    param_id_bytes = param_id.encode("utf-8")
    for attempt in range(1, retries + 1):
        # Drain any stale PARAM_VALUE messages before requesting
        while True:
            stale = mav.recv_match(type="PARAM_VALUE", blocking=False)
            if stale is None:
                break

        mav.mav.param_request_read_send(
            mav.target_system,
            mav.target_component,
            param_id_bytes,
            -1,  # use param_id string, not index
        )

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg is None:
                continue
            received_id = msg.param_id.rstrip("\x00")
            if received_id == param_id:
                return float(msg.param_value)
        print(f"  {_warn(f'Attempt {attempt}/{retries} timed out for {param_id}')}")

    return None


# ── Battery voltage prompt ────────────────────────────────────────────────────

def _prompt_battery_voltage(param_name: str, current_val: Optional[float]) -> float:
    """Ask the user what the expected voltage threshold should be.

    Provides 3S/4S suggestions based on common LiPo voltages.
    """
    suggestions = {
        "BATT_LOW_VOLT":  {"3S": 10.5, "4S": 14.0},
        "BATT_CRT_VOLT":  {"3S":  9.9, "4S": 13.2},
    }
    s = suggestions.get(param_name, {})
    print(f"\n  {_YELLOW}{param_name}{_RESET} current value: {current_val}")
    print(f"  Typical values — 3S: {s.get('3S','?')} V   4S: {s.get('4S','?')} V")
    while True:
        try:
            raw = input(f"  Enter expected value for {param_name} (or Enter to accept current): ").strip()
            if raw == "":
                return current_val if current_val is not None else 0.0
            return float(raw)
        except ValueError:
            print("  Invalid — enter a number.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="QuackOps preflight parameter check")
    parser.add_argument(
        "--connection", default="udpout:127.0.0.1:14551",
        help="MAVLink connection string (default: udpout:127.0.0.1:14551)",
    )
    args = parser.parse_args()

    print(f"\n{_BOLD}QuackOps Preflight Check{_RESET}")
    print(f"Connection: {args.connection}")
    print("Connecting (MAVProxy must already be running)...", flush=True)

    try:
        mav = mavutil.mavlink_connection(args.connection)
        mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )
        mav.wait_heartbeat(timeout=10)
    except Exception as exc:
        print(f"{_fail(f'Cannot connect: {exc}')}")
        print("Is MAVProxy running?  Run: ./outdoor_tests/start_mavproxy.sh")
        return 1

    print(f"Connected: sysid={mav.target_system}  compid={mav.target_component}\n")

    # Request all streams so we get responses
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 1, 1,
    )
    time.sleep(0.5)

    # ── Resolve expected values for battery voltages ──────────────────────────
    resolved_params: list[tuple[str, float, str]] = []
    for param_name, expected, note in _PARAMS:
        if expected is None:
            current = _read_param(mav, param_name)
            expected = _prompt_battery_voltage(param_name, current)
        resolved_params.append((param_name, expected, note))

    # ── Read all params and compare ───────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  {'PARAMETER':<24}  {'EXPECTED':>10}  {'ACTUAL':>10}  {'STATUS':<6}  NOTE")
    print(f"{'─'*72}")

    all_pass = True
    for param_name, expected, note in resolved_params:
        actual = _read_param(mav, param_name)
        if actual is None:
            status = _fail("TIMEOUT")
            actual_str = "???"
            all_pass = False
        elif abs(actual - expected) < 0.01:
            status = _ok("PASS  ")
            actual_str = f"{actual:.2f}"
        else:
            status = _fail("FAIL  ")
            actual_str = f"{actual:.2f}"
            all_pass = False

        print(f"  {param_name:<24}  {expected:>10.2f}  {actual_str:>10}  {status}  {note}")

    print(f"{'─'*72}")

    if all_pass:
        print(f"\n{_GREEN}{_BOLD}All checks PASSED — safe to fly.{_RESET}\n")
        return 0
    else:
        print(f"\n{_RED}{_BOLD}One or more checks FAILED — do NOT fly until corrected.{_RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
