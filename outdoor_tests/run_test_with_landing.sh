#!/bin/bash
# run_test_with_landing.sh — Orchestrate a QuackOps outdoor flight test with
# marker-triggered landing.
#
# Usage:
#   ./outdoor_tests/run_test_with_landing.sh hover [extra args...]
#   ./outdoor_tests/run_test_with_landing.sh lateral --pattern square --size 3 [extra args...]
#
# Unlike run_test.sh, this script does NOT launch camera_daemon.py. The flight
# script owns the camera and MJPEG stream internally.
#
# What it starts (in order):
#   1. MAVProxy  (serial → UDP bridge)
#   2. Flight script  (test_hover_with_landing.py or test_lateral_with_landing.py)
#      — foreground, camera/recording managed inside the script
#
# Ctrl+C or script exit tears everything down via trap.
set -euo pipefail

TEST="${1:?usage: run_test_with_landing.sh <hover|lateral> [extra args]}"
shift

case "$TEST" in
    hover)   FLIGHT_SCRIPT="outdoor_tests/test_hover_with_landing.py" ;;
    lateral) FLIGHT_SCRIPT="outdoor_tests/test_lateral_with_landing.py" ;;
    *)
        echo "Usage: $0 {hover|lateral} [flight script args...]" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOGDIR="${SCRIPT_DIR}/logs/${TIMESTAMP}_${TEST}_with_landing"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "  QuackOps outdoor test (with landing): $TEST"
echo "  Log dir: $LOGDIR"
echo "=========================================="

MAVPROXY_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    [[ -n "${MAVPROXY_PID}" ]] && kill -TERM "$MAVPROXY_PID" 2>/dev/null || true
    sleep 1
    wait 2>/dev/null || true
    echo "Done. Logs in: $LOGDIR"
}
trap cleanup EXIT INT TERM

# ── 1. MAVProxy ───────────────────────────────────────────────────────────────
echo "[1/2] Starting MAVProxy..."
"$SCRIPT_DIR/start_mavproxy.sh" "$LOGDIR" &
MAVPROXY_PID=$!
echo "      MAVProxy PID: $MAVPROXY_PID"
sleep 3  # allow MAVProxy to open serial and bind UDP ports

# ── 2. Flight script (foreground) ─────────────────────────────────────────────
echo "[2/2] Starting flight script: $FLIGHT_SCRIPT"
echo "      Args: $*"
echo "------------------------------------------"

cd "$REPO_ROOT"
source venv/bin/activate 2>/dev/null || true

python3 "$FLIGHT_SCRIPT" \
    --log-dir "$LOGDIR" \
    "$@"
