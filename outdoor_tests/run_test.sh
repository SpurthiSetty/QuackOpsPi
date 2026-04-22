#!/bin/bash
# run_test.sh — Orchestrate a QuackOps outdoor flight test.
#
# Usage:
#   ./outdoor_tests/run_test.sh hover [extra args...]
#   ./outdoor_tests/run_test.sh lateral --pattern square --size 3 [extra args...]
#
# What it starts (in order):
#   1. MAVProxy  (serial → UDP bridge)
#   2. camera_daemon.py  (PiCamera2 capture + MJPEG stream)
#   3. Flight script  (test_hover.py or test_lateral.py) — foreground
#
# Ctrl+C or script exit tears everything down via trap.
set -euo pipefail

TEST="${1:?usage: run_test.sh <hover|lateral> [extra args]}"
shift

# Validate test name
if [[ "$TEST" != "hover" && "$TEST" != "lateral" ]]; then
    echo "ERROR: TEST must be 'hover' or 'lateral' (got: $TEST)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOGDIR="${SCRIPT_DIR}/logs/${TIMESTAMP}_${TEST}"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "  QuackOps outdoor test: $TEST"
echo "  Log dir: $LOGDIR"
echo "=========================================="

MAVPROXY_PID=""
CAMERA_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    [[ -n "${CAMERA_PID}" ]] && kill -TERM "$CAMERA_PID" 2>/dev/null || true
    [[ -n "${MAVPROXY_PID}" ]] && kill -TERM "$MAVPROXY_PID" 2>/dev/null || true
    # Give processes a moment to shut down before waiting
    sleep 1
    wait 2>/dev/null || true
    echo "Done. Logs in: $LOGDIR"
}
trap cleanup EXIT INT TERM

# ── 1. MAVProxy ───────────────────────────────────────────────────────────────
echo "[1/3] Starting MAVProxy..."
"$SCRIPT_DIR/start_mavproxy.sh" "$LOGDIR" &
MAVPROXY_PID=$!
echo "      MAVProxy PID: $MAVPROXY_PID"
sleep 3  # allow MAVProxy to open serial and bind UDP ports

# ── 2. Camera daemon ──────────────────────────────────────────────────────────
echo "[2/3] Starting camera daemon..."
cd "$REPO_ROOT"
source venv/bin/activate 2>/dev/null || true

python3 outdoor_tests/camera_daemon.py \
    --output-dir "$LOGDIR" \
    > "$LOGDIR/camera.log" 2>&1 &
CAMERA_PID=$!
echo "      Camera daemon PID: $CAMERA_PID"
sleep 3  # allow camera to initialise and MJPEG server to bind

# ── 3. Flight script (foreground) ─────────────────────────────────────────────
echo "[3/3] Starting flight script: test_${TEST}.py"
echo "      Args: $*"
echo "------------------------------------------"

python3 "outdoor_tests/test_${TEST}.py" \
    --log-dir "$LOGDIR" \
    "$@"

# Exit code from flight script is preserved by set -e + subshell exit
