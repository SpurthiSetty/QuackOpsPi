#!/bin/bash
# start_mavproxy.sh — Launch MAVProxy with UDP outputs for Pi scripts and Mission Planner.
#
# Usage:
#   ./outdoor_tests/start_mavproxy.sh [LOG_DIR]
#
# Outputs:
#   udpout:192.168.137.1:14550  → Mission Planner on Windows laptop
#   udpin:127.0.0.1:14551       → preflight_check.py + flight scripts (udpout:127.0.0.1:14551)
set -euo pipefail

LOGDIR="${1:-/tmp}"
mkdir -p "$LOGDIR"

exec mavproxy.py \
  --master=/dev/ttyAMA0,57600 \
  --out=udpout:192.168.137.1:14550 \
  --out=udpin:127.0.0.1:14551 \
  --logfile="$LOGDIR/mavproxy.tlog" \
  --daemon
