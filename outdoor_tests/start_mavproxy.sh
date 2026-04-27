#!/bin/bash
# start_mavproxy.sh — Launch MAVProxy with UDP outputs for Pi scripts and Mission Planner.
#
# Usage:
#   ./outdoor_tests/start_mavproxy.sh [LOG_DIR]
#
# Outputs:
#   udpout:127.0.0.1:14550  → preflight_check.py + qpsFlightManager (udpin:127.0.0.1:14550)
#   udpin:0.0.0.0:14551     → Mission Planner connects here (UDP Client 192.168.1.128:14551)
set -euo pipefail

LOGDIR="${1:-/tmp}"
mkdir -p "$LOGDIR"

exec mavproxy.py \
  --master=/dev/ttyAMA0 \
  --baudrate=57600 \
  --out=udpout:127.0.0.1:14550 \
  --out=udpin:0.0.0.0:14551 \
  --logfile="$LOGDIR/mavproxy.tlog" \
  --daemon
