#!/usr/bin/env python3
"""Force SERIAL0_PROTOCOL=2 to restore USB MAVLink."""
import time
from pymavlink import mavutil

print("Connecting to /dev/ttyAMA0 @ 57600...")
conn = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
conn.wait_heartbeat()
print(f"Connected — sysid={conn.target_system}")

# Send PARAM_SET directly (bypasses any caching)
print("Setting SERIAL0_PROTOCOL = 2 (MAVLink2)...")
conn.mav.param_set_send(
    conn.target_system,
    conn.target_component,
    b'SERIAL0_PROTOCOL',
    2.0,
    mavutil.mavlink.MAV_PARAM_TYPE_INT32
)

conn.mav.param_set_send(
    conn.target_system,
    conn.target_component,
    b'ARMING_CHECK',
    0.0,
    mavutil.mavlink.MAV_PARAM_TYPE_INT32
)
# Read back to confirm
print("Reading back param...")
conn.mav.param_request_read_send(
    conn.target_system,
    conn.target_component,
    b'SERIAL0_PROTOCOL',
    -1
)

# Wait for response
deadline = time.time() + 5
while time.time() < deadline:
    msg = conn.recv_match(type='PARAM_VALUE', blocking=False)
    if msg and msg.param_id.strip('\x00') == 'SERIAL0_PROTOCOL':
        print(f"Confirmed: SERIAL0_PROTOCOL = {msg.param_value}")
        break
    time.sleep(0.1)
else:
    print("WARNING: No PARAM_VALUE response received")

print("Rebooting FC...")
conn.mav.command_long_send(
    conn.target_system,
    conn.target_component,
    mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
    0,
    1, 0, 0, 0, 0, 0, 0
)
print("Done. Wait 30s for FC to reboot, then try MP-USB.")
conn.close()