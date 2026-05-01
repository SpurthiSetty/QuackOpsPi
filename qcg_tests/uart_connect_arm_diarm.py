#!/usr/bin/env python3
"""Direct UART arm test - no MAVProxy."""
import time
from pymavlink import mavutil

# Direct UART connection
m = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
m.wait_heartbeat()
print(f"Connected: sys={m.target_system} comp={m.target_component}")

# Set STABILIZE
m.mav.command_long_send(
    1, 1, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0,
    0, 0, 0, 0, 0
)
time.sleep(1)

# Drain any STATUSTEXT
end = time.time() + 2
while time.time() < end:
    msg = m.recv_match(blocking=False)
    if msg and msg.get_type() == 'STATUSTEXT':
        print(f"  [FC] {msg.text}")

# Send ARM
print("Sending ARM...")
m.mav.command_long_send(
    1, 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
    1, 0, 0, 0, 0, 0, 0
)

# Wait for ack or any response
end = time.time() + 5
while time.time() < end:
    msg = m.recv_match(blocking=False)
    if msg:
        #print(msg)
        t = msg.get_type()
        if t == 'COMMAND_ACK':
            print(f"  ACK: command={msg.command} result={msg.result}")
        elif t == 'STATUSTEXT':
            print(f"  [FC] {msg.text}")
        elif t == 'HEARTBEAT' and (msg.base_mode & 128):
            print(f"  HEARTBEAT shows ARMED!")
            break

print("Done")
