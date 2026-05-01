#!/usr/bin/env python3
"""Quick hover test — arms, takes off to 1m, hovers 5s, lands."""
import time
from pymavlink import mavutil

# Connect to MAVProxy
m = mavutil.mavlink_connection('udpout:127.0.0.1:14551')
m.mav.heartbeat_send(
    mavutil.mavlink.MAV_TYPE_GCS,
    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
    0, 0, 0
)
print("Waiting for heartbeat...")
m.wait_heartbeat()
print(f"Connected: sys={m.target_system} comp={m.target_component}")

def send_command(cmd, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0, timeout=5):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        cmd, 0, p1, p2, p3, p4, p5, p6, p7
    )
    end = time.time() + timeout
    while time.time() < end:
        msg = m.recv_match(type=['COMMAND_ACK', 'STATUSTEXT'], blocking=False)
        if msg:
            t = msg.get_type()
            if t == 'COMMAND_ACK' and msg.command == cmd:
                print(f"  ACK: command={msg.command} result={msg.result}")
                return msg.result
            elif t == 'STATUSTEXT':
                print(f"  [FC] {msg.text}")
        time.sleep(0.01)
    print("  WARNING: No ACK received")
    return None

def get_altitude():
    msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg:
        return msg.relative_alt / 1000.0
    return None

# GUIDED mode
print("Setting GUIDED mode...")
result = send_command(
    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
    p1=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    p2=4
)

# ARM
print("Arming...")
result = send_command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=1)
if result != 0:
    print("Arm failed — aborting")
    exit(1)

# Wait for HEARTBEAT to confirm armed
end = time.time() + 10
while time.time() < end:
    msg = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
        print("  HEARTBEAT confirms ARMED")
        break

# TAKEOFF to 1m
print("Taking off to 1m...")
send_command(
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    p7=1.0,
    timeout=10
)

# Wait until actually at target altitude
print("Waiting to reach 1m...")
target_alt = 1.0
end = time.time() + 20
while time.time() < end:
    alt = get_altitude()
    if alt is not None:
        print(f"  Climbing... Alt: {alt:.2f}m")
        if alt >= target_alt * 0.85:
            print(f"  Reached target altitude: {alt:.2f}m")
            break
    time.sleep(0.2)
else:
    print("  WARNING: Timeout waiting for altitude — landing for safety")
    send_command(mavutil.mavlink.MAV_CMD_NAV_LAND, timeout=10)
    exit(1)

# Hover 5 seconds
print("Hovering for 5s...")
end = time.time() + 10
while time.time() < end:
    alt = get_altitude()
    if alt is not None:
        print(f"  Alt: {alt:.2f}m")
    time.sleep(1)

# LAND
print("Landing...")
send_command(mavutil.mavlink.MAV_CMD_NAV_LAND, timeout=10)

# Wait for disarm
print("Waiting for disarm...")
end = time.time() + 30
while time.time() < end:
    msg = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if msg and not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
        print("  HEARTBEAT confirms DISARMED")
        break

print("Done")
