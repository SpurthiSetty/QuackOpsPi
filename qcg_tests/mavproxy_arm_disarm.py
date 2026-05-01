#!/usr/bin/env python3
"""MAVProxy arm/disarm — matches udpin:127.0.0.1:14551 output."""
import time
from pymavlink import mavutil

m = mavutil.mavlink_connection('udpout:127.0.0.1:14551')

# Kick a heartbeat so MAVProxy knows our return address
m.mav.heartbeat_send(
    mavutil.mavlink.MAV_TYPE_GCS,
    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
    0, 0, 0
)

print("Waiting for heartbeat...")
m.wait_heartbeat()
print(f"Connected: sys={m.target_system} comp={m.target_component}")

def send_command(cmd, p1=0, p2=0, timeout=5):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        cmd, 0, p1, p2, 0, 0, 0, 0, 0
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

# ARM
print("Sending ARM...")
result = send_command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=1)

if result == 0:
    end = time.time() + 5
    while time.time() < end:
        msg = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("  HEARTBEAT confirms ARMED")
            break

    print("Holding for 3s...")
    time.sleep(3)

    print("Sending DISARM...")
    send_command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=0)

print("Done")