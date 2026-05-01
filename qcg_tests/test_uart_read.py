#!/usr/bin/env python3
"""Read-only UART telemetry test - confirms what flows FROM the FC."""
import time
from pymavlink import mavutil

m = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
print("Waiting for heartbeat...")
m.wait_heartbeat()
print(f"Connected: sys={m.target_system} comp={m.target_component}")
print()
print("Reading messages for 10 seconds. Will count message types.")
print()

counts = {}
start = time.time()
while time.time() - start < 10:
    msg = m.recv_match(blocking=False)
    if msg is None:
        time.sleep(0.01)
        continue
    t = msg.get_type()
    counts[t] = counts.get(t, 0) + 1
    
    # Print first occurrence of each message type
    if counts[t] == 1:
        print(f"  First {t}: ", end="")
        if t == "HEARTBEAT":
            print(f"mode_id={msg.custom_mode} armed={bool(msg.base_mode & 128)}")
        elif t == "BATTERY_STATUS":
            print(f"voltage={msg.voltages[0]/1000.0}V remaining={msg.battery_remaining}%")
        elif t == "GPS_RAW_INT":
            print(f"fix={msg.fix_type} sats={msg.satellites_visible}")
        elif t == "ATTITUDE":
            print(f"roll={msg.roll:.3f} pitch={msg.pitch:.3f}")
        elif t == "STATUSTEXT":
            print(f"{msg.text}")
        elif t == "RC_CHANNELS":
            print(f"ch3(throttle)={msg.chan3_raw}")
        else:
            print()

print()
print("=" * 50)
print("Message type counts in 10 seconds:")
print("=" * 50)
for t, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")
