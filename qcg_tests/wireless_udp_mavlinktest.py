from pymavlink import mavutil

m = mavutil.mavlink_connection('udpin:127.0.0.1:14551')
m.wait_heartbeat()

# Send ARM with EXPLICIT target
m.mav.command_long_send(
    1,  # target_system: autopilot is sysid 1
    1,  # target_component: autopilot is compid 1 (ArduPilot autopilot)
    400,  # MAV_CMD_COMPONENT_ARM_DISARM
    0,    # confirmation
    1,    # param1: 1=arm
    0, 0, 0, 0, 0, 0
)

import time
time.sleep(0.5)

# Drain ALL messages and look for ANYTHING related
import time
end = time.time() + 5
while time.time() < end:
    msg = m.recv_match(blocking=False)
    if msg:
        t = msg.get_type()
        if t in ['STATUSTEXT', 'COMMAND_ACK', 'HEARTBEAT']:
            print(f"{t}: {msg}")