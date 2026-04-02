import asyncio
from mavsdk import System

async def run():
    drone = System()
    print("Connecting to Pixhawk on /dev/ttyAMA0...")
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    print("Waiting for heartbeat...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("SUCCESS — Pixhawk is connected!")
            break

    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.remaining_percent:.1f}%")
        break

    async for gps in drone.telemetry.gps_info():
        print(f"GPS fix: {gps.fix_type}, Satellites: {gps.num_satellites}")
        break

    print("All checks passed.")

asyncio.run(run())