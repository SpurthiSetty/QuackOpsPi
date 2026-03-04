import asyncio
from mavsdk import System

async def run():
    drone = System()
    print("Connecting to PX4 SITL over UDP...")
    await drone.connect(system_address="udp://:14540")

    print("Waiting for heartbeat...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("SUCCESS — Connected to simulated Pixhawk!")
            break

    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.remaining_percent * 100:.1f}%")
        break

    async for gps in drone.telemetry.gps_info():
        print(f"GPS fix: {gps.fix_type}, Satellites: {gps.num_satellites}")
        break

    async for position in drone.telemetry.position():
        print(f"Position: {position.latitude_deg:.6f}, {position.longitude_deg:.6f}, Alt: {position.relative_altitude_m:.1f}m")
        break

    print("All checks passed.")

asyncio.run(run())