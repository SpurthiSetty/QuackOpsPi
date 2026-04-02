import asyncio
from mavsdk import System

SERIAL_ADDRESS = "serial:///dev/ttyAMA0:57600"


async def run():
    drone = System()
    print(f"Connecting to Pixhawk on {SERIAL_ADDRESS}...")
    await drone.connect(system_address=SERIAL_ADDRESS)

    print("Waiting for heartbeat...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("  CONNECTED to Pixhawk\n")
            break

    # Check health before attempting arm
    print("--- Pre-Arm Health Check ---")
    try:
        health = await asyncio.wait_for(
            read_one(drone.telemetry.health()),
            timeout=10
        )
        if health:
            print(f"  Gyro cal:    {health.is_gyrometer_calibration_ok}")
            print(f"  Accel cal:   {health.is_accelerometer_calibration_ok}")
            print(f"  Mag cal:     {health.is_magnetometer_calibration_ok}")
            print(f"  Local pos:   {health.is_local_position_ok}")
            print(f"  Global pos:  {health.is_global_position_ok}")
            print(f"  Home pos:    {health.is_home_position_ok}")
            print(f"  Armable:     {health.is_armable}")
    except asyncio.TimeoutError:
        print("  Could not read health (timeout)")

    # Check current armed state
    print("\n--- Current Armed State ---")
    try:
        armed = await asyncio.wait_for(
            read_one(drone.telemetry.armed()),
            timeout=10
        )
        print(f"  Armed: {armed}")
    except asyncio.TimeoutError:
        print("  Could not read armed state (timeout)")

    # Attempt to arm
    print("\n--- Attempting ARM ---")
    try:
        await drone.action.arm()
        print("  ARM command sent successfully!")
    except Exception as e:
        print(f"  ARM failed: {e}")

    # Wait a moment and check armed state
    await asyncio.sleep(2)
    print("\n--- Armed State After ARM ---")
    try:
        armed = await asyncio.wait_for(
            read_one(drone.telemetry.armed()),
            timeout=10
        )
        print(f"  Armed: {armed}")
    except asyncio.TimeoutError:
        print("  Could not read armed state (timeout)")

    # Wait 3 seconds then disarm
    print("\n  Waiting 3 seconds before disarm...")
    await asyncio.sleep(3)

    print("\n--- Attempting DISARM ---")
    try:
        await drone.action.disarm()
        print("  DISARM command sent successfully!")
    except Exception as e:
        print(f"  DISARM failed: {e}")

    # Final armed state check
    await asyncio.sleep(2)
    print("\n--- Final Armed State ---")
    try:
        armed = await asyncio.wait_for(
            read_one(drone.telemetry.armed()),
            timeout=10
        )
        print(f"  Armed: {armed}")
    except asyncio.TimeoutError:
        print("  Could not read armed state (timeout)")

    print("\n=============================")
    print("  Arm/Disarm test complete!")
    print("=============================")


async def read_one(async_generator):
    async for value in async_generator:
        return value


asyncio.run(run())