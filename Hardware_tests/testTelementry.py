import asyncio
from mavsdk import System

SERIAL_ADDRESS = "serial:///dev/ttyAMA0:57600"
TIMEOUT_SECONDS = 10


async def read_one(async_generator, name):
    """Read a single value from a telemetry stream with timeout."""
    try:
        async for value in async_generator:
            return value
    except Exception as e:
        print(f"  ERROR reading {name}: {e}")
        return None


async def test_with_timeout(async_generator, name):
    """Wrap a telemetry read with a timeout."""
    try:
        result = await asyncio.wait_for(
            read_one(async_generator, name),
            timeout=TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        print(f"  TIMEOUT: {name} (no data after {TIMEOUT_SECONDS}s)")
        return None


async def run():
    drone = System()
    print(f"Connecting to Pixhawk on {SERIAL_ADDRESS}...")
    await drone.connect(system_address=SERIAL_ADDRESS)

    # 1. Connection
    print("\n--- Connection ---")
    print("Waiting for heartbeat...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("  CONNECTED to Pixhawk")
            break

    # 2. Battery
    print("\n--- Battery ---")
    battery = await test_with_timeout(drone.telemetry.battery(), "battery")
    if battery:
        print(f"  ID:          {battery.id}")
        print(f"  Voltage:     {battery.voltage_v:.2f} V")
        print(f"  Remaining:   {battery.remaining_percent:.1f}%")
        print(f"  Current:     {battery.current_battery_a:.2f} A")

    # 3. GPS Info
    print("\n--- GPS Info ---")
    gps_info = await test_with_timeout(drone.telemetry.gps_info(), "gps_info")
    if gps_info:
        print(f"  Fix type:    {gps_info.fix_type}")
        print(f"  Satellites:  {gps_info.num_satellites}")

    # 4. Position (requires GPS fix)
    print("\n--- Position ---")
    position = await test_with_timeout(drone.telemetry.position(), "position")
    if position:
        print(f"  Latitude:    {position.latitude_deg:.6f} deg")
        print(f"  Longitude:   {position.longitude_deg:.6f} deg")
        print(f"  Abs Alt:     {position.absolute_altitude_m:.2f} m")
        print(f"  Rel Alt:     {position.relative_altitude_m:.2f} m")

    # 5. Flight Mode
    print("\n--- Flight Mode ---")
    flight_mode = await test_with_timeout(drone.telemetry.flight_mode(), "flight_mode")
    if flight_mode:
        print(f"  Mode:        {flight_mode}")

    # 6. Armed Status
    print("\n--- Armed Status ---")
    armed = await test_with_timeout(drone.telemetry.armed(), "armed")
    if armed is not None:
        print(f"  Armed:       {armed}")

    # 7. In-Air Status
    print("\n--- In-Air Status ---")
    in_air = await test_with_timeout(drone.telemetry.in_air(), "in_air")
    if in_air is not None:
        print(f"  In air:      {in_air}")

    # 8. Landed State
    print("\n--- Landed State ---")
    landed = await test_with_timeout(drone.telemetry.landed_state(), "landed_state")
    if landed:
        print(f"  State:       {landed}")

    # 9. Health
    print("\n--- Health ---")
    health = await test_with_timeout(drone.telemetry.health(), "health")
    if health:
        print(f"  Gyro cal:    {health.is_gyrometer_calibration_ok}")
        print(f"  Accel cal:   {health.is_accelerometer_calibration_ok}")
        print(f"  Mag cal:     {health.is_magnetometer_calibration_ok}")
        print(f"  Local pos:   {health.is_local_position_ok}")
        print(f"  Global pos:  {health.is_global_position_ok}")
        print(f"  Home pos:    {health.is_home_position_ok}")
        print(f"  Armable:     {health.is_armable}")

    # 10. Attitude (Euler angles)
    print("\n--- Attitude (Euler) ---")
    attitude = await test_with_timeout(drone.telemetry.attitude_euler(), "attitude_euler")
    if attitude:
        print(f"  Roll:        {attitude.roll_deg:.2f} deg")
        print(f"  Pitch:       {attitude.pitch_deg:.2f} deg")
        print(f"  Yaw:         {attitude.yaw_deg:.2f} deg")

    # 11. Heading
    print("\n--- Heading ---")
    heading = await test_with_timeout(drone.telemetry.heading(), "heading")
    if heading:
        print(f"  Heading:     {heading.heading_deg:.2f} deg")

    # 12. RC Status
    print("\n--- RC Status ---")
    rc = await test_with_timeout(drone.telemetry.rc_status(), "rc_status")
    if rc:
        print(f"  Available:   {rc.is_available}")
        print(f"  Signal:      {rc.signal_strength_percent:.1f}%")

    # 13. Status Text Messages
    print("\n--- Status Text (5s listen) ---")
    try:
        count = 0
        async for status_text in asyncio.timeout(5).__aenter__() or drone.telemetry.status_text():
            print(f"  [{status_text.type}] {status_text.text}")
            count += 1
            if count >= 5:
                break
    except (asyncio.TimeoutError, Exception):
        print("  (no status messages received)")

    # Summary
    print("\n=============================")
    print("  Telemetry test complete!")
    print("=============================")


asyncio.run(run())