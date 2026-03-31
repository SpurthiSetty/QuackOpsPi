import asyncio
import json
import websockets
from mavsdk import System

SERIAL_ADDRESS = "serial:///dev/ttyAMA0:57600"
LAPTOP_IP = "192.168.1.2"
WS_PORT = 3001
STREAM_RATE_HZ = 2


async def run():
    drone = System()
    print(f"Connecting to Pixhawk on {SERIAL_ADDRESS}...")
    await drone.connect(system_address=SERIAL_ADDRESS)

    print("Waiting for heartbeat...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("  CONNECTED to Pixhawk\n")
            break

    uri = f"ws://{LAPTOP_IP}:{WS_PORT}"
    print(f"Connecting to WebSocket server at {uri}...")

    async with websockets.connect(uri) as ws:
        print("  CONNECTED to laptop\n")
        print("Streaming telemetry... (Ctrl+C to stop)\n")

        while True:
            telemetry_data = {}

            try:
                async for position in drone.telemetry.position():
                    telemetry_data["position"] = {
                        "latitude_deg": position.latitude_deg,
                        "longitude_deg": position.longitude_deg,
                        "absolute_altitude_m": position.absolute_altitude_m,
                        "relative_altitude_m": position.relative_altitude_m
                    }
                    break
            except Exception as e:
                telemetry_data["position"] = {"error": str(e)}

            try:
                async for heading in drone.telemetry.heading():
                    telemetry_data["heading_deg"] = heading.heading_deg
                    break
            except Exception:
                telemetry_data["heading_deg"] = None

            try:
                async for battery in drone.telemetry.battery():
                    telemetry_data["battery"] = {
                        "voltage_v": battery.voltage_v,
                        "remaining_percent": battery.remaining_percent
                    }
                    break
            except Exception:
                telemetry_data["battery"] = None

            try:
                async for attitude in drone.telemetry.attitude_euler():
                    telemetry_data["attitude"] = {
                        "roll_deg": attitude.roll_deg,
                        "pitch_deg": attitude.pitch_deg,
                        "yaw_deg": attitude.yaw_deg
                    }
                    break
            except Exception:
                telemetry_data["attitude"] = None

            try:
                async for mode in drone.telemetry.flight_mode():
                    telemetry_data["flight_mode"] = str(mode)
                    break
            except Exception:
                telemetry_data["flight_mode"] = None

            try:
                async for armed in drone.telemetry.armed():
                    telemetry_data["armed"] = armed
                    break
            except Exception:
                telemetry_data["armed"] = None

            message = json.dumps(telemetry_data, default=str)
            await ws.send(message)
            print(f"  Sent: lat={telemetry_data.get('position', {}).get('latitude_deg', '?'):.6f}, "
                  f"lon={telemetry_data.get('position', {}).get('longitude_deg', '?'):.6f}, "
                  f"alt={telemetry_data.get('position', {}).get('relative_altitude_m', '?'):.2f}m, "
                  f"heading={telemetry_data.get('heading_deg', '?')}, "
                  f"mode={telemetry_data.get('flight_mode', '?')}")

            await asyncio.sleep(1 / STREAM_RATE_HZ)


asyncio.run(run())
