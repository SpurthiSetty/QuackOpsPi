import asyncio
import websockets
import json

LAPTOP_IP = "192.168.1.184"  # Replace with your laptop's actual IP

async def run():
    uri = f"ws://{LAPTOP_IP}:3001"
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        msg = json.dumps({"type": "status", "data": "Pi says hello"})
        await ws.send(msg)
        print(f"Sent: {msg}")
        response = await ws.recv()
        print(f"Received: {response}")
        print("SUCCESS — WebSocket communication works!")

asyncio.run(run())