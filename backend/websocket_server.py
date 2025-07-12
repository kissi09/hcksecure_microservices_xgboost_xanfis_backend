# websocket_server.py
import asyncio
import websockets
import json
import subprocess

async def handle_connection(websocket, path):
    async for message in websocket:
        print(f"Received data: {message}")
        data = json.loads(message)
        # Call inference.py
        result = subprocess.run(['python3', 'inference.py', json.dumps(data)], capture_output=True, text=True)
        await websocket.send(result.stdout)

# Start WebSocket server
start_server = websockets.serve(handle_connection, 'localhost', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()