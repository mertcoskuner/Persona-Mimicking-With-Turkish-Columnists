# websocket_client.py
import asyncio
import websockets
import json

async def test_websocket():
    token = ''
    uri = f"ws://127.0.0.1:8001/ws/llm/Turkish%20llama%208b/?token={token}"

    async with websockets.connect(uri) as websocket:
        # Send a message to the server
        message = {'message': 'Hello, LLM!'}
        await websocket.send(json.dumps(message))
        print(f"Sent: {message}")

        # Receive a response from the server
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Received: {data}")

asyncio.get_event_loop().run_until_complete(test_websocket())
