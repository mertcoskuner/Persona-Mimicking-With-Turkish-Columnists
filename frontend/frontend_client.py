import asyncio
import websockets
import json

async def test_websocket():
    # Replace with your JWT token
    token = ""  # Generate using generate_token.py
    uri = f"ws://127.0.0.1:8001/ws/llm/Turkish%20llama%208b/?token={token}"

    async with websockets.connect(uri) as websocket:
        # Send a message to the server
        message = {"message": "Hello, LLM!"}
        await websocket.send(json.dumps(message))
        print(f"Sent: {message}")

        # Receive a response from the server
        response = await websocket.recv()
        print(f"Received: {response}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
