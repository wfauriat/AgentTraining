import httpx
import json
from datetime import datetime

MODEL = "qwen3:8b"

messages = []

def get_current_time():
    return str(datetime.now().isoformat())

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns the current date and time",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}]

while True:
    if not messages or messages[-1].get("role") != "tool":
        query = input("Ask the model something: ")
        if query == "quit":
            print("Powering down")
            break
        messages.append({"role": "user", "content": query})
    response = httpx.post(
        "http://localhost:11434/api/chat",
        json={"model": MODEL,
            "messages": messages,
            "stream": False,
            "tools": tools},
        timeout=120)
    response.raise_for_status()
    payload = response.json()["message"]
    if payload.get("tool_calls"):
        messages.append(payload)
        tool_name = payload["tool_calls"][0]["function"]["name"]
        if tool_name == "get_current_time":
            current_time = get_current_time()
            messages.append({"role": "tool", "content":current_time})
        print(f"Called tool: {payload.get("tool_calls")}")
    else:
        print("> User: " + query)
        response.raise_for_status()
        reply = response.json()["message"]["content"]
        print("> " + MODEL + ": " + reply)
        messages.append({"role": "assistant", "content": reply})
