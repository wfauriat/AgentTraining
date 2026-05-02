import httpx
import json

response = httpx.post(
    "http://localhost:11434/api/chat",
    json={"model": "qwen3:8b",
           "messages": [{"role": "user", "content": "Hello, how are you ?"}],
           "stream": False},
    timeout=120
)
data = response.json()
print(data["message"]["content"])
 
# [print(str(key) + ":" + str(val)) for (key,val) in zip(data.keys(),data.values())]
print(json.dumps(data, indent=2))
total_seconds = data["total_duration"] / 1_000_000_000
print(f"{total_seconds:.2f}s")