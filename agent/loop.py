# loop.py — the agent loop
# Responsible for: sending messages, detecting tool calls,
# dispatching tools, and knowing when to stop.

import time

import httpx

import observability as obs
from config import MODEL, OLLAMA_URL, MAX_TURNS, MODEL_TIMEOUT


def run(messages: list, tools: list, dispatch_table: dict) -> str:
    """
    Run the agent loop for one user turn.

    Sends the current message history to the model, handles any
    tool calls, and returns the final text reply. Returns an
    error string if MAX_TURNS is reached or the model call fails.
    """
    for turn in range(MAX_TURNS):
        # — Send current history to the model —
        t0 = time.time()
        try:
            response = httpx.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                    "tools": tools,
                },
                timeout=MODEL_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            return f"[agent error: model call failed — {e}]"

        payload = response.json()["message"]
        obs.log_model_call(messages, time.time() - t0, payload)

        # — Branch: tool call or final text reply? —
        if payload.get("tool_calls"):
            # Append the model's tool call to history first
            messages.append(payload)

            # Handle every tool call in the response (model may request several)
            for call in payload["tool_calls"]:
                tool_name = call["function"]["name"]
                arguments = call["function"].get("arguments", {})

                if tool_name == "finish":
                    print(f"  [tool call: {tool_name}({arguments})]")
                    final = arguments.get("message", "[agent finished]")
                    obs.log_tool_call(tool_name, arguments, final, 0.0)
                    return final

                print(f"  [tool call: {tool_name}({arguments})]")
                t1 = time.time()
                if tool_name not in dispatch_table:
                    result = f"[error: unknown tool '{tool_name}']"
                else:
                    result = dispatch_table[tool_name](tool_name, arguments)
                obs.log_tool_call(tool_name, arguments, result, time.time() - t1)

                # Append tool result — model reads this on the next turn
                messages.append({"role": "tool", "content": result})

        else:
            # Model produced a text reply — we're done for this turn
            reply = payload["content"]
            if not reply.strip():
                continue  # keep going, model just produced filler
            messages.append({"role": "assistant", "content": reply})
            return reply

    # Reached turn limit without a text reply
    return f"[agent stopped: reached {MAX_TURNS}-turn limit]"