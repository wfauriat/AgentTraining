# tools/meta.py — trivial utility tools
from datetime import datetime


def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().isoformat()

def finish(message: str) -> str:
    return message


# Tool schema sent to Ollama.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this when the user's request is fully completed. "
                "Pass a final summary message to show to the user. "
                "Calling this tool ends the conversation turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description" : "The final message once the task is finished."
                    }
                },
                "required": ["message"],
            },
        },
    }
]

def dispatch(tool_name: str, arguments: dict) -> str:
    """
    Execute a tool by name and return its result as a string.
    Returns an error string if the tool is unknown — the model
    will see this and can recover gracefully.
    """
    if tool_name == "get_current_time":
        return get_current_time()
    if tool_name == "finish":
        message = arguments.get("message", "")
        return finish(message)

    return f"[error: unknown tool '{tool_name}']"