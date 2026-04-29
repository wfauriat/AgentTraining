# tools/meta.py — trivial utility tools
from datetime import datetime


def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().isoformat()


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

    return f"[error: unknown tool '{tool_name}']"