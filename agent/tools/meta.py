# tools/meta.py — trivial utility tools
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError


# ── Argument schemas ───────────────────────────────────────────────────────
# Each tool's inputs are described once as a Pydantic model. The dispatcher
# uses it to validate before calling the tool function. If validation fails,
# we return a string the model can read and recover from.

class FinishArgs(BaseModel):
    message: str = Field(..., description="The final message once the task is finished.")


# ── Tool implementations ───────────────────────────────────────────────────

def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().isoformat()


def finish(message: str) -> str:
    return message


# ── Schema sent to Ollama ──────────────────────────────────────────────────

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
            "parameters": FinishArgs.model_json_schema(),
        },
    },
]


# ── Dispatch ───────────────────────────────────────────────────────────────

def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "get_current_time":
        return get_current_time()

    if tool_name == "finish":
        try:
            args = FinishArgs(**arguments)
        except ValidationError as e:
            return f"[error: invalid arguments for finish — {e}]"
        return finish(args.message)

    return f"[error: unknown tool '{tool_name}']"