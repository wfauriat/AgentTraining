# tools/files.py — sandboxed read/write inside WORKSPACE_DIR
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from config import WORKSPACE_DIR


# ── Argument schemas ───────────────────────────────────────────────────────

class ReadFileArgs(BaseModel):
    filepath: str = Field(..., min_length=1, description="The path of the file to read")


class WriteFileArgs(BaseModel):
    filepath: str = Field(..., min_length=1, description="The path of the file to write")
    content: str = Field("", description="The content to write into the file (may be empty)")


# ── Sandbox helper ─────────────────────────────────────────────────────────

def safe_path(user_path: str) -> Path | None:
    workspace = Path(WORKSPACE_DIR).resolve()
    candidate = (workspace / user_path).resolve()
    if not candidate.is_relative_to(workspace):
        return None
    return candidate


# ── Tool implementations ───────────────────────────────────────────────────

def read_file(filepath: str) -> str:
    safepath = safe_path(filepath)
    if safepath is None:
        return "[error: path outside workspace or invalid]"
    try:
        with open(safepath, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"[error reading file: {e}]"


def write_file(filepath: str, content: str) -> str:
    safepath = safe_path(filepath)
    if safepath is None:
        return "[error: path outside workspace or invalid]"
    try:
        # Make sure parent directory exists for paths like "subdir/notes.txt"
        safepath.parent.mkdir(parents=True, exist_ok=True)
        with open(safepath, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Content successfully written in file {safepath}"
    except Exception as e:
        return f"[error writing file: {e}]"


# ── Schemas sent to Ollama ─────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file in the local workspace.",
            "parameters": ReadFileArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write into a file in the local workspace. Default behavior is overwriting the present content of the file.",
            "parameters": WriteFileArgs.model_json_schema(),
        },
    },
]


# ── Dispatch ───────────────────────────────────────────────────────────────

def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "read_file":
        try:
            args = ReadFileArgs(**arguments)
        except ValidationError as e:
            return f"[error: invalid arguments for read_file — {e}]"
        return read_file(args.filepath)

    if tool_name == "write_file":
        try:
            args = WriteFileArgs(**arguments)
        except ValidationError as e:
            return f"[error: invalid arguments for write_file — {e}]"
        return write_file(args.filepath, args.content)

    return f"[error: unknown tool '{tool_name}']"