from pathlib import Path
from config import WORKSPACE_DIR

def safe_path(user_path: str) -> Path | None:
    workspace = Path(WORKSPACE_DIR).resolve()
    candidate = (workspace / user_path).resolve()
    
    # Is candidate inside workspace? Use is_relative_to (Python 3.9+)
    if not candidate.is_relative_to(workspace):
        return None
    return candidate


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
        with open(safepath, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Content successfully written in file {safepath}"
    except Exception as e:
        return f"[error writing file: {e}]"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of the file in the local workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The path of the file to read",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write into a file in the local workspace. Default behavior is overriding the present content of the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The path of the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write into the file."
                    }
                },
                "required": ["filepath", "content"],
            },
        },
    },
]


def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "read_file":
        filepath = arguments.get("filepath", "")
        if not filepath:
            return "[error: read_file requires a filepath argument]"
        return read_file(filepath)

    if tool_name == "write_file":
        filepath = arguments.get("filepath", "")
        content = arguments.get("content", "")
        if not filepath:
            return "[error: write_file requires a filepath argument]"
        return write_file(filepath, content)

    return f"[error: unknown tool '{tool_name}']"