# tools/files.py — sandboxed file tools. Same semantics as agent/tools/files.py,
# rewritten in LangChain @tool style so ToolNode handles schema + validation.

from pathlib import Path

from langchain_core.tools import tool

from config import WORKSPACE_DIR


def _safe_path(user_path: str) -> Path | None:
    """Resolve user_path under WORKSPACE_DIR; reject anything escaping it.

    Handles ../ traversal, absolute paths, and symlink trickery via resolve()
    before the is_relative_to() check.
    """
    workspace = Path(WORKSPACE_DIR).resolve()
    candidate = (workspace / user_path).resolve()
    if not candidate.is_relative_to(workspace):
        return None
    return candidate


@tool
def read_file(filepath: str) -> str:
    """Read a UTF-8 text file from the local workspace.

    Args:
        filepath: path of the file, relative to the workspace root.
    """
    safe = _safe_path(filepath)
    if safe is None:
        return "[error: path outside workspace or invalid]"
    try:
        return safe.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[error: file not found — {filepath}]"
    except Exception as e:
        return f"[error reading file: {e}]"


@tool
def write_file(filepath: str, content: str) -> str:
    """Write text into a file in the local workspace. Overwrites by default.
    Creates parent directories if missing.

    Args:
        filepath: path of the file, relative to the workspace root.
        content: text to write into the file (may be empty).
    """
    safe = _safe_path(filepath)
    if safe is None:
        return "[error: path outside workspace or invalid]"
    try:
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {safe.relative_to(Path(WORKSPACE_DIR).resolve())}"
    except Exception as e:
        return f"[error writing file: {e}]"
