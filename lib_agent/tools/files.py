# tools/files.py — sandboxed file tools. Same semantics as agent/tools/files.py,
# rewritten in LangChain @tool style so ToolNode handles schema + validation.

import shutil
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


# ── bash-equivalents (pure Python, sandboxed) ─────────────────────────────
# Implemented without shelling out — no subprocess, no shell metacharacters
# to escape, no injection risk. Same _safe_path defense as read/write.

@tool
def list_directory(path: str = ".") -> str:
    """List files and subdirectories in a workspace directory (sorted, dirs first).

    Args:
        path: directory relative to the workspace root. Defaults to root.
    """
    safe = _safe_path(path)
    if safe is None:
        return "[error: path outside workspace or invalid]"
    if not safe.exists():
        return f"[error: path does not exist — {path}]"
    if not safe.is_dir():
        return f"[error: not a directory — {path}]"
    try:
        entries = sorted(safe.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception as e:
        return f"[error listing directory: {e}]"
    lines = [f"{p.name}{'/' if p.is_dir() else ''}" for p in entries]
    return "\n".join(lines) if lines else "[empty directory]"


@tool
def make_directory(path: str) -> str:
    """Create a directory in the workspace (idempotent; mkdir -p semantics).

    Args:
        path: directory path relative to the workspace root.
    """
    safe = _safe_path(path)
    if safe is None:
        return "[error: path outside workspace or invalid]"
    try:
        safe.mkdir(parents=True, exist_ok=True)
        rel = safe.relative_to(Path(WORKSPACE_DIR).resolve())
        return f"Created directory {rel}"
    except Exception as e:
        return f"[error creating directory: {e}]"


@tool
def find_files(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """Find files matching a glob pattern under a workspace directory.

    Args:
        pattern: glob pattern (e.g. "*.md" for top level, "**/*.txt" for recursive).
        path: starting directory relative to the workspace root.
        max_results: cap on returned matches (clamped to 500).
    """
    safe = _safe_path(path)
    if safe is None:
        return "[error: path outside workspace or invalid]"
    if not safe.is_dir():
        return f"[error: not a directory — {path}]"
    cap = max(1, min(max_results, 500))  # cap result count to bound output
    workspace = Path(WORKSPACE_DIR).resolve()
    try:
        matches: list[str] = []
        for i, m in enumerate(safe.rglob(pattern)):
            if i >= cap:
                matches.append(f"…[truncated at {cap}]")
                break
            try:
                matches.append(str(m.relative_to(workspace)))
            except ValueError:
                continue  # any path that escaped is silently dropped
    except Exception as e:
        return f"[error searching: {e}]"
    return "\n".join(matches) if matches else "[no matches]"


@tool
def delete_file(path: str) -> str:
    """Delete a single file from the workspace. Refuses directories.

    Args:
        path: file path relative to the workspace root.
    """
    safe = _safe_path(path)
    if safe is None:
        return "[error: path outside workspace or invalid]"
    if not safe.exists():
        return f"[error: file does not exist — {path}]"
    if safe.is_dir():
        return f"[error: refusing to delete a directory — {path}]"
    try:
        safe.unlink()
        rel = safe.relative_to(Path(WORKSPACE_DIR).resolve())
        return f"Deleted {rel}"
    except Exception as e:
        return f"[error deleting file: {e}]"


@tool
def copy_file(src: str, dst: str) -> str:
    """Copy a file inside the workspace. Refuses directory copies. If dst
    exists, it is overwritten. Creates intermediate directories under dst.

    Args:
        src: source file path, relative to the workspace root.
        dst: destination file path, relative to the workspace root.
    """
    safe_src = _safe_path(src)
    safe_dst = _safe_path(dst)
    if safe_src is None or safe_dst is None:
        return "[error: path outside workspace or invalid]"
    if not safe_src.exists():
        return f"[error: source file does not exist — {src}]"
    if safe_src.is_dir():
        return f"[error: refusing to copy a directory — {src}]"
    try:
        safe_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(safe_src), str(safe_dst))
        workspace = Path(WORKSPACE_DIR).resolve()
        return f"Copied {safe_src.relative_to(workspace)} → {safe_dst.relative_to(workspace)}"
    except Exception as e:
        return f"[error copying file: {e}]"


@tool
def move_file(src: str, dst: str) -> str:
    """Move or rename a file inside the workspace. Refuses directory moves.
    If dst exists, it is overwritten. Creates intermediate directories under dst.

    Args:
        src: source file path, relative to the workspace root.
        dst: destination file path, relative to the workspace root.
    """
    safe_src = _safe_path(src)
    safe_dst = _safe_path(dst)
    if safe_src is None or safe_dst is None:
        return "[error: path outside workspace or invalid]"
    if not safe_src.exists():
        return f"[error: source file does not exist — {src}]"
    if safe_src.is_dir():
        return f"[error: refusing to move a directory — {src}]"
    try:
        safe_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(safe_src), str(safe_dst))
        workspace = Path(WORKSPACE_DIR).resolve()
        return f"Moved {src} → {safe_dst.relative_to(workspace)}"
    except Exception as e:
        return f"[error moving file: {e}]"


@tool
def edit_file(
    filepath: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Surgical string replacement in a workspace file. Reads the file,
    replaces `old_string` with `new_string`, writes it back.

    Safety: by default `old_string` must appear EXACTLY ONCE in the file —
    otherwise the call is rejected. This prevents accidentally clobbering
    matching text in unrelated places. To replace every occurrence, set
    `replace_all=True`. To pick one of several occurrences, include enough
    surrounding context in `old_string` to make it unique.

    Args:
        filepath: path of the file, relative to the workspace root.
        old_string: text to find. Must be non-empty and (by default) unique.
        new_string: replacement text. Must differ from old_string.
        replace_all: replace every occurrence (default: only when unique).
    """
    safe = _safe_path(filepath)
    if safe is None:
        return "[error: path outside workspace or invalid]"
    if not safe.exists():
        return f"[error: file does not exist — {filepath}]"
    if safe.is_dir():
        return f"[error: not a file — {filepath}]"
    if not old_string:
        return "[error: old_string is empty]"
    if old_string == new_string:
        return "[error: old_string and new_string are identical — no edit needed]"
    try:
        content = safe.read_text(encoding="utf-8")
    except Exception as e:
        return f"[error reading file: {e}]"

    count = content.count(old_string)
    if count == 0:
        return f"[error: old_string not found in {filepath}]"
    if count > 1 and not replace_all:
        return (
            f"[error: old_string appears {count} times in {filepath}; "
            f"set replace_all=True to replace all, or include more context "
            f"in old_string to make it unique]"
        )

    new_content = content.replace(old_string, new_string)
    try:
        safe.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"[error writing file: {e}]"

    workspace = Path(WORKSPACE_DIR).resolve()
    return f"Replaced {count} occurrence(s) in {safe.relative_to(workspace)}"
