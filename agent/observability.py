# observability.py — minimal JSONL session logger.
#
# Every model call and every tool call is written as one JSON line to
# logs/<timestamp>.jsonl. Cheap to enable, invaluable when something
# weird happens.
#
# Design notes:
#   - Session-scoped file: one file per CLI run, started in start_session().
#   - Synchronous writes: one line per event, flushed immediately. Good
#     enough for v1; if throughput ever becomes an issue, batch later.
#   - No-op gracefully if start_session() was never called — keeps the
#     logger optional and safe to import from anywhere.

import json
import time
from datetime import datetime
from pathlib import Path

_log_path: Path | None = None
_session_started_at: float | None = None


def start_session(log_dir: str = "./logs") -> Path:
    """Open a new log file for this session and return its path."""
    global _log_path, _session_started_at

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = Path(log_dir) / f"session_{timestamp}.jsonl"
    _session_started_at = time.time()

    _write({
        "event": "session_start",
        "log_path": str(_log_path),
    })
    return _log_path


def log_model_call(messages: list, latency_s: float, response_payload: dict) -> None:
    """Record a model round-trip."""
    tool_calls = response_payload.get("tool_calls") or []
    _write({
        "event": "model_call",
        "latency_s": round(latency_s, 3),
        "n_messages": len(messages),
        "n_tool_calls": len(tool_calls),
        "tool_call_names": [c["function"]["name"] for c in tool_calls],
        "content_preview": (response_payload.get("content") or "")[:200],
    })


def log_tool_call(
    tool_name: str,
    arguments: dict,
    result: str,
    latency_s: float,
    error: str | None = None,
) -> None:
    """Record one tool execution."""
    _write({
        "event": "tool_call",
        "tool": tool_name,
        "arguments": arguments,
        "result_preview": result[:300],
        "result_length": len(result),
        "latency_s": round(latency_s, 3),
        "error": error,
    })


def log_user_message(content: str) -> None:
    _write({"event": "user_message", "content": content})


def log_agent_reply(reply: str) -> None:
    _write({"event": "agent_reply", "reply": reply})


def _write(record: dict) -> None:
    """Append one JSON line. Silently no-ops if no session is active."""
    if _log_path is None:
        return
    record["t"] = round(time.time() - (_session_started_at or 0), 3)
    record["wall_time"] = datetime.now().isoformat()
    with open(_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")